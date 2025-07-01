import os
import sys
import shutil
import numpy as np
from ase.io import write
from ase import Atoms
from ase.constraints import FixAtoms

from config import ELEMENT, REPEAT, LATTICE_CONSTANT, CAVITY_BUFFER, DISPLACEMENT_THRESHOLD, USE_MOLECULAR_CRYSTAL
from crystal_utils import create_crystal, carve_elliptical_cavity
from molecule_utils import load_and_center_molecule, compute_cavity_radii
from relaxation import relax_with_lj
from output_utils import print_cell_dimensions, print_crystal_info


def write_plain_xyz(filename, atoms, title="Structure"):
    with open(filename, 'w') as f:
        f.write(f"{len(atoms)}\n")
        f.write(f"{title}\n")
        for s, pos in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
            f.write(f"{s:<3}  {pos[0]:>12.6f}  {pos[1]:>12.6f}  {pos[2]:>12.6f}\n")


def freeze_by_distance(atoms, center, cutoff):
    freeze_indices = []
    tags = np.zeros(len(atoms), dtype=int)
    for i, pos in enumerate(atoms.get_positions()):
        if np.linalg.norm(pos - center) > cutoff:
            freeze_indices.append(i)
            tags[i] = 1
    atoms.set_tags(tags)
    atoms.set_constraint(FixAtoms(indices=freeze_indices))
    print(f"[Freeze] {len(freeze_indices)} atoms frozen beyond {cutoff} Å.")
    return freeze_indices


def main():
    if len(sys.argv) > 1:
        molecule_filename = sys.argv[1]
    else:
        molecule_filename = 'water.xyz'

    xyz_title = sys.argv[5] if len(sys.argv) > 5 else "relaxed_structure"

    


    BASE_OUTPUT_DIR = "output"
    molecule_name = os.path.splitext(os.path.basename(molecule_filename))[0]
    OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, molecule_name)
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    crystal = create_crystal(ELEMENT, REPEAT, LATTICE_CONSTANT)
    print_cell_dimensions(crystal, output_dir=OUTPUT_DIR)
    print_crystal_info(crystal, label="Original", output_dir=OUTPUT_DIR)
    
    cavity_center = crystal.get_cell().sum(axis=0) / 2
    print(f"[Centering] Automatically using cavity center: {cavity_center}")

    if cavity_center is None:
        cavity_center = crystal.get_cell().sum(axis=0) / 2

    molecule = load_and_center_molecule(molecule_filename, cavity_center)
    radius_x, radius_y, radius_z = compute_cavity_radii(molecule, CAVITY_BUFFER)
    crystal_cavity = carve_elliptical_cavity(crystal, cavity_center, radius_x, radius_y, radius_z)
    print_crystal_info(crystal_cavity, label="After Cavity", output_dir=OUTPUT_DIR)

    combined = crystal_cavity + molecule

    if USE_MOLECULAR_CRYSTAL:
        from molecule_utils import freeze_distant_molecules
        freeze_distant_molecules(combined[:-len(molecule)], cavity_center, cutoff_distance=5.0, molecule_size=3)
    else:
        freeze_by_distance(combined[:-len(molecule)], cavity_center, cutoff=5.0)

    # Freeze guest molecule
    num_mol_atoms = len(molecule)
    mol_indices = list(range(len(combined) - num_mol_atoms, len(combined)))

    # Safely get all existing constraint indices
    existing_constraints = []
    for constraint in combined.constraints:
        if hasattr(constraint, 'get_indices'):
            existing_constraints.extend(constraint.get_indices())

    all_constraints = existing_constraints + mol_indices
    combined.set_constraint(FixAtoms(indices=all_constraints))

    original_positions = combined.get_positions().copy()
    write(os.path.join(OUTPUT_DIR, "before_lj.xyz"), combined)

    relaxed, displacements, num_displaced = relax_with_lj(
        combined, original_positions, DISPLACEMENT_THRESHOLD, OUTPUT_DIR
    )
    
    print(f"[Debug] Total atoms: {len(relaxed)}")
    
    # --- Split relaxed structure ---
    relaxed_positions = relaxed.get_positions()
    relaxed_symbols = relaxed.get_chemical_symbols()
    relaxed_cell = relaxed.get_cell()
    relaxed_pbc = relaxed.get_pbc()

    # Extract relaxed molecule
    relaxed_molecule = Atoms(
        symbols=relaxed_symbols[-num_mol_atoms:], 
        positions=relaxed_positions[-num_mol_atoms:],
        cell=relaxed_cell,
        pbc=relaxed_pbc
    )

    # Extract relaxed crystal (everything except the last num_mol_atoms)
    relaxed_crystal_only = Atoms(
        symbols=relaxed_symbols[:-num_mol_atoms],
        positions=relaxed_positions[:-num_mol_atoms],
        cell=relaxed_cell,
        pbc=relaxed_pbc
    )

    # Write each .xyz
    write_plain_xyz(os.path.join(OUTPUT_DIR, "relaxed_crystal_only.xyz"), relaxed_crystal_only, title="Relaxed Crystal (No Molecule)")
    write_plain_xyz(os.path.join(OUTPUT_DIR, "relaxed_molecule_only.xyz"), relaxed_molecule, title="Relaxed Molecule Only")

    
    # Identify frozen atoms
    frozen_indices = set()
    for constraint in relaxed.constraints:
        if hasattr(constraint, 'get_indices'):
            frozen_indices.update(constraint.get_indices())
            
    print(f"[Debug] Number of frozen atoms: {len(frozen_indices)}")
    print(f"[Debug] Sample frozen atom indices: {list(frozen_indices)[:10]}")

    displacements = np.linalg.norm(relaxed.get_positions() - original_positions, axis=1)
   

    movable_indices = [i for i in range(len(relaxed)) if i not in frozen_indices]
    displaced_movable = [i for i in movable_indices if displacements[i] > DISPLACEMENT_THRESHOLD]

    print(f"[Debug] Number of movable atoms: {len(movable_indices)}")
    print(f"[Debug] Number of displaced movable atoms: {len(displaced_movable)}")
    
    
    num_displaced_movable = len(displaced_movable)

    print(f"[LJ] Displaced MOVABLE atoms > {DISPLACEMENT_THRESHOLD} Å: {num_displaced_movable}")
    with open(os.path.join(OUTPUT_DIR, "info.txt"), "a") as f:
        f.write(f"[LJ] Displaced MOVABLE atoms > {DISPLACEMENT_THRESHOLD} Å: {num_displaced_movable}\n")

    # Tag displaced atoms
    tags = relaxed.get_tags()
    tags[displacements > DISPLACEMENT_THRESHOLD] = 2
    relaxed.set_tags(tags)

    from ase.io import Trajectory
    Trajectory(os.path.join(OUTPUT_DIR, "after_lj_full.traj"), 'w').write(relaxed)

    relaxed_stripped = Atoms(
        symbols=relaxed.get_chemical_symbols(),
        positions=relaxed.get_positions(),
        cell=relaxed.get_cell(),
        pbc=relaxed.get_pbc()
    )
    write_plain_xyz(os.path.join(OUTPUT_DIR, "after_lj.xyz"), relaxed_stripped, title=xyz_title)

    with open(os.path.join(OUTPUT_DIR, "displacements.txt"), "w") as f:
        f.write("Index  Displacement (Å)\n")
        for i, d in enumerate(displacements):
            if d > DISPLACEMENT_THRESHOLD:
                f.write(f"{i:<6}  {d:.6f}\n")


if __name__ == "__main__":
    main()

