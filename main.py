import os
import numpy as np
from ase.io import read, write
from ase.constraints import FixAtoms
from config import (
    ELEMENT, REPEAT, LATTICE_CONSTANT,
    DISPLACEMENT_THRESHOLD, OUTPUT_DIR,
    CAVITY_BUFFER, CAVITY_RADII, MOLECULE_PATH
)
from molecule_utils import load_and_center_molecule, compute_cavity_radii
from crystal_utils import (
    create_crystal, create_cavity, insert_molecule,
    remove_unbound_hydrogens, align_single_molecule,
    align_molecule_in_structure, extract_spherical_region,
    ellipsoid_atoms, min_clearance_to_crystal,
    ellipsoid_markers
)
from relaxation import relax_with_lj
from write_molecule_with_cavity_markers import write_molecule_with_cavity_markers

# Main workflow: builds crystal, carves cavity, inserts molecule, relaxes, and writes outputs

def main():
    # Load and center guest molecule at origin
    mol = load_and_center_molecule(MOLECULE_PATH, np.array([0.0, 0.0, 0.0]))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Build and save raw crystal cluster
    crystal = create_crystal(REPEAT, LATTICE_CONSTANT)
    write(os.path.join(OUTPUT_DIR, "step1_crystal.xyz"), crystal)

    # Step 2: Orient molecule along its key axis (atoms 3-4) to +Y
    mol = align_single_molecule(mol, atom1_idx=3, atom2_idx=4, target_axis=np.array([0.0,1.0,0.0]))

    # Step 3: Compute cavity shape from molecule and carve it out
    radii = compute_cavity_radii(mol, CAVITY_BUFFER)
    cavity_cryst = create_cavity(crystal, radii=radii)
    write(os.path.join(OUTPUT_DIR, "step2_cavity.xyz"), cavity_cryst)

    # Step 4: Insert oriented molecule into cavity and save
    combined = insert_molecule(cavity_cryst, mol, center=True, cavity_radii=radii)
    write(os.path.join(OUTPUT_DIR, "step3_molecule_inserted.xyz"), combined)

    # Step 5: Freeze molecule atoms, check clearance to crystal
    start_idx = len(cavity_cryst)
    mol_idx = list(range(start_idx, start_idx + len(mol)))
    combined.set_constraint(FixAtoms(indices=mol_idx))
    dmin, _, _ = min_clearance_to_crystal(combined, mol_idx)
    print(f"[Cavity check] Minimum distance = {dmin:.3f} Å")

    # Step 6: Relax local region with LJ and BFGS
    orig = combined.get_positions()
    relaxed, disp, _ = relax_with_lj(combined, orig, DISPLACEMENT_THRESHOLD,
                                      OUTPUT_DIR, molecule_indices=mol_idx)
    write(os.path.join(OUTPUT_DIR, "step5_relaxed_full.xyz"), relaxed)

    # Step 7: Strip out molecule and any unbound H, save crystal fragment
    frag = relaxed.copy()
    del frag[mol_idx]
    frag = remove_unbound_hydrogens(frag)
    write(os.path.join(OUTPUT_DIR, "step6_relaxed_crystal_cavity_no_molecule.xyz"), frag)

    # Step 8: Save guest molecule separately
    write(os.path.join(OUTPUT_DIR, "step7_molecule.xyz"), mol)

    # Step 9: Re-align full relaxed structure so molecule axis is +Y at origin
    aligned = align_molecule_in_structure(relaxed, mol_idx,
                                          atom1_idx=0, atom2_idx=1,
                                          target_axis=np.array([0.0,1.0,0.0]),
                                          translate_to_origin=True)
    write(os.path.join(OUTPUT_DIR, "step8_aligned_structure.xyz"), aligned)

    # Step 10: Extract spherical fragment around molecule, save
    sphere = extract_spherical_region(aligned, mol_idx, radius=12.0)
    write(os.path.join(OUTPUT_DIR, "step9_spherical_fragment.xyz"), sphere)


if __name__ == "__main__":
    main()
