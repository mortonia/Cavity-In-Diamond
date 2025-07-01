# molecule_utils.py

import numpy as np
from ase.io import read
from ase.constraints import FixAtoms

def load_and_center_molecule(filename, center):
    mol = read(filename)
    mol_center = mol.get_positions().mean(axis=0)
    mol.translate(center - mol_center)
    return mol

def compute_cavity_radii(molecule, buffer):
    positions = molecule.get_positions()
    min_pos = positions.min(axis=0)
    max_pos = positions.max(axis=0)
    size = max_pos - min_pos
    return size[0]/2 + buffer, size[1]/2 + buffer, size[2]/2 + buffer

def freeze_distant_molecules(structure, cavity_center, cutoff_distance, molecule_size=3):
    """
    Freeze all atoms in molecules (assumed size=N) whose center of mass is beyond cutoff_distance.
    """
    num_atoms = len(structure)
    freeze_indices = []
    tags = np.zeros(num_atoms, dtype=int)

    for i in range(0, num_atoms, molecule_size):
        group_indices = list(range(i, i + molecule_size))
        if max(group_indices) >= num_atoms:
            break

        group_positions = structure.get_positions()[group_indices]
        com = group_positions.mean(axis=0)
        distance = np.linalg.norm(com - cavity_center)

        if distance > cutoff_distance:
            freeze_indices.extend(group_indices)
            tags[group_indices] = 1  # tag frozen for visualization

    structure.set_tags(tags)
    structure.set_constraint(FixAtoms(indices=freeze_indices))
    print(f"[Freeze] {len(freeze_indices)} atoms frozen beyond {cutoff_distance} Å.")
    return freeze_indices

