import numpy as np
from ase.io import read
from ase.constraints import FixAtoms

def load_and_center_molecule(filename, center):
    mol = read(filename)
    mol.translate(center - mol.get_positions().mean(axis=0))
    return mol

def compute_cavity_radii(molecule, buffer):
    pos = molecule.get_positions()
    size = pos.max(axis=0) - pos.min(axis=0)
    return tuple(size / 2 + buffer)

def freeze_distant_molecules(structure, cavity_center, cutoff_distance, molecule_size=3):
    n = len(structure)
    freeze = []
    tags = np.zeros(n, int)
    for i in range(0, n, molecule_size):
        idx = range(i, min(i + molecule_size, n))
        com = structure.get_positions()[idx].mean(axis=0)
        if np.linalg.norm(com - cavity_center) > cutoff_distance:
            freeze.extend(idx)
            tags[list(idx)] = 1
    structure.set_tags(tags)
    structure.set_constraint(FixAtoms(indices=freeze))
    print(f"[Freeze] {len(freeze)} atoms frozen beyond {cutoff_distance} Å.")
    return freeze
