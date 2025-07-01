# crystal_utils.py

import numpy as np
from ase.build import bulk
from ase.io import read
from config import USE_MOLECULAR_CRYSTAL, MOLECULAR_CRYSTAL_FILE

def create_crystal(element, repeat, lattice_constant):
    if USE_MOLECULAR_CRYSTAL:
        unitcell = read(MOLECULAR_CRYSTAL_FILE)

        # Fix for .xyz files with no cell info:
        unitcell.set_cell([[10, 0, 0], [0, 10, 0], [0, 0, 10]])
        unitcell.set_pbc([True, True, True])

        return unitcell.repeat(repeat)
    else:
        crystal = bulk(name=element, crystalstructure='fcc', a=lattice_constant)
        return crystal.repeat(repeat)

def carve_elliptical_cavity(crystal, center, radius_x, radius_y, radius_z):
    new_atoms = crystal.copy()
    positions = new_atoms.get_positions()
    mask = []
    for pos in positions:
        val = ((pos[0] - center[0]) / radius_x) ** 2 + \
              ((pos[1] - center[1]) / radius_y) ** 2 + \
              ((pos[2] - center[2]) / radius_z) ** 2
        mask.append(val > 1.0)
    return new_atoms[mask]

def tag_cavity_boundary_atoms(crystal, center, radius_x, radius_y, radius_z, delta=0.1):
    positions = crystal.get_positions()
    tags = np.zeros(len(crystal), dtype=int)

    for i, pos in enumerate(positions):
        val = ((pos[0] - center[0]) / radius_x) ** 2 + \
              ((pos[1] - center[1]) / radius_y) ** 2 + \
              ((pos[2] - center[2]) / radius_z) ** 2
        if 1.0 - delta <= val <= 1.0 + delta:
            tags[i] = 3  # Tag value 3 means cavity boundary

    crystal.set_tags(tags)
    return crystal

