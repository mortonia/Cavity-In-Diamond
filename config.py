import numpy as np
from ase.io import read
from molecule_utils import compute_cavity_radii

USE_MOLECULAR_CRYSTAL = False
MOLECULAR_CRYSTAL_FILE = 'ice_unitcell.xyz'
MOLECULE_PATH = 'molecules_xyz/propiolic.xyz'
molecule = read(MOLECULE_PATH)

ELEMENT = 'Ar'
LATTICE_CONSTANT = 5.30
REPEAT = (10, 10, 10)
CAVITY_BUFFER = 3.0
CAVITY_RADII = compute_cavity_radii(molecule, CAVITY_BUFFER)
DISPLACEMENT_THRESHOLD = 0.1

LJ_PARAMS = {
    'C': {'epsilon':0.00284,'sigma':3.40},
    'H': {'epsilon':0.00236,'sigma':2.96},
    'O': {'epsilon':0.00674,'sigma':3.00},
    'N': {'epsilon':0.00739,'sigma':3.20},
    'Ar':{'epsilon':0.0103,'sigma':3.40},
    'Ne':{'epsilon':0.0031,'sigma':2.75},
}

OUTPUT_DIR = 'outputs'
