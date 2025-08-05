import numpy as np
from ase.io import read
from molecule_utils import compute_cavity_radii

# Path to the guest molecule file
MOLECULE_PATH = 'molecules_xyz/propiolic.xyz'

# Read molecule once for calculating default cavity sizes
molecule = read(MOLECULE_PATH)

# Crystal building parameters
ELEMENT = 'Ar'                 # Host crystal element
LATTICE_CONSTANT = 5.30        # Lattice constant in Ångstroms
REPEAT = (10, 10, 10)          # Number of unit cells in each direction

# Cavity parameters
CAVITY_BUFFER = 3.0            # Extra padding around molecule
CAVITY_RADII = compute_cavity_radii(molecule, CAVITY_BUFFER)

# Relaxation settings
DISPLACEMENT_THRESHOLD = 0.1   # Å threshold for reporting large moves

# Lennard-Jones potential parameters by element
LJ_PARAMS = {
    'C': {'epsilon': 0.00284, 'sigma': 3.40},
    'H': {'epsilon': 0.00236, 'sigma': 2.96},
    'O': {'epsilon': 0.00674, 'sigma': 3.00},
    'N': {'epsilon': 0.00739, 'sigma': 3.20},
    'Ar':{'epsilon': 0.0103,  'sigma': 3.40},
    'Ne':{'epsilon': 0.0031,  'sigma': 2.75},
}

# Directory for all output files
OUTPUT_DIR = 'outputs'

