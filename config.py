# --- config.py ---
import numpy as np
from ase.io import read
from molecule_utils import compute_cavity_radii

# Path to the guest molecule file
MOLECULE_PATH = 'molecules_xyz/propiolic.xyz'

# Read molecule once for calculating default cavity sizes
molecule = read(MOLECULE_PATH)

# Crystal building parameters (defaults; CLI/INI can override)
ELEMENT = 'Ne'
LATTICE_CONSTANT = 4.46368   # Å
REPEAT = (9, 9, 9)

# Cavity parameters
CAVITY_BUFFER = 0.1          # Å
CAVITY_RADII = compute_cavity_radii(molecule, CAVITY_BUFFER)

# Relaxation settings
DISPLACEMENT_THRESHOLD = 0.1 # Å

# Lennard-Jones parameters by element
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
