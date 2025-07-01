# config.py

# Toggle between atomic and molecular crystals
USE_MOLECULAR_CRYSTAL = False
MOLECULAR_CRYSTAL_FILE = 'ice_unitcell.xyz'

# For atomic crystal fallback
ELEMENT = 'C'
LATTICE_CONSTANT = 3.57  # Å

REPEAT = (7, 7, 7)
CAVITY_BUFFER = 2.0
DISPLACEMENT_THRESHOLD = 0.1

# Lennard-Jones parameters per element (eV, Å)
LJ_PARAMS = {
    'C': {'epsilon': 0.00284, 'sigma': 3.40},
    'H': {'epsilon': 0.00236, 'sigma': 2.96},
    'O': {'epsilon': 0.00674, 'sigma': 3.00},
    'N': {'epsilon': 0.00739, 'sigma': 3.20},
    'Ar': {'epsilon': 0.0103,  'sigma': 3.40},
}

