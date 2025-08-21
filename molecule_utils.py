# --- molecule_utils.py ---
import numpy as np
from ase.io import read

# Utility functions for loading and manipulating the guest molecule

def load_and_center_molecule(filename, center):
    """
    Read a molecule from file and translate it so its geometric center
    lies at the specified `center` coordinate.
    """
    mol = read(filename)
    mol.translate(center - mol.get_positions().mean(axis=0))
    return mol


def compute_cavity_radii(molecule, buffer):
    """
    Determine semi-axes (rx, ry, rz) for an ellipsoidal cavity as
    half the molecule's bounding box plus a buffer.
    """
    pos = molecule.get_positions()
    size = pos.max(axis=0) - pos.min(axis=0)
    return tuple(size / 2 + buffer)
