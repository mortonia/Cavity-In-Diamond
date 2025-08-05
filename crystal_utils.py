import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.neighborlist import NeighborList
from config import CAVITY_RADII, ELEMENT, LATTICE_CONSTANT, REPEAT


def create_crystal(size=REPEAT, lattice_constant=LATTICE_CONSTANT):
    crystal = bulk(ELEMENT, 'fcc', a=lattice_constant).repeat(size)
    crystal.set_pbc(False)
    return crystal


def remove_unbound_hydrogens(atoms, max_bond=1.2):
    cutoffs = [0.5 if s=='H' else 0.8 for s in atoms.get_chemical_symbols()]
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)
    keep = []
    for i,s in enumerate(atoms):
        if s.symbol!='H': keep.append(i)
        else:
            neigh,_ = nl.get_neighbors(i)
            if any(atoms[j].symbol!='H' for j in neigh): keep.append(i)
    return atoms[keep]


def create_cavity(atoms, center=None, radii=CAVITY_RADII):
    if center is None:
        center = atoms.get_positions().mean(axis=0)
    rx,ry,rz = radii
    return atoms[[i for i,pos in enumerate(atoms.get_positions())
                  if (pos-center)**2 @ np.array([1/rx**2,1/ry**2,1/rz**2])>1]]


def ellipsoid_atoms(atoms, center=None, radii=CAVITY_RADII):
    if center is None:
        center = atoms.get_positions().mean(axis=0)
    rx,ry,rz=radii
    return atoms[[i for i,pos in enumerate(atoms.get_positions())
                  if (pos-center)**2 @ np.array([1/rx**2,1/ry**2,1/rz**2])<=1]]


def min_clearance_to_crystal(structure, molecule_indices, cutoff=8.0):
    pos = structure.get_positions()
    mol = pos[molecule_indices]
    cry = np.delete(pos, molecule_indices, axis=0)
    dmin= np.linalg.norm((cry[:,None,:]-mol[None,:,:]),axis=2).min()
    i,j = divmod(np.argmin((cry[:,None,:]-mol[None,:,:])**2), mol.shape[0])
    return dmin, j, i
