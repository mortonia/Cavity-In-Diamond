import numpy as np
from ase.build import bulk
from ase import Atoms
from ase.neighborlist import NeighborList
from config import CAVITY_RADII, ELEMENT

# Functions to build and modify the host crystal cluster

def create_crystal(size, lattice_constant):
    """
    Generate a finite FCC crystal cluster of the specified element,
    repeated `size` times, with periodic boundary conditions turned off.
    """
    crystal = bulk(ELEMENT, 'fcc', a=lattice_constant).repeat(size)
    crystal.set_pbc(False)
    return crystal


def remove_unbound_hydrogens(atoms, max_bond=1.2):
    """
    Remove hydrogen atoms that are not bonded (within `max_bond` Å)
    to any heavier atom. Useful after carving out a cavity.
    """
    cutoffs = [0.5 if s=='H' else 0.8 for s in atoms.get_chemical_symbols()]
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)
    keep = []
    for i, atom in enumerate(atoms):
        if atom.symbol != 'H':
            keep.append(i)
        else:
            neigh, _ = nl.get_neighbors(i)
            if any(atoms[j].symbol != 'H' for j in neigh):
                keep.append(i)
    return atoms[keep]


def create_cavity(atoms, center=None, radii=CAVITY_RADII):
    """
    Remove all atoms within the ellipsoid defined by `center` and `radii`.
    Returns the hollowed-out crystal cluster.
    """
    if center is None:
        center = atoms.get_positions().mean(axis=0)
    rx, ry, rz = radii
    mask = [i for i, pos in enumerate(atoms.get_positions())
            if ((pos - center)**2 @ np.array([1/rx**2, 1/ry**2, 1/rz**2])) > 1]
    return atoms[mask]


def ellipsoid_atoms(atoms, center=None, radii=CAVITY_RADII):
    """
    Extract the atoms inside the ellipsoidal cavity region for analysis.
    """
    if center is None:
        center = atoms.get_positions().mean(axis=0)
    rx, ry, rz = radii
    mask = [i for i, pos in enumerate(atoms.get_positions())
            if ((pos - center)**2 @ np.array([1/rx**2, 1/ry**2, 1/rz**2])) <= 1]
    return atoms[mask]


def insert_molecule(crystal, molecule, center=True, cavity_center=None,
                    cavity_radii=CAVITY_RADII, safety_margin=1.0):
    """
    Place the guest molecule into the cavity:
      1. Optionally recenter molecule at origin,
      2. Compute or use given cavity center,
      3. Translate molecule into the cavity,
      4. Check that no atoms protrude beyond the cavity (with margin).
    """
    mol = molecule.copy()
    # Determine cavity center from crystal bounding box
    if cavity_center is None:
        pos = crystal.get_positions()
        minc, maxc = pos.min(axis=0), pos.max(axis=0)
        cavity_center = (minc + maxc) / 2
    # Recentering molecule if desired
    if center:
        mpos = mol.get_positions()
        mcenter = (mpos.max(axis=0) + mpos.min(axis=0)) / 2
        mol.translate(-mcenter)
    # Translate into cavity
    mol.translate(cavity_center)
    # Fit check
    rx, ry, rz = np.array(cavity_radii) * safety_margin
    rel = mol.get_positions() - cavity_center
    if np.any((np.abs(rel[:,0])>rx) | (np.abs(rel[:,1])>ry) | (np.abs(rel[:,2])>rz)):
        max_ext = np.max(np.abs(rel), axis=0)
        raise ValueError(f"Molecule does not fit: extents {max_ext}, radii {(rx,ry,rz)}")
    # Combine and return
    return crystal + mol


def extract_spherical_region(structure, molecule_indices, radius):
    """
    Return atoms within `radius` of the molecule's center of mass for a reduced fragment.
    """
    pos = structure.get_positions()
    mcenter = pos[molecule_indices].mean(axis=0)
    dist = np.linalg.norm(pos - mcenter, axis=1)
    return structure[dist <= radius]


def align_single_molecule(mol, atom1_idx, atom2_idx, target_axis):
    """
    Rotate a molecule so the vector between two atom indices aligns with `target_axis`.
    Useful for orienting the guest before insertion.
    """
    mol = mol.copy()
    pos = mol.get_positions()
    vec = pos[atom2_idx] - pos[atom1_idx]
    vec /= np.linalg.norm(vec)
    targ = target_axis / np.linalg.norm(target_axis)
    if np.allclose(vec, targ):
        return mol  # Already aligned
    # Build Rodrigues rotation
    v = np.cross(vec, targ)
    s = np.linalg.norm(v)
    c = np.dot(vec, targ)
    vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    R = np.eye(3) + vx + vx@vx * ((1-c)/(s**2))
    cm = mol.get_center_of_mass()
    mol.set_positions((pos-cm) @ R.T + cm)
    return mol


def align_molecule_in_structure(atoms, molecule_indices, atom1_idx, atom2_idx,
                                target_axis, translate_to_origin=False):
    """
    Rotate the entire structure so the molecule's axis aligns with `target_axis`,
    then optionally translate molecule center to origin.
    """
    atoms = atoms.copy()
    p1 = atoms.positions[molecule_indices[atom1_idx]]
    p2 = atoms.positions[molecule_indices[atom2_idx]]
    vec = p2-p1
    vec /= np.linalg.norm(vec)
    targ = target_axis / np.linalg.norm(target_axis)
    if not np.allclose(vec, targ):
        cross = np.cross(vec, targ)
        if np.linalg.norm(cross) < 1e-6:
            axis, angle = np.array([1,0,0]), np.pi
        else:
            axis = cross / np.linalg.norm(cross)
            angle = np.arccos(np.clip(np.dot(vec, targ), -1, 1))
        K = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
        R = np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K@K)
        com = atoms.get_center_of_mass()
        atoms.set_positions((atoms.positions-com) @ R.T + com)
    if translate_to_origin:
        center = atoms.positions[molecule_indices].mean(axis=0)
        atoms.translate(-center)
    return atoms


def min_clearance_to_crystal(structure, molecule_indices, cutoff=8.0):
    """
    Compute the minimum distance between any molecule atom and any crystal atom.
    Returns (distance, index_in_molecule, index_in_crystal).
    """
    pos = structure.get_positions()
    mol_pos = pos[molecule_indices]
    cry_pos = np.delete(pos, molecule_indices, axis=0)
    dmat = np.linalg.norm(cry_pos[:,None,:] - mol_pos[None,:,:], axis=2)
    i,j = divmod(np.argmin(dmat), mol_pos.shape[0])
    return dmat.min(), j, i


def ellipsoid_markers(center, radii, n_theta=18, n_phi=36, symbol='X'):
    """
    Generate dummy atoms on the ellipsoid surface defined by `center` and `radii`.
    Useful for visualizing the cavity boundary.
    """
    a,b,c = radii
    ctr = np.array(center)
    thetas = np.linspace(0, np.pi, n_theta)
    phis = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    pts = []
    for th in thetas:
        for ph in phis:
            x = a*np.sin(th)*np.cos(ph)
            y = b*np.sin(th)*np.sin(ph)
            z = c*np.cos(th)
            pts.append(ctr + np.array([x,y,z]))
    from ase import Atoms as _A
    return _A([symbol]*len(pts), positions=np.array(pts))
