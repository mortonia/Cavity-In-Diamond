# --- crystal_utils.py ---
import numpy as np
from ase.build import bulk
from ase import Atoms
from ase.neighborlist import NeighborList
from config import CAVITY_RADII

# =========================
# Builder & geometry utils
# =========================

def create_crystal(element, crystal_structure, size, lattice_constant):
    """Build a finite cluster (no PBC) for the requested structure.
    crystal_structure: 'fcc' | 'bcc' | 'hcp' | 'diamond'
    """
    crystal = bulk(element, crystalstructure=crystal_structure, a=lattice_constant).repeat(size)
    crystal.set_pbc(False)
    return crystal


def remove_unbound_hydrogens(atoms, max_bond=1.2):
    """Remove H atoms that aren't bonded to any heavy atom."""
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


def create_cavity(atoms, center=None, radii=CAVITY_RADII, eps=1e-9):
    """Remove atoms STRICTLY inside the ellipsoid; boundary atoms are kept."""
    if center is None:
        center = atoms.get_positions().mean(axis=0)
    rx, ry, rz = radii
    inv2 = np.array([1/rx**2, 1/ry**2, 1/rz**2])
    keep = []
    for i, pos in enumerate(atoms.get_positions()):
        val = ((pos - center)**2) @ inv2
        if not (val < 1.0 - eps):  # strict interior only
            keep.append(i)
    return atoms[keep]


def ellipsoid_atoms(atoms, center=None, radii=CAVITY_RADII, eps=0.0):
    """Return atoms inside the ellipsoid (optionally strict via eps)."""
    if center is None:
        center = atoms.get_positions().mean(axis=0)
    rx, ry, rz = radii
    inv2 = np.array([1/rx**2, 1/ry**2, 1/rz**2])
    idx = []
    for i, pos in enumerate(atoms.get_positions()):
        val = ((pos - center)**2) @ inv2
        if val < 1.0 - eps:
            idx.append(i)
    return atoms[idx]


def insert_molecule(crystal, molecule, center=True, cavity_center=None,
                    cavity_radii=None, safety_margin=1.0, check_fit=True):
    """Place molecule at cavity_center; optionally verify ellipsoidal fit."""
    mol = molecule.copy()
    if cavity_center is None:
        pos = crystal.get_positions()
        cavity_center = 0.5 * (pos.min(axis=0) + pos.max(axis=0))
    if center:
        mpos = mol.get_positions()
        mcenter = 0.5 * (mpos.max(axis=0) + mpos.min(axis=0))
        mol.translate(-mcenter)
    mol.translate(cavity_center)
    if check_fit and (cavity_radii is not None):
        rx, ry, rz = np.array(cavity_radii) * safety_margin
        rel = mol.get_positions() - cavity_center
        if np.any((np.abs(rel[:,0])>rx) | (np.abs(rel[:,1])>ry) | (np.abs(rel[:,2])>rz)):
            max_ext = np.max(np.abs(rel), axis=0)
            raise ValueError(f"Molecule does not fit: extents {max_ext}, radii {(rx,ry,rz)}")
    return crystal + mol


def extract_spherical_region(structure, molecule_indices, radius):
    pos = structure.get_positions()
    mcenter = pos[molecule_indices].mean(axis=0)
    dist = np.linalg.norm(pos - mcenter, axis=1)
    return structure[dist <= radius]


def align_single_molecule(mol, atom1_idx, atom2_idx, target_axis):
    mol = mol.copy()
    pos = mol.get_positions()
    vec = pos[atom2_idx] - pos[atom1_idx]
    vec /= np.linalg.norm(vec)
    targ = target_axis / np.linalg.norm(target_axis)
    if np.allclose(vec, targ):
        return mol
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
    atoms = atoms.copy()
    p1 = atoms.positions[molecule_indices[atom1_idx]]
    p2 = atoms.positions[molecule_indices[atom2_idx]]
    vec = p2-p1; vec /= np.linalg.norm(vec)
    targ = target_axis/np.linalg.norm(target_axis)
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
    pos = structure.get_positions()
    mol_pos = pos[molecule_indices]
    cry_pos = np.delete(pos, molecule_indices, axis=0)
    dmat = np.linalg.norm(cry_pos[:,None,:] - mol_pos[None,:,:], axis=2)
    i,j = divmod(np.argmin(dmat), mol_pos.shape[0])
    return dmat.min(), j, i


def ellipsoid_markers(center, radii, n_theta=18, n_phi=36, symbol='X'):
    a,b,c = radii; ctr = np.array(center)
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

# ----- placement helpers -----

def supercell_center_from_fractional(crystal, frac_xyz):
    pos = crystal.get_positions(); lo = pos.min(axis=0); hi = pos.max(axis=0)
    L = hi - lo
    return lo + np.array(frac_xyz, dtype=float) * L


def midplane_center(crystal, axis, index, lattice_constant, repeat):
    pos = crystal.get_positions(); lo = pos.min(axis=0)
    nx, ny, nz = repeat; a = lattice_constant
    c = lo.copy()
    if axis == 'x':
        if not (0 <= index < nx - 1):
            raise ValueError(f"index must be in [0, {nx-2}] for axis x")
        c[0] = lo[0] + (index + 0.5) * a
        c[1] = lo[1] + 0.5 * ny * a
        c[2] = lo[2] + 0.5 * nz * a
    elif axis == 'y':
        if not (0 <= index < ny - 1):
            raise ValueError(f"index must be in [0, {ny-2}] for axis y")
        c[0] = lo[0] + 0.5 * nx * a
        c[1] = lo[1] + (index + 0.5) * a
        c[2] = lo[2] + 0.5 * nz * a
    elif axis == 'z':
        if not (0 <= index < nz - 1):
            raise ValueError(f"index must be in [0, {nz-2}] for axis z")
        c[0] = lo[0] + 0.5 * nx * a
        c[1] = lo[1] + 0.5 * ny * a
        c[2] = lo[2] + (index + 0.5) * a
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")
    return c

# ----- single-vacancy helpers -----

def _inside_ellipsoid_mask(positions, center, radii, eps=0.0):
    rx, ry, rz = radii
    inv2 = np.array([1.0/max(rx,1e-12)**2, 1.0/max(ry,1e-12)**2, 1.0/max(rz,1e-12)**2])
    rel = positions - center
    val = (rel**2) @ inv2
    return val < (1.0 - eps)


def bias_toward_single_nearest(crystal, center, a, frac=1e-3):
    pos = crystal.get_positions()
    d = np.linalg.norm(pos - center, axis=1)
    j0, j1 = np.argsort(d)[:2]
    if abs(d[j0] - d[j1]) < 1e-8:
        v = pos[j0] - center; n = np.linalg.norm(v)
        if n > 0:
            center = center + (frac * a) * (v / n)
    return center


def carve_exact_k(atoms, center, radii, k=1, max_iter=60, eps=1e-12):
    pos = atoms.get_positions()
    lo, hi = 0.0, 1.0
    chosen_scale = None
    for _ in range(max_iter):
        s = 0.5 * (lo + hi)
        mask = _inside_ellipsoid_mask(pos, center, np.array(radii) * s, eps=eps)
        cnt = int(mask.sum())
        if cnt > k:
            hi = s
        elif cnt < k:
            lo = s
        else:
            chosen_scale = s
            break
    if chosen_scale is None:
        d = np.linalg.norm(pos - center, axis=1)
        idx = np.argsort(d)[:k]
        keep = np.ones(len(atoms), dtype=bool); keep[idx] = False
        return atoms[keep], idx, (0.0,)
    mask = _inside_ellipsoid_mask(pos, center, np.array(radii) * chosen_scale, eps=eps)
    remove_idx = np.where(mask)[0]
    keep = ~mask
    return atoms[keep], remove_idx, (chosen_scale,)


def create_cavity_single_atom(atoms, center):
    pos = atoms.get_positions()
    idx = int(np.argmin(np.linalg.norm(pos - center, axis=1)))
    keep = np.ones(len(atoms), dtype=bool); keep[idx] = False
    return atoms[keep], np.array([idx])
