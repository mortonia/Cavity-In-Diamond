# --- crystal_utils.py ---
import numpy as np
from ase.build import bulk
from ase import Atoms
from ase.neighborlist import NeighborList
from config import CAVITY_RADII, LJ_PARAMS

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
    cutoffs = [0.5 if s == 'H' else 0.8 for s in atoms.get_chemical_symbols()]
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
    inv2 = np.array([1.0 / rx**2, 1.0 / ry**2, 1.0 / rz**2])
    keep = []
    for i, pos in enumerate(atoms.get_positions()):
        val = ((pos - center) ** 2) @ inv2
        if not (val < 1.0 - eps):  # strict interior only
            keep.append(i)
    return atoms[keep]


def ellipsoid_atoms(atoms, center=None, radii=CAVITY_RADII, eps=0.0):
    """Return atoms inside the ellipsoid (optionally strict via eps)."""
    if center is None:
        center = atoms.get_positions().mean(axis=0)
    rx, ry, rz = radii
    inv2 = np.array([1.0 / rx**2, 1.0 / ry**2, 1.0 / rz**2])
    idx = []
    for i, pos in enumerate(atoms.get_positions()):
        val = ((pos - center) ** 2) @ inv2
        if val < 1.0 - eps:
            idx.append(i)
    return atoms[idx]


def insert_molecule(crystal,
                    molecule,
                    center=True,
                    cavity_center=None,
                    cavity_radii=None,
                    safety_margin=1.0,
                    check_fit=True,
                    center_strategy="com",
                    anchor_idx=None):
    """
    Place molecule at cavity_center; optionally verify ellipsoidal fit.

    center_strategy: "com" | "bbox" | "atom"
      - "com": anchor by center-of-mass
      - "bbox": anchor by bounding-box center
      - "atom": anchor by a specific atom index (anchor_idx required)
    """
    mol = molecule.copy()
    if cavity_center is None:
        pos = crystal.get_positions()
        cavity_center = 0.5 * (pos.min(axis=0) + pos.max(axis=0))

    if center:
        if center_strategy == "com":
            ref_point = mol.get_center_of_mass()
        elif center_strategy == "bbox":
            p = mol.get_positions()
            ref_point = 0.5 * (p.max(axis=0) + p.min(axis=0))
        elif center_strategy == "atom":
            if anchor_idx is None:
                raise ValueError("anchor_idx required when center_strategy='atom'")
            ref_point = mol.positions[int(anchor_idx)]
        else:
            raise ValueError(f"Unknown center_strategy='{center_strategy}'")
        mol.translate(-ref_point)

    mol.translate(cavity_center)

    if check_fit and (cavity_radii is not None):
        rx, ry, rz = np.array(cavity_radii) * safety_margin
        rel = mol.get_positions() - cavity_center
        if np.any((np.abs(rel[:, 0]) > rx) |
                  (np.abs(rel[:, 1]) > ry) |
                  (np.abs(rel[:, 2]) > rz)):
            max_ext = np.max(np.abs(rel), axis=0)
            raise ValueError(f"Molecule does not fit: extents {max_ext}, radii {(rx, ry, rz)}")
    return crystal + mol


def extract_spherical_region(structure, molecule_indices, radius):
    """Return atoms within `radius` of the molecule's center of mass."""
    pos = structure.get_positions()
    mcenter = pos[molecule_indices].mean(axis=0)
    dist = np.linalg.norm(pos - mcenter, axis=1)
    return structure[dist <= radius]

# =========================
# Alignment (robust)
# =========================

def _pick_alignment_pair(mol):
    """Choose two atom indices that define a good molecular axis.
    1) If exactly one heavy atom exists (non-H), use heavy -> farthest atom.
    2) Else, use the farthest pair of atoms.
    Returns (i, j) or None if the molecule has < 2 atoms.
    """
    n = len(mol)
    if n < 2:
        return None

    pos = mol.get_positions()
    symbols = mol.get_chemical_symbols()

    # Prefer a single heavy atom -> farthest neighbor (nice for e.g. H2O)
    heavy = [i for i, s in enumerate(symbols) if s != 'H']
    if len(heavy) == 1:
        i = heavy[0]
        d = np.linalg.norm(pos - pos[i], axis=1)
        j = int(np.argmax(d))
        if j == i:
            j = 1 if i == 0 else 0
        return (i, j)

    # Fallback: farthest pair
    max_d = -1.0
    pair = (0, 1)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pos[j] - pos[i])
            if d > max_d:
                max_d = d
                pair = (i, j)
    return pair


def align_single_molecule(
    mol,
    atom1_idx=None,
    atom2_idx=None,
    target_axis=np.array([0.0, 1.0, 0.0]),
    return_pair=False
):
    """Rotate a molecule so that (atom2-atom1) aligns with target_axis.
    If atom indices are None, choose a robust pair automatically.
    Returns the rotated molecule (and the pair if return_pair=True).
    """
    mol = mol.copy()
    n = len(mol)
    if n < 2:
        return (mol, None) if return_pair else mol

    if atom1_idx is None or atom2_idx is None:
        pair = _pick_alignment_pair(mol)
        if pair is None:
            return (mol, None) if return_pair else mol
        atom1_idx, atom2_idx = pair
    else:
        pair = (atom1_idx, atom2_idx)

    pos = mol.get_positions()
    vec = pos[atom2_idx] - pos[atom1_idx]
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return (mol, pair) if return_pair else mol

    vec /= norm
    targ = target_axis / np.linalg.norm(target_axis)

    if np.allclose(vec, targ, atol=1e-6):
        return (mol, pair) if return_pair else mol

    v = np.cross(vec, targ)
    s = np.linalg.norm(v)
    c = np.dot(vec, targ)
    if s < 1e-12:
        v = np.array([1.0, 0.0, 0.0])

    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / max(s**2, 1e-18))

    cm = mol.get_center_of_mass()
    mol.set_positions((pos - cm) @ R.T + cm)

    return (mol, pair) if return_pair else mol


def align_molecule_in_structure(atoms, molecule_indices, atom1_idx, atom2_idx,
                                target_axis, translate_to_origin=False):
    """Rotate whole structure so the selected molecular axis aligns with target_axis."""
    atoms = atoms.copy()
    p1 = atoms.positions[molecule_indices[atom1_idx]]
    p2 = atoms.positions[molecule_indices[atom2_idx]]
    vec = p2 - p1
    vec /= np.linalg.norm(vec)
    targ = target_axis / np.linalg.norm(target_axis)
    if not np.allclose(vec, targ):
        cross = np.cross(vec, targ)
        if np.linalg.norm(cross) < 1e-6:
            axis, angle = np.array([1, 0, 0]), np.pi
        else:
            axis = cross / np.linalg.norm(cross)
            angle = np.arccos(np.clip(np.dot(vec, targ), -1, 1))
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        com = atoms.get_center_of_mass()
        atoms.set_positions((atoms.positions - com) @ R.T + com)
    if translate_to_origin:
        center = atoms.positions[molecule_indices].mean(axis=0)
        atoms.translate(-center)
    return atoms


def min_clearance_to_crystal(structure, molecule_indices, cutoff=8.0):
    """Minimum distance between any molecule atom and any host atom."""
    pos = structure.get_positions()
    mol_pos = pos[molecule_indices]
    cry_pos = np.delete(pos, molecule_indices, axis=0)
    dmat = np.linalg.norm(cry_pos[:, None, :] - mol_pos[None, :, :], axis=2)
    i, j = divmod(np.argmin(dmat), mol_pos.shape[0])
    return dmat.min(), j, i


def ellipsoid_markers(center, radii, n_theta=18, n_phi=36, symbol='X'):
    """Dummy atoms on ellipsoid surface for visualization."""
    a, b, c = radii
    ctr = np.array(center)
    thetas = np.linspace(0, np.pi, n_theta)
    phis = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    pts = []
    for th in thetas:
        for ph in phis:
            x = a * np.sin(th) * np.cos(ph)
            y = b * np.sin(th) * np.sin(ph)
            z = c * np.cos(th)
            pts.append(ctr + np.array([x, y, z]))
    from ase import Atoms as _A
    return _A([symbol] * len(pts), positions=np.array(pts))

# ----- placement helpers -----

def supercell_center_from_fractional(crystal, frac_xyz):
    """Map fractional (0..1 across the whole supercell) to Cartesian center."""
    pos = crystal.get_positions()
    lo = pos.min(axis=0); hi = pos.max(axis=0)
    L = hi - lo
    return lo + np.array(frac_xyz, dtype=float) * L


def midplane_center(crystal, axis, index, lattice_constant, repeat):
    """Center on the plane halfway between unit cells 'index' and 'index+1' along axis."""
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
    inv2 = np.array([1.0 / max(rx, 1e-12)**2,
                     1.0 / max(ry, 1e-12)**2,
                     1.0 / max(rz, 1e-12)**2])
    rel = positions - center
    val = (rel ** 2) @ inv2
    return val < (1.0 - eps)


def bias_toward_single_nearest(crystal, center, a, frac=1e-3):
    """Nudge center slightly toward nearest atom to break midplane degeneracy."""
    pos = crystal.get_positions()
    d = np.linalg.norm(pos - center, axis=1)
    j0, j1 = np.argsort(d)[:2]
    if abs(d[j0] - d[j1]) < 1e-8:
        v = pos[j0] - center; n = np.linalg.norm(v)
        if n > 0:
            center = center + (frac * a) * (v / n)
    return center


def carve_exact_k(atoms, center, radii, k=1, max_iter=60, eps=1e-12):
    """Scale ellipsoid until EXACTLY k atoms are strictly inside (or fallback to k nearest)."""
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
    """Deterministically remove ONLY the single nearest atom to center."""
    pos = atoms.get_positions()
    idx = int(np.argmin(np.linalg.norm(pos - center, axis=1)))
    keep = np.ones(len(atoms), dtype=bool); keep[idx] = False
    return atoms[keep], np.array([idx])

# ----- clearance pre-trim (optional) -----

def _center_point(mol, strategy="com", anchor_idx=None):
    if strategy == "com":
        return mol.get_center_of_mass()
    elif strategy == "bbox":
        p = mol.get_positions(); return 0.5 * (p.max(axis=0) + p.min(axis=0))
    elif strategy == "atom":
        if anchor_idx is None:
            raise ValueError("anchor_idx required when strategy='atom'")
        return mol.positions[int(anchor_idx)]
    else:
        raise ValueError(f"Unknown strategy='{strategy}'")

def _rmin_lj(s1, s2):
    sigma = 0.5 * (LJ_PARAMS[s1]["sigma"] + LJ_PARAMS[s2]["sigma"])
    return (2.0 ** (1.0 / 6.0)) * sigma  # location of LJ minimum

def pretrim_host_for_clearance(
    crystal,
    molecule,
    cavity_center,
    mode="lj",
    min_clearance=3.0,
    rmin_scale=1.0,
    center_strategy="com",
    anchor_idx=None
):
    """
    Remove only those host atoms that would sit closer than a threshold to the
    molecule when inserted at `cavity_center`.

    mode = 'lj'   -> threshold_ij = rmin(sigma_mix(i,j)) * rmin_scale
           'fixed'-> threshold_ij = min_clearance (same for all pairs)
           'off'  -> do nothing

    Returns: (trimmed_crystal, removed_indices)
    """
    if mode == "off":
        return crystal, np.array([], dtype=int)

    # Place a copy of the molecule at the exact cavity center
    mol = molecule.copy()
    ref = _center_point(mol, center_strategy, anchor_idx)
    mol.translate(np.array(cavity_center, float) - ref)

    host_pos = crystal.get_positions()
    host_sym = crystal.get_chemical_symbols()
    mol_pos = mol.get_positions()
    mol_sym = mol.get_chemical_symbols()

    keep = np.ones(len(crystal), dtype=bool)

    if mode == "fixed":
        for i in range(len(crystal)):
            d = np.linalg.norm(mol_pos - host_pos[i], axis=1)
            if np.any(d < float(min_clearance)):
                keep[i] = False

    elif mode == "lj":
        for i in range(len(crystal)):
            ok = True
            for j in range(len(mol)):
                rmin = _rmin_lj(host_sym[i], mol_sym[j]) * float(rmin_scale)
                if np.linalg.norm(mol_pos[j] - host_pos[i]) < rmin:
                    ok = False
                    break
            keep[i] = ok
    else:
        raise ValueError("mode must be 'lj', 'fixed', or 'off'")

    trimmed = crystal[keep]
    removed = np.where(~keep)[0]
    print(f"[Clearance] Removed {len(removed)} host atoms to satisfy clearance (mode={mode}).")
    return trimmed, removed

# ----- orientation optimization for max clearance (optional) -----

def _rotate_about_com(mol, R):
    m = mol.copy()
    cm = m.get_center_of_mass()
    p = m.get_positions()
    m.set_positions((p - cm) @ R.T + cm)
    return m

def _rot_matrix_from_euler(yaw, pitch, roll):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Ry = np.array([[ cy, 0, sy],
                   [  0, 1,  0],
                   [-sy, 0, cy]])
    Rx = np.array([[1,  0,   0],
                   [0, cp, -sp],
                   [0, sp,  cp]])
    Rz = np.array([[cr, -sr, 0],
                   [sr,  cr, 0],
                   [ 0,   0, 1]])
    # Z * X * Y
    return Rz @ Rx @ Ry

def _min_clearance_after_single_vacancy(crystal, mol, cav_center, radii,
                                        center_strategy="com",
                                        carve_exact_k_fn=None, insert_fn=None):
    """Carve k=1 on a copy, insert mol at cav_center, return min host–guest distance."""
    assert carve_exact_k_fn is not None and insert_fn is not None
    cryst_copy = crystal.copy()
    carved, _, _ = carve_exact_k_fn(cryst_copy, cav_center, radii, k=1)
    placed = insert_fn(carved, mol, center=True, cavity_center=cav_center,
                       cavity_radii=None, check_fit=False, center_strategy=center_strategy)
    pos = placed.get_positions()
    n_host = len(carved)
    host_pos = pos[:n_host]
    guest_pos = pos[n_host:]
    dmin = np.inf
    for g in guest_pos:
        d = np.linalg.norm(host_pos - g, axis=1).min()
        if d < dmin:
            dmin = d
    return dmin

def orient_for_max_clearance(crystal, mol, cav_center, radii,
                             yaw_steps=12, pitch_steps=6, roll_steps=1,
                             clearance_floor=None,  # e.g., 3.3 Å to early-exit
                             center_strategy="com",
                             carve_exact_k_fn=None, insert_fn=None, verbose=False):
    """
    Search a coarse Euler grid to maximize the minimum host–guest distance
    after a single-vacancy carve at cav_center. Returns the rotated molecule.
    """
    if carve_exact_k_fn is None or insert_fn is None:
        raise ValueError("Provide carve_exact_k_fn and insert_fn callables.")

    best = -np.inf
    best_mol = mol

    yaw_vals   = np.linspace(0, 2*np.pi, yaw_steps, endpoint=False)
    pitch_vals = np.linspace(-np.pi/2, np.pi/2, pitch_steps)
    roll_vals  = np.linspace(0, 2*np.pi, roll_steps, endpoint=False)

    for yaw in yaw_vals:
        for pitch in pitch_vals:
            for roll in roll_vals:
                R = _rot_matrix_from_euler(yaw, pitch, roll)
                cand = _rotate_about_com(mol, R)
                dmin = _min_clearance_after_single_vacancy(
                    crystal, cand, cav_center, radii,
                    center_strategy=center_strategy,
                    carve_exact_k_fn=carve_exact_k_fn, insert_fn=insert_fn
                )
                if verbose:
                    print(f"[Orient] yaw={yaw:.2f} pitch={pitch:.2f} roll={roll:.2f} -> dmin={dmin:.3f} Å")
                if dmin > best:
                    best = dmin
                    best_mol = cand
                if clearance_floor is not None and dmin >= clearance_floor:
                    return cand
    if verbose:
        print(f"[Orient] Best min-clearance = {best:.3f} Å")
    return best_mol

# ----- choose the best single vacancy (keep k=1) -----

def _min_clearance_host_guest(host_atoms, guest_atoms):
    hp = host_atoms.get_positions()
    gp = guest_atoms.get_positions()
    dmin = np.inf
    for g in gp:
        d = np.linalg.norm(hp - g, axis=1).min()
        if d < dmin:
            dmin = d
    return dmin

def _insert_at_center(crystal, mol, center, center_strategy="com"):
    """Return a new Atoms with mol inserted at 'center' (COM/bbox/atom), no fit check."""
    return insert_molecule(
        crystal, mol, center=True, cavity_center=np.array(center, float),
        cavity_radii=None, check_fit=False, center_strategy=center_strategy
    )

def min_clearance_with_specific_vacancy(crystal, mol, cav_center, remove_idx, center_strategy="com"):
    """Remove ONLY 'remove_idx', insert mol at cav_center, return (clearance, carved_struct)."""
    mask = np.ones(len(crystal), dtype=bool)
    mask[int(remove_idx)] = False
    carved = crystal[mask]
    placed = _insert_at_center(carved, mol, cav_center, center_strategy=center_strategy)
    n_host = len(carved)
    host = placed[:n_host]
    guest = placed[n_host:]
    return _min_clearance_host_guest(host, guest), carved

def choose_best_single_vacancy(crystal, mol, cav_center, candidates=12, center_strategy="com", verbose=False):
    """
    Try removing each of the 'candidates' nearest host atoms to cav_center (one at a time),
    insert the molecule at the same center, and pick the removal that maximizes the
    minimum host–guest distance. Returns (best_carved_crystal, [best_idx], best_clearance).
    """
    pos = crystal.get_positions()
    d = np.linalg.norm(pos - cav_center, axis=1)
    cand_idx = np.argsort(d)[:max(1, int(candidates))]

    best_idx = None
    best_clear = -np.inf
    best_carved = None

    for idx in cand_idx:
        clr, carved = min_clearance_with_specific_vacancy(
            crystal, mol, cav_center, idx, center_strategy=center_strategy
        )
        if verbose:
            print(f"[VacancyTest] remove idx={int(idx)} -> min_clearance={clr:.3f} Å")
        if clr > best_clear:
            best_clear = clr
            best_idx = int(idx)
            best_carved = carved

    if verbose:
        print(f"[VacancyPick] chose idx={best_idx} with min_clearance={best_clear:.3f} Å")

    return best_carved, np.array([best_idx], dtype=int), best_clear

# ----- in-plane (midplane) center optimization + best single vacancy -----

def _axes_for_plane(axis):
    if axis == 'x':
        return (1, 2)  # free: y, z
    if axis == 'y':
        return (0, 2)  # free: x, z
    if axis == 'z':
        return (0, 1)  # free: x, y
    raise ValueError("axis must be 'x','y','z'")

def optimize_inplane_center_and_vacancy(crystal, mol, cav_center, plane_axis, lattice_constant,
                                        max_shift_frac=0.25, n_steps=5, candidates=24,
                                        center_strategy="com", verbose=False):
    """
    Search small in-plane shifts around `cav_center` (keeping the coordinate along `plane_axis`
    fixed) and, for each shifted center, choose the best SINGLE vacancy. Return the best combo.

    Returns: (best_carved_crystal, best_center, removed_idx_array, best_clearance)
    """
    free_a, free_b = _axes_for_plane(plane_axis)
    a = float(lattice_constant)
    max_shift = max_shift_frac * a
    grid = np.linspace(-max_shift, max_shift, n_steps)
    base = np.array(cav_center, dtype=float)

    best_carved = None
    best_center = None
    best_idx = None
    best_clear = -np.inf

    for da in grid:
        for db in grid:
            test_center = base.copy()
            test_center[free_a] += da
            test_center[free_b] += db

            carved, idx, clear = choose_best_single_vacancy(
                crystal, mol, test_center, candidates=candidates,
                center_strategy=center_strategy, verbose=False
            )
            if verbose:
                print(f"[InPlane] shift=({da:.3f},{db:.3f}) -> idx={idx.tolist()} clr={clear:.3f} Å")

            if clear > best_clear:
                best_clear = clear
                best_idx = idx
                best_center = test_center
                best_carved = carved

    if verbose:
        print(f"[InPlane] Best clearance={best_clear:.3f} Å at center={best_center} with idx={best_idx.tolist()}")
    return best_carved, best_center, best_idx, best_clear

