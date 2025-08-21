# --- main.py ---
import os
import argparse
import configparser
from datetime import datetime
import numpy as np
from ase.io import write
from ase.constraints import FixAtoms

# Defaults from config.py (used as fallbacks)
from config import (
    ELEMENT as CFG_ELEMENT,
    REPEAT as CFG_REPEAT,
    LATTICE_CONSTANT as CFG_A,
    DISPLACEMENT_THRESHOLD as CFG_DTH,
    OUTPUT_DIR as CFG_OUT,
    CAVITY_BUFFER as CFG_BUF,
    MOLECULE_PATH as CFG_MOLPATH,
)

from molecule_utils import load_and_center_molecule, compute_cavity_radii
from crystal_utils import (
    create_crystal, create_cavity, insert_molecule,
    remove_unbound_hydrogens, align_single_molecule,
    align_molecule_in_structure, extract_spherical_region,
    min_clearance_to_crystal, midplane_center,
    bias_toward_single_nearest, carve_exact_k,
)
from relaxation import relax_with_lj


def _parse_tuple3(text, tp=float):
    parts = [p.strip() for p in str(text).split(',')]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 comma-separated values, got: {text}")
    return tuple(tp(p) for p in parts)


def _resolve_molecule_path(name_or_path):
    cands = [
        name_or_path,
        os.path.join("molecules", name_or_path),
        os.path.join("molecules_xyz", name_or_path),
    ]
    for p in cands:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Molecule file not found: {name_or_path} (looked in ./, ./molecules/, ./molecules_xyz/)"
    )


def _ensure_output_dir(desired_path, policy="increment"):
    policy = (policy or "increment").lower()
    if not os.path.exists(desired_path):
        os.makedirs(desired_path, exist_ok=True)
        return desired_path
    if policy == "overwrite":
        return desired_path
    elif policy == "fail":
        raise FileExistsError(f"Output directory exists: {desired_path}")
    elif policy == "timestamp":
        stamped = f"{desired_path}_{datetime.now():%Y%m%d-%H%M%S}"
        os.makedirs(stamped, exist_ok=True)
        return stamped
    elif policy == "increment":
        base = desired_path; i = 1
        candidate = f"{base}_{i:03d}"
        while os.path.exists(candidate):
            i += 1
            candidate = f"{base}_{i:03d}"
        os.makedirs(candidate, exist_ok=True)
        return candidate
    else:
        raise ValueError(f"Unknown collision_policy: {policy}")


def load_settings():
    parser = argparse.ArgumentParser(description="Guest-in-crystal workflow (INI-configurable).")
    parser.add_argument("--input", "-i", default="inputs/example.ini", help="Path to INI file.")
    # Optional direct overrides (CLI flags win over INI; INI wins over config.py)
    parser.add_argument("--element")
    parser.add_argument("--structure")
    parser.add_argument("--lattice-constant", type=float)
    parser.add_argument("--repeat")
    parser.add_argument("--molecule")
    parser.add_argument("--cavity-buffer", type=float)
    parser.add_argument("--cavity-radii")
    parser.add_argument("--extract-radius", type=float)
    parser.add_argument("--output-dir")
    parser.add_argument("--collision-policy", choices=["increment", "timestamp", "overwrite", "fail"])
    # Placement controls
    parser.add_argument("--placement-mode", choices=["midplane", "fractional", "cartesian"])
    parser.add_argument("--placement-axis", choices=["x", "y", "z"])
    parser.add_argument("--placement-index", type=int)
    parser.add_argument("--placement-center-frac", help="fx, fy, fz in [0,1] across supercell (mode=fractional)")
    parser.add_argument("--placement-center-cart", help="cx, cy, cz in Å (mode=cartesian)")
    parser.add_argument("--placement-nudge", help="dx, dy, dz in Å (optional Cartesian nudge)")
    args = parser.parse_args()

    # Allow inline comments like "4.46368 ; Å"
    cp = configparser.ConfigParser(inline_comment_prefixes=(";", "#"), strict=False)
    cp.read(args.input)

    # Host
    element = args.element or cp.get("host", "element", fallback=CFG_ELEMENT)
    structure = args.structure or cp.get("host", "crystal_structure", fallback="fcc")
    a = args.lattice_constant or cp.getfloat("host", "lattice_constant", fallback=CFG_A)
    repeat_text = args.repeat or cp.get("host", "repeat", fallback=f"{CFG_REPEAT[0]}, {CFG_REPEAT[1]}, {CFG_REPEAT[2]}")
    repeat = _parse_tuple3(repeat_text, tp=int)

    # Guest
    mol_name = args.molecule or cp.get("guest", "molecule", fallback=os.path.basename(CFG_MOLPATH))
    molecule_path = _resolve_molecule_path(mol_name)

    # Cavity
    buffer_ = args.cavity_buffer or cp.getfloat("cavity", "buffer", fallback=CFG_BUF)
    radii_override = args.cavity_radii or cp.get("cavity", "radii", fallback=None)
    cavity_radii = _parse_tuple3(radii_override, tp=float) if radii_override else None

    # Extraction
    extract_radius = args.extract_radius or cp.getfloat("extract", "radius", fallback=12.0)

    # Output dir & policy
    out_dir_req = args.output_dir or cp.get("output", "dir", fallback=CFG_OUT)
    collision_policy = args.collision_policy or cp.get("output", "collision_policy", fallback="increment")
    output_dir = _ensure_output_dir(out_dir_req, policy=collision_policy)

    # Placement section
    placement_mode = args.placement_mode or cp.get("placement", "mode", fallback=None)
    placement_axis = args.placement_axis or cp.get("placement", "axis", fallback=None)
    placement_index = args.placement_index or cp.getint("placement", "index", fallback=None)
    center_frac_txt = args.placement_center_frac or cp.get("placement", "center", fallback=None)
    center_cart_txt = args.placement_center_cart or cp.get("placement", "center_cart", fallback=None)
    nudge_txt = args.placement_nudge or cp.get("placement", "nudge", fallback=None) \
                or cp.get("placement", "epsilon_xyz", fallback=None)

    center_frac = _parse_tuple3(center_frac_txt, tp=float) if center_frac_txt else None
    center_cart = _parse_tuple3(center_cart_txt, tp=float) if center_cart_txt else None
    nudge = np.array(_parse_tuple3(nudge_txt, tp=float)) if nudge_txt else None

    return {
        "element": element,
        "structure": structure,
        "a": a,
        "repeat": repeat,
        "molecule_path": molecule_path,
        "buffer": buffer_,
        "cavity_radii": cavity_radii,
        "extract_radius": extract_radius,
        "output_dir": output_dir,
        "disp_thresh": CFG_DTH,
        # placement
        "placement_mode": placement_mode,
        "placement_axis": placement_axis,
        "placement_index": placement_index,
        "center_frac": center_frac,
        "center_cart": center_cart,
        "nudge": nudge,
    }


def _fractional_to_cartesian_center(crystal, frac_xyz):
    pos = crystal.get_positions(); lo = pos.min(axis=0); hi = pos.max(axis=0)
    L = hi - lo
    return lo + np.array(frac_xyz, dtype=float) * L


def main():
    S = load_settings()

    # Load & orient molecule
    mol = load_and_center_molecule(S["molecule_path"], np.array([0.0, 0.0, 0.0]))
    mol = align_single_molecule(mol, atom1_idx=3, atom2_idx=4, target_axis=np.array([0.0, 1.0, 0.0]))

    # Build crystal
    crystal = create_crystal(S["element"], S["structure"], S["repeat"], S["a"])
    write(os.path.join(S["output_dir"], "step1_crystal.xyz"), crystal)

    # Compute / override cavity radii
    radii = S["cavity_radii"] or compute_cavity_radii(mol, S["buffer"])

    # Compute cavity center based on placement settings
    cav_center = None
    if S["placement_mode"] == "midplane":
        if S["placement_axis"] is None or S["placement_index"] is None:
            raise ValueError("mode=midplane requires --placement-axis and --placement-index")
        cav_center = midplane_center(crystal, axis=S["placement_axis"], index=S["placement_index"],
                                     lattice_constant=S["a"], repeat=S["repeat"])
    elif S["placement_mode"] == "fractional":
        if S["center_frac"] is None:
            raise ValueError("mode=fractional requires --placement-center-frac fx,fy,fz")
        cav_center = _fractional_to_cartesian_center(crystal, S["center_frac"])
    elif S["placement_mode"] == "cartesian":
        if S["center_cart"] is None:
            raise ValueError("mode=cartesian requires --placement-center-cart cx,cy,cz")
        cav_center = np.array(S["center_cart"], dtype=float)

    # Fallback to bbox center if not specified
    if cav_center is None:
        pos = crystal.get_positions()
        cav_center = 0.5 * (pos.min(axis=0) + pos.max(axis=0))

    # Optional tiny Cartesian nudge
    if S["nudge"] is not None:
        cav_center = np.array(cav_center, dtype=float) + S["nudge"]

    # Break symmetry near midplanes to ensure a single vacancy
    cav_center = bias_toward_single_nearest(crystal, cav_center, S["a"], frac=1e-3)

    # Carve EXACTLY ONE atom
    cavity_cryst, removed_idx, (scale_used,) = carve_exact_k(crystal, cav_center, radii, k=1)
    print(f"[Cavity] Removed indices={removed_idx.tolist()} (scale={scale_used:.6f})")
    write(os.path.join(S["output_dir"], "step2_cavity.xyz"), cavity_cryst)

    # Insert molecule at SAME center; skip fit check (we deliberately removed only 1 atom)
    combined = insert_molecule(cavity_cryst, mol, center=True, cavity_center=cav_center,
                               cavity_radii=None, check_fit=False)
    write(os.path.join(S["output_dir"], "step3_molecule_inserted.xyz"), combined)

    # Fix molecule atoms
    start_idx = len(cavity_cryst)
    mol_idx = list(range(start_idx, start_idx + len(mol)))
    combined.set_constraint(FixAtoms(indices=mol_idx))

    # Clearance check
    dmin, _, _ = min_clearance_to_crystal(combined, mol_idx)
    print(f"[Cavity check] Minimum distance = {dmin:.3f} Å")

    # Relax locally
    orig = combined.get_positions()
    relaxed, disp, _ = relax_with_lj(combined, orig, S["disp_thresh"], S["output_dir"], molecule_indices=mol_idx)
    write(os.path.join(S["output_dir"], "step5_relaxed_full.xyz"), relaxed)

    # Remove guest + cleanup H
    frag = relaxed.copy(); del frag[mol_idx]
    frag = remove_unbound_hydrogens(frag)
    write(os.path.join(S["output_dir"], "step6_relaxed_crystal_cavity_no_molecule.xyz"), frag)

    # Save guest
    write(os.path.join(S["output_dir"], "step7_molecule.xyz"), mol)

    # Align final, extract sphere
    aligned = align_molecule_in_structure(relaxed, mol_idx, atom1_idx=0, atom2_idx=1,
                                          target_axis=np.array([0.0, 1.0, 0.0]), translate_to_origin=True)
    write(os.path.join(S["output_dir"], "step8_aligned_structure.xyz"), aligned)

    sphere = extract_spherical_region(aligned, mol_idx, radius=S["extract_radius"])
    write(os.path.join(S["output_dir"], "step9_spherical_fragment.xyz"), sphere)

    print("[Done] Outputs in:", S["output_dir"])

if __name__ == "__main__":
    main()
