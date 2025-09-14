# --- main.py ---
import os
import argparse
import configparser
from datetime import datetime
import numpy as np
from ase.io import write
from ase.constraints import FixAtoms

# Fallback defaults from config.py
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
from relaxation import relax_with_lj
from crystal_utils import (
    create_crystal, create_cavity, insert_molecule,
    remove_unbound_hydrogens, align_single_molecule,
    align_molecule_in_structure, extract_spherical_region,
    min_clearance_to_crystal, midplane_center,
    bias_toward_single_nearest, carve_exact_k,  # kept for compatibility / orientation search
    pretrim_host_for_clearance, orient_for_max_clearance,
    choose_best_single_vacancy, optimize_inplane_center_and_vacancy,
)


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
        base = desired_path
        i = 1
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
    parser.add_argument("--input", "-i", default="inputs/input.ini", help="Path to INI file.")

    # Optional direct overrides (CLI > INI > config.py)
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
    parser.add_argument("--placement-bias-frac", type=float, help="fraction of a to bias toward nearest site (default 0.0)")

    # Clearance pre-trim (optional; keep 'off' to strictly remove one atom)
    parser.add_argument("--clearance-mode", choices=["off", "lj", "fixed"])
    parser.add_argument("--clearance-rmin-scale", type=float)
    parser.add_argument("--clearance-min", type=float)

    # Orientation search (optional)
    parser.add_argument("--orient-enable", action="store_true")
    parser.add_argument("--orient-yaw-steps", type=int)
    parser.add_argument("--orient-pitch-steps", type=int)
    parser.add_argument("--orient-roll-steps", type=int)
    parser.add_argument("--orient-clearance-floor", type=float)

    args = parser.parse_args()

    # INI with inline comments allowed
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
    bias_frac = args.placement_bias_frac
    if bias_frac is None:
        try:
            bias_frac = cp.getfloat("placement", "bias_frac", fallback=0.0)
        except Exception:
            bias_frac = 0.0

    center_frac = _parse_tuple3(center_frac_txt, tp=float) if center_frac_txt else None
    center_cart = _parse_tuple3(center_cart_txt, tp=float) if center_cart_txt else None
    nudge = np.array(_parse_tuple3(nudge_txt, tp=float)) if nudge_txt else None

    # Clearance (default off to preserve single-vacancy rule)
    clearance_mode = args.clearance_mode or cp.get("clearance", "mode", fallback="off")
    clearance_rmin_scale = args.clearance_rmin_scale
    if clearance_rmin_scale is None:
        try:
            clearance_rmin_scale = cp.getfloat("clearance", "rmin_scale", fallback=1.0)
        except Exception:
            clearance_rmin_scale = 1.0
    clearance_min = args.clearance_min
    if clearance_min is None:
        try:
            clearance_min = cp.getfloat("clearance", "min", fallback=3.0)
        except Exception:
            clearance_min = 3.0

    # Orientation search
    orient_enabled = bool(args.orient_enable or cp.getboolean("orient", "enable", fallback=False))
    orient_yaw_steps = args.orient_yaw_steps or cp.getint("orient", "yaw_steps", fallback=12)
    orient_pitch_steps = args.orient_pitch_steps or cp.getint("orient", "pitch_steps", fallback=6)
    orient_roll_steps = args.orient_roll_steps or cp.getint("orient", "roll_steps", fallback=1)
    orient_clearance_floor = args.orient_clearance_floor
    if orient_clearance_floor is None:
        val = cp.get("orient", "clearance_floor", fallback=None)
        orient_clearance_floor = float(val) if (val is not None) and (val != "") else None

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
        "bias_frac": bias_frac,
        # clearance
        "clearance_mode": clearance_mode,
        "clearance_rmin_scale": clearance_rmin_scale,
        "clearance_min": clearance_min,
        # orientation
        "orient_enabled": orient_enabled,
        "orient_yaw_steps": orient_yaw_steps,
        "orient_pitch_steps": orient_pitch_steps,
        "orient_roll_steps": orient_roll_steps,
        "orient_clearance_floor": orient_clearance_floor,
    }


def _fractional_to_cartesian_center(crystal, frac_xyz):
    pos = crystal.get_positions()
    lo = pos.min(axis=0)
    hi = pos.max(axis=0)
    L = hi - lo
    return lo + np.array(frac_xyz, dtype=float) * L


def main():
    S = load_settings()

    # Load molecule (anchor to origin for convenience; exact anchoring on insert)
    mol = load_and_center_molecule(S["molecule_path"], np.array([0.0, 0.0, 0.0]))

    # Robust orientation: pick a good atom pair, align to +Y; remember the pair
    mol, align_pair = align_single_molecule(
        mol,
        atom1_idx=None,  # auto-pick
        atom2_idx=None,  # auto-pick
        target_axis=np.array([0.0, 1.0, 0.0]),
        return_pair=True
    )

    # Build crystal
    crystal = create_crystal(S["element"], S["structure"], S["repeat"], S["a"])
    write(os.path.join(S["output_dir"], "step1_crystal.xyz"), crystal)

    # Compute / override cavity radii (used for orientation search; single-vacancy ignores fit)
    radii = S["cavity_radii"] or compute_cavity_radii(mol, S["buffer"])

    # Compute cavity center based on placement settings
    cav_center = None
    if S["placement_mode"] == "midplane":
        if S["placement_axis"] is None or S["placement_index"] is None:
            raise ValueError("mode=midplane requires --placement-axis and --placement-index (or INI equivalents).")
        cav_center = midplane_center(
            crystal,
            axis=S["placement_axis"],
            index=S["placement_index"],
            lattice_constant=S["a"],
            repeat=S["repeat"]
        )
    elif S["placement_mode"] == "fractional":
        if S["center_frac"] is None:
            raise ValueError("mode=fractional requires --placement-center-frac fx,fy,fz (or INI placement.center).")
        cav_center = _fractional_to_cartesian_center(crystal, S["center_frac"])
    elif S["placement_mode"] == "cartesian":
        if S["center_cart"] is None:
            raise ValueError("mode=cartesian requires --placement-center-cart cx,cy,cz (or INI placement.center_cart).")
        cav_center = np.array(S["center_cart"], dtype=float)

    # Fallback: geometric center of the cluster
    if cav_center is None:
        pos = crystal.get_positions()
        cav_center = 0.5 * (pos.min(axis=0) + pos.max(axis=0))

    # Optional tiny Cartesian nudge
    if S["nudge"] is not None:
        cav_center = np.array(cav_center, dtype=float) + S["nudge"]

    # Optional bias toward nearest site to break degeneracy (default 0.0 = off)
    if S["bias_frac"] and S["bias_frac"] != 0.0:
        cav_center = bias_toward_single_nearest(crystal, cav_center, S["a"], frac=float(S["bias_frac"]))

    # Optional: rotate molecule to maximize clearance for single-vacancy placement
    if S["orient_enabled"]:
        mol = orient_for_max_clearance(
            crystal, mol, cav_center, radii,
            yaw_steps=S["orient_yaw_steps"],
            pitch_steps=S["orient_pitch_steps"],
            roll_steps=S["orient_roll_steps"],
            clearance_floor=S["orient_clearance_floor"],
            center_strategy="com",
            carve_exact_k_fn=carve_exact_k,
            insert_fn=insert_molecule,
            verbose=False
        )

    # --- In-plane center optimization + best single vacancy (keeps midplane coordinate fixed) ---
    plane_axis = S["placement_axis"] if S["placement_mode"] == "midplane" else (S["placement_axis"] or "x")
    cavity_cryst, cav_center_opt, removed_idx, best_clear = optimize_inplane_center_and_vacancy(
        crystal, mol, cav_center, plane_axis=plane_axis, lattice_constant=S["a"],
        max_shift_frac=0.25,   # try 0.15–0.30 for bulky guests
        n_steps=5,             # 5x5 grid
        candidates=32,         # larger pool for big molecules
        center_strategy="com",
        verbose=False
    )
    print(f"[Cavity] In-plane optimized. Removed idx={removed_idx.tolist()}, "
          f"best pre-relax clearance={best_clear:.3f} Å, center={cav_center_opt}")
    write(os.path.join(S["output_dir"], "step2_cavity.xyz"), cavity_cryst)

    # Optional: pre-trim any host atoms that would violate a clearance threshold
    # NOTE: keep 'off' to strictly remove only one atom.
    if S["clearance_mode"] != "off":
        cavity_cryst, extra_removed = pretrim_host_for_clearance(
            cavity_cryst,
            mol,
            cav_center_opt,
            mode=S["clearance_mode"],
            rmin_scale=S["clearance_rmin_scale"],
            min_clearance=S["clearance_min"],
            center_strategy="com"
        )
        if len(extra_removed) > 0:
            print(f"[Clearance] Extra host atoms removed: {len(extra_removed)}")
        write(os.path.join(S["output_dir"], "step2_cavity.xyz"), cavity_cryst)  # overwrite if changed

    # Insert molecule at the optimized center; COM anchor; skip fit check (single-vacancy path)
    combined = insert_molecule(
        cavity_cryst, mol, center=True, cavity_center=cav_center_opt,
        cavity_radii=None, check_fit=False, center_strategy="com"
    )
    write(os.path.join(S["output_dir"], "step3_molecule_inserted.xyz"), combined)

    # Fix molecule atoms
    start_idx = len(cavity_cryst)
    mol_idx = list(range(start_idx, start_idx + len(mol)))
    combined.set_constraint(FixAtoms(indices=mol_idx))

    # Clearance check (pre-relax)
    dmin, _, _ = min_clearance_to_crystal(combined, mol_idx)
    print(f"[Cavity check] Minimum distance (pre-relax) = {dmin:.3f} Å")

    # Relax locally
    orig = combined.get_positions()
    relaxed, disp, _ = relax_with_lj(
        combined, orig, S["disp_thresh"], S["output_dir"], molecule_indices=mol_idx
    )
    write(os.path.join(S["output_dir"], "step5_relaxed_full.xyz"), relaxed)

    # Remove guest + cleanup H
    frag = relaxed.copy()
    del frag[mol_idx]
    frag = remove_unbound_hydrogens(frag)
    write(os.path.join(S["output_dir"], "step6_relaxed_crystal_cavity_no_molecule.xyz"), frag)

    # Save guest (as oriented/used)
    write(os.path.join(S["output_dir"], "step7_molecule.xyz"), mol)

    # Align final structure to the same molecular axis and move molecule to origin
    if align_pair is not None and len(align_pair) == 2:
        a1, a2 = align_pair
        aligned = align_molecule_in_structure(
            relaxed, mol_idx, atom1_idx=a1, atom2_idx=a2,
            target_axis=np.array([0.0, 1.0, 0.0]), translate_to_origin=True
        )
    else:
        aligned = align_molecule_in_structure(
            relaxed, mol_idx, atom1_idx=0, atom2_idx=1 if len(mol) > 1 else 0,
            target_axis=np.array([0.0, 1.0, 0.0]), translate_to_origin=True
        )
    write(os.path.join(S["output_dir"], "step8_aligned_structure.xyz"), aligned)

    # Extract a spherical fragment around the molecule
    sphere = extract_spherical_region(aligned, mol_idx, radius=S["extract_radius"])
    write(os.path.join(S["output_dir"], "step9_spherical_fragment.xyz"), sphere)

    print("[Done] Outputs in:", S["output_dir"])


if __name__ == "__main__":
    main()
