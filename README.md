# Guest-in-Crystal (Single-Vacancy, Midplane Placement)

## What this does
Places an arbitrary molecule inside an fcc-like rare-gas crystal (e.g., Ne), **keeps the molecule exactly between two unit cells**, **removes only one host atom**, maximizes the **minimum** host–guest distance, and relaxes the environment with a custom Lennard-Jones (LJ) model while keeping the guest rigid.

---

## Key features
- **Crystal builder**: finite fcc (or other ASE bulk structures), no PBC.
- **Guest handling**: robust, molecule-agnostic alignment (auto-picks a good axis if none given).
- **Single-vacancy guarantee**: removes exactly **one** host atom (unless you explicitly enable clearance pre-trim).
- **In-plane micro-optimization**: slides the placement **within the midplane only** and re-chooses the best vacancy to maximize the worst Ne–guest gap.
- **Rigid-guest relaxation**: custom LJ calculator + BFGS; guest remains rigid, host breathes.
- **Convenient I/O**: INI-configurable workflow, XYZ outputs for each step.

---

## What changed (and why overlaps went away)

1) **Robust alignment for any molecule**  
   If atom indices aren’t provided, the code auto-selects an axis (single heavy atom → farthest atom; else farthest pair) and aligns to +Y. Works for tiny guests (e.g., H₂O) and bulky organics.

2) **Smart single-vacancy selection** (still **one** atom removed)  
   Instead of deleting the nearest host, we test a small candidate set and remove the **one** that maximizes the **worst** (smallest) host–guest distance after insertion.

3) **In-plane micro-optimization (midplane preserved)**  
   We keep the coordinate along your midplane axis fixed (still “between two cells”) and scan tiny **in-plane** shifts (e.g., ±0.25·a on a 5×5 grid). For each shift we re-choose the best single vacancy. This breaks symmetry and finds local “pockets,” increasing clearance **without** removing more atoms.

4) **Optional orientation search**  
   Coarse Euler sweep (yaw/pitch/roll) to maximize the minimal Ne–guest gap before carving. Off by default.

5) **Optional safety pre-trim**  
   If you need a guaranteed pre-relax gap, enable a light LJ-based or fixed-Å pre-trim (may remove extra atoms). Leave **off** to enforce single-vacancy.

---

## A pinch of math (why the micro-optimization works)

Let host atoms be $\{h_i\}$, the oriented guest atoms $\{p_j\}$, guest COM $p_{\mathrm{com}}$, intended midplane center $c$, and an **in-plane** shift $s$ (component along midplane axis is zero). Placing by COM:

$$
g_j(s) = p_j - p_{\mathrm{com}} + c + s, \qquad r_{ij}(s) = \lVert h_i - g_j(s) \rVert_2.
$$

If you remove exactly one host $k$, the remaining worst (closest) separation is

$$
d_{\min}(s,k) = \min_{i \neq k,\; j} \, r_{ij}(s).
$$

We pick the best vacancy for that placement and then the best in-plane shift:

$$
s^* = \arg\max_{s \in \mathcal{S}} \; \max_{k \in \mathcal{K}} \; \min_{i \neq k,\, j} \lVert h_i - g_j(s) \rVert_2.
$$

We solve this via a small 2D grid in the plane $\mathcal{S}$ and a shortlist $\mathcal{K}$ of nearby host atoms. This directly maximizes the **minimum** host–guest distance while keeping (i) the guest on the midplane and (ii) exactly one vacancy.

---

## File layout
- `main.py` — CLI entry, loads INI, orchestrates workflow (placement, single-vacancy + in-plane optimization, relaxation, outputs).
- `config.py` — defaults (element, lattice constant, repeat, LJ params, paths).
- `molecule_utils.py` — load/center molecule, compute cavity radii (if you use ellipsoids).
- `crystal_utils.py` — build crystal; midplane center; **single-vacancy selection**; **in-plane optimization**; alignment; extraction; (optional) orientation and clearance pre-trim.
- `lj_calculator.py` — custom LJ that respects constraints.
- `relaxation.py` — BFGS with rigid-guest wrapper.

---

## Input configuration (INI)

Create/edit `inputs/input.ini`. Example:

```ini
[host]
; Element symbol for the host crystal (rare gas recommended)
element = Ne

; ASE bulk crystal structure name: fcc | bcc | hcp | diamond
crystal_structure = fcc

; Lattice constant in Å
lattice_constant = 4.46368

; Supercell repetition counts: nx, ny, nz
repeat = 9, 9, 9


[guest]
; Path to the guest molecule XYZ (relative to repo root is fine)
molecule = molecules_xyz/water.xyz


[cavity]
; Extra padding used by some utilities (e.g., ellipsoid sizing or orientation search)
buffer = 0.10

; OPTIONAL override for cavity radii in Å (uncomment to use)
; radii = 3.0, 2.5, 3.2


[placement]
; Placement mode: midplane | fractional | cartesian
mode = midplane

; Midplane axis: x | y | z  (used when mode = midplane)
axis = x

; Midplane index (between unit cells index and index+1)  (used when mode = midplane)
index = 4

; Small degeneracy-breaking nudge as a fraction of the lattice constant (e.g., 0.02–0.08)
bias_frac = 0.05

; OPTIONAL small Cartesian tweak in Å (applied after center selection)
; nudge = 0.0, 0.0, 0.0

; OPTIONAL fractional center across entire supercell (used when mode = fractional)
; center = 0.5, 0.5, 0.5

; OPTIONAL explicit Cartesian center in Å (used when mode = cartesian)
; center_cart = 10.0, 20.0, 15.0


[orient]
; Enable coarse orientation search to maximize minimal host–guest distance (recommended for bulky guests)
enable = false

; Number of yaw samples (around +Y by default)
yaw_steps = 12

; Number of pitch samples
pitch_steps = 6

; Number of roll samples
roll_steps = 1

; OPTIONAL early stop if minimal distance ≥ this value (Å)
; clearance_floor = 3.3


[clearance]
; Pre-trim mode before insertion:
;   off   = do nothing (enforces single vacancy only)
;   lj    = remove hosts that violate r_min(sigma_mix) * rmin_scale (may remove >1 atom)
;   fixed = remove hosts closer than 'min' Å (may remove >1 atom)
mode = off

; Scale factor for LJ r_min threshold when mode = lj
rmin_scale = 1.00

; Fixed Å threshold when mode = fixed
min = 3.0


[extract]
; Radius in Å for the final spherical fragment around the guest
radius = 12.0


[output]
; Output directory (created if missing; collision policy applies)
dir = outputs

; Handling when the output directory already exists:
;   increment | timestamp | overwrite | fail
collision_policy = increment

---

## How to run

### Basic
```bash
python main.py --input inputs/input.ini
```

### Helpful overrides
```bash
# Use a different molecule + enable orientation search
python main.py -i inputs/input.ini --molecule molecules_xyz/water.xyz --orient-enable

# Place between y-cells 5 and 6 with small in-plane bias
python main.py -i inputs/input.ini --placement-mode midplane --placement-axis y --placement-index 5 --placement-bias-frac 0.05

# Fractional placement (not midplane), with fixed clearance pre-trim at 3.2 Å
python main.py -i inputs/input.ini --placement-mode fractional --placement-center-frac 0.5,0.5,0.5   --clearance-mode fixed --clearance-min 3.2
```

---

## Outputs (in `output.dir`)
- `step1_crystal.xyz` — pristine finite crystal  
- `step2_cavity.xyz` — **one atom removed** (or more if you enabled pre-trim)  
- `step3_molecule_inserted.xyz` — guest inserted at optimized center (guest fixed)  
- `step5_relaxed_full.xyz` — post-LJ relaxation (guest rigid, host relaxed)  
- `step6_relaxed_crystal_cavity_no_molecule.xyz` — host only (guest removed)  
- `step7_molecule.xyz` — oriented guest used for insertion  
- `step8_aligned_structure.xyz` — final realigned structure (guest axis to +Y, origin)  
- `step9_spherical_fragment.xyz` — local sphere around guest  

---

## Tuning for bulky/asymmetric guests
- **Orientation search**: set `[orient] enable = true`; try `yaw_steps=24`, `pitch_steps=12`.  
- **In-plane optimizer** (edit in `main.py` if needed):  
  `max_shift_frac = 0.20–0.30`, `n_steps = 7`, `candidates = 32–64`.  
- **Pre-trim (last resort)**: `[clearance] mode = lj` with `rmin_scale ≈ 1.00–1.05` or `mode = fixed` with a modest Å cutoff. (May remove >1 atom; keep **off** if you must enforce single vacancy.)

---

## Troubleshooting
- **Overlap persists pre-relax**: increase in-plane grid/range and candidate pool; enable orientation search.  
- **More than one host removed**: you likely enabled `[clearance] mode`. Set to `off` to enforce single-vacancy.  
- **Not on the midplane**: use `placement.mode = midplane` and set `axis`/`index`; avoid large `nudge`.  
- **Tiny or zero displacements on relax**: check that the guest is constrained (rigid) and the host is free; ensure LJ parameters include all involved elements.

---

## Why this matches the physics
LJ repulsion grows rapidly at small $r$. By maximizing the **minimum** host–guest distance after removing **one** well-chosen host atom—and allowing only **in-plane** micro-shifts—the initial geometry avoids pathological contacts while preserving a visually fcc environment. The subsequent rigid-guest relaxation then equilibrates the first shell without distorting the molecule.
