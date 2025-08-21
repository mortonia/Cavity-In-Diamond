# README — Guest-in-Crystal (Single‑Vacancy) Workflow

## What this code does

This project builds a finite crystal cluster (no PBC), carves a **minimal cavity that removes exactly one host atom**, inserts a guest molecule at a specified location, and performs a **localized Lennard–Jones relaxation** while preserving the guest’s internal geometry (rigid‑body constraint). The workflow is configurable via an **INI file** with optional **CLI overrides**.

Key features:

* **Single‑vacancy carving**: robustly removes *one* lattice site using `carve_exact_k(..., k=1)`, with a tiny symmetry‑breaking bias to avoid midplane degeneracy.
* **Flexible placement**: choose the cavity center by **midplane** (between unit cells), **fractional** supercell coordinates, or explicit **Cartesian** coordinates.
* **Deterministic orientation**: optionally align the guest along a chosen axis (e.g., +Y) before insertion.
* **Localized relaxation**: only atoms near the guest are free to move; far atoms are frozen to reduce compute and preserve the outer lattice.
* **Reproducible outputs**: each pipeline step writes an XYZ for inspection.

---

## Repository structure

```
├── molecule_utils.py        # Load/center guest, compute ellipsoidal radii (semi‑axes)
├── config.py                # Default params (used as fallbacks: CLI > INI > config.py)
├── crystal_utils.py         # Build crystal; placement helpers; single‑vacancy carving; insertion; alignment
├── lj_calculator.py         # Custom LJ calculator (respects frozen atoms)
├── relaxation.py            # BFGS + wrapper to keep guest rigid (remove internal forces)
└── main.py                  # Orchestrates: load → build → carve(1) → insert → relax → export
```

---

## Installation

* Python **3.8+**
* `ase`, `numpy`

```bash
pip install ase numpy
```

---

## Quick start

1. Put your INI at `inputs/input.ini` (example below).
2. From the repo root:

```bash
python3 main.py -i inputs/input.ini
```

(Use `python` or `py -3` if appropriate on your system.)

Results are written to the directory specified in the INI (or overridden on the CLI). Each stage is saved as `outputs/step*.xyz`; BFGS logs to `outputs/relaxation.log`.

---

## INI configuration (recommended)

The workflow reads an **INI** file and supports inline comments with `;` or `#`.

### Full parameter reference

**\[host]**

* `element` — host element symbol (e.g., `Ne`, `Ar`).
* `crystal_structure` — one of `fcc`, `bcc`, `hcp`, `diamond`.
* `lattice_constant` — Å.
* `repeat` — `nx, ny, nz` supercell repeats (integers).

**\[guest]**

* `molecule` — filename of the guest (searched in `./`, `./molecules/`, `./molecules_xyz/`).

**\[cavity]**

* `buffer` — Å added to **semi‑axes** computed from the guest’s bounding box.
* `radii` — optional `rx, ry, rz` in Å to override computed radii.

**\[placement]** (choose one mode)

* `mode` — `midplane` | `fractional` | `cartesian`.
* For `midplane`: `axis` in `x|y|z`, and `index` in `[0 .. repeat[axis]-2]`.
* For `fractional`: `center = fx, fy, fz` in `[0,1]` across the whole supercell.
* For `cartesian`: `center_cart = cx, cy, cz` in Å.
* Optional: `nudge = dx, dy, dz` in Å to break lattice symmetry (tiny bias).

**\[extract]**

* `radius` — Å for spherical fragment extraction around the guest (post‑relaxation).

**\[output]**

* `dir` — output directory (created if missing).
* `collision_policy` — `increment` | `timestamp` | `overwrite` | `fail` (controls what happens if `dir` exists).

### Example INI

```ini
; inputs/input.ini
[host]
element = Ne
crystal_structure = fcc
lattice_constant = 4.46368   ; Å
repeat = 9, 9, 9

[guest]
molecule = propiolic.xyz

[cavity]
buffer = 0.10                 ; Å
; radii = 3.0, 3.0, 3.0       ; (optional) override computed semi‑axes

[placement]
mode = midplane
axis = y
index = 3
; nudge = 0.004, 0, 0         ; small bias (Å), optional

[extract]
radius = 12.0

[output]
dir = outputs
collision_policy = increment
```

---

## Command‑line overrides

CLI overrides **win over INI**, and INI wins over `config.py`.

**Use a specific INI**

```bash
python3 main.py -i inputs/input.ini
```

**Override host & output on the fly**

```bash
python3 main.py -i inputs/input.ini \
  --element Ne --structure fcc --lattice-constant 4.46368 --repeat "9, 9, 9" \
  --output-dir outputs/ne_run --collision-policy increment
```

**Pick a placement mode**

```bash
# Midplane between unit cells along y (index must be 0..ny-2)
python3 main.py -i inputs/input.ini --placement-mode midplane --placement-axis y --placement-index 3

# Fractional supercell coordinates (0..1)
python3 main.py -i inputs/input.ini --placement-mode fractional --placement-center-frac "0.5, 0.5, 0.5"

# Explicit Cartesian center (Å)
python3 main.py -i inputs/input.ini --placement-mode cartesian --placement-center-cart "12.0, 10.5, 8.0"

# Optional tiny symmetry‑breaking bias (Å)
python3 main.py -i inputs/input.ini --placement-nudge "0.004, 0, 0"
```

---

## Outputs

* `step1_crystal.xyz` — pristine cluster
* `step2_cavity.xyz` — cluster after **single‑atom** cavity removal
* `step3_molecule_inserted.xyz` — guest placed at the cavity center
* `step5_relaxed_full.xyz` — post‑relaxation structure (guest + host)
* `step6_relaxed_crystal_cavity_no_molecule.xyz` — host only, unbound H removed
* `step7_molecule.xyz` — oriented guest molecule
* `step8_aligned_structure.xyz` — relaxed system realigned to guest axis at origin
* `step9_spherical_fragment.xyz` — spherical fragment around the guest
* `relaxation.log` — BFGS optimization log

During carving you’ll also see a console line such as:

```
[Cavity] Removed indices=[1234] (scale=0.543210)
```

`indices` is the removed atom(s); `scale` is the ellipsoid scaling used to hit `k=1`.

---

## How the single‑vacancy carving works

1. We compute an ellipsoid from the guest’s bounding box (semi‑axes + buffer) or use radii you supply.
2. The chosen center is **nudged slightly** toward the nearest lattice site to break midplane ties.
3. A bisection search (`carve_exact_k`) scales the ellipsoid to include **exactly one** atom (strict interior test keeps boundary atoms). If geometry makes this impossible, it falls back to removing the **nearest** atom.
4. We insert the molecule at the same center and **skip fit checks** (since we deliberately carved only one site).

---

## Tips & troubleshooting

* **Too many atoms removed**: ensure your INI sets `mode` correctly; keep a small `nudge` (e.g., `0.004,0,0`) to break symmetry; rely on the default single‑vacancy path via `carve_exact_k(..., k=1)`.
* **No relaxation**: your `fmax` in `relaxation.py` is conservative (`0.01`). For stronger relaxations, consider `0.05` and/or increase LJ cutoff `rc` in `lj_calculator.py`.
* **Units**: all distances are **Å**.
* **Performance**: large `repeat` values grow the number of host atoms cubically.

---

*Developed for minimal‑footprint guest insertion in nearly perfect fcc lattices; tuned for deterministic single‑site removal with clear, reproducible outputs.*

