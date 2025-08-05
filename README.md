# Simulation Workflow README

## Overview

This codebase automates the process of embedding a guest molecule into an inert host crystal (argon by default), carving an ellipsoidal cavity around the molecule, performing a localized Lennard–Jones relaxation, and extracting various structural outputs for analysis. It is useful for studying molecular insertion and local structural relaxation effects in clusters.

## Motivations

* **Guest–Host Modeling**: Understand how a small molecule (e.g., propiolic acid) interacts physically when inserted into a solid matrix.
* **Cavity Carving**: Create a tailored void exactly matching the guest geometry, avoiding overlaps and ensuring accurate fit.
* **Localized Relaxation**: Perform energy minimization only in the region around the molecule to reduce computational cost while preserving long-range lattice integrity.
* **Visual and Quantitative Outputs**: Generate XYZ snapshots at each stage and calculate key metrics (minimum clearance, cavity geometry, displacements).

## Repository Structure

```
├── molecule_utils.py        # Load and center guest molecule; compute cavity dimensions
├── config.py                # Global parameters (elements, buffer, LJ params, paths)
├── crystal_utils.py         # Build host crystal; carve cavity; insert molecule; fragment extraction; alignment
├── lj_calculator.py         # Custom Lennard–Jones energy & force calculator that respects frozen atoms
├── relaxation.py            # Wrap LJ calculator in BFGS optimizer; preserve molecular geometry
├── write_molecule_with_cavity_markers.py  # (external) visualize cavity boundary markers
└── main.py                  # Orchestrates steps 1–11 and writes output files
```

## Prerequisites

* Python 3.8+
* [ASE (Atomic Simulation Environment)](https://wiki.fysik.dtu.dk/ase/)
* NumPy

Ensure these packages are installed, e.g.:

```bash
pip install ase numpy
```

## Usage

1. **Adjust Parameters** (optional):

   * Edit `config.py` to change:

     * `ELEMENT`: host atom type (default `'Ar'`)
     * `REPEAT`: crystal dimensions
     * `CAVITY_BUFFER`: extra padding around the molecule
     * `LJ_PARAMS`: Lennard–Jones parameters per element

2. **Run Main Script**:
   Navigate to the project directory and execute:

   ```bash
   python main.py
   ```

(Note - may need to use "python3" instead of "python". This depends on what you have installed)

```
This will create an `outputs/` folder (if not present) and generate the following files:

| File                                      | Description                                      |
|-------------------------------------------|--------------------------------------------------|
| `step1_crystal.xyz`                       | Pristine crystal cluster                         |
| `step2_cavity.xyz`                        | Cluster after carving cavity                     |
| `step3_molecule_inserted.xyz`             | Molecule placed into cavity                      |
| `step5_relaxed_full.xyz`                  | Cluster + molecule after localized relaxation    |
| `step6_relaxed_crystal_cavity_no_molecule.xyz` | Crystal fragment without guest, unbound H removed |
| `step7_molecule.xyz`                      | Oriented guest molecule                          |
| `step8_aligned_structure.xyz`             | Relaxed system realigned to guest axis at origin |
| `step9_spherical_fragment.xyz`            | Spherical fragment extraction                    |

3. **Inspect Outputs**:  
   - Visualize XYZ files in any molecular viewer (e.g., ASE GUI, VMD).  
   - Review `outputs/relaxation.log` for optimization details.  
   - Analyze printed metrics (minimum clearance, maximum displacement) in console output.

## Extending the Workflow
- **Different Molecules**: Change `MOLECULE_PATH` in `config.py` to another XYZ file.  
- **Host Materials**: Modify `ELEMENT` and `LJ_PARAMS` for different host–guest systems.  
- **Parameter Sweeps**: Automate loops over `CAVITY_BUFFER` or relaxation thresholds by wrapping calls to `main()` in a custom script.

---
*Developed for streamlined host–guest insertion and localized relaxation studies.*

```
