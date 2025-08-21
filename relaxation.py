# --- relaxation.py ---
import os
import numpy as np
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from ase.calculators.calculator import Calculator, all_changes
from config import LJ_PARAMS
from lj_calculator import CustomLennardJones

class RigidMoleculeWrapper(Calculator):
    implemented_properties=['energy','forces']
    def __init__(self, base_calc, molecule_indices):
        super().__init__()
        self.base = base_calc
        self.molecule_indices = molecule_indices
    def calculate(self, atoms=None, properties=['energy','forces'], system_changes=all_changes):
        self.base.calculate(atoms, properties, system_changes)
        res = self.base.results
        forces = res['forces'].copy()
        mf = forces[self.molecule_indices]
        com_force = mf.mean(axis=0)
        mf -= com_force
        for idx, f in zip(self.molecule_indices, mf):
            forces[idx] -= f
        self.results = {'energy': res['energy'], 'forces': forces}


def relax_with_lj(atoms, orig_pos, disp_thresh, output_dir,
                  max_local_dist=100.0, molecule_indices=None):
    if molecule_indices is None:
        raise ValueError("Must provide molecule_indices for geometry preservation")
    atoms = atoms.copy()
    cen = atoms.get_positions()[molecule_indices].mean(axis=0)
    dist = np.linalg.norm(atoms.get_positions() - cen, axis=1)
    fixed = [i for i, d in enumerate(dist) if d > max_local_dist]
    atoms.set_constraint(FixAtoms(indices=fixed))
    base = CustomLennardJones(LJ_PARAMS, rc=10.0)
    atoms.set_calculator(RigidMoleculeWrapper(base, molecule_indices))
    log = os.path.join(output_dir, "relaxation.log")
    dyn = BFGS(atoms, logfile=log, maxstep=0.005)
    dyn.run(fmax=0.01, steps=100)
    new_pos = atoms.get_positions()
    disp = np.linalg.norm(new_pos - orig_pos, axis=1)
    count_large = (disp > disp_thresh).sum()
    print(f"[Relaxation] Max displacement = {disp.max():.3f} Å at atom {disp.argmax()}")
    print(f"[Relaxation] Atoms with displacement > {disp_thresh:.2f} Å: {count_large}")
    return atoms, disp, count_large
