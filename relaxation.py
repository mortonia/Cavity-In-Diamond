# relaxation.py

import os
import numpy as np
from ase.optimize import BFGS
from lj_calculator import CustomLennardJones
from config import LJ_PARAMS

def relax_with_lj(atoms, original_positions, displacement_threshold, output_dir):
    lj = CustomLennardJones(lj_params=LJ_PARAMS, rc=10.0)
    atoms.set_calculator(lj)

    dyn = BFGS(atoms, logfile=os.path.join(output_dir, "relax.log"))
    dyn.run(fmax=0.01, steps=200)

    new_positions = atoms.get_positions()
    displacements = np.linalg.norm(new_positions - original_positions, axis=1)
    num_displaced = np.sum(displacements > displacement_threshold)

    with open(os.path.join(output_dir, "lj_displacements.txt"), "w") as f:
        for i, disp in enumerate(displacements):
            f.write(f"{i} {disp:.4f}\n")

    return atoms, displacements, num_displaced
