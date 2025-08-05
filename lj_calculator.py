from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import NeighborList
import numpy as np

# Custom LJ calculator that respects atom constraints (frozen indices)
class CustomLennardJones(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, lj_params, rc=10.0):
        super().__init__()
        self.lj_params = lj_params
        self.rc = rc

    def calculate(self, atoms=None, properties=['energy','forces'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        pos = atoms.get_positions()
        sym = atoms.get_chemical_symbols()
        # Collect frozen indices from any FixAtoms constraint
        frozen = {i for c in atoms.constraints if hasattr(c,'get_indices') for i in c.get_indices()}
        nl = NeighborList([self.rc/2]*len(atoms), self_interaction=False, bothways=True)
        nl.update(atoms)
        E = 0.0
        F = np.zeros((len(atoms),3))
        cell = atoms.get_cell()
        # Pairwise LJ loops
        for i in range(len(atoms)):
            nbrs, offs = nl.get_neighbors(i)
            for j, off in zip(nbrs, offs):
                if j <= i: continue
                r_vec = pos[j] + off @ cell - pos[i]
                r = np.linalg.norm(r_vec)
                sigma = 0.5*(self.lj_params[sym[i]]['sigma'] + self.lj_params[sym[j]]['sigma'])
                eps   = np.sqrt(self.lj_params[sym[i]]['epsilon'] * self.lj_params[sym[j]]['epsilon'])
                sr6 = (sigma/r)**6
                sr12= sr6**2
                E += 4*eps*(sr12 - sr6)
                fmag = 24*eps/r*(2*sr12 - sr6)
                fvec = fmag * r_vec / r
                if i not in frozen: F[i] -= fvec
                if j not in frozen: F[j] += fvec
        self.results = {'energy': E, 'forces': F}

