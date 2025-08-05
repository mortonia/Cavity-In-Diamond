from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import NeighborList
import numpy as np

class CustomLennardJones(Calculator):
    implemented_properties=['energy','forces']
    def __init__(self, lj_params, rc=10.0):
        super().__init__()
        self.lj_params=lj_params
        self.rc=rc

    def calculate(self, atoms=None, properties=['energy','forces'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        pos, sym = atoms.get_positions(), atoms.get_chemical_symbols()
        frozen = {i for c in atoms.constraints if hasattr(c,'get_indices') for i in c.get_indices()}
        nl = NeighborList([self.rc/2]*len(atoms), self_interaction=False, bothways=True)
        nl.update(atoms)
        E=0; F=np.zeros((len(atoms),3))
        for i in range(len(atoms)):
            nbrs, offs = nl.get_neighbors(i)
            for j,off in zip(nbrs,offs):
                if j<=i: continue
                r_vec = pos[j]+off@atoms.get_cell()-pos[i]
                r = np.linalg.norm(r_vec)
                s = 0.5*(self.lj_params[sym[i]]['sigma']+self.lj_params[sym[j]]['sigma'])
                e = np.sqrt(self.lj_params[sym[i]]['epsilon']*self.lj_params[sym[j]]['epsilon'])
                sr6=(s/r)**6; sr12=sr6**2
                E += 4*e*(sr12-sr6)
                f = 24*e/r*(2*sr12-sr6)*r_vec/r
                if i not in frozen: F[i]-=f
                if j not in frozen: F[j]+=f
        self.results['energy']=E
        self.results['forces']=F
