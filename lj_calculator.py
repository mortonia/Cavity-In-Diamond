from ase.calculators.calculator import Calculator, all_changes
import numpy as np
from ase.neighborlist import NeighborList

class CustomLennardJones(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, lj_params, rc=6.0):
        super().__init__()
        self.lj_params = lj_params
        self.rc = rc

    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = atoms.get_positions()
        symbols = atoms.get_chemical_symbols()
        n_atoms = len(atoms)

        # Identify frozen atoms (if any)
        frozen_indices = set()
        for constraint in atoms.constraints:
            if hasattr(constraint, 'get_indices'):
                frozen_indices.update(constraint.get_indices())

        # Create neighbor list
        cutoffs = [self.rc / 2.0] * n_atoms  # half cutoff per atom (added together later)
        nl = NeighborList(cutoffs=cutoffs, self_interaction=False, bothways=True)
        nl.update(atoms)

        energy = 0.0
        forces = np.zeros((n_atoms, 3))

        for i in range(n_atoms):
            neighbors, offsets = nl.get_neighbors(i)
            for j, offset in zip(neighbors, offsets):
                if j <= i:
                    continue  # Avoid double-counting

                # Compute distance with periodic offset
                r_vec = positions[j] + np.dot(offset, atoms.get_cell()) - positions[i]
                r = np.linalg.norm(r_vec)

                elem_i = symbols[i]
                elem_j = symbols[j]
                sigma_ij = 0.5 * (self.lj_params[elem_i]['sigma'] + self.lj_params[elem_j]['sigma'])
                epsilon_ij = np.sqrt(self.lj_params[elem_i]['epsilon'] * self.lj_params[elem_j]['epsilon'])

                sr6 = (sigma_ij / r) ** 6
                sr12 = sr6 ** 2
                pair_energy = 4 * epsilon_ij * (sr12 - sr6)
                energy += pair_energy

                force_mag = 24 * epsilon_ij / r * (2 * sr12 - sr6)
                force_vec = force_mag * r_vec / r

                # Apply forces only to movable atoms
                if i not in frozen_indices:
                    forces[i] -= force_vec
                if j not in frozen_indices:
                    forces[j] += force_vec

        self.results['energy'] = energy
        self.results['forces'] = forces
