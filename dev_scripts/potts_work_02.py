import numpy as np
from numba import njit

@njit
def potts_monte_carlo_3(lattice, q, steps):
    size = lattice.shape[0]

    for _ in range(steps):
        for i in range(size):
            for j in range(size):
                current_spin = lattice[i, j]

                energy_diff = 0
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:

                    neighbor_spin = lattice[i + dx, j + dy]
                    energy_diff += 1 if neighbor_spin != current_spin else 0

                # Flip the spin with probability prob
                if np.random.rand() < np.exp(-energy_diff):
                    lattice[i, j] = np.random.randint(low=1, high=q + 1)

    return lattice


q = 3  # Number of states

array = np.random.randint(low=1, high=q+1, size=(10, 10))
print(array)


steps = 100000  # Number of Monte Carlo steps

result = potts_monte_carlo_3(array, q, steps)
print(result)
