import numpy as np
from numba import njit

def potts_monte_carlo_1(lattice, q, steps):
    size = lattice.shape[0]

    for _ in range(steps):
        for i in range(size):
            for j in range(size):
                current_spin = lattice[i, j]

                # Calculate energy due to neighboring spins
                energy_diff = 0
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    ni = (i + dx) % size
                    nj = (j + dy) % size
                    neighbor_spin = lattice[ni, nj]
                    energy_diff += 1 if neighbor_spin == current_spin else 0

                # Calculate the probability of spin flip
                prob = np.exp(-energy_diff)

                # Flip the spin with probability prob
                if np.random.rand() < prob:
                    lattice[i, j] = np.random.randint(low=1, high=q+1)

    return lattice

def potts_monte_carlo_2(lattice, q, steps):
    size = lattice.shape[0]

    for _ in range(steps):
        # Calculate energy differences due to neighboring spins
        energy_diff = np.zeros_like(lattice)
        energy_diff[:-1] += (lattice[:-1] != lattice[1:]).astype(int)
        energy_diff[1:] += (lattice[:-1] != lattice[1:]).astype(int)
        energy_diff[:, :-1] += (lattice[:, :-1] != lattice[:, 1:]).astype(int)
        energy_diff[:, 1:] += (lattice[:, :-1] != lattice[:, 1:]).astype(int)

        # Calculate probabilities of spin flip
        probabilities = np.exp(-energy_diff)

        # Flip the spins with probabilities
        random_nums = np.random.rand(*lattice.shape)
        flip_mask = random_nums < probabilities
        random_spins = np.random.randint(low=1, high=q+1, size=lattice.shape)
        lattice = np.where(flip_mask, random_spins, lattice)

    return lattice


q = 4  # Number of states
array = np.random.randint(low=1, high=q, size=(10, 10))
print(array)
steps = 1000  # Number of Monte Carlo steps
result = potts_monte_carlo_1(array, q, steps)
print(result)
