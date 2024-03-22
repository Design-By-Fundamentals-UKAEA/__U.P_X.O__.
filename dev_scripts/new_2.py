import matplotlib.pyplot as plt
from itertools import product
import numpy as np

nx, ny, nz = 25, 25, 25  # Small grid dimensions for demonstration
S = 10  # Number of states
iterations = 100  # Number of iterations
temperature = 0.0001

# Initialize the state matrix with random states
state_matrix = np.random.randint(1, S + 1, size=(nx, ny, nz))
hamiltonians = np.zeros(iterations)  # Array to store Hamiltonian values for all iterations

STRUCTURE = np.array(list(product([-1, 0, 1], repeat=3)))

# Pre-calculate periodic boundary indices for the entire grid
indices = np.array([(x, y, z) for x in range(nx) for y in range(ny) for z in range(nz)])
x_indices, y_indices, z_indices = indices[:, 0], indices[:, 1], indices[:, 2]

plt.figure()
plt.imshow(state_matrix[:, :, 5])

# Main simulation loop
for iteration in range(iterations):
    print(f'Iteration: {iteration}')
    HAM = np.zeros((nx, ny, nz))

    for shift in STRUCTURE:
        x2 = (x_indices + shift[0]) % nx
        y2 = (y_indices + shift[1]) % ny
        z2 = (z_indices + shift[2]) % nz

        # Vectorized comparison for new and current state energies
        E1 = state_matrix[x_indices, y_indices, z_indices] == state_matrix[x2, y2, z2]

        # Only consider unique new states for changes
        new_states = np.random.randint(1, S + 1, size=state_matrix.shape)
        mask = new_states != state_matrix
        unique_new_states = np.where(mask, new_states, state_matrix)

        E2 = unique_new_states[x_indices, y_indices, z_indices] == state_matrix[x2, y2, z2]

        E1_total = np.sum(E1)
        E2_total = np.sum(E2)
        delE = E2_total - E1_total

        # Accept or reject the new state in a vectorized manner
        accept = (delE < 0) | (np.random.rand(nx, ny, nz) < np.exp(-delE / temperature))
        state_matrix = np.where(accept, unique_new_states, state_matrix)

    hamiltonians[iteration] = np.sum(HAM)

plt.figure()
plt.imshow(state_matrix[:, :, 5])
