import matplotlib.pyplot as plt
from itertools import product
import numpy as np

nx, ny, nz = 25, 25, 25  # Small grid dimensions for demonstration
S = 10 # Number of states
iterations = 100  # Number of iterations

temperature = 0.0001
temp_min = 0.01  # Minimum temperature
temp_max = 10  # Maximum temperature
temperature_gradient = np.linspace(temp_min, temp_max, nz)

plt.figure()
plt.plot(temperature_gradient)

# Initialize the state matrix with random states
state_matrix = np.random.randint(1, S + 1, size=(nx, ny, nz))
hamiltonians = np.zeros(iterations)  # Array to store Hamiltonian values for all iterations


STRUCTURE = list(product([-1, 0, 1], repeat=3))

plt.figure()
plt.imshow(state_matrix[1, :, :])

# Main simulation loop
for iteration in range(iterations):
    HAM = np.zeros((nx, ny, nz))
    print(f'Iteration: {iteration}')
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                temperature = temperature_gradient[z]
                # --------------------------------
                # Calculate energy at site
                E1, E2 = [], []
                current_state = state_matrix[x, y, z]
                for dx, dy, dz in STRUCTURE:
                    x2, y2, z2 = (x + dx) % nx, (y + dy) % ny, (z + dz) % nz
                    E1.append(current_state == state_matrix[x2, y2, z2])
                # --------------------------------
                # Flip the state
                new_state = np.random.randint(1, S + 1)
                while new_state == current_state:
                    new_state = np.random.randint(1, S + 1)
                # --------------------------------
                # Calculate new energy at site
                for dx, dy, dz in STRUCTURE:
                    x2, y2, z2 = (x + dx) % nx, (y + dy) % ny, (z + dz) % nz
                    E2.append(new_state == state_matrix[x2, y2, z2])
                # --------------------------------
                # Compare the energies
                E1_total, E2_total = 27-sum(E1), 27-sum(E2)
                delE = E2_total - E1_total
                # --------------------------------
                # Accept or reject the new state
                if delE < 0 or np.random.rand() < np.exp(-delE / temperature):
                    state_matrix[x, y, z] = new_state
                    HAM[x, y, z] = E2_total
                else:
                    HAM[x, y, z] = E1_total
    hamiltonians[iteration] = HAM.sum()

for i in range(nx):
    plt.figure()
    plt.imshow(state_matrix[i, :, :])
