import matplotlib.pyplot as plt
import numpy as np
def potts_model_grain_growth_updated_hamiltonian(nx, ny, nz, S, temperature, iterations):
    """
    Perform Q-state Potts model grain growth simulation on a 3D grid with updated Hamiltonian calculation.

    :param nx: Number of grid points along the x-axis
    :param ny: Number of grid points along the y-axis
    :param nz: Number of grid points along the z-axis
    :param S: Number of states (grains)
    :param temperature: Simulation temperature
    :param iterations: Number of iterations for the simulation
    :return: Updated state matrix, array of Hamiltonian values at each iteration
    """
    # Initialize the state matrix with random states
    state_matrix = np.random.randint(1, S + 1, size=(nx, ny, nz))
    hamiltonians = np.zeros(iterations)  # Array to store Hamiltonian values for all iterations

    # Define a function to calculate the Hamiltonian for a given state
    def calculate_hamiltonian(state_matrix):
        hamiltonian = 0
        # Sum over all nearest neighbor pairs
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    # Periodic boundary conditions
                    for dx, dy, dz in [(1,0,0), (0,1,0), (0,0,1), (-1,0,0), (0,-1,0), (0,0,-1)]:
                        x2 = (x + dx) % nx
                        y2 = (y + dy) % ny
                        z2 = (z + dz) % nz
                        if state_matrix[x, y, z] != state_matrix[x2, y2, z2]:
                            hamiltonian += 1
        return hamiltonian

    # Main simulation loop
    for iteration in range(iterations):
        print(f'Iteration: {iteration}')
        ham_ = 0
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    # Pick a random new state different from the current state
                    current_state = state_matrix[x, y, z]
                    new_state = np.random.randint(1, S + 1)
                    while new_state == current_state:
                        new_state = np.random.randint(1, S + 1)

                    # Calculate the change in Hamiltonian if we were to update the state
                    original_hamiltonian = calculate_hamiltonian(state_matrix)
                    state_matrix[x, y, z] = new_state
                    new_hamiltonian = calculate_hamiltonian(state_matrix)
                    delta_hamiltonian = new_hamiltonian - original_hamiltonian

                    # Decide whether to accept the new state
                    if delta_hamiltonian <= 0:
                        continue  # Accept new state
                    else:
                        state_matrix[x, y, z] = current_state  # Revert to original state

        # Store the Hamiltonian for this iteration
        hamiltonians[iteration] = calculate_hamiltonian(state_matrix)

    return state_matrix, hamiltonians

# Example usage with a small grid and parameters for testing purposes
nx, ny, nz = 10, 10, 10  # Small grid dimensions for demonstration
S = 10  # Number of states
temperature = 100  # Simulation temperature
iterations = 25  # Number of iterations

# Call the function with the same parameters for comparison
updated_state_matrix, hamiltonians_array = potts_model_grain_growth_updated_hamiltonian(nx, ny, nz, S, temperature, iterations)

# Display the Hamiltonians array for confirmation
plt.plot(hamiltonians_array)
