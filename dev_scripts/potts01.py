import random
import matplotlib.pyplot as plt


# Example usage:
N = 50  # Size of the lattice
q = 2  # Number of spin states
num_steps = 100000  # Number of simulation steps


# def initialize_lattice(N, q):
lattice = [[random.randint(0, q-1) for _ in range(N)] for _ in range(N)]
#return lattice

# Plotting the lattice configuration
plt.subplot(2, 2, 1)
plt.imshow(lattice, cmap='tab10')
plt.colorbar(ticks=range(q))
plt.title("Potts Model Lattice")
plt.show()




# def simulate_potts_model(N, q, num_steps):
#lattice = initialize_lattice(N, q)

for step in range(num_steps):
    i = random.randint(0, N-1)
    j = random.randint(0, N-1)

    current_spin = lattice[i][j]
    neighbors_sum = 0
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ni = (i + di) % N
        nj = (j + dj) % N
        neighbors_sum += 1 if lattice[ni][nj] == current_spin else 0

    new_spin = random.randint(0, q-1)
    energy_diff = neighbors_sum - 1 if new_spin == current_spin else neighbors_sum

    if energy_diff < 0:
        lattice[i][j] = new_spin

    #return lattice

## Example usage:
#N = 100  # Size of the lattice
#q = 4  # Number of spin states
#num_steps = 1000  # Number of simulation steps

#lattice = simulate_potts_model(N, q, num_steps)

# Plotting the lattice configuration
plt.subplot(2, 2, 2)
plt.imshow(lattice, cmap='tab10')
plt.colorbar(ticks=range(q))
plt.title("Potts Model Lattice")
plt.show()
