import random
import numpy as np
import matplotlib.pyplot as plt

N = 10  # Size of the lattice
q = 5  # Number of spin states
num_steps = 1000  # Number of simulation steps

# Initialize the lattice
states = np.array([[random.randint(0, q-1) for _ in range(N)] for _ in range(N)])


for step in range(num_steps):
    print(step)
    for ni in range(1,N-1):
        for nj in range(1,N-1):
            current_sate = states[ni][nj]
            a = current_sate == states[ni-1][nj+0]
            b = current_sate == states[ni-1][nj-1]
            c = current_sate == states[ni+0][nj-1]
            d = current_sate == states[ni+1][nj-1]
            e = current_sate == states[ni+1][nj+0]
            f = current_sate == states[ni+1][nj+1]
            g = current_sate == states[ni+0][nj+1]
            h = current_sate == states[ni-0][nj+1]
            e0 = 1*a+1*b+1*c+1*d+1*e+1*f+1*g+1*h
            new_state = random.randint(0, q-1)
            a = new_state == states[ni-1][nj+0]
            b = new_state == states[ni-1][nj-1]
            c = new_state == states[ni+0][nj-1]
            d = new_state == states[ni+1][nj-1]
            e = new_state == states[ni+1][nj+0]
            f = new_state == states[ni+1][nj+1]
            g = new_state == states[ni+0][nj+1]
            h = new_state == states[ni-0][nj+1]
            e1 = 1*a+1*b+1*c+1*d+1*e+1*f+1*g+1*h
            if e0-e1 < 0.0:
                states[ni][nj] = new_state

states = states[1:-1, 1:-1]

# Plotting the lattice configuration
plt.imshow(states, cmap='tab10')
plt.colorbar(ticks=range(q))
plt.title("Potts Model Lattice")
plt.show()