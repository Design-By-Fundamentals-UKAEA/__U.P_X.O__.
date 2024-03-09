import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label

N = 10  # Size of the lattice
q = 3  # Number of spin states
num_steps = 1000  # Number of simulation steps

# Initialize the lattice
states = np.array([[random.randint(0, q-1) for _ in range(N)] for _ in range(N)])
# Plotting the lattice configuration
plt.imshow(states, cmap='tab10')
plt.colorbar(ticks=range(q))
plt.title("Potts Model Lattice")
plt.show()


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

# Plotting the lattice configuration
plt.imshow(states, cmap='tab10')
plt.colorbar(ticks=range(q))
plt.title("Potts Model Lattice")
plt.show()

states = states[1:-1, 1:-1]

# Plotting the lattice configuration
plt.imshow(states, cmap='tab10')
plt.colorbar(ticks=range(q))
plt.title("Potts Model Lattice")
plt.show()


# Label connected components (clusters)
labeled_array, num_clusters = label(states)



def dfs(i, j, cluster_label, cluster, visited):
    stack = [(i, j)]
    while stack:
        x, y = stack.pop()
        if x < 0 or y < 0 or x >= N or y >= N or visited[x][y] or states[x][y] != cluster_label:
            continue
        visited[x][y] = True
        cluster.append((x, y))
        stack.extend([(x-1, y), (x+1, y), (x, y-1), (x, y+1)])

N = states.shape[0]
visited = np.zeros_like(states, dtype=bool)
clusters = []

for i in range(N):
    for j in range(N):
        if not visited[i][j]:
            cluster = []
            dfs(i, j, states[i][j], cluster, visited)
            clusters.append(cluster)

# Plot each cluster
for i, cluster in enumerate(clusters):
    plt.figure()
    cluster_array = np.zeros_like(states)
    for cell in cluster:
        cluster_array[cell] = 1
    plt.imshow(cluster_array, cmap='binary')
    plt.title(f"Cluster {i+1}")
    plt.show()