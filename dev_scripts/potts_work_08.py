import numpy as np
import matplotlib.pyplot as plt

def find_clusters(array):
    visited = np.zeros_like(array)
    clusters = []

    rows, cols = array.shape
    indices = np.arange(rows * cols).reshape(rows, cols)

    while True:
        unvisited = indices[visited == 0]
        if len(unvisited) == 0:
            break

        start_idx = unvisited[0]
        start_coord = np.unravel_index(start_idx, array.shape)
        target = array[start_coord]

        cluster = set()
        dfs(array, start_coord, visited, cluster, target)
        clusters.append(cluster)

    return clusters

def dfs(array, coord, visited, cluster, target):
    stack = [coord]

    while stack:
        x, y = stack.pop()
        if visited[x, y] == 1:
            continue
        visited[x, y] = 1
        cluster.add((x, y))

        neighbors = get_neighbors(array, x, y)
        stack.extend(neighbors[visited[neighbors[:, 0], neighbors[:, 1]] == 0])

def get_neighbors(array, x, y):
    rows, cols = array.shape
    neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
    valid_neighbors = [(nx, ny) for nx, ny in neighbors if 0 <= nx < rows and 0 <= ny < cols]
    return np.array(valid_neighbors)

def find_cluster_boundaries(clusters):
    boundaries = []

    for cluster in clusters:
        boundary = set()

        for point in cluster:
            x, y = point
            neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]

            for nx, ny in neighbors:
                if (nx, ny) not in cluster:
                    boundary.add((nx, ny))

        boundaries.append(boundary)

    return boundaries

# Given array
array = np.array([[2, 2, 1, 1, 1, 2, 2, 2, 2, 2],
                  [3, 2, 2, 1, 1, 2, 2, 2, 2, 2],
                  [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                  [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                  [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                  [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                  [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                  [2, 2, 3, 2, 2, 2, 2, 2, 2, 2],
                  [2, 2, 2, 2, 1, 2, 3, 3, 2, 2],
                  [2, 2, 3, 3, 1, 2, 3, 3, 2, 2]])

# Find individual clusters
clusters = find_clusters(array)

# Find cluster boundaries
boundaries = find_cluster_boundaries(clusters)

# Plot the clusters and boundaries
plt.figure(figsize=(6, 6))
plt.imshow(array, cmap='jet', origin='lower')

colors = plt.cm.get_cmap('tab10').colors  # Generate colors for clusters
for i, cluster in enumerate(clusters):
    for point in cluster:
        plt.text(point[1], point[0], str(array[point]), color=colors[i], ha='center', va='center')

    boundary_points = np.array(list(boundaries[i]))
    plt.plot(boundary_points[:, 1], boundary_points[:, 0], 'k.', markersize=5)

plt.xticks([])
plt.yticks([])
plt.title('Individual Clusters with Boundaries')
plt.show()









# Choose the index of the cluster to plot its boundary
cluster_index = 3

# Plot the boundary of the selected cluster
plt.figure(figsize=(6, 6))
plt.imshow(array, cmap='jet', origin='lower')

boundary_points = np.array(list(boundaries[cluster_index]))
plt.plot(boundary_points[:, 1], boundary_points[:, 0], 'ks', markersize=5)

plt.xticks([])
plt.yticks([])
plt.title(f'Boundary of Cluster {cluster_index}')
plt.show()





i = 0
cluster = clusters[0]
plt.figure(figsize=(6, 6))
plt.imshow(array, cmap='jet', origin='lower')
colors = plt.cm.get_cmap('tab10').colors  # Generate colors for clusters
for i, cluster in enumerate(clusters):
    for point in cluster:
        plt.text(point[1], point[0], str(array[point]), color=colors[i], ha='center', va='center')

    boundary_points = np.array(list(boundaries[i]))
    plt.plot(boundary_points[:, 1], boundary_points[:, 0], 'k.', markersize=5)

plt.xticks([])
plt.yticks([])
plt.title('Individual Clusters with Boundaries')
plt.show()
