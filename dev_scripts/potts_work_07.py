import numpy as np
import matplotlib.pyplot as plt

def find_clusters(array):
    visited = np.zeros_like(array)
    clusters = []

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if visited[i, j] == 0:
                cluster = set()
                dfs(array, i, j, visited, cluster)
                if len(cluster) > 0:
                    clusters.append(cluster)

    return clusters

def dfs(array, i, j, visited, cluster):
    stack = [(i, j)]
    target = array[i, j]

    while stack:
        x, y = stack.pop()
        if visited[x, y] == 1:
            continue
        visited[x, y] = 1
        cluster.add((x, y))

        if x - 1 >= 0 and array[x - 1, y] == target and visited[x - 1, y] == 0:
            stack.append((x - 1, y))
        if x + 1 < array.shape[0] and array[x + 1, y] == target and visited[x + 1, y] == 0:
            stack.append((x + 1, y))
        if y - 1 >= 0 and array[x, y - 1] == target and visited[x, y - 1] == 0:
            stack.append((x, y - 1))
        if y + 1 < array.shape[1] and array[x, y + 1] == target and visited[x, y + 1] == 0:
            stack.append((x, y + 1))



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

# Find cluster boundaries
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

# Find the number of clusters each boundary point is shared with
num_clusters_shared = find_boundary_clusters(boundaries, clusters)

print("Number of clusters each boundary point is shared with:")
for i, num_clusters in enumerate(num_clusters_shared):
    print(f"Boundary Point {i + 1}: {num_clusters}")
