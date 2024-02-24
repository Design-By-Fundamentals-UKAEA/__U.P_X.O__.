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

# Plot the clusters
plt.figure(figsize=(6, 6))
plt.imshow(array, cmap='jet', origin='lower')

colors = plt.cm.get_cmap('tab10').colors  # Generate colors for clusters
for i, cluster in enumerate(clusters):
    for point in cluster:
        plt.text(point[1], point[0], str(array[point]), color=colors[i], ha='center', va='center')

plt.xticks([])
plt.yticks([])
plt.title('Individual Clusters')
plt.show()
