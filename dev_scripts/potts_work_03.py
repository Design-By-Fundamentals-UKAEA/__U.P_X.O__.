import numpy as np
import matplotlib.pyplot as plt



def find_clusters(array):
    size = array.shape[0]
    visited = np.zeros_like(array)
    clusters = []

    for i in range(size):
        for j in range(size):
            if visited[i, j] == 0:
                cluster = set()
                dfs(array, i, j, visited, cluster)
                if len(cluster) > 0:
                    clusters.append(cluster)

    return clusters

def dfs(array, i, j, visited, cluster):
    size = array.shape[0]
    if i < 0 or i >= size or j < 0 or j >= size:
        return
    if visited[i, j] == 1:
        return
    visited[i, j] = 1
    cluster.add((i, j))
    if i - 1 >= 0 and array[i - 1, j] == array[i, j]:
        dfs(array, i - 1, j, visited, cluster)
    if i + 1 < size and array[i + 1, j] == array[i, j]:
        dfs(array, i + 1, j, visited, cluster)
    if j - 1 >= 0 and array[i, j - 1] == array[i, j]:
        dfs(array, i, j - 1, visited, cluster)
    if j + 1 < size and array[i, j + 1] == array[i, j]:
        dfs(array, i, j + 1, visited, cluster)

# Given array
array = np.array([[2, 2, 1, 1, 1, 2, 2, 2, 2, 2],
                  [3, 2, 2, 1, 1, 2, 2, 2, 2, 2],
                  [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                  [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                  [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                  [2, 2, 2, 2, 2, 3, 2, 2, 2, 2],
                  [2, 2, 2, 2, 3, 2, 3, 2, 2, 2],
                  [2, 2, 3, 2, 2, 3, 2, 2, 2, 2],
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


1.5 HR ATCTSD - 2
1.5 HR GCSE foundation math - 1
1.5 HR GCSE higher - 1
1.5 HR GCSE foundation Science - 1
1.0 HR Year 8 Math - 1
2.0 HR Year 6 Math - 2
1.5 HR Year 9 Math - 1
