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
        if y - 1 >= 0 and array[x, y - 1] == t arget and visited[x, y - 1] == 0:
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






def calculate_cluster_area(cluster):
    return len(cluster)

def calculate_cluster_perimeter(cluster, boundary):
    perimeter = 0
    for point in boundary:
        x, y = point
        neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        for nx, ny in neighbors:
            if (nx, ny) not in cluster and (nx, ny) not in boundary:
                perimeter += 1
    return perimeter

def calculate_cluster_properties(clusters, boundaries):
    areas = []
    perimeters = []

    for i, cluster in enumerate(clusters):
        boundary = boundaries[i]
        area = calculate_cluster_area(cluster)
        perimeter = calculate_cluster_perimeter(cluster, boundary)
        areas.append(area)
        perimeters.append(perimeter)

    return areas, perimeters


##############################################################
##############################################################
##############################################################



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





# Find cluster boundaries
boundaries = find_cluster_boundaries(clusters)





# Plot the boundaries
plt.figure(figsize=(6, 6))
plt.imshow(array, cmap='jet', origin='lower')

for boundary in boundaries:
    for point in boundary:
        x, y = point
        plt.plot(y, x, 'ro', markersize=5)

plt.xticks([])
plt.yticks([])
plt.title('Cluster Boundaries')
plt.show()


# Calculate cluster areas and perimeters
areas, perimeters = calculate_cluster_properties(clusters, boundaries)

# Print cluster areas and perimeters
for i, cluster in enumerate(clusters):
    print(f"Cluster {i+1}: Area={areas[i]}, Perimeter={perimeters[i]}")
