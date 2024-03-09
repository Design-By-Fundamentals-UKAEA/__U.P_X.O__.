import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

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



N = states.shape[0]
visited = np.zeros_like(states, dtype=bool)
clusters = []

for i in range(N):
    for j in range(N):
        if visited[i, j]:
            continue
        
        cluster_label = states[i, j]
        cluster = []
        stack = [(i, j)]
        
        while stack:
            x, y = stack.pop()
            
            if visited[x, y] or states[x, y] != cluster_label:
                continue
            
            visited[x, y] = True
            cluster.append((x, y))
            
            if x > 0:
                stack.append((x - 1, y))
            if x < N - 1:
                stack.append((x + 1, y))
            if y > 0:
                stack.append((x, y - 1))
            if y < N - 1:
                stack.append((x, y + 1))
        
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
    







from skimage.measure import find_contours


# Calculate boundary points for all clusters
boundary_points_clusters = []

for cluster in clusters:
    cluster_array = np.zeros_like(states)
    for point in cluster:
        cluster_array[point] = 1
    
    # Find the contours of the cluster
    contours = find_contours(cluster_array, 0.5, fully_connected='low')
    
    # Get the outer boundary (largest contour)
    boundary = contours[np.argmax([len(contour) for contour in contours])]
    
    # Sort boundary points in clockwise order
    centroid = np.mean(boundary, axis=0)
    angles = np.arctan2(boundary[:, 0] - centroid[0], boundary[:, 1] - centroid[1])
    boundary_points = np.array([list(boundary[i]) for i in np.argsort(angles)])
    
    boundary_points_clusters.append(boundary_points)

# Plot each cluster with boundary points
for i, (cluster, boundary_points) in enumerate(zip(clusters, boundary_points_clusters)):
    plt.figure()
    cluster_array = np.zeros_like(states)
    for cell in cluster:
        cluster_array[cell] = 1
    plt.imshow(cluster_array, cmap='binary')
    plt.plot(boundary_points[:, 1], boundary_points[:, 0], 'ro-', linewidth=2)
    plt.title(f"Cluster {i+1}")
    plt.show()





# Calculate area for each cluster
cluster_areas = [len(cluster) for cluster in clusters]


# Print area for each cluster
for i, area in enumerate(cluster_areas):
    print(f"Cluster {i+1} area: {area}")
    
    
    
    


# Calculate aspect ratio for each cluster
cluster_aspect_ratios = []

for cluster in clusters:
    cluster_array = np.array(cluster)
    min_x, min_y = np.min(cluster_array, axis=0)
    max_x, max_y = np.max(cluster_array, axis=0)
    
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    
    aspect_ratio = width / height
    cluster_aspect_ratios.append(aspect_ratio)

# Print aspect ratio for each cluster
for i, aspect_ratio in enumerate(cluster_aspect_ratios):
    print(f"Cluster {i+1} aspect ratio: {aspect_ratio}")
    






# Calculate morphological orientation for each cluster
cluster_orientations = []

for cluster in clusters:
    cluster_array = np.array(cluster)
    x = cluster_array[:, 0]
    y = cluster_array[:, 1]
    
    # Calculate second-order moments
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_centered = x - x_mean
    y_centered = y - y_mean
    Ixx = np.sum(x_centered**2)
    Iyy = np.sum(y_centered**2)
    Ixy = np.sum(x_centered * y_centered)
    
    # Calculate orientation angle
    theta = 0.5 * np.arctan2(2 * Ixy, (Ixx - Iyy))
    cluster_orientations.append(theta)

# Print morphological orientation for each cluster
for i, orientation in enumerate(cluster_orientations):
    print(f"Cluster {i+1} orientation: {orientation} radians")
    
    
    
    




# Generate random color for each cluster
colors = []
for _ in range(len(clusters)):
    r = random.random()
    g = random.random()
    b = random.random()
    colors.append((r, g, b))

# Create figure and axes
fig, ax = plt.subplots()

# Plot each cluster
for cluster, color in zip(clusters, colors):
    cluster_array = np.array(cluster)
    x = cluster_array[:, 1]
    y = cluster_array[:, 0]
    ax.plot(x, y, 's', color=color, markersize=20)

# Set limits and aspect ratio of the plot
ax.set_xlim([0, N-1])
ax.set_ylim([0, N-1])
ax.set_aspect('equal')

# Show the plot
plt.show()



# Calculate the equivalent circle diameter of each cluster
diameter = [2 * np.sqrt(area / np.pi) for area in cluster_areas]



