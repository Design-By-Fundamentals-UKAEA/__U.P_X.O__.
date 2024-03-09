import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from scipy.ndimage import convolve

# State matrix S as provided
S = np.array([[12, 12, 11, 11, 11, 11, 11, 12, 12],
              [12, 12, 11, 11, 11, 11, 11, 12, 12],
              [12, 12, 11, 11, 11, 11, 11, 12, 12],
              [12, 12, 11, 11, 11, 11, 11,  6,  6],
              [ 6,  5,  5, 11, 11, 11,  6,  6,  6],
              [ 6,  5,  5,  5,  6,  6,  6,  6,  6],
              [ 6,  5,  5,  5,  6,  6,  6,  6,  6],
              [12,  5,  5,  5,  6,  6,  6,  6,  6],
              [12, 12, 11, 11, 11,  6,  6, 12, 12]])

S = PXGS.gs[8].s

# Label the grains
labeled_grains = label(S, connectivity=2)
unique_grains = np.unique(labeled_grains)

# Prepare to collect boundary info for each grain
grain_boundaries = {grain: [] for grain in unique_grains if grain != 0}  # Exclude background
junction_points = []

# Kernel to identify neighbors
kernel = np.array([[1, 1, 1],
                   [1, 0, 1],
                   [1, 1, 1]])

for grain in unique_grains:
    if grain == 0:  # Skip background
        continue

    # Create a mask for the current grain
    grain_mask = labeled_grains == grain

    # Identify the boundary of the grain
    grain_boundary = convolve(grain_mask.astype(int), kernel, mode='constant', cval=0) < kernel.sum()
    grain_boundary = np.logical_and(grain_boundary, grain_mask)

    # Store the boundary pixels for the grain
    boundary_pixels = np.argwhere(grain_boundary)
    grain_boundaries[grain] = boundary_pixels

    # Identify potential junction points (for simplicity, here we consider any boundary pixel with less than 8 neighbors)
    for y, x in boundary_pixels:
        neighbors = labeled_grains[max(y-1, 0):y+2, max(x-1, 0):x+2]
        unique_neighbors = np.unique(neighbors)
        if len(unique_neighbors) - (0 in unique_neighbors) > 2:  # More than two grains meeting
            junction_points.append((y, x))

# Convert junction_points to a unique list of tuples
junction_points = list(set(junction_points))
junction_points = np.array(junction_points)
junction_points_x_all = junction_points.T[1]
junction_points_y_all = junction_points.T[0]
# Output information
print(f"Total grains: {len(unique_grains)-1}")  # Excluding background
print(f"Junction Points: {junction_points}")
for grain, boundaries in grain_boundaries.items():
    print(f"Grain {grain} has {len(boundaries)} boundary pixels")


plt.figure(figsize=(10, 10))
plt.imshow(S, cmap='viridis', interpolation='nearest')
plt.plot(junction_points_x_all, junction_points_y_all, 'k.', markersize=5, markerfacecolor=None, markeredgecolor='k')

plt.title('Grain Structure with Boundaries and Junction Points')
plt.axis('off')

# Note: This approach simplifies junction point identification; a more precise method may require analyzing neighbor configurations

gid = 20

#plt.figure(figsize=(10, 10))
#plt.imshow(S, cmap='viridis', interpolation='nearest')
jp_nparrays = {}
for gid in grain_boundaries.keys():
    print(20*'-')
    print(gid)
    gb = grain_boundaries[gid]
    print(gb)
    # Identify those junction point locations which belong to this grain boundary
    gb_jp_loc = list(np.where((gb[:, None] == junction_points).all(-1).any(0))[0])
    # Get the junction point cooerdinates
    print(gb_jp_loc)
    if gb_jp_loc:
        gb_jp = [junction_points[i] for i in gb_jp_loc]
        j_y, j_x = zip(*gb_jp)
        jp_nparrays[gid] = np.vstack((j_y, j_x))
        #plt.plot(j_x, j_y, 'gx', markersize=10, markerfacecolor=None, markeredgecolor='k')  #

plt.title('Grain Structure with Boundaries and Junction Points')
plt.axis('off')

jp_nparrays[343].T


# Plotting the grain structure, boundaries, and junction points
plt.figure(figsize=(10, 10))
# Plot the original grain structure
plt.imshow(S, cmap='viridis', interpolation='nearest')
plt.plot(j_x, j_y, 'gx', markersize=10, markerfacecolor=None, markeredgecolor='k')  #
plt.title('Grain Structure with Boundaries and Junction Points')
plt.axis('off')













# Plotting the grain structure, boundaries, and junction points
plt.figure(figsize=(10, 10))

# Plot the original grain structure
plt.imshow(S, cmap='nipy_spectral', interpolation='nearest')
plt.title('Grain Structure with Boundaries and Junction Points')
plt.axis('off')

# Plot grain boundaries for each grain
for grain, boundaries in grain_boundaries.items():
    y, x = zip(*boundaries)
    plt.plot(x, y, 'wx', markersize=2)  # White dots for boundaries

# Highlight junction points
j_y, j_x = zip(*junction_points)
plt.plot(j_x, j_y, 'go', markersize=5, markerfacecolor=None, markeredgecolor='black')  # Red 'o' for junction points

plt.show()

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from itertools import product


# Extract boundary and junction points for grain 6
boundary_pixels_grain_6 = np.array(grain_boundaries[6])
junction_points_grain_6 = [jp for jp in junction_points if tuple(jp) in boundary_pixels_grain_6.tolist()]

# Convert boundary pixels to a more convenient format for graph construction
pixel_to_node = {tuple(pixel): idx for idx, pixel in enumerate(boundary_pixels_grain_6)}
node_to_pixel = {idx: pixel for pixel, idx in pixel_to_node.items()}

# Prepare for graph construction
num_pixels = len(boundary_pixels_grain_6)
graph_matrix = np.zeros((num_pixels, num_pixels))

# Fill the adjacency matrix for the graph
for i, pixel in enumerate(boundary_pixels_grain_6):
    for j, neighbor_pixel in enumerate(boundary_pixels_grain_6):
        if i != j:
            # Check if pixels are neighbors (including diagonals)
            if np.linalg.norm(pixel - neighbor_pixel, ord=1) <= 2:
                graph_matrix[i, j] = 1  # Adjacent boundary pixels

# Convert the adjacency matrix to a CSR matrix
graph_csr = csr_matrix(graph_matrix)

# Map junction points to nodes in the graph
junction_nodes = [pixel_to_node[tuple(jp)] for jp in junction_points_grain_6]

# Calculate shortest paths between all pairs of junction points
dist_matrix, predecessors = dijkstra(csgraph=graph_csr, indices=junction_nodes, return_predecessors=True, directed=False)




def is_junction(y, x, junction_points):
    return (y, x) in junction_points

# Placeholder for boundary segments
boundary_segments = []

# For each grain, attempt to find segments between junction points
for grain, boundaries in grain_boundaries.items():
    # Placeholder for current segment's start point
    start_point = None

    for point in boundaries:
        y, x = point

        # If the current point is a junction, it could be a start or end of a segment
        if is_junction(y, x, junction_points):
            if start_point is None:
                # Mark as start of a new segment
                start_point = (y, x)
            else:
                # Found the end point of the segment
                end_point = (y, x)

                # Save the segment (start_point, end_point)
                boundary_segments.append((start_point, end_point))

                # Reset start_point for the next segment
                start_point = end_point  # Or set to None to start fresh for a new segment
