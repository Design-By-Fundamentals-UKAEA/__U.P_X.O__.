import numpy as np
from scipy.ndimage import label
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# Generating a 3D random integer matrix with values between 1 and 5, inclusive
matrix_size = 10  # Define the size of the matrix
random_matrix = np.random.randint(1, 6, size=(matrix_size, matrix_size, matrix_size))

def identify_grains(matrix):
    grains = []  # List to store the identified grains
    for value in np.unique(matrix):
        if value == 0:  # Skipping value 0, assuming it doesn't represent a grain
            continue

        # Creating a binary matrix for the current value
        binary_matrix = (matrix == value).astype(int)

        # Performing connected component labeling
        labeled_matrix, num_features = label(binary_matrix)

        # Storing grains, each represented by a unique label in labeled_matrix
        for i in range(1, num_features + 1):
            grain = np.where(labeled_matrix == i, value, 0)
            grains.append(grain)

    return grains

def is_boundary(matrix, x, y, z):
    current_value = matrix[x, y, z]
    neighbors = [
        (x-1, y, z), (x+1, y, z),
        (x, y-1, z), (x, y+1, z),
        (x, y, z-1), (x, y, z+1)
    ]

    for nx, ny, nz in neighbors:
        if nx < 0 or nx >= matrix_size or ny < 0 or ny >= matrix_size or nz < 0 or nz >= matrix_size:
            return True  # Outside the matrix bounds, consider it a boundary
        if matrix[nx, ny, nz] != current_value:
            return True  # Neighbor has a different value, it's a boundary point
    return False

def draw_cube(ax, position, color):
    # Cube vertices
    r = [0, 1]
    vertices = np.array([[x, y, z] for x in r for y in r for z in r])
    vertices += position

    # Create the sides of the cube
    faces = [
        [vertices[j] for j in [0, 1, 3, 2]], [vertices[j] for j in [4, 5, 7, 6]],
        [vertices[j] for j in [0, 1, 5, 4]], [vertices[j] for j in [2, 3, 7, 6]],
        [vertices[j] for j in [0, 2, 6, 4]], [vertices[j] for j in [1, 3, 7, 5]]
    ]

    # Create a 3D polygon collection and add it to the axes
    poly3d = [faces]
    ax.add_collection3d(Poly3DCollection(poly3d, facecolors=color, linewidths=0.1, edgecolors='k', alpha=0.66))

def plot_3d_grains(matrix):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Colors for different values in the matrix
    colors = ['red', 'green', 'blue', 'yellow', 'cyan']

    # Plotting each cube in the 3D matrix
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            for k in range(matrix.shape[2]):
                value = matrix[i, j, k]
                draw_cube(ax, (i, j, k), colors[value-1])
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('3D Visualization of Grains with Cubes')

    # Setting the aspect ratio for equal scaling along all axes
    ax.set_box_aspect([1,1,1])

    plt.show()

# Identifying boundary points
boundary_points = []
for x in range(matrix_size):
    for y in range(matrix_size):
        for z in range(matrix_size):
            if is_boundary(random_matrix, x, y, z):
                boundary_points.append((x, y, z))

grains = identify_grains(random_matrix)

print(f"Total number of grains identified: {len(grains)}")
# Optional: Print or analyze the individual grains
