import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.signal import argrelextrema

def potts_model1(x_size, y_size, q, nsteps):
  """
  Simulates grain growth using Q-state Potts model Monte-Carlo simulation.

  Args:
    x_size: The size of the domain in the x-axis.
    y_size: The size of the domain in the y-axis.
    q: The number of states in the Potts model.

  Returns:
    A numpy array of the grain boundaries.
  """

  grains = np.zeros((x_size, y_size), dtype=int)
  for i in range(x_size):
    for j in range(y_size):
      grains[i, j] = random.randint(0, q - 1)

  for _ in range(nsteps):
    current_index = random.randint(0, x_size * y_size - 1)
    x, y = current_index // y_size, current_index % y_size
    neighbors = []
    for i in range(max(0, x - 1), min(x + 2, x_size)):
      for j in range(max(0, y - 1), min(y + 2, y_size)):
        if i != x or j != y:
          neighbors.append(grains[i, j])
    new_state = random.choice(neighbors)
    grains[x, y] = new_state

  grain_boundaries = np.zeros((x_size, y_size), dtype=bool)
  for i in range(x_size):
    for j in range(y_size):
      grain_boundaries[i, j] = grains[i, j] != grains[i - 1, j] or grains[i, j] != grains[i, j - 1]

  return grains, grain_boundaries


def potts_model2(x_size, y_size, q, nsteps, temperature):
  """
  Simulates grain growth using Q-state Potts model Monte-Carlo simulation.

  Args:
    x_size: The size of the domain in the x-axis.
    y_size: The size of the domain in the y-axis.
    q: The number of states in the Potts model.

  Returns:
    A numpy array of the grain boundaries.
  """

  grains = np.zeros((x_size, y_size), dtype=int)
  for i in range(x_size):
    for j in range(y_size):
      grains[i, j] = random.randint(0, q - 1)

  for _ in range(10000):
    current_index = random.randint(0, x_size * y_size - 1)
    x, y = current_index // y_size, current_index % y_size
    neighbors = []
    for i in range(max(0, x - 1), min(x + 2, x_size)):
      for j in range(max(0, y - 1), min(y + 2, y_size)):
        if i != x or j != y:
          neighbors.append(grains[i, j])
    new_state = random.choice(neighbors)
    p = 1 / (1 + np.exp(-(grains[x, y] - new_state) / temperature))
    if random.random() < p:
      grains[x, y] = new_state

  grain_boundaries = np.zeros((x_size, y_size), dtype=bool)
  for i in range(x_size):
    for j in range(y_size):
      grain_boundaries[i, j] = grains[i, j] != grains[i - 1, j] or grains[i, j] != grains[i, j - 1]

  return grains, grain_boundaries

def potts_model3(x_size, y_size, q, nsteps, temperature):
  """
  Simulates grain growth using Q-state Potts model Monte-Carlo simulation.

  Args:
    x_size: The size of the domain in the x-axis.
    y_size: The size of the domain in the y-axis.
    q: The number of states in the Potts model.

  Returns:
    A numpy array of the grain boundaries.
  """

  grains = np.zeros((x_size, y_size), dtype=int)
  for i in range(x_size):
    for j in range(y_size):
      grains[i, j] = random.randint(0, q - 1)

  for _ in range(10000):
    current_index = random.randint(0, x_size * y_size - 1)
    x, y = current_index // y_size, current_index % y_size
    neighbors = []
    for i in range(max(0, x - 1), min(x + 2, x_size)):
      for j in range(max(0, y - 1), min(y + 2, y_size)):
        if i != x or j != y:
          neighbors.append(grains[i, j])
    new_state = random.choice(neighbors)
    energy_difference = grains[x, y] - new_state
    u = random.random()
    accept = False
    while not accept:
      if u < np.exp(-energy_difference / temperature):
        accept = True
      else:
        u = random.random()
    grains[x, y] = new_state

  grain_boundaries = np.zeros((x_size, y_size), dtype=bool)
  for i in range(x_size):
    for j in range(y_size):
      grain_boundaries[i, j] = grains[i, j] != grains[i - 1, j] or grains[i, j] != grains[i, j - 1]

  return grains, grain_boundaries

def find_clusters(array):
    visited = np.zeros_like(array)
    clusters = []

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if visited[i, j] == 0:
                cluster = []
                dfs(array, i, j, visited, cluster)
                if len(cluster) > 0:
                    clusters.append(cluster)

    return clusters

def find_clusters1(array):
    visited = np.zeros_like(array)
    clusters = []

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if visited[i, j] == 0:
                cluster = []
                stack = [(i, j)]
                target = array[i, j]

                while stack:
                    x, y = stack.pop()
                    if visited[x, y] == 1:
                        continue
                    visited[x, y] = 1
                    cluster.append((x, y))

                    if x - 1 >= 0 and array[x - 1, y] == target and visited[x - 1, y] == 0:
                        stack.append((x - 1, y))
                    if x + 1 < array.shape[0] and array[x + 1, y] == target and visited[x + 1, y] == 0:
                        stack.append((x + 1, y))
                    if y - 1 >= 0 and array[x, y - 1] == target and visited[x, y - 1] == 0:
                        stack.append((x, y - 1))
                    if y + 1 < array.shape[1] and array[x, y + 1] == target and visited[x, y + 1] == 0:
                        stack.append((x, y + 1))
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
        cluster.append((x, y))

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

def calculate_cluster_properties(clusters):
    areas = []
    perimeters = []
    aspect_ratios = []

    for cluster in clusters:
        cluster = np.array(cluster)

        area = len(cluster)
        perimeter = calculate_cluster_perimeter(cluster)

        min_x, min_y = np.min(cluster, axis=0)
        max_x, max_y = np.max(cluster, axis=0)
        width = max_x - min_x + 1
        height = max_y - min_y + 1
        aspect_ratio = width / height

        areas.append(area)
        perimeters.append(perimeter)
        aspect_ratios.append(aspect_ratio)

    return areas, perimeters, aspect_ratios



def calculate_cluster_perimeter(cluster):
    mask = np.zeros((cluster[:, 0].max() + 2, cluster[:, 1].max() + 2), dtype=bool)
    mask[cluster[:, 0], cluster[:, 1]] = True
    diff = np.diff(mask, axis=0)
    perimeter = np.sum(np.abs(diff)) + np.sum(np.abs(diff[:, :-1]))
    diff = np.diff(mask, axis=1)
    perimeter += np.sum(np.abs(diff)) + np.sum(np.abs(diff[:-1, :]))
    return perimeter

def find_neighboring_clusters(clusters, boundaries):
    neighboring_clusters = []

    for i in range(len(clusters)):
        neighboring_cluster_indices = []

        for j in range(len(clusters)):
            if i == j:
                continue

            if any(point in boundaries[j] for point in boundaries[i]):
                neighboring_cluster_indices.append(j)

        neighboring_clusters.append(neighboring_cluster_indices)
    return neighboring_clusters

def calculate_curvature(cluster_boundary):
    '''
    This code defines a new function calculate_curvature that takes a cluster boundary as input and performs the following steps:

    Converts the cluster boundary points to a numpy array.
    Calculates the first derivatives of the x and y coordinates using the np.gradient function.
    Calculates the second derivatives of the x and y coordinates using the np.gradient function.
    Calculates the curvature at each point using the formula (dx * d2y - dy * d2x) / ((dx ** 2 + dy ** 2) ** 1.5), where dx and dy are the first derivatives and d2x and d2y are the second derivatives.
    Returns the curvature array.
    '''
    # Convert cluster boundary points to numpy array
    points = np.array(list(cluster_boundary))

    # Calculate the first derivative
    dx = np.gradient(points[:, 0])
    dy = np.gradient(points[:, 1])

    # Calculate the second derivative
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)

    # Calculate the curvature
    curvature = (dx * d2y - dy * d2x) / ((dx ** 2 + dy ** 2) ** 1.5)

    return curvature

def calculate_curvatures_boundaries(boundaries):
    # Calculate curvatures for each cluster boundary
    curvatures = []
    for boundary in boundaries:
        curvature = calculate_curvature(boundary)
        curvatures.append(curvature)
    return curvatures

def calculate_morphological_orientation(cluster):
    '''
    In this code, the calculate_morphological_orientation function takes a
    cluster (boundary points) as input and performs the following steps:
        * Converts the cluster boundary points to a numpy array.
        * Calculates the centroid of the cluster using np.mean.
        * Calculates the covariance matrix of the cluster points.
        * *Performs eigenvalue decomposition on the covariance matrix using np.linalg.eig.
        * Sorts the eigenvalues and eigenvectors in descending order.
        * Retrieves the first eigenvector (major axis) corresponding to the largest eigenvalue.
        * Calculates the orientation angle (in radians) from the major axis using np.arctan2.
        * Converts the orientation angle from radians to degrees.
    The function returns the orientation angle in degrees.
    '''
    # Convert cluster boundary points to numpy array
    points = np.array(list(cluster))

    # Calculate the centroid of the cluster
    centroid = np.mean(points, axis=0)

    # Calculate the covariance matrix of the cluster points
    centered_points = points - centroid
    covariance_matrix = np.dot(centered_points.T, centered_points) / len(points)

    # Perform eigenvalue decomposition on the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    sort_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_indices]
    eigenvectors = eigenvectors[:, sort_indices]

    # Get the first eigenvector (major axis)
    major_axis = eigenvectors[:, 0]

    # Calculate the orientation angle from the major axis
    orientation_angle = np.arctan2(major_axis[1], major_axis[0])
    # Convert angle from radians to degrees
    orientation_angle_deg = np.degrees(orientation_angle)

    return orientation_angle_deg

def calculate_morphological_orientation_clusters(clusters):
    # Calculate morphological orientation for each cluster
    orientations = []
    for cluster in clusters:
        orientation = calculate_morphological_orientation(cluster)
        orientations.append(orientation)
    return orientations


def identify_kde_modes(data):
    # Calculate KDE
    kde = gaussian_kde(data)

    # Evaluate KDE on a set of x-values
    x_vals = np.linspace(np.min(data), np.max(data), 1000)
    y_vals = kde.evaluate(x_vals)

    # Find local maxima (peaks) of the KDE curve
    maxima_indices = argrelextrema(y_vals, np.greater)[0]
    modes = x_vals[maxima_indices]

    return modes

def plot_grains(grains):
    # Get the unique numbers in the array
    unique_numbers = np.unique(grains)

    # Generate random colors for each unique number
    num_colors = len(unique_numbers)
    colors = np.random.rand(num_colors, 3)

    # Create a colormap with the random colors
    cmap = plt.cm.colors.ListedColormap(colors)

    # Plot the array using imshow
    plt.imshow(grains, cmap=cmap)

    # Add colorbar to show the color-to-number mapping
    cbar = plt.colorbar(ticks=unique_numbers)
    cbar.set_label('Number')

    # Show the plot
    plt.show()


if __name__ == "__main__":
    x_size = 100
    y_size = 100
    q = 3
    nsteps = 100000
    temperature = -500
    #grains, grain_boundaries = potts_model1(x_size, y_size, q, nsteps)
    #grains, grain_boundaries = potts_model2(x_size, y_size, q, nsteps, temperature)
    grains, grain_boundaries = potts_model3(x_size, y_size, q, nsteps, temperature)
    print(grains)
    print(grain_boundaries)


    clusters = find_clusters1(grains)

    # Find cluster boundaries
    boundaries = find_cluster_boundaries(clusters)

    # Find neighboring clusters
    neighboring_clusters = find_neighboring_clusters(clusters, boundaries)


    curvatures =  calculate_curvatures_boundaries(boundaries)
    # Calculate cluster properties
    cluster_areas, perimeters, aspect_ratios = calculate_cluster_properties(clusters)

    # Identify modes of the KDE
    kde_modes = identify_kde_modes(cluster_areas)

    # Calculate kernel density estimation (KDE)
    kde = gaussian_kde(cluster_areas)
    x_vals = np.linspace(np.min(cluster_areas), np.max(cluster_areas), 100)
    y_vals = kde.evaluate(x_vals)

    # Plot KDE curve with modes
    #plt.plot(x_vals, y_vals)
    #plt.scatter(kde_modes, kde.evaluate(kde_modes), c='red', label='Modes')
    #plt.xlabel('Cluster Area')
    #plt.ylabel('Density')
    #plt.title('Kernel Density Estimation (KDE) of Cluster Areas with Modes')
    #plt.legend()

    plot_grains(grains)
