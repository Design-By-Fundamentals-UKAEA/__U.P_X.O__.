import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from matplotlib.patches import Rectangle

def identify_clusters_boundaries(array):
    # Label connected components in the array
    labeled_array = label(array)

    # Find unique labels
    unique_labels = np.unique(labeled_array)

    # Create a list to store cluster boundaries
    cluster_boundaries = []

    # Iterate over unique labels to identify clusters and boundaries
    for label_value in unique_labels:
        if label_value == 0:
            continue  # Skip background label

        # Create a mask for the current cluster
        cluster_mask = labeled_array == label_value

        # Find the boundaries of the cluster
        boundaries = find_boundaries(cluster_mask)

        # Append the boundaries to the cluster_boundaries list
        cluster_boundaries.append(boundaries)

    return labeled_array, cluster_boundaries

def find_boundaries(mask):
    # Pad the mask to include the borders
    padded_mask = np.pad(mask, pad_width=1, mode='constant')

    # Find the boundaries
    boundaries = padded_mask[1:, :-1] != padded_mask[:-1, :-1]
    boundaries |= padded_mask[:-1, 1:] != padded_mask[:-1, :-1]

    return boundaries

def plot_clusters(array):
    plt.imshow(array, cmap='viridis')
    plt.colorbar()
    plt.title('Clusters')
    plt.show()

def plot_boundaries(array, boundaries):
    plt.imshow(array, cmap='viridis')
    plt.colorbar()

    # Plot boundaries as rectangles
    for boundary in boundaries:
        min_row, min_col = np.min(np.where(boundary), axis=1)
        max_row, max_col = np.max(np.where(boundary), axis=1)

        rect = Rectangle((min_col - 0.5, min_row - 0.5), max_col - min_col + 1, max_row - min_row + 1,
                         edgecolor='red', facecolor='none')
        plt.gca().add_patch(rect)

    plt.title('Cluster Boundaries')
    plt.show()

# Example usage
S = 1  # Maximum value in the array
array = np.random.randint(0, S + 1, size=(20, 20))  # Generate a random 10x10 array

labeled_array, cluster_boundaries = identify_clusters_boundaries(array)
plot_clusters(labeled_array)
plot_boundaries(array, cluster_boundaries)
