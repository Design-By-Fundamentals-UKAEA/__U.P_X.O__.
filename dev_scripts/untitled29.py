import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries

# Given state matrix S
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

# Label grains
labeled_grains = label(S, connectivity=1)

# Find boundaries using skimage
boundaries = find_boundaries(labeled_grains, mode='outer')

# Kernel to identify neighbors
kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

# Convolve labeled_grains to identify junction points
convolved = convolve(labeled_grains > 0, kernel, mode='constant', cval=0)
junctions = convolved > 3  # Junctions where 3 or more grains meet

# Plotting
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# Original grain structure
ax[0].imshow(S, cmap='nipy_spectral')
ax[0].set_title('Original Grain Structure')
ax[0].axis('off')

# Grain boundaries
ax[1].imshow(S, cmap='nipy_spectral')
ax[1].imshow(boundaries, cmap='gray', alpha=0.5)  # Overlay boundaries
ax[1].set_title('Grain Boundaries')
ax[1].axis('off')

# Junction points
ax[2].imshow(S, cmap='nipy_spectral')
ax[2].imshow(junctions, cmap='hot', alpha=0.5)  # Overlay junctions
ax[2].set_title('Junction Points')
ax[2].axis('off')

plt.tight_layout()
plt.show()
