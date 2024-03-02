import numpy as np

# Example list of coordinates
coords = np.array([[2, 2], [1, 1], [3, 3], [2, 3], [1, 2]])
coords = jp_nparrays[343].T
# Step 1: Find the leftmost (and then topmost) point
leftmost_point = coords[np.lexsort((coords[:, 1], coords[:, 0]))][0]

# Step 2: Center the points around the leftmost point
centered_coords = coords - leftmost_point

# Step 3: Calculate angles
angles = np.arctan2(centered_coords[:, 1], centered_coords[:, 0])

# Step 4: Sort by angles
# Note: np.arctan2 returns angles in [-pi, pi], negative angles are in the 3rd and 4th quadrants
# Sorting in ascending order effectively gives us a counterclockwise direction,
# so we might need to adjust based on your clockwise requirement.
sorted_indices = np.argsort(angles)
sorted_coords = coords[sorted_indices]

# Adjusting for clockwise order (if necessary)
# This example assumes the sorting gives counterclockwise; if you need to adjust to clockwise,
# you might invert the order or adjust the sorting criteria based on your specific needs.

# Result
sorted_coords
