from shapely.geometry import Polygon
import numpy as np

def create_ellipse(center, width, height, num_points=100):
    """
    Create an ellipse as a Shapely Polygon.

    Parameters:
    - center: A tuple (x, y) representing the center of the ellipse.
    - width: The total width of the ellipse (major axis length).
    - height: The total height of the ellipse (minor axis length).
    - num_points: The number of points used to approximate the ellipse.

    Returns:
    - A Shapely Polygon object representing the ellipse.
    """
    angle = np.linspace(0, 2*np.pi, num_points)
    ellipse_x = center[0] + (width / 2) * np.cos(angle)
    ellipse_y = center[1] + (height / 2) * np.sin(angle)

    return Polygon(np.c_[ellipse_x, ellipse_y])


import random


def modify_ellipses(ellipses):
    """
    Randomly choose one ellipse as primary, make the other secondary,
    retain the primary completely, and remove the intersected portion from the secondary.
    """
    # Randomly assign primary and secondary
    primary_idx = random.randint(0, len(ellipses) - 1)
    secondary_idx = 1 - primary_idx  # Works because there are only two ellipses

    primary = ellipses[primary_idx]
    secondary = ellipses[secondary_idx]

    # Calculate the intersection
    intersection = primary.intersection(secondary)

    # Remove the intersected portion from the secondary
    modified_secondary = secondary.difference(intersection)

    # Return the new list [primary, modified_secondary]
    return [primary, modified_secondary] if primary_idx == 0 else [modified_secondary, primary]


ellipses = [create_ellipse((0, 0), 5, 5),
            create_ellipse((3, 0), 5, 5)]

# To display the ellipse, you can use matplotlib
import matplotlib.pyplot as plt

# Extract coordinates
coords = [np.array(ellipse.exterior.coords) for ellipse in ellipses]

# Plotting
plt.figure()
for coord in coords:
    plt.plot(coord[:,0], coord[:,1])  # Plot each ellipse
    plt.fill(coord[:,0], coord[:,1], alpha=0.5)  # Fill the ellipse with some transparency

plt.axis('equal')
plt.grid(True)
plt.show()



# Modify ellipses according to the requirements
modified_ellipses = modify_ellipses(ellipses)
modified_ellipses[1]



# Perform the union of the modified ellipses
union_ellipse = modified_ellipses[0].union(modified_ellipses[1])
