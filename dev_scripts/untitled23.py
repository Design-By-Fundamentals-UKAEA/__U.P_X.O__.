import numpy as np
import matplotlib.pyplot as plt

# Define the starting point A and ending point B of the line
A = np.array([-3, 5])  # Example: (0, 0)
B = np.array([10, 0])  # Example: (10, 0)

# Function to plot points
def plot_points(points):
    plt.plot(*zip(*np.vstack([points])), marker='o', linestyle='-')
    plt.xlim(min(points[:,0]) - 1, max(points[:,0]) + 1)
    plt.ylim(min(points[:,1]) - 1, max(points[:,1]) + 1)
    plt.grid(True)
    plt.show()


def constant_spacing(A, B, n):
    return np.linspace(A, B, n+2)

# Usage
n = 20  # Number of points to introduce
points = constant_spacing(A, B, n)
plot_points(points)


def quadratic_spacing(A, B, n, start='A'):
    t = np.linspace(0, 1, n + 2)**2  # Quadratic spacing
    if start == 'B':
        t = 1 - t  # Reverse if starting from B
    return A + (B - A) * t[:, None]

# Usage
points = quadratic_spacing(A, B, n)
plot_points(points)


def linear_spacing(A, B, n, m=0.5, start='A'):
    t = np.linspace(0, 1, n + 2)  # Linear spacing
    if start == 'B':
        t = 1 - t  # Reverse if starting from B
    return A + (B - A) * t[:, None]

# Usage
points = linear_spacing(A, B, n)
plot_points(points)


def bi_quadratic_spacing(A, B, n):
    mid = (A + B) / 2
    first_half = quadratic_spacing(A, mid, n // 2, start='A')
    second_half = quadratic_spacing(mid, B, n // 2, start='B')
    return np.vstack([first_half, second_half[1:]])

# Usage
points = bi_quadratic_spacing(A, B, n)
plot_points(points)


def random_spacing(A, B, max_points, min_dist):
    points = [A]
    while len(points) < max_points + 1:
        rand_point = A + (B - A) * np.random.rand()
        if all(np.linalg.norm(rand_point - p) >= min_dist for p in points):
            points.append(rand_point)
        if np.linalg.norm(B - points[-1]) < min_dist:
            break
    points.append(B)
    return np.array(points)

# Usage
max_points = 10
min_dist = 2
points = random_spacing(A, B, max_points, min_dist)
plot_points(points)



import numpy as np

def find_dividing_point(A, B, ratio, ratio_start):
    if ratio_start == 'A':
        C = (A[0] + ratio * (B[0] - A[0]), A[1] + ratio * (B[1] - A[1]))
    elif ratio_start == 'B':
        C = (B[0] + ratio * (A[0] - B[0]), B[1] + ratio * (A[1] - B[1]))
    return C

def quadratic_spacing(A, B, n_points, from_start=True):
    # Calculate the total distance between A and B
    total_distance = np.sqrt((B[0] - A[0])**2 + (B[1] - A[1])**2)

    # Generate n_points quadratic distances from 0 to 1
    t = np.linspace(0, 1, n_points)**2 if from_start else (1 - np.linspace(0, 1, n_points)**2)

    # Calculate the quadratic distances along the line
    distances = t * total_distance

    # Calculate the direction vector from A to B
    direction = ((B[0] - A[0]) / total_distance, (B[1] - A[1]) / total_distance)

    # Generate the points
    points = [(A[0] + distance * direction[0], A[1] + distance * direction[1]) for distance in distances]

    return points


# Example usage
A = np.array([-3, 5])  # Example: (0, 0)
B = np.array([10, -3])  # Example: (10, 0)

ratio = 0.5  # Ratio for dividing point
ratio_start = 'A'  # Starting from A
n_points = 20  # Number of points for quadratic spacing

C = find_dividing_point(A, B, ratio, ratio_start)

# Apply quadratic spacing to AC and CB
points_AC = quadratic_spacing(A, C, n_points//2, from_start=True)
points_CB = quadratic_spacing(C, B, n_points//2, from_start=False)

# Combine points from both segments, ensuring C is not duplicated
points = points_AC + points_CB[1:]
# Now, `points` contains the quadratically spaced points along AC and CB

# Convert the list of points to a numpy array
points_array = np.array(points)

points_array
plot_points(points)





def perturb_points(points, A, B, max_perturbation):
    # Calculate the direction vector from A to B
    direction_vector = (B[0] - A[0], B[1] - A[1])

    # Calculate the normal vector (rotate direction vector by 90 degrees)
    normal_vector = (-direction_vector[1], direction_vector[0])

    # Normalize the normal vector
    normal_length = np.sqrt(normal_vector[0]**2 + normal_vector[1]**2)
    normal_unit_vector = (normal_vector[0] / normal_length, normal_vector[1] / normal_length)

    # Perturb each point
    perturbed_points = []
    for point in points:
        # Generate a random perturbation length within the allowed maximum
        perturbation_length = np.random.uniform(-max_perturbation, max_perturbation)

        # Apply the perturbation along the normal direction
        perturbed_point = (
            point[0] + perturbation_length * normal_unit_vector[0],
            point[1] + perturbation_length * normal_unit_vector[1]
        )
        perturbed_points.append(perturbed_point)

    return perturbed_points

# Original line segment and points
A = (-3, 5)
B = (10, -3)

A = (3, 0)
B = (10, 0)

A = (0, -1)
B = (0, 10)

points = quadratic_spacing(A, B, 20, from_start=True)

# Determine maximum perturbation length
line_length = np.sqrt((B[0] - A[0])**2 + (B[1] - A[1])**2)
max_perturbation = min(1, line_length / 10)

# Perturb points
perturbed_points = perturb_points(points, A, B, max_perturbation)

# Plotting the original line, points, and perturbed points
plt.figure(figsize=(10, 6))
plt.plot([A[0], B[0]], [A[1], B[1]], 'k--', label='Line AB')
for point, perturbed_point in zip(points, perturbed_points):
    plt.plot(point[0], point[1], 'ro')  # Original point
    plt.plot(perturbed_point[0], perturbed_point[1], 'bo')  # Perturbed point
    plt.plot([point[0], perturbed_point[0]], [point[1], perturbed_point[1]], 'b--')  # Line showing perturbation

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Quadratic Spacing with Random Perturbations Normal to AB')
plt.legend()
plt.grid(True)
plt.show()


import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, CubicSpline, UnivariateSpline


# Sort the perturbed points by their x values
perturbed_points_sorted = sorted(perturbed_points, key=lambda p: p[0])

# Separate the sorted x and y coordinates
x = [p[0] for p in perturbed_points_sorted]
y = [p[1] for p in perturbed_points_sorted]


# Fine grid for spline evaluation
x_fine = np.linspace(min(x), max(x), 300)

# Linear spline using interp1d
linear_interp = interp1d(x, y, kind='linear')
y_linear = linear_interp(x_fine)

# Cubic spline using interp1d
cubic_interp = interp1d(x, y, kind='cubic')
y_cubic_interp = cubic_interp(x_fine)

# CubicSpline
# Specify the slopes at the endpoints
slope_at_A = 0  # Tangential to the x-axis at the start
slope_at_B = 0  # Tangential to the x-axis at the end

cubic_spline = CubicSpline(x, y, bc_type=((1, slope_at_A), (1, slope_at_B)))
y_cubic_spline = cubic_spline(x_fine)

# UnivariateSpline with a smoothing factor (s)
smoothing_factor = 20
degree = 2
univariate_spline = UnivariateSpline(x, y, k = degree, s=smoothing_factor)
y_univariate = univariate_spline(x_fine)

# Plotting
plt.figure(figsize=(14, 10))

# Original perturbed points
plt.plot(x, y, 'ro', label='Perturbed Points')

# Linear Spline
plt.plot(x_fine, y_linear, label='Linear Spline (interp1d)', linestyle='--')

# Cubic Spline (interp1d)
plt.plot(x_fine, y_cubic_interp, label='Cubic Spline (interp1d)', linestyle=':')

# CubicSpline
plt.plot(x_fine, y_cubic_spline, label='CubicSpline', linestyle='-.')

# UnivariateSpline
plt.plot(x_fine, y_univariate, label='UnivariateSpline', linestyle='-')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Splines Through Perturbed Points')
plt.legend()
plt.grid(True)
plt.show()



# Create the CubicSpline with specified endpoint slopes
cubic_spline = CubicSpline(x, y, bc_type=((1, slope_at_A), (1, slope_at_B)))

# Evaluate the spline
y_cubic_spline = cubic_spline(x_fine)

# Plotting
plt.plot(x_fine, y_cubic_spline, label='CubicSpline with Tangential Endpoints')
# Add other plot details as necessary
