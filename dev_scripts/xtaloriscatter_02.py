from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import numpy as np

# Lattice parameter for Copper (Cu)
a = 3.615  # Angstroms, approximate value

lattice = Lattice.cubic(a)
structure = Structure(lattice, ["Cu", "Cu", "Cu", "Cu"],
                      [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])


analyzer = SpacegroupAnalyzer(structure)
symmetric_ops = analyzer.get_symmetry_operations()


# --------------------------------
# Placeholder function to convert Euler angles to a rotation matrix
import numpy as np

def normalize_angle(angle):
  """Normalizes an angle to the range [0, 360)"""
  angle = angle % 360
  return angle if angle >= 0 else angle + 360

def euler_to_matrix(phi1, Phi, phi2):
    phi1, Phi, phi2 = np.deg2rad([phi1, Phi, phi2])
    Rz_phi1 = np.array([[np.cos(phi1), -np.sin(phi1), 0],
                        [np.sin(phi1), np.cos(phi1), 0],
                        [0, 0, 1]])
    Rx_Phi = np.array([[1, 0, 0],
                       [0, np.cos(Phi), -np.sin(Phi)],
                       [0, np.sin(Phi), np.cos(Phi)]])
    Rz_phi2 = np.array([[np.cos(phi2), -np.sin(phi2), 0],
                        [np.sin(phi2), np.cos(phi2), 0],
                        [0, 0, 1]])
    R = Rz_phi1 @ Rx_Phi @ Rz_phi2
    return R

def FCC_symmetries(phi1, Phi, phi2):
    """Calculates all symmetric equivalent Bunge Euler angles for FCC"""
    phi1, Phi, phi2 = normalize_angle(phi1), normalize_angle(Phi), normalize_angle(phi2)


    symmetries = [
        (phi1, Phi, phi2),
        (180 - phi1, Phi, 180 - phi2),
        (phi1, 180 - Phi, phi2),
        (180 - phi1, 180 - Phi, 180 - phi2),
        (90 - phi2, Phi, phi1),
        (90 + phi2, Phi, 360 - phi1),
        (270 + phi2, Phi, phi1),
        (270 - phi2, Phi, 360 - phi1),
    ]

    # Add additional symmetries based on the cube's 90-degree rotational symmetries
    for i in range(3):  # Repeat for 90, 180, and 270 degree rotations about Z
        new_symmetries = []
        for s in symmetries:
            s1, s2, s3 = s
            new_symmetries.append((s3, s1, s2))
        symmetries.extend(new_symmetries)

    return symmetries

phi1, Phi, phi2 = 0, 0, 0
sym_equivalents = FCC_symmetries(phi1, Phi, phi2)
for eq in sym_equivalents:
    print(f"phi1: {eq[0]:.2f}, Phi: {eq[1]:.2f}, phi2: {eq[2]:.2f}")

phi1, Phi, phi2 = 0, 0, 0
EA1, EA2, EA3 = [], [], []
sym_equivalents = FCC_symmetries(phi1, Phi, phi2)
for eq in sym_equivalents:
    EA1.append(eq[0])
    EA2.append(eq[1])
    EA3.append(eq[2])

np.unique(np.vstack([EA1, EA2, EA3]).T, axis=0)



import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(EA1, EA2, EA3, s=20, c='blue', marker='o')  # Plot as points
ax.set_xlim(0, 90)
ax.set_ylim(0, 90)
ax.set_zlim(0, 90)


# Example usage
phi1, Phi, phi2 = 90, 45, 30  # Euler angles in degrees
rotation_matrix = euler_to_matrix(phi1, Phi, phi2)
print("Rotation Matrix:\n", rotation_matrix)

phi1, Phi, phi2 = matrix_to_euler(rotation_matrix)
print(f"Recovered Euler angles: phi1: {phi1:.2f}, Phi: {Phi:.2f}, phi2: {phi2:.2f}")

rotation_matrix = euler_to_matrix(phi1, Phi, phi2)
print("Rotation Matrix:\n", rotation_matrix)




# Example usage
phi1, Phi, phi2 = 90, 45, 30  # Euler angles in degrees
rotation_matrix = euler_to_matrix(phi1, Phi, phi2)
print(rotation_matrix)

phi1, Phi, phi2 = matrix_to_euler(rotation_matrix)
print(f"phi1: {phi1:.2f}, Phi: {Phi:.2f}, phi2: {phi2:.2f}")



# Given Bunge's Euler angles for a brass texture component
euler_angles = (35, 45, 0)  # Example values in degrees

# Convert to rotation matrix
rotation_matrix = euler_to_matrix(euler_angles)

# Apply each symmetry operation and convert back to Euler angles
equivalent_euler_angles = []
for op in symmetric_ops:
    transformed_matrix = op.rotation_matrix @ rotation_matrix  # Matrix multiplication
    equivalent_euler_angles.append(matrix_to_euler(transformed_matrix))

# Now `equivalent_euler_angles` would contain all symmetrically equivalent Euler angles
