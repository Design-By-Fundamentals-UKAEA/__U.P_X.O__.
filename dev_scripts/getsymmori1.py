import numpy as np

def normalize_angle_360(angle):
  """Normalizes an angle to the range [0, 360)"""
  angle = angle % 360
  return angle if angle >= 0 else angle + 360

def normalize_angle_180(angle):
  """Normalizes an angle to the range [0, 180)"""
  angle = angle % 360
  if angle > 180:
    return angle - 180
  else:
    return angle

def FCC_symmetries(phi1, Phi, phi2):
  """Calculates symmetric equivalent Bunge Euler angles with the range 0-360, 0-180, 0-360"""
  phi1 = normalize_angle_360(phi1)
  Phi = normalize_angle_180(Phi)
  phi2 = normalize_angle_360(phi2)

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

      # Apply normalization to Phi (the second angle)
      new_symmetries = [(s1, normalize_angle_180(s2), s3) for s1, s2, s3 in new_symmetries]

      symmetries.extend(new_symmetries)

  return symmetries



phi1, Phi, phi2 = 45, 35, 0
EA1, EA2, EA3 = [], [], []
sym_equivalents = FCC_symmetries(phi1, Phi, phi2)
for eq in sym_equivalents:
    EA1.append(abs(eq[0]))
    EA2.append(abs(eq[1]))
    EA3.append(abs(eq[2]))

np.unique(np.vstack([EA1, EA2, EA3]).T, axis=0)
