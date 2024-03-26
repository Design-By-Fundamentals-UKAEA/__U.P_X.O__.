import numpy as np

def normalize_angle(angle):
  """Normalizes an angle to the range [0, 360)"""
  angle = angle % 360
  return angle if angle >= 0 else angle + 360

def reduce_angle(angle):
   """Reduces an angle to the range [0, 90] using symmetries"""
   angle = normalize_angle(angle)
   if angle > 270:
     return 360 - angle
   elif angle > 180:
     return 180 - angle
   elif angle > 90:
     return 180 - angle
   else:
     return angle

def FCC_symmetries(phi1, Phi, phi2):
  """Calculates symmetric equivalent Bunge Euler angles (all in the range of 0 to 90 degrees) for FCC"""
  phi1, Phi, phi2 = normalize_angle(phi1), normalize_angle(Phi), normalize_angle(phi2)

  symmetries = [
      (reduce_angle(phi1), reduce_angle(Phi), reduce_angle(phi2)),
      (reduce_angle(180 - phi1), reduce_angle(Phi), reduce_angle(180 - phi2)),
      (reduce_angle(phi1), reduce_angle(180 - Phi), reduce_angle(phi2)),
      (reduce_angle(180 - phi1), reduce_angle(180 - Phi), reduce_angle(180 - phi2)),
      (reduce_angle(90 - phi2), reduce_angle(Phi), reduce_angle(phi1)),
      (reduce_angle(90 + phi2), reduce_angle(Phi), reduce_angle(360 - phi1)),
      (reduce_angle(270 + phi2), reduce_angle(Phi), reduce_angle(phi1)),
      (reduce_angle(270 - phi2), reduce_angle(Phi), reduce_angle(360 - phi1)),
  ]

  # Add additional symmetries based on the cube's 90-degree rotational symmetries
  for i in range(3):  # Repeat for 90, 180, and 270 degree rotations about Z
      new_symmetries = []
      for s in symmetries:
          s1, s2, s3 = s
          new_symmetries.append((reduce_angle(s3), reduce_angle(s1), reduce_angle(s2)))
      symmetries.extend(new_symmetries)

  return symmetries

# Example usage
phi1, Phi, phi2 = 60, 35, 45
sym_equivalents = FCC_symmetries(phi1, Phi, phi2)

for eq in sym_equivalents:
    print(f"phi1: {eq[0]:.2f}, Phi: {eq[1]:.2f}, phi2: {eq[2]:.2f}")




phi1, Phi, phi2 = 45, 35, 0
EA1, EA2, EA3 = [], [], []
sym_equivalents = FCC_symmetries(phi1, Phi, phi2)
for eq in sym_equivalents:
    EA1.append(abs(eq[0]))
    EA2.append(abs(eq[1]))
    EA3.append(abs(eq[2]))

EA = np.unique(np.vstack([EA1, EA2, EA3]).T, axis=0)
