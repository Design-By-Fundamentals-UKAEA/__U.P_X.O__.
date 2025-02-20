import numpy as np
from upxo.geoEntities.plane import Plane
from upxo.geoEntities.mulpoint3d import MPoint3d as mp3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

xspec, yspec, zspec = [0, 1, 0.1], [0, 1, 0.1], [0, 1, 0.1]
mpnt3d = mp3d.from_xyz_grid(xspec=xspec, yspec=yspec, zspec=zspec,
                            dxyz=[0.0, 0.0, 0.0],
                            translate_ref=[0.5, 0.5, 0.5],
                            rot=[0.0, 0.0, 0.0],
                            rot_ref=[0.5, 0.5, 0.5],
                            degree=True)
plane = Plane(point=(0.0, 0.0, 0.0), normal=(1, 1, 1))
num_planes, translation_vector = 3, np.array([0.25, 0.25, 0.25])

planes = plane.create_translated_planes(translation_vector, num_planes)
D = [plane.calc_perp_distances(mpnt3d.coords, signed=False)
     for plane in planes]

# Mean and standard deviation for the normal distribution
cod_mean = 0.2
cod_std = (0.1 - 0.0) / 6  # Standard deviation is one-sixth of the range
cod = np.random.normal(cod_mean, cod_std, num_planes)

coords = [mpnt3d.coords[np.argwhere(d <= cod_)].squeeze() for d, cod_ in zip(D, cod)]
# --------------------------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(mpnt3d.coords[:, 0], mpnt3d.coords[:, 1], mpnt3d.coords[:, 2],
           c='b', marker='o', alpha=0.00, s=100,
           edgecolors='black')

for coord in coords:
    if coord is not None:
        ax.scatter(coord[:, 0], coord[:, 1], coord[:, 2],
                   c=np.random.random(3), marker='o', alpha=0.8, s=50,
                   edgecolors='black')

xbound, ybound, zbound = xspec[:2], yspec[:2], zspec[:2]
vertices = np.array([[xbound[0], ybound[0], zbound[0]],  # 0
                     [xbound[1], ybound[0], zbound[0]],  # 1
                     [xbound[1], ybound[1], zbound[0]],  # 2
                     [xbound[0], ybound[1], zbound[0]],  # 3
                     [xbound[0], ybound[0], zbound[1]],  # 4
                     [xbound[1], ybound[0], zbound[1]],  # 5
                     [xbound[1], ybound[1], zbound[1]],  # 6
                     [xbound[0], ybound[1], zbound[1]]])  # 7
# Define the edges of the cuboid
edges = [[0, 1], [1, 2], [2, 3], [3, 0],
         [4, 5], [5, 6], [6, 7], [7, 4],
         [0, 4], [1, 5], [2, 6], [3, 7]]
for edge in edges:
    ax.plot(*zip(*vertices[edge]), color='k', linewidth=2.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()



COORDS = {i: coord for i, coord in enumerate(coords) if coord.size > 0}



plane_number = 2

coords = COORDS[plane_number]

plane = Plane(point=np.mean(coords, axis=0)-0.05, normal=(1, 1, 1))
translation_vector = (0.05, 0.05, 0.05)
planes = plane.create_translated_planes(translation_vector, 5)

D = [plane.calc_perp_distances(coords, signed=False)
     for plane in planes]

cod = 0.05
coords_subset = [coords[np.argwhere(d <= cod)].squeeze() for d in D]

COORDS[plane_number] = {i: coord for i, coord in enumerate(coords_subset) if coord.size > 0}

for COORD in COORDS[plane_number].values():
    for row in COORD:
        index_to_remove = np.where(( coords == row ).all(axis=1))
        coords = np.delete(coords, index_to_remove, axis=0)

COORDS[plane_number][max(COORDS[plane_number].keys())+1] = coords


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for coords in COORDS.values():
    if type(coords) == np.ndarray:
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                   c=np.random.random(3), marker='o', alpha=0.8, s=50,
                   edgecolors='black')
    if type(coords) == dict:
        for coords_ in coords.values():
            ax.scatter(coords_[:, 0], coords_[:, 1], coords_[:, 2],
                       c=np.random.random(3), marker='o', alpha=1.0, s=50,
                       edgecolors='black')


xbound, ybound, zbound = xspec[:2], yspec[:2], zspec[:2]
vertices = np.array([[xbound[0], ybound[0], zbound[0]],  # 0
                     [xbound[1], ybound[0], zbound[0]],  # 1
                     [xbound[1], ybound[1], zbound[0]],  # 2
                     [xbound[0], ybound[1], zbound[0]],  # 3
                     [xbound[0], ybound[0], zbound[1]],  # 4
                     [xbound[1], ybound[0], zbound[1]],  # 5
                     [xbound[1], ybound[1], zbound[1]],  # 6
                     [xbound[0], ybound[1], zbound[1]]])  # 7
# Define the edges of the cuboid
edges = [[0, 1], [1, 2], [2, 3], [3, 0],
         [4, 5], [5, 6], [6, 7], [7, 4],
         [0, 4], [1, 5], [2, 6], [3, 7]]
for edge in edges:
    ax.plot(*zip(*vertices[edge]), color='k', linewidth=2.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
