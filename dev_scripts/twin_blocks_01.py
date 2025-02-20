import numpy as np
from upxo.geoEntities.plane import Plane
from upxo.geoEntities.mulpoint3d import MPoint3d as mp3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

xspec, yspec, zspec = [0, 1, 0.05], [0, 1, 0.05], [0, 1, 0.05]
mpnt3d = mp3d.from_xyz_grid(xspec=xspec, yspec=yspec, zspec=zspec,
                            dxyz=[0.0, 0.0, 0.0],
                            translate_ref=[0.5, 0.5, 0.5],
                            rot=[0.0, 0.0, 0.0],
                            rot_ref=[0.5, 0.5, 0.5],
                            degree=True)
plane = Plane(point=(0.0, 0.0, 0.0), normal=(1, 1, 0))
num_planes, translation_vector = 8, np.array([0.2, 0.2, 0.2])

planes = plane.create_translated_planes(translation_vector, num_planes)
D = [plane.calc_perp_distances(mpnt3d.coords, signed=False)
     for plane in planes]

# Mean and standard deviation for the normal distribution
cod_mean = 0.125
cod_std = (0.15 - 0.0) / 6  # Standard deviation is one-sixth of the range
cod = np.random.normal(cod_mean, cod_std, num_planes)

coords = [mpnt3d.coords[np.argwhere(d <= cod_)].squeeze() for d, cod_ in zip(D, cod)]
# --------------------------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(mpnt3d.coords[:, 0], mpnt3d.coords[:, 1], mpnt3d.coords[:, 2],
           c='b', marker='o', alpha=0.01, s=100,
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
