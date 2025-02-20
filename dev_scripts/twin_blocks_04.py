import numpy as np
import matplotlib.pyplot as plt
from upxo.geoEntities.sline2d import Sline2d as sl2d

x, y = np.meshgrid(np.arange(0, 1, 0.025), np.arange(0, 1, 0.025))
GRAIN = (x.ravel(), y.ravel())
# GRAIN = np.random.random((2, 100))
remaining_indices = list(range(GRAIN[0].size))

lines = sl2d.by_LFAL(location=[-0.1, 0.5], factor=0.5, angle=75,
                     length=5).array_translation(ncopies=10, vector=[0.2, 0.0],
                                                 spacing='constant')
twin_indices = []
for line in lines:
    _twin_indices_ = line.find_neigh_point_by_perp_distance(GRAIN, 0.4, use_bounding_rec=True)
    if _twin_indices_:
        twin_indices.append(_twin_indices_)
        remaining_indices = list(set(remaining_indices) - set(_twin_indices_))

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(GRAIN[0][remaining_indices], GRAIN[1][remaining_indices], 'bo', markersize=8, alpha=0.1)

for _twin_indices_ in twin_indices:
    plt.plot(GRAIN[0][_twin_indices_], GRAIN[1][_twin_indices_], 'ko', markersize=6, alpha=0.8, mfc=np.random.random(3))

ax.set_aspect('equal')
