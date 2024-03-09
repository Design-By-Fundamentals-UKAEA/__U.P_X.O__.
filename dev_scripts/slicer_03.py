import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import MultiPoint
from shapely.ops import voronoi_diagram
from shapely.geometry import MultiPolygon

#vertices = [[0.0, 0.0], [0.8, 0.0], [0.5, 1.0]]
vertices = [[0.0, 0.0], [0.5, -0.5], [1.5, -0.5], [2.0, 0.0], [1.5, 0.5], [0.5, 0.5]]
envelope = Polygon(vertices)

x_bounds = envelope.boundary.xy[0]
y_bounds = envelope.boundary.xy[1]

xmin = min(x_bounds)
xmax = max(x_bounds)
ymin = min(y_bounds)
ymax = max(y_bounds)
base_rect = Polygon([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])

n_points = 10
x = 1.0*xmin + 1.0*(xmax-xmin)*np.random.random(n_points)
y = 1.0*ymin + 1.0*(ymax-ymin)*np.random.random(n_points)
points = MultiPoint([[x[i], y[i]] for i in range(n_points)])
pxtal = voronoi_diagram(points, tolerance = 0.0, edges=False)

plt.fill(base_rect.boundary.xy[0], base_rect.boundary.xy[1],
         facecolor = 'cyan', edgecolor = 'black', alpha = 1.0, linewidth = 3)
plt.fill(envelope.boundary.xy[0], envelope.boundary.xy[1],
         facecolor = 'blue', edgecolor = 'black', alpha = 1.0, linewidth = 3)
plt.plot(x, y, 'ko')
# =============================================================================
#     plt.fill(grain.boundary.xy[0], grain.boundary.xy[1],
#              facecolor = np.random.random(3), edgecolor = 'black', alpha = 1.0, linewidth = 1)
# for grain_count in range(len(pxtal.geoms)):
#     grain = pxtal.geoms[grain_count]
#     plt.fill(grain.boundary.xy[0], grain.boundary.xy[1],
#              facecolor = np.random.random(3), edgecolor = 'black', alpha = 1.0, linewidth = 1)
# plt.axis('equal')
# plt.show()
# =============================================================================

dir(pxtal)

contained = []
contained_cropped = []
for count in range(len(pxtal.geoms)):
    if pxtal.geoms[count].intersects(envelope):
        contained.append(pxtal.geoms[count])
        contained_cropped.append(pxtal.geoms[count].intersection(envelope))

mpc = MultiPolygon(contained)
mpc_cropped = MultiPolygon(contained_cropped)
mpc_cropped

for grain_count in range(len(pxtal.geoms)):
    grain = pxtal.geoms[grain_count]
    plt.fill(grain.boundary.xy[0], grain.boundary.xy[1],
             facecolor = np.random.random(3), edgecolor = 'black', alpha = 1.0, linewidth = 1)
plt.fill(envelope.boundary.xy[0], envelope.boundary.xy[1],
         facecolor = 'blue', edgecolor = 'black', alpha = 0.5, linewidth = 3)
plt.axis('equal')
plt.show()

for grain_count in range(len(mpc.geoms)):
    grain = mpc.geoms[grain_count]
    plt.fill(grain.boundary.xy[0], grain.boundary.xy[1],
             facecolor = np.random.random(3), edgecolor = 'black', alpha = 1.0, linewidth = 1)
plt.axis('equal')
plt.show()

sub_grains = []
for sub_grain_count in range(len(mpc_cropped.geoms)):
    grain = mpc_cropped.geoms[sub_grain_count]
    plt.fill(grain.boundary.xy[0], grain.boundary.xy[1],
             facecolor = np.random.random(3), edgecolor = 'black', alpha = 1.0, linewidth = 1)
    sub_grains.append(grain)
plt.axis('equal')
plt.show()

from shapely.strtree import STRtree
sub_grain_tree = STRtree(sub_grains)

querry_grain_id = -1
query_geom = sub_grains[querry_grain_id].buffer(-0.02, cap_style = 2)
sub_grains[querry_grain_id].length
sub_grains[querry_grain_id] - query_geom
MultiPolygon(sub_grain_tree.query(query_geom))