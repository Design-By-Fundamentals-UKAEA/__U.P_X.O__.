from shapely.ops import voronoi_diagram
from shapely.geometry import Point
from shapely.geometry import MultiPoint
from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from shapely.geometry import LinearRing

points_list = [[0, 0], [0.5, 0.5], [0, 1], [2, 2], [0, 3], [3, 2]]
#points_list = [[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]]
env = MultiPoint([[-0.5,-0.5],[1.5, -0.5],[0.5, 0.5]])
points = MultiPoint(points_list)
pxtal = voronoi_diagram(points, envelope=env, tolerance=0.0, edges=False)

gb_length_min = []
for grain_count in range(len(pxtal.geoms)):
    gr = pxtal.geoms[grain_count]
    # Get the individual edges of the grain boundary
    x_this , y_this  = np.array(gr.boundary.xy[0][:-1]), np.array(gr.boundary.xy[1][:-1])
    x_front, y_front = np.roll(x_this, +1), np.roll(y_this, +1)
    gb_length_min.append(min(np.sqrt(np.square(x_this - x_front) + np.square(y_this - y_front))))
smallest_gb_length = min(gb_length_min)

# Calculate the grid on the poly-xtal
rec = pxtal.envelope
recx, recy = rec.boundary.xy[0][:-1], rec.boundary.xy[1][:-1]
############################################################################
RECT_GRID_FACTOR = 15
distribution = 'ru'
if distribution == 'rectgrid':
    x = np.linspace(min(recx), max(recx), int((max(recx) - min(recx))*RECT_GRID_FACTOR/smallest_gb_length))
    y = np.linspace(min(recy), max(recy), int((max(recy) - min(recy))*RECT_GRID_FACTOR/smallest_gb_length))
    grid_spacing_x = min([x[1] - x[0], y[1] - y[0]])
    grid_spacing_y = grid_spacing_x
    x, y = np.meshgrid(x, y)
else:
    COORD_SIZE = 100
    xmin, xmax = min(pxtal.envelope.boundary.xy[0]), max(pxtal.envelope.boundary.xy[0])
    ymin, ymax = min(pxtal.envelope.boundary.xy[1]), max(pxtal.envelope.boundary.xy[1])
    grid_spacing_x = 0.1
    grid_spacing_y = 0.1
    if distribution == 'ru':
        # RANDOM RANDOM
        x = np.random.random((COORD_SIZE, COORD_SIZE))
        y = np.random.random((COORD_SIZE, COORD_SIZE))
    elif distribution == 'rp':
        # RANDOM POWER
        exponent = 3
        x = np.reshape(np.random.power(exponent, size = COORD_SIZE**2), (COORD_SIZE, COORD_SIZE))
        y = np.reshape(np.random.power(exponent, size = COORD_SIZE**2), (COORD_SIZE, COORD_SIZE))
    elif distribution == 're':
        # RANDOM EXPONENTIAL
        exponent = 1
        x = np.reshape(np.random.exponential(exponent, size = COORD_SIZE**2), (COORD_SIZE, COORD_SIZE))
        y = np.reshape(np.random.exponential(exponent, size = COORD_SIZE**2), (COORD_SIZE, COORD_SIZE))
    x = x/x.max()
    y = y/y.max()
    x = x*(xmax-xmin) + xmin
    y = y*(ymax-ymin) + ymin
############################################################################
# OVERLAY GRID ON THE PXTAL
fig = plt.figure(dpi = 100)
for count in range(len(pxtal.geoms)):
    grain = pxtal.geoms[count]
    gb = grain.boundary
    plt.fill(gb.xy[0], gb.xy[1],
             facecolor = 'cyan', edgecolor = 'black', alpha = 1.0, linewidth = 2)
for count in range(len(points_list)):
    p = points_list[count]
    plt.scatter(p[0], p[1], color = 'red')
plt.plot(x, y, 'k+', markersize = 3)
plt.axis('equal')
plt.show()
############################################################################
GRAINS = []
GRAIN_BOUNDARY_ZONES = []
GRAIN_CORES = []

fig = plt.figure(dpi = 100)
for count in range(len(pxtal.geoms)):
    grain = pxtal.geoms[count]
    plt.fill(grain.boundary.xy[0], grain.boundary.xy[1],
             facecolor = 'white', edgecolor = 'black', alpha = 1.0, linewidth = 3)

bz_offset_MINIMUM = 1.25*min([grid_spacing_x, grid_spacing_y])

for grain_count in range(len(pxtal.geoms)):
    gr = pxtal.geoms[grain_count]
    GRAINS.append(gr)
    #-----------------------------------------------------
    bz_offset = bz_offset_MINIMUM # Boundary zone offset
    grdf = Polygon(gr.boundary.parallel_offset(-bz_offset, 'left', resolution = 1, join_style=3))
    GRAIN_BOUNDARY_ZONES.append(gr - grdf)
    #-----------------------------------------------------
    GRAIN_CORES.append(grdf)
    #-----------------------------------------------------
    plt.fill(grdf.boundary.xy[0], grdf.boundary.xy[1], linestyle = ':', facecolor = 'white', edgecolor = 'black', alpha = 1.0, linewidth = 2)
    #-----------------------------------------------------
    # For each point check the validity for being contained within grdf
    mask_gr = np.zeros(np.shape(x))
    mask_x_gr = []
    mask_y_gr = []
    gr_contains = gr.contains
    gr_boundary_contains = gr.boundary.contains
    mask_core = np.zeros(np.shape(x))
    mask_x_core = []
    mask_y_core = []
    grdf_contains = grdf.contains
    grdf_boundary_contains = grdf.boundary.contains
    mask_x_zone = []
    mask_y_zone = []
    for i in range(np.shape(x)[0]): # Across rows
        for j in range(np.shape(x)[1]): # Across columns
            point_object = Point(x[i, j], y[i, j])
            if gr_contains(point_object) or gr_boundary_contains(point_object):
                mask_gr[i, j] = 1
                mask_x_gr.append(x[i, j])
                mask_y_gr.append(y[i, j])
            if grdf_contains(point_object) or grdf_boundary_contains(point_object):
                mask_core[i, j] = 1
                mask_x_core.append(x[i, j])
                mask_y_core.append(y[i, j])
    plt.plot(mask_x_core, mask_y_core, 'k+', markersize = 3)
    mask_zone = mask_gr - mask_core
    for i in range(np.shape(x)[0]): # Across rows
        for j in range(np.shape(x)[1]): # Across columns
            if mask_zone[i, j] == 1:
                mask_x_zone.append(x[i, j])
                mask_y_zone.append(y[i, j])
    plt.plot(mask_x_zone, mask_y_zone, 'x', color = np.random.random((1,3)))
plt.axis('equal')
fig.tight_layout()
plt.show()
