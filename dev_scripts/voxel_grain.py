from shapely.ops import voronoi_diagram
from shapely.geometry import Point
from shapely.geometry import MultiPoint
from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial import ConvexHull, convex_hull_plot_2d

points_list = [(0, 0), (0.5, 0.5), (0, 1), (2, 2), (0, 3), (3, 2)]
points = MultiPoint(points_list)
regions = voronoi_diagram(points, envelope=None, tolerance=0.0, edges=False)
############## THE POLY-XTAL ##############
fig = plt.figure(dpi = 100)
for count in range(len(regions.geoms)):
    grain = regions.geoms[count]
    gb = grain.boundary
    plt.fill(gb.xy[0], gb.xy[1],
             facecolor = 'red', edgecolor = 'black', alpha = 1.0, linewidth = 2)
for count in range(len(points_list)):
    p = points_list[count]
    plt.scatter(p[0], p[1], color = 'black')
plt.axis('equal')
plt.show()
###################################################################
###################################################################
##### LETS ESTABLISH THE PROOF OF CONCEPT FOR 1 GRAIN #############
###################################################################
###################################################################
gr   = regions.geoms[0] # Grain number 0

# Get the individual edges of the grain boundary
x_this, y_this   = np.array(gr.boundary.xy[0][:-1]), np.array(gr.boundary.xy[1][:-1])
x_front, y_front = np.roll(x_this, +1), np.roll(y_this, +1)

# Get gb edges as [[x(i), y(i)], [x(i+1), y(i+1)]]
gb_edges = [np.c_[x_this.ravel(), y_this.ravel()], np.c_[x_front.ravel(), y_front.ravel()]]
gb_lengths = np.sqrt(np.square(gb_edges[0][:,0]-gb_edges[1][:,0]) + 
                     np.square(gb_edges[0][:,1]-gb_edges[1][:,1]))
gb_lengths = np.sqrt(np.square(x_this - x_front) + np.square(y_this - y_front))

bz_offset = 0.3 # Boundary zone offset
grdf = Polygon(gr.boundary.parallel_offset(-bz_offset, 'left', resolution = 1, join_style=3))
rec  = gr.envelope
#rec  = grdf.envelope
rrec = grdf.minimum_rotated_rectangle
MultiPolygon([gr, rec, rrec])

############## BOUNDING RECTANGLE ##############
# PARENT POLYGONS
fig = plt.figure(dpi = 100)
plt.fill(gr.boundary.xy[0], gr.boundary.xy[1], facecolor = 'red', edgecolor = 'black', alpha = 1.0, linewidth = 2)
plt.fill(rec.boundary.xy[0], rec.boundary.xy[1], facecolor = 'cyan', edgecolor = 'black', alpha = 1.0, linewidth = 2)
plt.fill(grdf.boundary.xy[0], grdf.boundary.xy[1], facecolor = 'yellow', edgecolor = 'black', alpha = 1.0, linewidth = 2)
plt.axis('equal')
fig.tight_layout()
plt.show()

# GENERATE and OVERLAY THE GRID
recx, recy = rec.boundary.xy[0][:-1], rec.boundary.xy[1][:-1]
x = np.linspace(min(recx), max(recx), int((max(recx) - min(recx))*2/bz_offset))
y = np.linspace(min(recy), max(recy), int((max(recy) - min(recy))*2/bz_offset))
grid_spacing_x = min([x[1] - x[0], y[1] - y[0]])
grid_spacing_y = grid_spacing_x
x, y = np.meshgrid(x, y)
fig = plt.figure(dpi = 100)
plt.fill(gr.boundary.xy[0], gr.boundary.xy[1], facecolor = 'red', edgecolor = 'black', alpha = 1.0, linewidth = 2)
plt.fill(rec.boundary.xy[0], rec.boundary.xy[1], facecolor = 'cyan', edgecolor = 'black', alpha = 1.0, linewidth = 2)
plt.fill(grdf.boundary.xy[0], grdf.boundary.xy[1], facecolor = 'yellow', edgecolor = 'black', alpha = 1.0, linewidth = 2)
plt.scatter(x, y)
plt.axis('equal')
fig.tight_layout()
plt.show()

# Make a mask array like x and y
mask = np.zeros(np.shape(x))
# For each point check the validity for being contained within grdf
mask_x = []
mask_y = []
grdf_contains = grdf.contains
grdf_boundary_contains = grdf.boundary.contains
for i in range(np.shape(x)[0]): # Across rows
    for j in range(np.shape(x)[1]): # Across columns
        point_object = Point(x[i, j], y[i, j])
        if grdf_contains(point_object) or grdf_boundary_contains(point_object):
            mask[i, j] = 1
            mask_x.append(x[i, j])
            mask_y.append(y[i, j])
            
# OVERLAY THE CPONTAINED GRID POINTS
fig = plt.figure(dpi = 100)
plt.fill(gr.boundary.xy[0], gr.boundary.xy[1], facecolor = 'red', edgecolor = 'black', alpha = 1.0, linewidth = 2)
plt.fill(rec.boundary.xy[0], rec.boundary.xy[1], facecolor = 'cyan', edgecolor = 'black', alpha = 1.0, linewidth = 2)
plt.fill(grdf.boundary.xy[0], grdf.boundary.xy[1], facecolor = 'yellow', edgecolor = 'black', alpha = 1.0, linewidth = 2)
plt.scatter(x, y, color = 'green')
plt.scatter(mask_x, mask_y, color = 'black')
plt.axis('equal')
fig.tight_layout()
plt.show()

# PLOT ONLY THE CONTAINED GRID POINTS
fig = plt.figure(dpi = 100)
plt.scatter(mask_x, mask_y, color = 'black')
plt.axis('equal')
fig.tight_layout()
plt.show()

########### DETOUR: USE OF KDTREE TO QUERRY NEAREST NEIGHBOURS
mask_xy = np.hstack([np.array(mask_x)[np.newaxis].transpose(),
                     np.array(mask_y)[np.newaxis].transpose()])
kd_tree = cKDTree(mask_xy)# Make cKDTree of all contained points
cutoff_radius = np.sqrt(grid_spacing_x**2+grid_spacing_y**2)# the cut-off radius
query_point = mask_xy[8]# Choose a query point
# Find neighbours for this query point within the cut-off radius
neigh = kd_tree.query_ball_point(query_point, 1.0001*cutoff_radius, eps = 0) 
# plot the results
fig = plt.figure(dpi = 100)
plt.scatter(query_point[0], query_point[1])
plt.plot(mask_xy[:, 0], mask_xy[:, 1], "xk", markersize=5)
for count in neigh:
    plt.plot([query_point[0], mask_x[count]], [query_point[1], mask_y[count]], 'red', alpha = 1.0)
plt.axis('equal')
plt.show()

# Make multi-point from the list of contained points
mpoint_coord = []
for count in range(len(mask_x)):
    mpoint_coord.append((mask_x[count], mask_y[count]))
mp = MultiPoint(mpoint_coord)

# Get convex hull of 
mp.convex_hull

hull = ConvexHull(mask_xy)

# OVERLAY CONVEX HULL ON THE CONTAINED GRID POINTS
fig = plt.figure(dpi = 100)
for count in range(len(regions.geoms)):
    grain = regions.geoms[count]
    gb = grain.boundary
    plt.fill(gb.xy[0], gb.xy[1],
             facecolor = 'red', edgecolor = 'black', alpha = 1.0, linewidth = 2)
plt.fill(gr.boundary.xy[0], gr.boundary.xy[1], facecolor = 'yellow', edgecolor = 'black', alpha = 1.0, linewidth = 2)
plt.fill(grdf.boundary.xy[0], grdf.boundary.xy[1], facecolor = 'yellow', edgecolor = 'black', alpha = 1.0, linewidth = 2)
plt.plot(mask_x, mask_y, 'ks', markersize = 2)
for simplex in hull.simplices:
    plt.plot([mask_x[simplex[0]], mask_x[simplex[1]]],
             [mask_y[simplex[0]], mask_y[simplex[1]]],
             '-o', linewidth = 2, markersize = 8, alpha = 0.8)
plt.axis('equal')
plt.axis('off')
plt.show()




# Make a mask array like x and y
mask_gr = np.zeros(np.shape(x))
# For each point check the validity for being contained within grdf
gr_contains = gr.contains
gr_boundary_contains = gr.boundary.contains
grdf_contains = grdf.contains
grdf_boundary_contains = grdf.boundary.contains
for i in range(np.shape(x)[0]): # Across rows
    for j in range(np.shape(x)[1]): # Across columns
        point_object = Point(x[i, j], y[i, j])
        if gr_contains(point_object) or gr_boundary_contains(point_object):
            mask_gr[i, j] = 1

mask_zone = mask_gr-mask
mask_x_zone = []
mask_y_zone = []
for i in range(np.shape(x)[0]): # Across rows
    for j in range(np.shape(x)[1]): # Across columns
        if mask_zone[i, j] == 1:
            mask_x_zone.append(x[i, j])
            mask_y_zone.append(y[i, j])


# EXTRACT GRAIN BOUNDARY ZONE
# gbz = gr - grdf





