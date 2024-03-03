import numpy as np
from shapely.geometry import LineString, MultiPolygon, Polygon, Point
from shapely.ops import polygonize
from shapely.ops import SplitOp
from scipy import interpolate
import random
##########################################################################
# make a complex geometry
grain = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
centroid = grain.centroid.xy
centroid = [centroid[0][0], centroid[1][0]]
##########################################################################
x_this , y_this  = np.array(grain.boundary.xy[0][:-1]), np.array(grain.boundary.xy[1][:-1])
x_front, y_front = np.roll(x_this, +1), np.roll(y_this, +1)
gb_edges_x = np.c_[x_this.ravel(), x_front.ravel()]
gb_edges_y = np.c_[y_this.ravel(), y_front.ravel()]
gb_lengths = np.sqrt(np.square(x_this - x_front) + np.square(y_this - y_front))
n_edges = len(gb_edges_x)
p_edges_to_choose = 0.8
n_edges_to_choose = int(p_edges_to_choose * n_edges)
# Two numbers between 0 and 1. RULE: 1st element < 2nd element
edge_domain = [0.4, 0.6]
inter_point_selection = 'linear'# linear, ru

# Choose 3 edges at random
f = interpolate.interp1d(gb_edges_x[0], gb_edges_y[0], kind = 'linear')
N_interp_points = 1
xnew = []
ynew = []
for count in range(N_interp_points):
    if inter_point_selection == 'linear':
        factor = edge_domain[0]
    elif inter_point_selection == 'ru':
        factor = edge_domain[0]+random.random()*(edge_domain[1]-edge_domain[0])
    x_coord_temp = min([gb_edges_x[0][1], gb_edges_x[0][0]])+factor*abs(gb_edges_x[0][1]-gb_edges_x[0][0])
    xnew.append(x_coord_temp)
    if sum(gb_edges_x[0]) == 0.0:
        ynew.append(min([gb_edges_y[0][1], gb_edges_y[0][0]])+factor*abs(gb_edges_y[0][1]-gb_edges_y[0][0]))
    else:
        ynew.append(f(x_coord_temp))
sort_index = sorted(range(len(ynew)), key=ynew.__getitem__)
ynew = [ynew[i] for i in sort_index]
xnew = [xnew[i] for i in sort_index]

f = interpolate.interp1d([xnew[0], centroid[0]], [ynew[0], centroid[0]], kind = 'linear')
factor = 1.01
x_coord_temp = min([xnew[0], centroid[0]]) + factor*abs(xnew[0]-centroid[0])
if xnew[0] == centroid[0]:
    y_coord_temp = min([ynew[0], centroid[1]]) + factor*abs(ynew[0]-centroid[1])
else:
    y_coord_temp = f(x_coord_temp)

LINE1_x = [centroid[0], x_coord_temp]
LINE1_y = [centroid[1], float(y_coord_temp)]

##########################################################################
splitting_line = LineString([[0.5, 1.5], [0.5, 0.5], [1.5, 0.5]])
##########################################################################
PXTAL = SplitOp._split_polygon_with_line(grain, splitting_line)
##########################################################################
# CONVEX HULL WITH ALPHA OPTION FROM ALPHASHAPE PACKAGE
import alphashape
import matplotlib.pyplot as plt
import numpy as np

x = np.random.random((30, 30))
y = np.random.random((30, 30))
coordinates = np.c_[x.ravel(), y.ravel()]

points = np.hstack((np.array(mask_x_core)[np.newaxis].transpose(),
                    np.array(mask_y_core)[np.newaxis].transpose()))
#alpha = 0.95 * alphashape.optimizealpha(points)
alpha = 5
hull = alphashape.alphashape(points, alpha)
hull_pts = hull.exterior.coords.xy

fig = plt.figure(dpi = 200)
plt.plot(mask_x_core, mask_y_core, 'x', markersize = 5)
plt.plot(hull_pts[0], hull_pts[1], '-ko', color='red', alpha = 0.6)
plt.show()
