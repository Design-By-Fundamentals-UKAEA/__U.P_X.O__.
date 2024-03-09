import numpy as np
from shapely.geometry import LineString, MultiPolygon, Polygon, Point, MultiPoint
from shapely.ops import polygonize
from shapely.ops import split, SplitOp
from shapely.ops import voronoi_diagram
from shapely import affinity
from scipy import interpolate
import copy
import random
import matplotlib.pyplot as plt
import math
##########################################################################
def cosd(rad_angle):
    return np.cos(rad_angle*np.pi/180)
def sind(rad_angle):
    return np.sin(rad_angle*np.pi/180) 
##########################################################################   
xstretch = 1
ystretch = 1
A = 1

x0 = 0
x1 = x0 + A*cosd(60)
x2 = x1 + A
x3 = x2 + A*cosd(60)
x4 = x2
x5 = x1

y0 = 0
y1 = y0 - A*sind(60)
y2 = y1
y3 = y0
y4 = y0 + A*sind(60)
y5 = y4

vertices = [[x0, y0],
            [x1, y1],
            [x2, y2],
            [x3, y3],
            [x4, y4],
            [x5, y5]]

grain = Polygon(vertices)
##########################################################################
x_this, y_this  = np.array(grain.boundary.xy[0][:-1]), np.array(grain.boundary.xy[1][:-1])
x_front, y_front = np.roll(x_this, +1), np.roll(y_this, +1)
gb_edges_x = [list(row) for row in list(np.c_[x_this.ravel(), x_front.ravel()])]
gb_edges_y = [list(row) for row in list(np.c_[y_this.ravel(), y_front.ravel()])]
gb_lengths = np.sqrt(np.square(x_this - x_front) + np.square(y_this - y_front))
gb_edges_n = len(gb_lengths)
gb_lengths_max = max(gb_lengths)
##########################################################################
grain_edges_objects = []
for i in range(gb_edges_n):
    _point_a = Point(gb_edges_x[i][0], gb_edges_y[i][0])
    _point_b = Point(gb_edges_x[i][1], gb_edges_y[i][1])
    grain_edges_objects.append(LineString([_point_a, _point_b]))
##########################################################################
grain_edge_points = []
for gb_edge_count in range(gb_edges_n):
    point_distances = np.linspace(0, 1, 10)
    This_Edge_Points = []
    for point_count in range(len(point_distances)):
        This_Edge_Points.append(grain_edges_objects[gb_edge_count].interpolate(gb_lengths[gb_edge_count]*point_distances[point_count]))
    grain_edge_points.append(This_Edge_Points)
##########################################################################
plt.figure('vaasu', figsize = (3.5, 3.5), dpi=100)
#for gb_edge_count in range(gb_edges_n):
gb_edge_count = 5
plt.plot(gb_edges_x[gb_edge_count][0], gb_edges_y[gb_edge_count][0], 'kx', markersize = 12)
pcount = 0
for p in grain_edge_points[gb_edge_count]:
    plt.plot(p.x, p.y, 'ro')
    plt.text(p.x, p.y, str(pcount))
    pcount += 1
plt.plot(gb_edges_x[gb_edge_count][1], gb_edges_y[gb_edge_count][1], 'kx', markersize = 12)
plt.xlim([0, 2.0])
plt.ylim([-1, 1])
##########################################################################
# haha = LineString(grain_edge_points[0])
##########################################################################