import numpy as np
from shapely.geometry import LineString, MultiPolygon, Polygon, Point
from shapely.ops import polygonize
from shapely.ops import SplitOp
from shapely.ops import voronoi_diagram
from scipy import interpolate
import random
import matplotlib.pyplot as plt
##########################################################################
vertices = [[0.0, 0.0], [0.5, -0.5], [1.5, -0.5], [2.0, 0.0], [1.5, 0.5], [0.5, 0.5]]
vertices = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
vertices = [[0.0, 0.0], [1.0, 0.0], [2.5, 1.0], [0.75, 1.0]]
vertices = [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]
grain = Polygon(vertices)
grain = Polygon(vertices)
#grain = Polygon(vertices)
grain = Polygon(vertices)

centroid = grain.centroid.xy
centroid = [centroid[0][0], centroid[1][0]]
centroid = [.5, 0.5]
##########################################################################
x_this , y_this  = np.array(grain.boundary.xy[0][:-1]), np.array(grain.boundary.xy[1][:-1])
x_front, y_front = np.roll(x_this, +1), np.roll(y_this, +1)
gb_edges_x = np.c_[x_this.ravel(), x_front.ravel()]
gb_edges_y = np.c_[y_this.ravel(), y_front.ravel()]
gb_centre_x = np.mean(gb_edges_x, axis = 1)
gb_centre_y = np.mean(gb_edges_y, axis = 1)
gb_lengths = np.sqrt(np.square(x_this - x_front) + np.square(y_this - y_front))
n_edges = len(gb_edges_x)
edges = list(np.arange(0, n_edges, 1, dtype = 'int'))
##########################################################################
#~~~~~~~~~~~~~
plt.fill(gb_edges_x, gb_edges_y, color = 'cyan', edgecolor = 'b')
plt.plot(centroid[0], centroid[1], 'k+', markersize = 25)
for count in range(len(gb_centre_x)):
    plt.text(gb_centre_x[count], gb_centre_y[count], str(count), fontsize = 20, weight = 'normal')
    plt.plot(x_this[count], y_this[count], 'kd', markersize = 8)
    plt.text(x_this[count], y_this[count], str(count), fontsize = 16, color = 'brown', weight = 'bold')
##########################################################################
p_edges_to_choose = 1.0
##########################################################################
if p_edges_to_choose == 0.0: p_edges_to_choose += 0.01

min_num_edges_for_pap = 2
n_edges_to_choose = int(p_edges_to_choose * n_edges)
edges_to_choose   = list(np.sort(random.sample(edges, n_edges_to_choose)))
edges_to_choose
if len(edges_to_choose) <= min_num_edges_for_pap:
    while len(edges_to_choose) < min_num_edges_for_pap:
        p_edges_to_choose *= 1.01
        n_edges_to_choose = int(p_edges_to_choose * n_edges)
        edges_to_choose = list(np.sort(random.sample(edges, n_edges_to_choose)))
p_edges_to_choose
edges_to_choose
##########################################################################
# Two numbers between 0 and 1. RULE: 1st element < 2nd element
edge_domain = [0.1, 0.9]
inter_point_selection = 'linear'# linear, ru
N_interp_points = 1
##########################################################################
plt.fill(gb_edges_x, gb_edges_y, color = 'white', edgecolor = 'b')
plt.plot(centroid[0], centroid[1], 'k+', markersize = 25)
for count in range(len(gb_centre_x)):
    plt.text(gb_centre_x[count], gb_centre_y[count], str(count), fontsize = 20, weight = 'normal')
    plt.plot(x_this[count], y_this[count], 'kd', markersize = 8)
    plt.text(x_this[count], y_this[count], str(count), fontsize = 16, color = 'brown', weight = 'bold')
##########################################################################
SPLITTER_BOUNDARY_POINTS_X = []
SPLITTER_BOUNDARY_POINTS_Y = []
for edge_count in edges_to_choose:
    ed_x = gb_edges_x[edge_count]
    ed_y = gb_edges_y[edge_count]
    #L = gb_lengths[edge_count]
    INTERP = interpolate.interp1d(ed_x, ed_y, kind = 'linear')
    #------------
    if N_interp_points == 1:
        points_norm_distances = np.mean([edge_domain[0], edge_domain[1]])
        interp_point_pert_factor = 0.0
        if ed_x[0] == ed_x[1]:
            points_norm_distances *= (np.max(ed_y) - np.min(ed_y))
        else:
            points_norm_distances *= (np.max(ed_x) - np.min(ed_x))
        domain_length = edge_domain[1] - edge_domain[0]
        if random.random()<0.5:
            points_norm_distances = points_norm_distances + interp_point_pert_factor*domain_length*random.random()
        else:
            points_norm_distances = points_norm_distances - interp_point_pert_factor*domain_length*random.random()
    else:
        xpoints = []
        ypoints = []
        if inter_point_selection == 'linear':
            points_norm_distances = np.linspace(edge_domain[0], edge_domain[1], N_interp_points)
            points_norm_distances *= (np.max(ed_x) - np.min(ed_x))
        elif inter_point_selection == 'ru':
            interp_point_pert_factor = 0.2
            domain_length = edge_domain[1] - edge_domain[0]
            points_norm_distances = np.linspace(edge_domain[0], edge_domain[1], N_interp_points)
            points_norm_distances *= (np.max(ed_x) - np.min(ed_x))
            points_norm_distances[0] = points_norm_distances[0]
            points_norm_distances[-1] = points_norm_distances[-1]
            if N_interp_points > 2:
                for inner_domain_count in range(N_interp_points-2):
                    true_count = inner_domain_count+1
                    if random.random()<0.5:
                        points_norm_distances[true_count] = points_norm_distances[true_count]+interp_point_pert_factor*domain_length*random.random()
                    else:
                        points_norm_distances[true_count] = points_norm_distances[true_count]-interp_point_pert_factor*domain_length*random.random()
    if ed_x[0] == ed_x[1]:
        print('VERTICAL')
        y_min = np.min(abs(np.array([ed_y[0], ed_y[1]])))
        y_coord = y_min + points_norm_distances
        if N_interp_points == 1:
            xpoints = ed_x[0]
            ypoints = y_coord
        else:
            xpoints.append(ed_x[0])
            ypoints.append(y_coord)
            xpoints, ypoints = xpoints[0], ypoints[0]
    elif ed_y[0] == ed_y[1]:
        print('HORIZONTAL')
        x_min = np.min(abs(np.array([ed_x[0], ed_x[1]])))
        x_coord = x_min + points_norm_distances
        y_coord = INTERP(x_coord)
        if N_interp_points == 1:
            xpoints = x_coord
            ypoints = float(y_coord)
        else:
            xpoints.append(x_coord)
            ypoints.append(y_coord)
            xpoints, ypoints = xpoints[0], ypoints[0]
    else:
        print('NOT HORIZONTAL')
        x_min = np.min(abs(np.array([ed_x[0], ed_x[1]])))
        x_coord = x_min + points_norm_distances
        y_coord = INTERP(x_coord)
        if N_interp_points == 1:
            xpoints = x_coord
            ypoints = float(y_coord)
        else:
            xpoints.append(x_coord)
            ypoints.append(y_coord)
            xpoints, ypoints = xpoints[0], ypoints[0]
    #------------
    if N_interp_points == 1:
        plt.text(xpoints, ypoints, str(0), color = 'blue', fontsize = 14)
    elif N_interp_points > 1:
        for count in range(len(xpoints)):
            plt.text(xpoints[count], ypoints[count], str(count), color = 'blue', fontsize = 14)
    #------------
    SPLITTER_BOUNDARY_POINTS_X.append(xpoints)
    SPLITTER_BOUNDARY_POINTS_Y.append(ypoints)
##########################################################################
SPLITTER_BOUNDARY_POINTS_X
SPLITTER_BOUNDARY_POINTS_Y
centroid
##########################################################################
PAP = []
if N_interp_points == 1:
    for split_count in range(len(SPLITTER_BOUNDARY_POINTS_X)):
        #print(split_count)
        slc_x = SPLITTER_BOUNDARY_POINTS_X[split_count] # split_line_coords
        slc_y = SPLITTER_BOUNDARY_POINTS_Y[split_count]
        if split_count+1 < len(SPLITTER_BOUNDARY_POINTS_X):
            slc_x_next = SPLITTER_BOUNDARY_POINTS_X[split_count+1]
            slc_y_next = SPLITTER_BOUNDARY_POINTS_Y[split_count+1]
        elif split_count+1 == len(SPLITTER_BOUNDARY_POINTS_X):
            slc_x_next = SPLITTER_BOUNDARY_POINTS_X[0]
            slc_y_next = SPLITTER_BOUNDARY_POINTS_Y[0]
        
        splitting_line = LineString([[slc_x, slc_y],
                                     [centroid[0], centroid[1]],
                                     [slc_x_next, slc_y_next],
                                     ]
                                    )
        PARTITION = SplitOp._split_polygon_with_line(grain, splitting_line)
        #======================================================================
        # edges_to_choose
        # edges
        # n_edges
        #edge_1 = edges_to_choose[split_count]
        #edge_2 = edge_1 + 1
        #check_point = Point(gb_centre_x[edge_2], gb_centre_y[edge_2])
        
        #plt.fill(PARTITION[0].boundary.xy[0], PARTITION[0].boundary.xy[1], color = 'cyan')
        #plt.fill(PARTITION[1].boundary.xy[0], PARTITION[1].boundary.xy[1], color = 'green')
        #plt.plot(check_point.x, check_point.y, 'kx')
        
        PAP.append(PARTITION[0])
        PAP.append(PARTITION[1])
        #if check_point.touches(PARTITION[0]):
        #    PAP.append(PARTITION[0])
        #else:
        #    PAP.append(PARTITION[1])
        #======================================================================
##########################################################################
# SUB-SELECTION OF PAP PARTITIONS as not all in above list are valid
# valid PAP sections do not intersect. Adjacent ones will only touch
#paps = [0, 1, 2, 3, 4, 5]
paps = list(range(len(PAP)))
from itertools import combinations
haha = [comb for comb in combinations(paps, len(edges_to_choose))]

valid_combo = []
for cell_set in haha:
    #cell_set = haha[7]
    cells = [PAP[i] for i in cell_set]
    #sum([cell.area for cell in cells]) == grain.area
    if sum([cell.area for cell in cells]) == grain.area:
        truth_table_intersect = list(np.repeat(0, len(edges_to_choose) - 1))
        intersection_areas = list(np.repeat(0, len(edges_to_choose) - 1))
        for boundary_check_count in range(len(truth_table_intersect)):
            intersection_areas[boundary_check_count] = cells[0].intersection(cells[boundary_check_count+1]).area
            if intersection_areas[boundary_check_count] > 0.0:
                truth_table_intersect[boundary_check_count] = True
            else:
                truth_table_intersect[boundary_check_count] = False
        if sum(truth_table_intersect)==0 and sum(intersection_areas)==0:
            valid_combo.append(cell_set)
            break

# =============================================================================
# valid_combo = []
# for cell_set in haha:
#     cells = [PAP[i] for i in cell_set]
#     if sum([cell.area for cell in cells]) == grain.area:
#         truth_table = list(range( len(edges_to_choose) - 1))
#         for boundary_check_count in range(len(truth_table)):
#             if cells[0].touches(cells[boundary_check_count+1]):
#                 truth_table[boundary_check_count] = True
#         if sum(truth_table) == len(truth_table):
#             valid_combo.append (cell_set)
#             #break
# =============================================================================

print(valid_combo)

#KNOWN_VALID_COMBO = [0, 3, 5]
PAP_valid = [PAP[i] for i in valid_combo[0]]
##########################################################################
grain_with_paps = MultiPolygon(PAP_valid)
##########################################################################
xmin = np.vstack(vertices[:])[:, 0].min()
xmax = np.vstack(vertices[:])[:, 0].max()
ymin = np.vstack(vertices[:])[:, 1].min()
ymax = np.vstack(vertices[:])[:, 1].max()
##########################################################################
plt.figure(dpi = 50)
plt.fill(gb_edges_x, gb_edges_y, color = 'cyan', edgecolor = 'k', linewidth = 3)
for pap_count in range(len(grain_with_paps.geoms)):
    thispap = grain_with_paps.geoms[pap_count]
    x = thispap.boundary.xy[0]
    y = thispap.boundary.xy[1]
    plt.fill(x, y, color = np.random.random(3), alpha = 0.8, edgecolor = 'brown', linewidth = 2)
    plt.text(thispap.centroid.x, thispap.centroid.y, str(pap_count), fontsize = 20, weight = 'normal', backgroundcolor = 'white')
plt.plot(centroid[0], centroid[1], 'k+', markersize = 25)
for count in range(len(gb_centre_x)):
    plt.text(gb_centre_x[count], gb_centre_y[count], str(count), fontsize = 20, weight = 'normal')
    plt.plot(x_this[count], y_this[count], 'kd', markersize = 8)
    plt.text(x_this[count], y_this[count], str(count), fontsize = 16, color = 'brown', weight = 'bold')
#plt.axis('equal')
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
#plt.tight_layout()
plt.show()

vertices = [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]
a = []
a = copy.deepcopy(vertices)
a.append(vertices[0])

env    = LineString(a)# MultiPoint(vertices)
points = MultiPoint([[0.2, 0.1], [0.6, 0.1], [0.6, 0.6], [0.45, 0.5]])
pxtal = voronoi_diagram(points, envelope=env, tolerance=0.0, edges=False)

points.coords


