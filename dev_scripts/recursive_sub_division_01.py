from shapely.ops import voronoi_diagram
from shapely.geometry import Point
from shapely.geometry import MultiPoint
from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon
from shapely.geometry import LineString
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from shapely.geometry import LinearRing
import random
import copy
from scipy import histogram

points_list = [[0, 0], [0.5, 0.5], [0, 1], [2, 2], [0, 3], [3, 2]]

xmin = -2.0
ymin = -2.0
xmax = 4.0
ymax = 4.0

PXTAL_boundary = Polygon([[xmin,ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])

points = MultiPoint(points_list)
pxtal  = voronoi_diagram(points, tolerance = 0.0, edges = False)

for count in range(len(pxtal.geoms)):
    this = pxtal.geoms[count]
    x = this.boundary.xy[0]
    y = this.boundary.xy[1]
    plt.fill(x, y, color = np.random.random(3), alpha = 0.8, edgecolor = 'brown', linewidth = 2)
    plt.text(this.centroid.x, this.centroid.y, str(count), fontsize = 20, weight = 'normal', backgroundcolor = 'white', alpha = 0.5)

plt.fill(PXTAL_boundary.boundary.xy[0], PXTAL_boundary.boundary.xy[1], 'cyan', alpha = 0.5, edgecolor = 'black')

for count in range(len(points)):
    xy = points[count].xy
    plt.plot(xy[0], xy[1], 'ko', markersize = 15)


contained = []
contained_cropped = []
for count in range(len(pxtal.geoms)):
    if pxtal.geoms[count].intersects(PXTAL_boundary):
        contained.append(pxtal.geoms[count])
        contained_cropped.append(pxtal.geoms[count].intersection(PXTAL_boundary))
#mpc = MultiPolygon(contained)
PXTAL_bound = MultiPolygon(contained_cropped)


pxtal__xtal_list = []
for xtal_count in range(len(PXTAL_bound.geoms)):
    pxtal__xtal_list.append(PXTAL_bound.geoms[xtal_count])


grains = []
for sub_grain_count in range(len(PXTAL_bound.geoms)):
    grain = PXTAL_bound.geoms[sub_grain_count]
    grains.append(grain)

plt.figure('GSerrbar', figsize=(3.5, 3.5), dpi=200)
for grain_count in range(len(grains)):
    plt.fill(grains[grain_count].boundary.xy[0], grains[grain_count].boundary.xy[1],
             facecolor = 'cyan', edgecolor = 'black', alpha = 1.0, linewidth = 1)
    #plt.plot(grains[grain_count].centroid.x, grains[grain_count].centroid.y, 'ro', markersize = 15)
    plt.text(grains[grain_count].centroid.x, grains[grain_count].centroid.y, str(grain_count), fontsize = 20, weight = 'bold', backgroundcolor = 'none', alpha = 1.0)
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
plt.show()
#==============================================================================
#==============================================================================
def voronoi_subdivision(xtal_object, n_seed_points, seed_lattice_type, combine_small_subs):
    '''
    xtal_object: 
        crystal object
        type: shapely Polygon
    n_seed_points: 
        number of seed points for voronoi tessellation inside this grain
        type: shapely MultiPoint
    seed_lattice_type:
        distribution of seed points
        type: str
        values: 'ru', 'hex', 'tri', 'rec', 'rn_xtal_centroid'
    combine_small_subs:
        Whether to combine small Voronoi cells with neighbours
        type: bool - True / False

    Access: 
        voronoi_subdivision(pxtal[2], seed_points, combine_small_subs)
    '''
    x_bounds = xtal_object.boundary.xy[0]
    y_bounds = xtal_object.boundary.xy[1]

    xmin = min(x_bounds)
    xmax = max(x_bounds)
    ymin = min(y_bounds)
    ymax = max(y_bounds)
    
    x = 0.9*xmin + 1.2*(xmax-xmin)*np.random.random(n_seed_points)
    y = 0.9*ymin + 1.2*(ymax-ymin)*np.random.random(n_seed_points)
    
    points = MultiPoint([[x[i], y[i]] for i in range(n_seed_points)])
    sub_grain_pxtal = voronoi_diagram(points, tolerance = 0.0, edges=False)
    
    #contained = []
    contained_cropped = []
    for count in range(len(sub_grain_pxtal.geoms)):
        if sub_grain_pxtal.geoms[count].intersects(xtal_object):
            #contained.append(sub_grain_pxtal.geoms[count])
            contained_cropped.append(sub_grain_pxtal.geoms[count].intersection(xtal_object))

    #mpc = MultiPolygon(contained)
    pxtal_object_modified = MultiPolygon(contained_cropped)
    
    return contained_cropped, pxtal_object_modified
#==============================================================================
pxtal = copy.deepcopy(PXTAL_bound)
#==============================================================================
n_grains = len(pxtal.geoms)
Vf = 0.5
gr_with_sub_cells = list(set(random.sample(list(range(n_grains)), int(n_grains*Vf))))
#==============================================================================
pxtal_new__xtal_list = copy.deepcopy(pxtal__xtal_list)
#==============================================================================
for chosen_grain_id in gr_with_sub_cells:
    xtal_object = copy.deepcopy(pxtal.geoms[chosen_grain_id])
    n_seed_points = 25 
    #==============================================================================
    #==============================================================================
    grain_cells, grain_pxtal = voronoi_subdivision(xtal_object,
                                      n_seed_points,
                                      'ru',
                                      'False')

    pxtal_new__xtal_list[chosen_grain_id] = grain_cells

PXTAL_cells_list = []
for count_level0 in range(len(pxtal_new__xtal_list)):
    if isinstance(pxtal_new__xtal_list[count_level0], list):
        for count_level1 in range(len(pxtal_new__xtal_list[count_level0])):
            PXTAL_cells_list.append(pxtal_new__xtal_list[count_level0][count_level1])
    else:
        PXTAL_cells_list.append(pxtal_new__xtal_list[count_level0])

PXTAL_new = MultiPolygon(PXTAL_cells_list)

plt.figure('GSerrbar', figsize=(3.5, 3.5), dpi=200)
for count in range(len(PXTAL_new.geoms)):
    this = PXTAL_new.geoms[count]
    x = this.boundary.xy[0]
    y = this.boundary.xy[1]
    plt.fill(x, y, color = 'cyan', alpha = 0.8, edgecolor = 'blue', linewidth = 3)
    plt.plot(this.centroid.x, this.centroid.y, 'k+')
for count in range(len(PXTAL_bound.geoms)):
    this = PXTAL_bound.geoms[count]
    x = this.boundary.xy[0]
    y = this.boundary.xy[1]
    plt.fill(x, y, color = 'none', edgecolor = 'black', linewidth = 3)
    plt.plot(this.centroid.x, this.centroid.y, 'ro', markersize = 12)

# NUMBER THE VERTICES
vertices = []
vertices_celled = []
for pol in PXTAL_new.geoms:
    vert = pol.boundary.xy
    coords = [[vert[0][i], vert[1][i]] for i in range(len(vert[0]))]
    for c in coords:
        vertices.append(c)
    vertices_celled.append(coords)
vertices = np.unique(np.array(vertices), axis = 0)
# IDENTITIES OF VERTICES
PXTAL_vertices_ID = tuple(range(len(vertices)))

plt.figure('GSerrbar', figsize=(3.5, 3.5), dpi=200)
for count in range(len(PXTAL_new.geoms)):
    this = PXTAL_new.geoms[count]
    x = this.boundary.xy[0]
    y = this.boundary.xy[1]
    plt.fill(x, y, color = 'cyan', alpha = 0.8, edgecolor = 'blue', linewidth = 1)
    plt.plot(this.centroid.x, this.centroid.y, 'k+')
for point in vertices:
    plt.scatter(point[0], point[1], color = 'darkred')


# IDENTIFY VERTICES BY THEIR ID ON THE GRAIN BOUNDARIES
PXTAL_vertices = []
PXTAL_vertices_celled = []
for pol in PXTAL_new.geoms:
    vert = pol.boundary.xy
    coords = [[vert[0][i], vert[1][i]] for i in range(len(vert[0]))]
    for c in coords:
        pass
        #Search in vertices asnf find relavant ID from PXTAL_vertices_ID
        #vertices.append(c)
    vertices_celled.append(coords)
vertices = np.unique(np.array(vertices), axis = 0)

# IDENTITIES OF VERTICES
ID_VERT = tuple(range(len(vertices)))

# AREAS
area_list_levels = []
area_list = []
for cell_0_count in range(len(pxtal_new__xtal_list)):
    if isinstance(pxtal_new__xtal_list[cell_0_count], list):
        area_list_levels.append([])
        for cell_1_count in range(len(pxtal_new__xtal_list[cell_0_count])):
            _area = pxtal_new__xtal_list[cell_0_count][cell_1_count].area
            area_list_levels[cell_0_count].append(_area)
            area_list.append(_area)
    else:
        _area = pxtal_new__xtal_list[cell_0_count].area
        area_list_levels.append(_area)
        area_list.append(_area)

# LENGTHS
perimeters_list_levels = []
perimeters_list = []
for cell_0_count in range(len(pxtal_new__xtal_list)):
    if isinstance(pxtal_new__xtal_list[cell_0_count], list):
        perimeters_list_levels.append([])
        for cell_1_count in range(len(pxtal_new__xtal_list[cell_0_count])):
            _perimeter = pxtal_new__xtal_list[cell_0_count][cell_1_count].length
            perimeters_list_levels[cell_0_count].append(_perimeter)
            perimeters_list.append(_perimeter)
    else:
        _perimeter = pxtal_new__xtal_list[cell_0_count].length
        perimeters_list_levels.append(_perimeter)
        perimeters_list.append(_perimeter)

# Edge LENGTHS - AS PER LEVEL ORDER
edgelength_list_levels = []
for cell_0_count in range(len(pxtal_new__xtal_list)):
    if isinstance(pxtal_new__xtal_list[cell_0_count], list):
        _edges = []
        for cell_1_count in range(len(pxtal_new__xtal_list[cell_0_count])):
            b = pxtal_new__xtal_list[cell_0_count][cell_1_count].boundary.coords
            linestrings = [LineString(b[k:k+2]) for k in range(len(b) - 1)]
            _edges.append(linestrings)
    else:
        b = pxtal_new__xtal_list[cell_0_count].boundary.coords
        _edges = [LineString(b[k:k+2]) for k in range(len(b) - 1)]
    edgelength_list_levels.append(_edges)
#------------------------------------------------------------------------------
# Grain boundary Edge - DUMPED and uniqued

# Get non-unique list of grain boundary edge objects
edge_list = []
for L_a_count in range(len(edgelength_list_levels)):
    if not isinstance(edgelength_list_levels[L_a_count], list):
        edge_list.append(edgelength_list_levels[L_a_count])
    else:
        for L_b_count in range(len(edgelength_list_levels[L_a_count])):
            if isinstance(edgelength_list_levels[L_a_count][L_b_count], list):
                for L_c_count in range(len(edgelength_list_levels[L_a_count][L_b_count])):
                    edge_list.append(edgelength_list_levels[L_a_count][L_b_count][L_c_count])
            else:
                edge_list.append(edgelength_list_levels[L_a_count][L_b_count])
# Memory addresses of Edge objects
edge_list_memaddr = [id(edge) for edge in edge_list]
# Find overlapping Linestrings:
EDGE_NUMBERS = []
counts_non_unique = list(range(len(edge_list)))
#for edge_count in range(len(edge_list)):
#    print(edge_list[0].within(edge_list[edge_count]))
#------------------------------------------------------------------------------
edgelength_list = []
edgelength_list_mem_address = []
for cell_0_count in range(len(pxtal_new__xtal_list)):
    if isinstance(pxtal_new__xtal_list[cell_0_count], list):
        _edges = []
        _edges_mem_address = []
        for cell_1_count in range(len(pxtal_new__xtal_list[cell_0_count])):
            b = pxtal_new__xtal_list[cell_0_count][cell_1_count].boundary.coords
            linestrings = [LineString(b[k:k+2]) for k in range(len(b) - 1)]
            _edges.append(linestrings)
            edgelength_list.append(linestrings)
            _edges_mem_address_temp = [id(LineString(b[k:k+2])) for k in range(len(b) - 1)]
        for _edge in _edges_mem_address_temp:
            edgelength_list_mem_address.append(_edge)
            #coordinates = [list(ls.coords) for ls in linestrings]
            #_L = 
            #_edgelengths = .ridge
            #perimeters_list_levels[cell_0_count].append(_perimeter)
            #perimeters_list.append(_perimeter)
    else:
        b = pxtal_new__xtal_list[cell_0_count].boundary.coords
        _edges = [LineString(b[k:k+2]) for k in range(len(b) - 1)]
        edgelength_list.append(_edges)
        _edges_mem_address_temp = [id(_edge) for _edge in _edges]
        edgelength_list_mem_address.append(_edges_mem_address_temp)
    edgelength_list_levels.append(_edges)
#    edgelength_list_mem_address.append(_edges_mem_address)
# =============================================================================
# mem_address_EDGES = []
# for L0_edges in edgelength_list_mem_address:
#     for L0_edge in L0_edges:
#         mem_address_EDGES.append(L0_edge)
# mem_address_EDGES = list(set(mem_address_EDGES))
# 
# # Get and plot histogram
# data = area_list
# data = perimeters_list
# bins = list(set([int(data[count]*100)/100 for count in range(len(data))]))
# bins.sort()
# bins = list(set([int(data[count]) for count in range(len(data))]))
# count_hist, value_count_hist = histogram(data, bins = bins)
# plt.bar(value_count_hist[:-1], count_hist, width = 0.01, edgecolor = 'black', color = 'lightgray')
# 
# =============================================================================
