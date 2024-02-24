from shapely.ops import voronoi_diagram
from shapely.geometry import LineString, MultiPolygon, Polygon, Point, MultiPoint
from shapely.ops import polygonize
from shapely.ops import split, SplitOp
from shapely.ops import voronoi_diagram
from shapely import affinity
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from shapely.geometry import LinearRing
############################################################################
from pxtalops import bounding_rectangle_pxtal
#-----------------------------------------------------------------
from object_list_builder import build_xtal_list_xtal_level0
from object_list_builder import build_xtal_list_centroid_level0
from object_list_builder import build_xtal_list_gb_level0
#-----------------------------------------------------------------
from characterization import charz_L0_PROP_pxt_areas
from characterization import charz_L0_PROP_pxt_peri
from characterization import charz_L0_PROP_pxt_vert_coords
from characterization import charz_L0_PROP_pxt_gb
from characterization import charz_L0_FID_xt_vert
#-----------------------------------------------------------------
from characterization import FIDs_near_propvalue
#-----------------------------------------------------------------
from vis_grain_structure_vtgs import plot_pxtal
############################################################################
# VISUALIZATION PARAMETERS
vis_pxtal_controls = {'vis_xtal_face_bool': True,
                      'vis_xtal_centroids_bool': True,
                      'vis_gbe_edges_bool': True,
                      'vis_gbe_vertices_bool': False,
                      'vis_overlay_grid': True
                      }

vis_parameters = {'figsize': (3.5, 3.5),
                  'dpi': 100,
                  'pxtalevel': 0,
                  'vis_xtal_clr': 'white',
                  'vis_xtal_alpha': 1.0,
                  'vis_gbe_clr': 'black',
                  'vis_gbe_lw': 2,
                  'vis_text_xtaln': True
                  }
############################################################################
points_list = [[0, 0], [0.5, 0.5], [0, 1], [2, 2], [0, 3], [3, 2]]
points = MultiPoint(points_list)
pxtal = voronoi_diagram(points, tolerance=0.0, edges=False)
#-----------------------------------------------------------------
# Determine the coords of bounding rectangle and limits of the bound
recx, recy, xlimits, ylimits = bounding_rectangle_pxtal(pxtal)
#-----------------------------------------------------------------
# Coordinates of the vertices in the pxtal: level-0
FD_PXTAL_vertices_xy = charz_L0_PROP_pxt_vert_coords(pxtal)
#-----------------------------------------------------------------
# IDs of the vertices in the pxtal: level-0
FID_PXTAL_vertices_ID = charz_L0_FID_xt_vert(FD_PXTAL_vertices_xy)
#-----------------------------------------------------------------
# list of grain polygon objects: level-0
xtal_list_xtal = build_xtal_list_xtal_level0(pxtal)
# list of grain's centroid point objects: level-0
xtal_centroid_list = build_xtal_list_centroid_level0(xtal_list_xtal)
# list of grain boundary objects and properties: level-0
pxtal_edges_objects = build_xtal_list_gb_level0(xtal_list_xtal)
#-----------------------------------------------------------------
xtal_list_area = charz_L0_PROP_pxt_areas(xtal_list_xtal)
xtal_list_peri = charz_L0_PROP_pxt_peri(xtal_list_xtal)
PXT_GB_L0 = charz_L0_PROP_pxt_gb(pxtal_edges_objects)
#-----------------------------------------------------------------

property_value_list = xtal_list_area['a']
property_value = 15
percentage_difference = 50.0
tolerance = 0.00001
direction = 'positive'

XTAL_IDs, values = FIDs_near_propvalue(property_value_list,
                                       property_value,
                                       percentage_difference,
                                       tolerance,
                                       direction
                                       )

fig = plt.figure(figsize = vis_parameters['figsize'], dpi = vis_parameters['dpi'])
ax_obj = plt.axes()
xtals_to_highlight = XTAL_IDs
plot_pxtal(xtal_list_xtal,
           xtals_to_highlight,
           xtal_centroid_list,
           vis_pxtal_controls,
           vis_parameters,
           ax_obj
           )
############################################################################
from gridops import make_grid_pxtal

grid_mul_factor = 5/PXT_GB_L0['l.MIN']
distribution_type = 'rectgrid'
distribution_subtype = 'uniform'
domain_size_count = 50

x, y, grid_spacing_x = make_grid_pxtal(distribution_type,
                                       method_bounds = 'frombounds',
                                       usefactor = True,
                                       xlimits = xlimits,
                                       ylimits = ylimits,
                                       grid_mul_factor = 5
                                       )
############################################################################
############################################################################
# OVERLAY GRID ON THE PXTAL
figsize = vis_parameters['figsize']
dpi     = vis_parameters['dpi']
from vis_grain_structure_vtgs import plot_pxtal
from vis_grain_structure_vtgs import plot_grid_overlay

fig = plt.figure(figsize = figsize, dpi = dpi)
ax_obj = plt.axes()

plot_pxtal(xtal_list_xtal, xtal_centroid_list, vis_pxtal_controls, vis_parameters, ax_obj)
plot_grid_overlay(x, y, ax_obj)
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
GRAINS = []
GRAIN_BOUNDARY_ZONES = []
GRAIN_CORES = []
############################################################################
fig = plt.figure(dpi = 100)
for count in range(len(pxtal.geoms)):
    grain = pxtal.geoms[count]
    plt.fill(grain.boundary.xy[0], grain.boundary.xy[1],
             facecolor = 'white', edgecolor = 'black', alpha = 1.0, linewidth = 3)
############################################################################
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
############################################################################