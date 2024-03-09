from mulpoint2d import mulpoint2d
from polyxtal import polyxtal
import pandas as pd
import numpy as np
#-----------------------------------------------
xbound = [-1, 1]
ybound = [-1, 1]
    # Iteratively adjusting the number of seeds
    # RANDOM >> BRIDSON SAMPLING
target_npoints_seeds = 400
target_npoints_seeds_tolerance = 10

target_area_mean = 0.01
target_area_mean_tolerance = 0.1*target_area_mean

CL = 0.20
CL_del = 0.005
max_iterations = 100
variable_list = [CL]
iteration = 0
m = mulpoint2d(method = 'random',
               gridding_technique = 'pds',
               sampling_technique = 'bridson1',
               space = 'linear',
               xbound = xbound,
               ybound = ybound,
               char_length = [CL, 0.10],
               lean = 'high'
               )
n = m.npoints
m.plot()

pxtal = polyxtal(gsgen_method = 'vt',
                 vt_base_tool = 'shapely',
                 point_method = 'mulpoints',
                 mulpoint_object = m,
                 xbound = xbound,
                 ybound = ybound,
                 vis_vtgs = False
                 )

pxtal.plot(dpi = 200,
           default_par_faces = {'clr': 'teal', 'alpha': 0.7, },
           default_par_lines = {'width': 1.5, 'clr': 'black', },
           xtal_marker_vertex = False,
           )

areas = [np.array([_xtal.area for _xtal in pxtal.L0.xtals])]
len(areas[0])
len(pxtal.L0.mpo_xtals_reppoints.points)

for i in range(pxtal.L0.xtals_n):
    pxtal.L0.mpo_xtals_reppoints.points[0].polygonal_area

hits = [n]
not_within_tolerance = True
while not_within_tolerance:
    print(f'Iteration: {iteration}')
    #if target_npoints_seeds - n <= target_npoints_seeds_tolerance:
    #    CL = CL + CL_del
    #else:
    #    CL = CL - CL_del
    if target_area_mean - areas[iteration].mean() <= target_area_mean_tolerance:
        CL = CL - CL_del
    else:
        CL = CL + CL_del
    m = mulpoint2d(method = 'random',
                   gridding_technique = 'pds',
                   sampling_technique = 'bridson1',
                   space = 'linear',
                   xbound = xbound,
                   ybound = ybound,
                   char_length = [CL, 0.10],
                   lean = 'high'
                   )
    n = m.npoints


    residual = target_area_mean - areas[iteration].mean()
    #iteration += 1
    #residual = target_npoints_seeds - n
    #if abs(residual) <= target_npoints_seeds_tolerance or iteration == max_iterations:
    #    not_within_tolerance = False
    if abs(residual) <= target_area_mean_tolerance or iteration == max_iterations:
        not_within_tolerance = False

    variable_list.append(CL)
    hits.append(n)
    pxtal = polyxtal(gsgen_method = 'vt',
                     vt_base_tool = 'shapely',
                     point_method = 'mulpoints',
                     mulpoint_object = m,
                     xbound = xbound,
                     ybound = ybound,
                     vis_vtgs = False
                     )
    iteration += 1
    areas.append(np.array([_xtal.area for _xtal in pxtal.L0.xtals]))

    if iteration == max_iterations:
        print(f'Maximum iterations of {max_iterations} reached')
    print(f'Residual = {residual}')

areas_mean = [area.mean() for area in areas]
areas_min = [area.min() for area in areas]
areas_max = [area.max() for area in areas]

import matplotlib.pyplot as plt
fig = plt.figure(dpi = 100)
ax1 = plt.axes()
ax2 = ax1.twiny()
ax1.plot(variable_list, hits, '-ko')
ax1.set_xlabel('characteristic length')
ax1.set_ylabel('Number of seed points')
ax2.set_xlim(ax1.get_xlim())
ax2.set_xlabel('Number of iterations')
ax2.set_xticks(list(range(0, len(hits), 5)))
plt.show()

fig = plt.figure(dpi = 200)
ax1 = plt.axes()
iteration_list = list(range(len(areas_mean)))
ax1.plot([min(iteration_list), max(iteration_list)],
         [target_area_mean, target_area_mean],
         '--kx',
         linewidth = 2, markersize = 6, markerfacecolor = 'white', label = 'target mean area')
ax1.plot(iteration_list, areas_mean, '-ko', label = 'mean', markerfacecolor = 'white', markersize = 5)
ax1.plot(iteration_list, areas_min, '-g', label = 'min')
ax1.plot(iteration_list, areas_max, '-b', label = 'max')
ax1.set_xlabel('Iteration number')
ax1.set_ylabel('xtal area, unit^2')
ax1.legend()
plt.show()


#target_area_mean = 0.01
#target_area_mean_tolerance = 0.1*target_area_mean

import pandas as pd
a = []
for i, area in enumerate(areas):
    a.append(pd.Series(area, name = i))
area_pdf = pd.concat(a, axis = 1)
import seaborn as sns

fig = plt.figure(dpi = 200)
ax = plt.axes()
for i in range(len(areas)):
    if i == 0:
        color = 'red'
        lw = 3
        alpha = 1.0
        legend = True
    elif i == len(areas)-1:
        color = 'green'
        lw = 3
        alpha = 1.0
        legend = True
    else:
        color = 'gray'
        lw = 1
        alpha = 0.5
        legend = False
    if i == 0 or i == len(areas)-1:
        sns.kdeplot(data = area_pdf,
                    x = i,
                    bw_adjust = 1.0,
                    cut = 0,
                    common_norm = True,
                    color = color,
                    alpha = alpha,
                    linewidth = lw,
                    label = f'Iteration {i}',
                    legend = legend
                    )
    else:
        sns.kdeplot(data = area_pdf,
                    x = i,
                    bw_adjust = 1.0,
                    cut = 0,
                    common_norm = True,
                    color = color,
                    alpha = alpha,
                    linewidth = lw,
                    legend = legend
                    )
ax.set_xlabel('xtal areas')
ax.legend()
plt.show()
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
pxtal.plot(dpi = 200,
           default_par_faces = {'clr': 'teal', 'alpha': 0.7, },
           default_par_lines = {'width': 1.5, 'clr': 'black', },
           xtal_marker_vertex = False,
           )
#print(pxtal.print_summary())
#-----------------------------------------------
pxtal.identify_L0_xtals_boundary(domain_shape = 'rectangular',
                                 base_data_structure_to_use = 'shapely',
                                 build_scalar_fields = True,
                                 scalar_field_names = ['bx_ape'],
                                 viz = True,
                                 vis_dpi = 200,
                                 throw = False
                                 )
# pxtal.L0.xtal_ss_boundary
#-----------------------------------------------
pxtal.identify_L0_xtals_internal(domain_shape = 'rectangular',
                                 base_data_structure_to_use = 'shapely',
                                 build_scalar_fields = True,
                                 scalar_field_names = ['bx_ape'],
                                 viz = True,
                                 vis_dpi = 200,
                                 throw = False
                                 )
# pxtal.L0.xtal_ss_internal
#-----------------------------------------------
neigh_0, neigh_1, neigh_1_, neigh_2, neigh_2_ = pxtal.get_neigh_xtal_of_xtals(rebuild_neigh_database = True,
                                                                              central_grain_ids = [0],
                                                                              n_near_neighbours = 2
                                                                              )
#-----------------------------------------------
pxtal.make_tree_vt_base_seeds(recalculate = True)
pxtal.make_tree_xtals_centroids(recalculate = True)
pxtal.make_tree_xtals_reppoints(recalculate = True)
pxtal.make_tree_xtals_pbjp(recalculate = True, vertex_type = 'L0_xtals_pbjp')

pxtal.check_fractions(par = 'boundary_and_internal')
# =============================================================================
# pxtal.calculate_lengths(level = 0,
#                         length_type = 'xtal.polygonal.pbjp'
#                         )
# #pxtal.L0.xtal_ble_val
# pxtal.calculate_lengths(level = 0,
#                         length_type = 'xtal.polygonal.perimeter'
#                         )
# #pxtal.L0.xtal_pe_val
#
# =============================================================================
pxtal.write_abapy_input_coords(identification_point_type = 'reppoint')
#-----------------------------------------------
# pxtal.plot(xtal_face_annot_count = True)
#-----------------------------------------------
pxtal.plot(highlight_specific_grains = True,
           specific_grains = pxtal.L0.xtal_ss_boundary.ids,
           highlight_specific_grains_annot = False,
           highlight_specific_grains_colour = True,
           highlight_specific_grains_hatch = False
           )

pxtal.plot(highlight_specific_grains = True,
           specific_grains = pxtal.L0.xtal_ss_internal.ids,
           highlight_specific_grains_annot = False,
           highlight_specific_grains_colour = True,
           highlight_specific_grains_hatch = False
           )
#-----------------------------------------------
pxtal.build_scalar_fields_from_xtal_list(scalar_field = 'bx_ape',
                                         xtal_list = pxtal.L0.xtal_ss_boundary.ids,
                                         save_to_attribute = True,
                                         throw = False
                                         )
len(pxtal.L0.xtal_ss_boundary.ape_val)
pxtal.build_scalar_fields_from_xtal_list(scalar_field = 'ix_ape',
                                         xtal_list = pxtal.L0.xtal_ss_internal.ids,
                                         save_to_attribute = True,
                                         throw = False
                                         )
len(pxtal.L0.xtal_ss_internal.ape_val) # <--------------
#-----------------------------------------------
pxtal.plot(highlight_specific_grains = True,
           specific_grains = [0, 5, 10],
           highlight_specific_grains_annot = False,
           highlight_specific_grains_colour = True,
           highlight_specific_grains_hatch = False
           )
#-----------------------------------------------
pxtal.make_tree_xtals_reppoints(recalculate = True)
pxtal.make_tree_xtals_centroids(recalculate = True)
pxtal.make_tree_xtals_pbjp(recalculate = True,
                           vertex_type = 'L0_xtals_pbjp',
                           )
#-----------------------------------------------
vertx, verty, _ = pxtal.extract_shapely_coords(shapely_grains_list = None, coord_of = 'L0_xtals_reppoints')
vertx, verty, _ = pxtal.extract_shapely_coords(shapely_grains_list = None, coord_of = 'L0_xtals_centroids')
vertx, verty, vertxy = pxtal.extract_shapely_coords(shapely_grains_list = None, coord_of = 'L0_xtal_vertices_pbjp')
#-----------------------------------------------
# pxtal.find_neighs()
# neigh_0, neigh_1, neigh_1_ = pxt.get_neigh_xtal_of_xtals(central_grain_ids = [0, 5], n_near_neighbours = 1)
neigh_0, neigh_1, neigh_1_, neigh_2, neigh_2_ = pxtal.get_neigh_xtal_of_xtals(rebuild_neigh_database = True,
                                                                              central_grain_ids = [13, 35, 57, 91],
                                                                              n_near_neighbours = 2
                                                                              )

pxtal.plot(renderer = 'matplotlib',
           database = 'shapely',
           figsize = [3.5, 3.5],
           dpi = 200,
           level = 0,
           xtal_clr_field = False,
           field_variable = 'areas_polygonal_exterior',
           xtal_face_annot_count = False,
           xtal_marker_centroid = False,
           xtal_marker_reppoint = False,
           xtal_marker_vertex = False,
           plot_neigh_0 = True,
           plot_neigh_1 = True,
           plot_neigh_2 = True,
           colour_neigh_0 = True,
           colour_neigh_1 = True,
           colour_neigh_2 = True,
           xtal_neigh_0_annot_count = False,
           xtal_neigh_1_annot_count = False,
           xtal_neigh_2_annot_count = False,
           neigh_0 = neigh_0,
           neigh_1_ = neigh_1_,
           neigh_2_ = neigh_2_,
           )
from distr_01 import distribution as dstr
import numpy as np
# STANDARD NAME in UPXO: Data for stats
DFS = {'areas_polygonal_exterior': np.array(pxtal.L0.xtals_ids),
       'areas_polygonal_exterior__boundary_xtals': np.array(pxtal.L0.xtal_ss_boundary.ids)
       }

distributions = {_key: dstr(data_name = _key,
                            data = _data_source,
                            ) for (_key, _data_source) in zip(DFS.keys(),
                                                              DFS.values()
                                                              )
                 }

#[_key for _key in DFS.keys()]
#[_val for _val in DFS.values()]

distributions['areas_polygonal_exterior'].H.be
distributions['areas_polygonal_exterior'].H.hv
distributions['areas_polygonal_exterior'].H.data
distributions['areas_polygonal_exterior'].plot_histogram(be_estimator = 'auto')

field_values, xtal_ids = pxtal.identify_grains_from_field_threshold(field_name = 'areas_polygonal_exterior',
                                                                    threshold_definition = 'percentiles',
                                                                    threshold_limits_values = [[0.0, 0.05], [0.05, 0.10], [0.10, 0.15]],
                                                                    threshold_limits_percentiles = [[0, 10], [10, 90], [90, 100]],
                                                                    inequality_definitions = [['>=', '<='], ['>=', '<='], ['>=', '<=']],
                                                                    exclude_grains = None,
                                                                    save_as_attribute = True,
                                                                    throw = True
                                                                    )

for xtal_ids_ in xtal_ids:
    pxtal.plot(highlight_specific_grains = True,
               specific_grains = xtal_ids_,
               highlight_specific_grains_annot = False,
               highlight_specific_grains_colour = True,
               highlight_specific_grains_hatch = False,
               xtal_face_annot_count = False,
               dpi = 200
               )



pxtal.write_abapy_input_coords(identification_point_type = 'reppoint')

neigh_0, neigh_1, neigh_1_, neigh_2, neigh_2_ = pxtal.get_neigh_xtal_of_xtals(rebuild_neigh_database = False,
                                                                  method = 'from_grain_list',
                                                                  query_grain_id_method = 'fromid',
                                                                  central_grain_ids = xtal_ids[0],
                                                                  n_near_neighbours = 2
                                                                  )
pxtal.plot(renderer = 'matplotlib',
           database = 'shapely',
           figsize = [3.5, 3.5],
           dpi = 200,
           level = 0,
           xtal_clr_field = True,
           field_variable = 'areas_polygonal_exterior',
           xtal_face_annot_count = False,
           xtal_marker_centroid = False,
           xtal_marker_reppoint = False,
           xtal_marker_vertex = False,
           plot_neigh_0 = True,
           plot_neigh_1 = True,
           plot_neigh_2 = True,
           colour_neigh_0 = True,
           colour_neigh_1 = True,
           colour_neigh_2 = True,
           xtal_neigh_0_annot_count = False,
           xtal_neigh_1_annot_count = False,
           xtal_neigh_2_annot_count = False,
           neigh_0 = neigh_0,
           neigh_1_ = neigh_1_,
           neigh_2_ = neigh_2_,
           )
#-----------------------------------------------
elshape = 'tri'
elorder = 1
algorithm = 6
#-----------------------------------------------
# Lets determine the appropriate global element size
#gb_length_min = []
#for grain_count in range(len(pxtal.L0.geoms)):
#    gr = pxtal.geoms[grain_count]
#    # Get the individual edges of the grain boundary
#    x_this , y_this  = np.array(gr.boundary.xy[0][:-1]), np.array(gr.boundary.xy[1][:-1])
#    x_front, y_front = np.roll(x_this, +1), np.roll(y_this, +1)
#    gb_length_min.append(min(np.sqrt(np.square(x_this - x_front) + np.square(y_this - y_front))))
#smallest_gb_length = min(gb_length_min)
#-----------------------------------------------
from pxtalmesh_01 import pxtalmesh
pxtal_mesh = pxtalmesh(meshing_tool = 'pygmsh',
                       pxtal = pxtal,
                       level = 0,
                       elshape = elshape,
                       elorder = elorder,
                       algorithm = algorithm,
                       elsize_global = [0.01, 0.02, 0.02],
                       optimize = True,
                       sta = True,
                       wtfs = True,
                       ff = ['vtk', 'inp'],
                       throw = False
                       )
#-----------------------------------------------
filename = 'femesh'
fileformat = 'vtk'
#-----------------------------------------------
mesh_quality_measures = ['aspect_ratio',
                         'skew',
                         'min_angle',
                         'area',
                         ]
#-----------------------------------------------
INTERACTIVE_MESH_QUALITY_ASSESSMENT = True
#-----------------------------------------------
# READ THE MESH FILES
#-----------------------------------------------
import pyvista as pv
#--------------------------
# make filename for storing meshdatain vtk format
mesh_filename = f'{filename}.{fileformat}'
# access gmsh mesh data variable mesh
mesh = pxtal_mesh.mesh[3]
# Write the vtk mesh file
mesh.write(mesh_filename)
#--------------------------
# Load the vtk mesh file
grid = pv.read(mesh_filename)
#--------------------------
# Visualize using pyvista
pxtal_mesh.vis_pyvista(data_to_vis = 'mesh > mesh > all',
                       rfia = True,
                       grid = grid
                       )
#--------------------------
# Extract the t and q elements
tel, qel = pxtal_mesh.get_pygmsh_qt_elements(mesh = mesh)
# Get the number of t and q elements
tel_n, qel_n, allel_n = pxtal_mesh.get_pygmsh_tq_n(method = 'from_el_list',
                                                   tel = tel,
                                                   qel = qel
                                                   )
print(f'Average number of elements per xtal: {int(round(allel_n/pxtal.L0.xtals_n, -1))}')
#--------------------------
# Calculate mesh quality
mqm_data, mqm_dataframe = pxtal_mesh.assess_pygmsh(grid = grid,
                                                   mesh_quality_measures = mesh_quality_measures,
                                                   elshape = elshape,
                                                   elorder = elorder,
                                                   algorithm = algorithm
                                                   )
#--------------------------
# Visualize the mesh quality field parameter
mesh_quality_measures = ['aspect_ratio',
                         'skew',
                         'min_angle',
                         'area']
clims = [[1.0, 2.5],
         [-1.0, 1.0],
         [0.0, 90.0],
         [0.0*mqm_dataframe['area'].max(), mqm_dataframe['area'].max()]
         ]
pxtal_mesh.vis_pyvista(data_to_vis = 'mesh > quality > field',
                       mesh_qual_fields = mqm_data,
                       mesh_qual_field_vis_par = {'mesh_quality_measures': mesh_quality_measures,
                                                  'cpos': 'xy',
                                                  'scalars': 'CellQuality',
                                                  'show_edges': False,
                                                  'cmap': 'viridis',
                                                  'clims': clims,
                                                  'below_color': 'white',
                                                  'above_color': 'black',
                                                  }
                       )
#--------------------------
# KDE- plot of data
band_widths = 4*[0.25]
colors = ['red', 'green', 'blue', 'gray']
pxtal_mesh.vis_kde(data = mqm_dataframe,
                   datatype = 'pandas_df',
                   df_names = mesh_quality_measures,
                   clips = clims,
                   cumulative = False,
                   band_widths = band_widths,
                   colors = colors
                   )
#--------------------------
