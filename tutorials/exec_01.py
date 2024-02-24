from mulpoint2d_3 import mulpoint2d
#-----------------------------------------------
xbound = [-1, 1]
ybound = [-1, 1]
#-----------------------------------------------
m = mulpoint2d(method = 'random',
               gridding_technique = 'pds',
               xbound = xbound,
               ybound = ybound,
               char_length = [0.2, 0.10],
               bridson_sampling_radius = 0.01,
               bridson_sampling_k = 30,
               lean = 'low'
               )
m.plot(dpi = 50)
#-----------------------------------------------
from polyxtal import polyxtal
pxtal = polyxtal(gsgen_method = 'vt',
                 vt_base_tool = 'shapely',
                 point_method = 'mulpoints',
                 mulpoint_object = m,
                 xbound = xbound,
                 ybound = ybound,
                 vis_vtgs = False
                 )
pxtal.plot(dpi = 100, )
#-----------------------------------------------
pxtal.identify_L0_xtals_boundary(domain_shape = 'rectangular',
                                 base_data_structure_to_use = 'shapely',
                                 build_scalar_fields = True,
                                 scalar_field_names = ['bx_ape'],
                                 viz = True,
                                 vis_dpi = 75,
                                 throw = False
                                 )
# pxtal.L0.xtal_ss_boundary
#-----------------------------------------------
pxtal.identify_L0_xtals_internal(domain_shape = 'rectangular',
                                 base_data_structure_to_use = 'shapely',
                                 build_scalar_fields = True,
                                 scalar_field_names = ['bx_ape'],
                                 viz = True,
                                 vis_dpi = 75,
                                 throw = False
                                 )

# pxtal.L0.xtal_ss_internal
#-----------------------------------------------
neigh_0, neigh_1, neigh_1_, neigh_2, neigh_2_ = pxtal.get_neigh_xtal_of_xtals(rebuild_neigh_database = True,
                                                                              central_grain_ids = [0],
                                                                              n_near_neighbours = 2
                                                                              )

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
                                                                              central_grain_ids = [0],
                                                                              n_near_neighbours = 2
                                                                              )

pxtal.plot(renderer = 'matplotlib',
           database = 'shapely',
           figsize = [3.5, 3.5],
           dpi = 100,
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
# =============================================================================
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
               highlight_specific_grains_hatch = False
               )



pxtal.write_abapy_input_coords(identification_point_type = 'reppoint')

neigh_0, neigh_1, neigh_1_, neigh_2, neigh_2_ = pxtal.get_neigh_xtal_of_xtals(rebuild_neigh_database = False,
                                                                  method = 'from_grain_list',
                                                                  query_grain_id_method = 'fromid',
                                                                  central_grain_ids = xtal_ids[2],
                                                                  n_near_neighbours = 2
                                                                  )

pxtal.plot(renderer = 'matplotlib',
           database = 'shapely',
           figsize = [3.5, 3.5],
           dpi = 100,
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