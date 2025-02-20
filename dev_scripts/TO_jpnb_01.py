# Lets begin by importing the multi-point module
from upxo.geoEntities.mulpoint2d import mulpoint2d
# -----------------------------------------------
m = mulpoint2d(method = 'random',
               gridding_technique = 'random',
               sampling_technique = 'uniform',
               nrndpnts = 100,
               randuni_calc = 'by_points',
               space = 'linear',
               xbound = [-1, 1],
               ybound = [-1, 1],
               lean = 'veryhigh',
               make_point_objects = True,
               make_ckdtree = True,
               vis = False
               )
# -----------------------------------------------
from upxo.pxtal.polyxtal import vtpolyxtal2d as polyxtal
pxtal = polyxtal(gsgen_method = 'vt',
                 vt_base_tool = 'shapely',
                 point_method = 'mulpoints',
                 mulpoint_object = m,
                 xbound = [-1, 1],
                 ybound = [-1, 1],
                 vis_vtgs = True
                 )
# -----------------------------------------------
m.plot()
pxtal.plot(dpi = 100,
           default_par_faces = {'clr': 'teal', 'alpha': 0.7, },
           default_par_lines = {'width': 1.5, 'clr': 'black', },
           xtal_marker_vertex = True,
           )
# -----------------------------------------------
pxtal.find_neighs()
pxtal.get_ids_L0_all_xtals()
pxtal.plot_neigh_1()
dir(pxtal.L0)
