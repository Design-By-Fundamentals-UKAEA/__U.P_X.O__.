from mulpoint2d_3 import mulpoint2d
from polyxtal import polyxtal2d as polyxtal
#-----------------------------------------------
xbound = [-1, 1]
ybound = [-1, 1]
#-----------------------------------------------
m = mulpoint2d(method = 'random',
               gridding_technique = 'random',
               sampling_technique = 'uniform',
               nrndpnts = 25,
               randuni_calc = 'by_points',
               space = 'linear',
               xbound = xbound,
               ybound = ybound,
               lean = 'veryhigh',
               make_point_objects = True,
               make_ckdtree = True,
               vis = False
               )
#-----------------------------------------------
pxtal = polyxtal(gsgen_method = 'vt',
                 vt_base_tool = 'shapely',
                 point_method = 'mulpoints',
                 mulpoint_object = m,
                 xbound = xbound,
                 ybound = ybound,
                 vis_vtgs = True
                 )
#-----------------------------------------------
m.plot()
pxtal.plot(dpi = 100,
           default_par_faces = {'clr': 'teal', 'alpha': 0.7, },
           default_par_lines = {'width': 1.5, 'clr': 'black', },
           xtal_marker_vertex = True,
           )
#-----------------------------------------------
# SCALAR FIELD ASSIGNMENT: GRAIN AREA
for i in range(pxtal.L0.xtals_n):
    pxtal.L0.mpo_xtals_reppoints.points[i].polygonal_area = pxtal.L0.xtals[i].area
#-----------------------------------------------
# SCALAR FIELD ASSIGNMENT: GRAIN PERIMETER
for i in range(pxtal.L0.xtals_n):
    pxtal.L0.mpo_xtals_reppoints.points[i].length = pxtal.L0.xtals[i].length

for i in range(pxtal.L0.xtals_n):
    print(pxtal.L0.xtals[i].length)
#-----------------------------------------------
# SCALAR FIELD ASSIGNMENT: MEAN GRAIN PERIMETER
edge_lengths = []
el_mean = []
el_std = []
for i in range(pxtal.L0.xtals_n):
    xy = pxtal.L0.xtals[i].boundary.xy
    x = np.array(list(xy[0][:-1]))
    y = np.array(list(xy[1][:-1]))
    x_ = np.roll(x, 1, axis = 0)
    y_ = np.roll(y, 1, axis = 0)
    edge_lengths.append(np.sqrt((x-x_)**2+(y-y_)**2))
    el_mean.append(edge_lengths[i].mean())
    el_std.append(edge_lengths[i].std())
#-----------------------------------------------
# SCALAR FIELD ASSIGNMENT: NUMBER OF VERTICES
#-----------------------------------------------
