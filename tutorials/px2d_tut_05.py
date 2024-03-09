from mulpoint2d_3 import mulpoint2d
from polyxtal import polyxtal2d as polyxtal
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
