import numpy as np
from point2d_04 import point2d
from mulpoint2d_3 import mulpoint2d
#-----------------------------------------------
m = mulpoint2d(method = 'coord',
               coordx = [-1,  1, 1, -1, 0.5, 0.5001],
               coordy = [-1, -1, 1,  1, 0.5, 0.5001],
               )

m.locx
pxtal = polyxtal(gsgen_method = 'vt',
                 vt_base_tool = 'shapely',
                 point_method = 'mulpoints',
                 mulpoint_object = m,
                 xbound = xbound,
                 ybound = ybound,
                 vis_vtgs = False
                 )
pxtal.plot()
#-----------------------------------------------
m.plot(dpi = 100)
#-----------------------------------------------
''' Accessing constituent point objects'''
m.points
#-----------------------------------------------
'''Changing position of a constituent point object'''
m.points[0].align_to(method = 'coord', x = 0.5, y = 0.5)
#-----------------------------------------------
''' Accessing the coordinates of the points '''