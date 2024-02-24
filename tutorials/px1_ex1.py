from get_mp_1 import get_mp
from polyxtal import polyxtal2d as polyxtal
#-----------------------------------------------
xbound, ybound = [-1, 1], [-1, 1]
#-----------------------------------------------
'''
values @METHOD
    1. 1: random > random > uniform
    2. 2: random > pds > bridson
    3. 3: random > random > dart
    4. 4: recgrid > linear
    5. 5: trigrid1 > linear
'''
m = get_mp(METHOD = 1, xbound = xbound, ybound = ybound)
#-----------------------------------------------
pxtal = polyxtal(gsgen_method = 'vt',
                 vt_base_tool = 'scipy',
                 point_method = 'mulpoints',
                 mulpoint_object = m,
                 xbound = xbound,
                 ybound = ybound,
                 vis_vtgs = False
                 )
pxtal.plot()
pxtal.L0.xtal_coord_centroid_x
#-----------------------------------------------
pxtal.identify_L0_xtals_boundary(domain_shape = 'rectangular',
                                 base_data_structure_to_use = 'shapely',
                                 build_scalar_fields = True,
                                 scalar_field_names = ['bx_ape'],
                                 viz = False,
                                 vis_dpi = 200,
                                 throw = False
                                 )
#-----------------------------------------------
pxtal.identify_L0_xtals_internal(domain_shape = 'rectangular',
                                 base_data_structure_to_use = 'shapely',
                                 build_scalar_fields = True,
                                 scalar_field_names = ['bx_ape'],
                                 viz = False,
                                 vis_dpi = 200,
                                 throw = False
                                 )
#-----------------------------------------------
