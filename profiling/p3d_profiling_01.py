# =============================================================================
#                     make_partition = False,
#                     partition_n = 4,
#                     partition_char_lengths = [1.0],
#                     partition_char_length_type = 'radius',
#                     partition_make_polygon = True,
#                     partition_tool = 'mpl',
#                     partition_rotation = 0,
#                     feed_partition = False,
#                     feed_partition_name = 'grain',
#                     feed_partition_tool = 'shapely',
#                     feed_partition_polygon = None
# =============================================================================

import timeit
import functools
import numpy as np
from point2d_04 import point3d
from point2d_04 import point3d_lean_highest
from point2d_04 import point3d_lean_highest_mc0
from point2d_04 import point3d_lean_highest_mc1
'''
OC: OBJECT CREATION
AA: Attribute Access
'''
def p2d_OC_profile_speed_case_00_A():
    for _ in range(0, 1_000_000):
        p = point3d(x = 0.0,
                    y = 0.0,
                    z = 0.0,
                    lean = 'low',
                    set_mid = False,
                    set_dim = False, dim = 3,
                    set_ptype = False, ptype = 'vt2dseed',
                    set_jn = False, jn = 3,
                    set_loc = False, loc = 'internal',
                    store_original_coord = False,
                    set_phase = False, phid = 1, phname = 'ukaea',
                    set_polygonal_area = False, polygonal_area = 0,
                    set_tcname = False, tcname = 'B',
                    set_ea = False, earepr = 'Bunge', ea = [45, 35, 0],
                    set_oo = False, oo = False,
                    set_tdist = False, tdist = 0.0000000000001,
                    store_vis_prop = False,
                    )
#//////////////////////////////////////////////////////////////////////////////
if __name__ == '__main__':
    time_taken = []
    repeats = 1
    for _ in range(0, 5):
        #-----------------------
        t = timeit.Timer(functools.partial(p2d_OC_profile_speed_case_00_A, ))
        time_taken.append(t.timeit(repeats)/repeats)
        print(_)
        #-----------------------
    print(np.array(time_taken).mean())
#//////////////////////////////////////////////////////////////////////////////
'''
NO.  CASE                            #INSTANCES   #TIMES (avg. of 5)
1.   point2d_profiling_OC_case_0     1 Million    3.27
1.   point2d_profiling_OC_case_0_    1 Million    2.99
1.   point2d_profiling_OC_case_0__   1 Million    3.49
1.   point2d_profiling_OC_case_1     10 thousand  6.16
1.   point2d_profiling_OC_case_1_    1 Million    15.91
1.   point2d_profiling_OC_case_2     1 Million    7.14
1.   point2d_profiling_OC_case_3     1 Million    6.18
1.   point2d_profiling_OC_case_20    1 Million    0.28
1.   point2d_profiling_OC_case_21    1 Million    0.31
1.   point2d_profiling_OC_case_22    1 Million    0.36
1.   point2d_profiling_OC_case_23    1 Million    0.41

COMMENTS:
    1. SLOWEST: point2d_profiling_OC_case_1
        5 TO 6 SECONDS FOR JUST 10,000 INSTANCES
        This is because, this is calling MATPLOTLIB polygon creation routines to make partition
        # TODO: instead of matplorlib routines, use ones native to UPXO instead
'''
#//////////////////////////////////////////////////////////////////////////////