# =============================================================================
# #===============================================
# from point2d_04 import point2d
# p = point2d(x = 1.0,
#             y = 1.0,
#             lean = 'low',
#             set_rid = True, rid_length = 4,
#             set_mid = True,
#             set_dim = True, dim = 2,
#             set_ptype = True, ptype = 'vt2dseed',
#             set_jn = True, jn = 3,
#             set_loc = True, loc = 'internal',
#             store_original_coord = True,
#             set_phase = True, phid = 1, phname = 'ukaea',
#             set_xtal_area=True, xtal_area = 0,
#             set_tcname = True, tcname = 'B',
#             set_ea = True, earepr = 'Bunge', ea = [45, 35, 0],
#             set_oo = True, oo = False,
#             set_tdist = True, tdist = 0.0000000000001,
#             store_vis_prop = True,
#             make_partition = True,
#             partition_n = 4,
#             partition_char_lengths = [1.0],
#             partition_char_length_type = 'radius',
#             partition_make_polygon = True,
#             partition_tool = 'mpl',
#             partition_rotation = 0,
#             feed_partition = False,
#             feed_partition_name = 'grain',
#             feed_partition_tool = 'shapely',
#             feed_partition_polygon = None
#             )
# #===============================================
# p.make_partition(restart = False,
#                  saa = True,
#                  n = 10,
#                  char_lengths = [1.0],
#                  char_length_type = 'radius',
#                  make_polygon = True,
#                  tool = 'shapely',
#                  rotation = 50,
#                  feed_partition = False,
#                  feed_partition_name = 'grain',
#                  feed_partition_tool = 'shapely',
#                  feed_partition_polygon = None
#                  )
# p.partition.ppos
# #--------------------
# p._print_summary()
# #--------------------
# p.plot()
# p.plot(vprop = p.set_vis_prop(mtype = 'o', mew = 2.0, mec = 'r',
#                                 msz = 20, mfill = 'cyan', malpha = 0.8,
#                                 bfill = 'teal')
#        )
# =============================================================================
#===============================================
import timeit, functools
import numpy as np
from point2d_04 import point2d
#--------------------
def point2d_profiling_case_0():
    n = 1_000
    _x = np.random.random(n)
    _y = np.random.random(n)
    p = [point2d(x = 1.0,
                y = 1.0,
                lean = 'low',
                set_rid = False, rid_length = 4,
                set_mid = False,
                set_dim = False, dim = 2,
                set_ptype = False, ptype = 'vt2dseed',
                set_jn = False, jn = 3,
                set_loc = False, loc = 'internal',
                store_original_coord = False,
                set_phase = False, phid = 1, phname = 'ukaea',
                set_xtal_area=False, xtal_area = 0,
                set_tcname = False, tcname = 'B',
                set_ea = False, earepr = 'Bunge', ea = [45, 35, 0],
                set_oo = False, oo = False,
                set_tdist = False, tdist = 0.0000000000001,
                store_vis_prop = False,
                make_partition = False,
                partition_n = 4,
                partition_char_lengths = [1.0],
                partition_char_length_type = 'radius',
                partition_make_polygon = True,
                partition_tool = 'mpl',
                partition_rotation = 0,
                feed_partition = False,
                feed_partition_name = 'grain',
                feed_partition_tool = 'shapely',
                feed_partition_polygon = None
                ) for __x, __y in zip(_x, _y)
          ]
    return p
t = timeit.Timer(functools.partial(point2d_profiling_case_0, ))
repeats = 1
print(t.timeit(repeats)/repeats)
#//////////////////////////////////////////////////////////////////////////////
def point2d_profiling_case_1():
    n = 1_000_000
    _x = np.random.random(n)
    _y = np.random.random(n)
    p = [point2d(x = 1.0,
                y = 1.0,
                lean = 'low',
                set_rid = True, rid_length = 4,
                set_mid = True,
                set_dim = True, dim = 2,
                set_ptype = True, ptype = 'vt2dseed',
                set_jn = True, jn = 3,
                set_loc = True, loc = 'internal',
                store_original_coord = True,
                set_phase = True, phid = 1, phname = 'ukaea',
                set_xtal_area=True, xtal_area = 0,
                set_tcname = True, tcname = 'B',
                set_ea = True, earepr = 'Bunge', ea = [45, 35, 0],
                set_oo = True, oo = False,
                set_tdist = True, tdist = 0.0000000000001,
                store_vis_prop = True,
                make_partition = False,
                partition_n = 4,
                partition_char_lengths = [1.0],
                partition_char_length_type = 'radius',
                partition_make_polygon = True,
                partition_tool = 'mpl',
                partition_rotation = 0,
                feed_partition = False,
                feed_partition_name = 'grain',
                feed_partition_tool = 'shapely',
                feed_partition_polygon = None
                ) for __x, __y in zip(_x, _y)
          ]
    return p
t = timeit.Timer(functools.partial(point2d_profiling_case_1, ))
repeats = 10
print(t.timeit(repeats)/repeats)
#//////////////////////////////////////////////////////////////////////////////
def point2d_profiling_case_2():
    n = 1_000_000
    _x = np.random.random(n)
    _y = np.random.random(n)
    p = [point2d(x = 1.0,
                y = 1.0,
                lean = 'low',
                set_rid = False, rid_length = 4,
                set_mid = False,
                set_dim = True, dim = 2,
                set_ptype = True, ptype = 'vt2dseed',
                set_jn = True, jn = 3,
                set_loc = True, loc = 'internal',
                store_original_coord = True,
                set_phase = True, phid = 1, phname = 'ukaea',
                set_xtal_area=True, xtal_area = 0,
                set_tcname = True, tcname = 'B',
                set_ea = True, earepr = 'Bunge', ea = [45, 35, 0],
                set_oo = True, oo = False,
                set_tdist = True, tdist = 0.0000000000001,
                store_vis_prop = True,
                make_partition = False,
                partition_n = 4,
                partition_char_lengths = [1.0],
                partition_char_length_type = 'radius',
                partition_make_polygon = True,
                partition_tool = 'mpl',
                partition_rotation = 0,
                feed_partition = False,
                feed_partition_name = 'grain',
                feed_partition_tool = 'shapely',
                feed_partition_polygon = None
                ) for __x, __y in zip(_x, _y)
          ]
    return p
t = timeit.Timer(functools.partial(point2d_profiling_case_2, ))
repeats = 10
print(t.timeit(repeats)/repeats)
#//////////////////////////////////////////////////////////////////////////////
def point2d_profiling_case_3():
    n = 1_000_000
    _x = np.random.random(n)
    _y = np.random.random(n)
    p = [point2d(x = 1.0,
                y = 1.0,
                lean = 'low',
                set_rid = False, rid_length = 4,
                set_mid = False,
                set_dim = True, dim = 2,
                set_ptype = True, ptype = 'vt2dseed',
                set_jn = True, jn = 3,
                set_loc = True, loc = 'internal',
                store_original_coord = True,
                set_phase = True, phid = 1, phname = 'ukaea',
                set_xtal_area=True, xtal_area = 0,
                set_tcname = True, tcname = 'B',
                set_ea = True, earepr = 'Bunge', ea = [45, 35, 0],
                set_oo = True, oo = False,
                set_tdist = True, tdist = 0.0000000000001,
                store_vis_prop = False,
                make_partition = False,
                partition_n = 4,
                partition_char_lengths = [1.0],
                partition_char_length_type = 'radius',
                partition_make_polygon = True,
                partition_tool = 'mpl',
                partition_rotation = 0,
                feed_partition = False,
                feed_partition_name = 'grain',
                feed_partition_tool = 'shapely',
                feed_partition_polygon = None
                ) for __x, __y in zip(_x, _y)
          ]
    return p
t = timeit.Timer(functools.partial(point2d_profiling_case_3, ))
repeats = 10
print(t.timeit(repeats)/repeats)
#//////////////////////////////////////////////////////////////////////////////
def point2d_profiling_case_4():
    n = 1_000_000
    _x = np.random.random(n)
    _y = np.random.random(n)
    p = [point2d(x = 1.0,
                y = 1.0,
                lean = 'low',
                set_rid = False, rid_length = 4,
                set_mid = False,
                set_dim = False, dim = 2,
                set_ptype = False, ptype = 'vt2dseed',
                set_jn = False, jn = 3,
                set_loc = False, loc = 'internal',
                store_original_coord = False,
                set_phase = False, phid = 1, phname = 'ukaea',
                set_xtal_area = False, xtal_area = 0,
                set_tcname = False, tcname = 'B',
                set_ea = False, earepr = 'Bunge', ea = [45, 35, 0],
                set_oo = False, oo = False,
                set_tdist = False, tdist = 0.0000000000001,
                store_vis_prop = False,
                make_partition = False,
                partition_n = 4,
                partition_char_lengths = [1.0],
                partition_char_length_type = 'radius',
                partition_make_polygon = False,
                partition_tool = 'mpl',
                partition_rotation = 0,
                feed_partition = False,
                feed_partition_name = 'grain',
                feed_partition_tool = 'shapely',
                feed_partition_polygon = None
                ) for __x, __y in zip(_x, _y)
          ]
    return p
t = timeit.Timer(functools.partial(point2d_profiling_case_4, ))
repeats = 10
print(t.timeit(repeats)/repeats)
#//////////////////////////////////////////////////////////////////////////////
def point2d_profiling_case_5():
    n = 1_000_000
    _x = np.random.random(n)
    _y = np.random.random(n)
    p = [point2d(x = 1.0,
                y = 1.0,
                lean = 'highest',
                set_rid = False, rid_length = 4,
                set_mid = False,
                set_dim = False, dim = 2,
                set_ptype = False, ptype = 'vt2dseed',
                set_jn = False, jn = 3,
                set_loc = False, loc = 'internal',
                store_original_coord = False,
                set_phase = False, phid = 1, phname = 'ukaea',
                set_xtal_area=False, xtal_area = 0,
                set_tcname = False, tcname = 'B',
                set_ea = False, earepr = 'Bunge', ea = [45, 35, 0],
                set_oo = False, oo = False,
                set_tdist = False, tdist = 0.0000000000001,
                store_vis_prop = False,
                make_partition = False,
                partition_n = 4,
                partition_char_lengths = [1.0],
                partition_char_length_type = 'radius',
                partition_make_polygon = False,
                partition_tool = 'mpl',
                partition_rotation = 0,
                feed_partition = False,
                feed_partition_name = 'grain',
                feed_partition_tool = 'shapely',
                feed_partition_polygon = None
                ) for __x, __y in zip(_x, _y)
          ]
    return p
t = timeit.Timer(functools.partial(point2d_profiling_case_5, ))
repeats = 10
print(t.timeit(repeats)/repeats)
#//////////////////////////////////////////////////////////////////////////////
#p = point2d_profiling()
#--------------------
#gen_exp = False
#if gen_exp:
#    x, y, p3 = point2d_profiling()
#    for i in range(10):
#        point_ = next(p3)
#        print(x[i], y[i], point_, point_.phase_id, point_.phase_name)
#        #print(x[i], y[i], point_, point_.rid, point_.dim, point_.ptype, point_.jn,
#        #      point_.loc)
#--------------------
# t = timeit.Timer(functools.partial(point2d_profiling_, ))




#print(t.timeit(repeats)/repeats)
#===============================================
#import cProfile
#cProfile.run("point2d_profiling()")
#===============================================