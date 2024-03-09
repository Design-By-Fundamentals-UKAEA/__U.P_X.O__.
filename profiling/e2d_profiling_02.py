import timeit
import functools
import numpy as np
from point2d_04 import point2d
from edge2d_05 import edge2d
from edge2d_05 import edge2d_lean_highest
from edge2d_05 import edge2d_from_point2d
'''
OC: OBJECT CREATION
AA: Attribute Access
'''
#------------------------------------------------------------------------------
def e2d_OC_profile_speed_case_00_A(p1, p2):
    '''
    1. Create "n" number of edges, each from two point objects
    2. Point object creation is not timed
    '''
    n = 1_000_000
    for _ in range(0, n):
        e = edge2d(method = 'points',
                   edge_lean = 'lowest',
                   pointa = p1,
                   pointb = p2,
                   points_lean = 'lowest')
#------------------------------------------------------------------------------
def e2d_OC_profile_speed_case_00_B(p1, p2):
    '''
    1. Create "n" number of edges, each from two point objects
    2. Point object creation is not timed
    3. Store each ovject in a list.
    4. Timing includes list creatioin and updating too
    ADVICE: USE LIST COMPREHENSION INSTEAD OF LIST APPEND
    '''
    n = 1_000_000
    e = [edge2d(method = 'points',
                end_points = [p1, p2],
                points_lean = 'lowest'
                ) for _ in range(0, n)]
#------------------------------------------------------------------------------
def e2d_OC_profile_speed_case_01_A(p1, p2):
    '''
    1. Create "n" number of edges, each from two point objects
    2. Point object creation is not timed
    3. Store each ovject in a list.
    4. Timing includes list creatioin and updating too
    ADVICE: USE LIST COMPREHENSION INSTEAD OF LIST APPEND
    '''
    n = 1_000_000
    e = [edge2d(method = 'points',
                end_points = [p1, p2],
                point_lean = 'lowest',
                make_xy = False,
                calc_centre = False,
                calc_length = False,
                set_loc_xtal = False,
                set_loc_pxtal = False,
                ) for _ in range(0, n)]
#------------------------------------------------------------------------------
def e2d_OC_profile_speed_case_20_A(p1, p2):
    '''
    1. Create "n" number of edges, each from two point objects
    2. Point object creation is not timed
    3. Store each ovject in a list.
    '''
    n = 1_000_000
    for _ in range(0, n):
        e = edge2d_lean_highest(start = [0.0, 0.0], end = [1.0, 1.0])
#------------------------------------------------------------------------------
def e2d_OC_profile_speed_case_20_B(p1, p2):
    '''
    1. Create "n" number of edges, each from two point objects
    2. Point object creation is not timed
    3. Store each ovject in a list.
    4. Timing includes list creatioin and updating too
    ADVICE: USE LIST COMPREHENSION INSTEAD OF LIST APPEND
    '''
    n = 1_000_000
    e = [edge2d_lean_highest(start = [0.0, 0.0], end = [1.0, 1.0]) for _ in range(0, n)]
#------------------------------------------------------------------------------
def e2d_OC_profile_speed_case_30_A(p1, p2):
    '''
    1. Create "n" number of edges, each from two point objects
    2. Point object creation is not timed
    3. Store each ovject in a dictionary against key being the edge object id.
    4. Timing includes list creatioin and updating too
    '''
    n = 1_000_000
    e = {}
    for _ in range(0, n):
        edge = edge2d_lean_highest(start = [0.0, 0.0], end = [1.0, 1.0])
        e[id(edge)] = edge
#------------------------------------------------------------------------------
def e2d_OC_profile_speed_case_40_A(p1, p2):
    '''
    1. Create "n" number of edges, each from two point objects
    2. Point object creation is not timed
    3. Store each ovject in a dictionary against key being the edge object id.
    4. Timing includes list creatioin and updating too
    '''
    n = 1_000_000
    e = [edge2d_from_point2d(p1, p2) for _ in range(0, n)]
#//////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////
if __name__ == '__main__':
    time_taken = []
    repeats = 1
    #-----------------------
    p1 = point2d(x = 0, y = 0)
    p2 = point2d(x = 1, y = 1)
    #-----------------------
    for _ in range(0, 5):
        t = timeit.Timer(functools.partial(e2d_OC_profile_speed_case_00_A, p1, p2))
        time_taken.append(t.timeit(repeats)/repeats)
        print(_)
        #-----------------------
    print(np.array(time_taken).mean())