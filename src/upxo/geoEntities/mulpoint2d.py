'''
This module hasthe collection of edge2d classes
This is a core module
NOTE: NOT TO BE SHARED WITH ANYONE OTHER THAN:
    *@UKAEA: Vaasu Anandatheertha, Chris Hardie, Vikram Phalke
    *@UKAEA:  Ben Poole, Allan Harte, Cori Hamelin
    *@OX,UKAEA:  Eralp Demir, Ed Tarleton
'''
__name__ = "UPXO >> MULTI-2D.POINT class"
__authors__ = ["Vaasu Anandatheertha"]
__lead_developer__ = ["Vaasu Anandatheertha"]
__emails__ = ["vaasu.anandatheertha@ukaea.uk", ]
__version__ = ["0.1@ upto.271022",
               "0.2@ from.281022",
               "0.3@ from.091122",
               "0.4@ from.211122"]
__license__ = "GPL v3"
# /////////////////////////////////////////////////////////////////////////////
import numpy as np
from collections import deque
from ..geoEntities import point2d
import matplotlib.pyplot as plt
from .._sup import dataTypeHandlers as dth
# /////////////////////////////////////////////////////////////////////////////


class mulpoint2d():
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    __POINT_TDIST__ = 0.001
    __CONST_PI__ = 3.141592653589793238
    ROUND_ZERO_DEC_PLACE = 10
    ε = 0.000000000001
    EPS = 0.000000000001
    EPS_contains = 0.000000000001
    EPS_above = EPS
    EPS_below = EPS
    EPS_left = EPS
    EPS_right = EPS
    EPS_divisor = EPS
    EPS_angle = EPS
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    __slots__ = ('points', 'mpoints', 'mulpoint_type',
                 'rid',
                 'lean',
                 'name',
                 'npoints',
                 'locx', 'locy', 'locxy',
                 'xpert', 'ypert',
                 'xbound', 'ybound',
                 'centroid',
                 'duparray',
                 'coupled_array',
                 '__state_change',
                 'tree',
                 'pdom',
                 'covhull',
                 'reprstate'
                 )

    def __init__(self,
                 mulpoint_type: str = 'seed',
                 method: str = 'points',
                 gridding_technique: str = 'random',
                 sampling_technique: str = 'uniform',
                 nrndpnts: int = 25,
                 randuni_calc: str = 'by_points',
                 char_length_mean: float = 0.10,
                 char_length_min: float = 0.05,
                 char_length_max: float = 0.15,
                 n_trials: int = 10,
                 n_iterations: int = 10,
                 point_objects: list = [],
                 make_point_objects: bool = True,
                 mulpoint_objects: list = None,
                 coordx: list = [],
                 coordy: list = [],
                 coordxy: list = [],
                 space: str = 'linear',
                 xbound: list = [0, 1],
                 ybound: list = [0, 1],
                 char_length: list = [0.25, 0.25],
                 n_char_lengths: list = [10, 10],
                 latvecs: list = [0.1, 0.1],
                 angles: list = [0, 60],
                 bridson_sampling_radius: float = 0.01,
                 bridson_sampling_k: float = 30,
                 perturb_flag: bool = False,
                 perturb_type: str = 'local_uniform',
                 perturb_mag: list = [0.05, 0.05],
                 lean: str = 'ignore',
                 pdom: str = 'grain',
                 make_rid: bool = True,
                 make_ckdtree: bool = True,
                 vis: bool = False,
                 print_summary: bool = True,
                 name: str = 'multi-point2d',
                 reprstate: bool = False
                 ):
        self.lean = lean
        self.name = name
        self.xbound = xbound
        self.ybound = ybound
        self.mulpoint_type = mulpoint_type
        self.mpoints = None
        self.reprstate = reprstate

        if method in dth.opt.upxo_point2d_list:
            '''
            BRANCH: 'up2d_list'

            # PREPARE BASE DATA
            up2d_list = [point2d(x=0, y=0, lean='ignore'),
                         point2d(x=1, y=0, lean='ignore'),
                         point2d(x=0, y=0, lean='ignore'),
                         point2d(x=1, y=1.01, lean='ignore'),
                         point2d(x=2, y=2, lean='ignore'),]

            # MAKE MULTI-POINT OBJECT
            mp = mulpoint2d(method='up2d_list', point_objects=up2d_list)
            print(mp.points,'#',mp.npoints,'#',[mp.locx,mp.locy],'#',mp.centroid)
            '''
            self.points = point_objects
            self.npoints = len(point_objects)
            self.locx, self.locy = zip(*np.array([(pobj.x, pobj.y)
                                                  for pobj in point_objects]))
            # -------------------
            # Following does have some redundant ops. Retain for now.
            self.__state_change = True
            self.recompute_from_mids()
            self.recompute_from_dist(tolerance_distance=0.0)
            self.recompute_basics(npoints_flag=True,
                                  locx_locy_flag=False,
                                  centroid_flag=True)
            self.__state_change = False

        elif method in dth.opt.coord_point2d_list:
            '''
            BRANCH: 'xy_list'

            # PREPARE BASE DATA
            xy_list = [[0, 1, 0, 1, 2], [0, 0, 0, 1.01, 2]]

            # MAKE MULTI-POINT OBJECT
            mp = mulpoint2d(method='xy_list', coordxy=xy_list)
            print(mp.points,'#',mp.npoints,'#',[mp.locx,mp.locy],'#',mp.centroid)
            '''
            self.locx, self.locy = np.array(coordxy[0]), np.array(coordxy[1])
            self.npoints = len(coordx)
            self.points = [point2d(x=_x, y=_y)
                           for (_x, _y) in zip(self.locx, self.locy)]
            # -------------------
            # Following does have some redundant ops. Retain for now.
            self.__state_change = True
            self.recompute_from_mids()
            self.recompute_from_dist(tolerance_distance=0.0)
            self.recompute_basics(npoints_flag=True,
                                  locx_locy_flag=True,
                                  centroid_flag=True)
            self.__state_change = False

        elif method in dth.opt.coord_pairs_point2d_list:
            '''
            BRANCH: 'xy_pair_list'

            # PREPARE BASE DATA
            xy_pair_list = [[0, 0], [1, 0], [0, 0], [1, 1.01], [2, 2]]

            # MAKE MULTI-POINT OBJECT
            mp = mulpoint2d(method='xy_pair_list', coordxy=xy_pair_list)
            print(mp.points,'#',mp.npoints,'#',[mp.locx,mp.locy],'#',mp.centroid)
            '''
            coordxy = np.array(coordxy).T
            self.locx, self.locy = np.array(coordxy[0]), np.array(coordxy[1])
            self.npoints = len(coordx)
            self.points = [point2d(x=_x, y=_y)
                           for (_x, _y) in zip(self.locx, self.locy)]
            # -------------------
            # Following does have some redundant ops. Retain for now.
            self.__state_change = True
            self.recompute_from_mids()
            self.recompute_from_dist(tolerance_distance=0.0)
            self.recompute_basics(npoints_flag=True,
                                  locx_locy_flag=True,
                                  centroid_flag=True)
            self.__state_change = False

        elif method in ('mulpoints', 'mulpoint', 'mp', 'mpoints'):
            '''
            from mulpoint2d import mulpoint2d
            x = [0, 0, 1, 2, 3]
            y = [0, 0, 1, 2, 3]
            m1 = mulpoint2d(method='xlist_ylist', coordx=x, coordy=y)

            m2 = mulpoint2d(method='xy_coord_list', coordxy=[x, y])
            m3 = mulpoint2d(method='xlist_ylist', coordx=[10, 12],
                            coordy=[20, 24])
            mulpoint_objects = [m1, m2, m3]
            m4 = mulpoint2d(method='mp',
                            mulpoint_objects=mulpoint_objects)

            from point2d import point2d
            p1 = point2d(x=0.1, y=0.2)
            p2 = p1 + 1
            p3 = p2 * 0.6498
            p4 = p1*0.468 + p3/p2
            from mulpoint2d import mulpoint2d
            m1 = mulpoint2d(method='points', point_objects=[p1, p2])
            m2 = mulpoint2d(method='points', point_objects=[p3, p4, p1 + p4])
            m4 = mulpoint2d(method='mp', mulpoint_objects=[m1, m2])
            '''
            self.mpoints = mulpoint_objects
            _points = []
            for mpobj in mulpoint_objects:
                for mpobj_point in mpobj.points:
                    _points.append(mpobj_point)
            self.points = _points
            self.npoints = len(self.points)
            self.locx, self.locy = zip(*np.array([(pobj.x, pobj.y)
                                                  for pobj in self.points]))
            # -------------------
            # Following does have some redundant ops. Retain for now.
            self.recompute_from_mids()
            self.recompute_from_dist(tolerance_distance=0.0)
            self.recompute_basics(npoints_flag=True,
                                  locx_locy_flag=False,
                                  centroid_flag=True)

        elif method in ('trigrid1', 'trigrid2', 'recgrid', 'hexgrid',
                        'random'):
            '''
            Generate triangular, rectangular and hexagonal grid.
            '''
            # --------------------------------
            self.makegrid(method=method,
                          gridding_technique=gridding_technique,
                          sampling_technique=sampling_technique,
                          nrndpnts=nrndpnts,
                          randuni_calc=randuni_calc,
                          char_length_mean=char_length_mean,
                          char_length_min=char_length_min,
                          char_length_max=char_length_max,
                          n_trials=n_trials,
                          n_iterations=n_iterations,
                          space=space,
                          xbound=xbound,
                          ybound=ybound,
                          char_length=char_length,
                          angles=angles,
                          bridson_sampling_k=bridson_sampling_k,
                          vis=vis
                          )
            # --------------------------------
            # Make UPXO point objects
            if self.points is None:
                if make_point_objects:
                    self.points = [point2d(x=__x, y=__y,
                                           lean='ignore',
                                           set_mid=True,
                                           set_dim=True, dim=2,
                                           set_ptype=True, ptype='vt2dseed',
                                           set_jn=True, jn=3,
                                           set_loc=True, loc='internal',
                                           store_original_coord=True,
                                           set_phase=True, phase_id=1,
                                           phase_name='ukaea',
                                           set_tcname=True, tcname='B',
                                           set_sfv_ea=True,
                                           sfv_repr_ea='Bunge',
                                           sfv_ea=[45, 35, 0],
                                           set_orientation_object=True,
                                           orientation_object=False,
                                           set_tdist=True,
                                           tdist=mulpoint2d.__POINT_TDIST__,
                                           store_vis_prop=True
                                           ) for __x, __y in zip(self.locx,
                                                                 self.locy)
                                   ]
                    # --------------------------------
            self.recompute_basics(npoints_flag=True,
                                  locx_locy_flag=True,
                                  centroid_flag=True)
        # __state_change: A flag variable to indicate a change in any of
        # the constituent points object. Default should be False.
        # A change should make this True. self.recompute_from_mids() to be
        # implemented when True.
        self.__state_change = False
        # A NOTE ON NEXT LINE:
        # Keep depth = 2 for the moment until the tree method is fully
        # developed and tested
        # WHEN depth = 2, no cleaning operation happens and duplicates
        # are not removed
        self.clean(depth=0)
        # -------------------------------------------
        self.assign_rid(flag=make_rid, idlength=6)
        # -------------------------------------------
        if make_ckdtree:
            self.maketree(treeType='ckdtree')
        # -------------------------------------------
        self.plot(flag=vis, xbound=xbound, ybound=ybound)
        # -------------------------------------------
        if self.reprstate:
            self._summary_(print_summary,
                                make_point_objects=make_point_objects,
                                )

    def recompute(self, flag_mids=True, flag_dist=True, flag_basics=True,
                  flag_npoint=True, flag_locxy=True, flag_centroid=True):
        if flag_mids:
            self.recompute_from_mids()
        if flag_dist:
            self.recompute_from_dist(tolerance_distance=0.0)
        if flag_basics:
            self.recompute_basics(npoints_flag=flag_npoint,
                                  locx_locy_flag=flag_locxy,
                                  centroid_flag=flag_centroid,
                                  )

    def __repr__(self, recompute_flag=False):
        if recompute_flag:
            if self.lean in ('no', 'low', 'medium'):
                self.recompute_from_mids()
                self.recompute_from_dist(method='points_normal',
                                         tolerance_distance=0.0)
            if self.lean in ('ignore', 'high', 'veryhigh'):
                self.recompute_from_dist(method='coord_normal',
                                         tolerance_distance=0.0)
        str1 = f'Multi-Point ({self.npoints})'
        str2 = f' Centroid:({round(self.centroid[0], 6)},'
        str3 = f' {round(self.centroid[1], 6)})'
        return str1 + str2 + str3

    def __len__(self):
        '''
        Returns [[mp1, mp2, ... mpn], [p1, p1, ..., pn]]
        Where,
              mp1 till mpn: child multi-point objects
              p1: total number of unique points in mp1, and so on
        '''
        return self.npoints

    def __iter__(self, __behaviour = '__iterate__over__points__'):
        '''
        RESTRICTION: operates on a single multi-point object
                     if there are child multi-point objects,
                     then user must ensure iter operating on each of them,
                     seperately !
        OPERATION:
                     Iterates through individual points and returns the
                     tuple (m_n.locx[i], m_n.locy[i]), where, n is the
                     n^th multi-point object
        '''
        # "__behaviour" options include:
            # (a) '__iterate__over__coordinates__'
            # (b) '__iterate__over__points__'
        __behaviour = '__iterate__over__points__'
        if __behaviour == '__iterate__over__coordinates__':
            return zip(self.locx, self.locy)
        elif __behaviour == '__iterate__over__points__':
            if self.lean in ('no', 'low', 'medium'):
                return self.points
            elif self.lean in ('high', 'veryhigh'):
                return zip(self.locx, self.locy)

    def __add__(self, obj=0.0, indices=[],
                recomp=True, operate_onetoone=True):
        '''
        BEHAVIOUR 1
        If obj is a single number of a single list of numbers, then this
        is usual addition to both x and y coordinates

        BEHAVIOUR 2
        if obj is a [~~  upxo point ~~] OR a [~~ list of upxo points ~~] OR
        a [~~ x- and y-coordinates ~~] OR a [~~ a list of c-y-coordinate
        pairs ~~], then this results in the obj elements being \\APPENDED\\
        to the present multi-point list of points. NOTE: multi-point properties
        will then be calculated automatically.

        NOTE: lean of the self.points will not be altered
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # PREPARE BASE DATA
        # ------------------------
        constant = 2
        # ------------------------
        constants = [1, 2, 3, 4]
        # ------------------------
        up2d_list = [point2d(x=0, y=0, lean='ignore'),
                     point2d(x=1, y=0, lean='ignore'),
                     point2d(x=0, y=0, lean='ignore'),
                     point2d(x=1, y=1.01, lean='ignore'),
                     point2d(x=2, y=2, lean='ignore'),]
        mp = mulpoint2d(method='up2d_list', point_objects=up2d_list)
        # ------------------------
        p1 = point2d(x=20, y=50, lean='ignore')
        # ------------------------
        p2 = point2d(x=2, y=2, lean='ignore')
        p12 = [p1, p2, p1+10, p2+50]
        # ------------------------
        # EXAMPLE - add single number 1
        constant = 1.0
        print(mp.points,'#',mp.npoints,'#',[mp.locx,mp.locy],'#',mp.centroid)
        mp + constant    # OR: mp.__add__(obj=constant, recomp=True)
        print(mp.points,'#',mp.npoints,'#',[mp.locx,mp.locy],'#',mp.centroid)
        # ------------------------
        # EXAMPLE - add list of numbers 1
        constants = [10, 2, 3]
        print(mp.points,'#',mp.npoints,'#',[mp.locx,mp.locy],'#',mp.centroid)
        mp + constants   # OR: mp.__add__(obj=constants,operate_onetoone=True)
        print(mp.points,'#',mp.npoints,'#',[mp.locx,mp.locy],'#',mp.centroid)
        # ------------------------
        # EXAMPLE - add single upxo point2d object - 1
        print(mp.points,'#',mp.npoints,'#',[mp.locx,mp.locy],'#',mp.centroid)
        mp + p1    # OR: mp.__add__(obj=p1,operate_onetoone=True)
        print(mp.points,'#',mp.npoints,'#',[mp.locx,mp.locy],'#',mp.centroid)
        # ------------------------
        # EXAMPLE - add list of upxo point2d objects - 1
        print(mp.points,'#',mp.npoints,'#',[mp.locx,mp.locy],'#',mp.centroid)
        mp + p12    # OR: mp.__add__(obj=p12,operate_onetoone=True)
        print(mp.points,'#',mp.npoints,'#',[mp.locx,mp.locy],'#',mp.centroid)
        # ------------------------
        # (x, y) coordinate of a single point
        obj = [-136, -465]
        print(mp.points,'#',mp.npoints,'#',[mp.locx,mp.locy],'#',mp.centroid)
        mp + obj
        print(mp.points,'#',mp.npoints,'#',[mp.locx,mp.locy],'#',mp.centroid)
        # ------------------------
        # coordinates of points
        obj = [[1.05, 1.06, 9.065],[1.05, 1.06, 9.065]]
        print(mp.points,'#',mp.npoints,'#',[mp.locx,mp.locy],'#',mp.centroid)
        mp + obj
        print(mp.points,'#',mp.npoints,'#',[mp.locx,mp.locy],'#',mp.centroid)
        # ------------------------
        # ------------------------
        '''
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if type(obj) in dth.dt.NUMBERS:
            '''
            Refer EXAMPLE - add single number 1
            '''
            for i, _ in enumerate(self.points):
                self.points[i].__add__(k=obj,
                                       saa=True,
                                       make_new=False,
                                       throw=False)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if type(obj) in dth.dt.ITERABLES:
            '''
            '''
            unique_datatypes = dth.unique_of_datatypes(obj)
            if len(unique_datatypes) == 1:
                # Refer EXAMPLE - list of numbers 1
                if unique_datatypes[0] == "<class 'int'>":
                    if operate_onetoone:
                        if self.npoints == len(obj):
                            for i, _ in enumerate(self.points):
                                self.points[i].__add__(k=obj[i],
                                                       saa=True,
                                                       make_new=False,
                                                       throw=False)
                        else:
                            self.points.append(point2d(x=obj[0], y=obj[1],
                                                       lean='ignore'))
                    else:
                        print('obj must be an iterable for this case')
                elif unique_datatypes[0] in ("<class 'UPXO-point.point2d'>",
                                             "<class 'list'>"):
                    if unique_datatypes[0] == "<class 'UPXO-point.point2d'>":
                        # Refer EXAMPLE - list of upxo point2d object
                        '''
                        PROOF OF CONCEPT CODE:

                        a = deque([point2d(0, 0),
                                   point2d(1, 0),
                                   point2d(1, 1),
                                   point2d(0, 1),
                                   point2d(2, 2),
                                  ])

                        b = [point2d(0.651, 0.984),
                             point2d(4.046, 4.165),
                             point2d(9.136, 9.013),
                             point2d(12.13, 12.89)]

                        indices = [1, 2, 3, 4]

                        if not indices:
                            indices = list(range(len(b)))

                        if indices not in dth.dt.ITERABLES:
                            indices = list(indices)

                        # Sort the indices and points in order
                        ind_order = np.argsort(indices)
                        indices = [indices[i] for i in ind_order]
                        b = [b[i] for i in ind_order]

                        a.insert(indices[0], b[0])

                        if len(indices) > 1:
                            if len(indices) > len(b):
                                indices = indices[:len(b)]
                            if len(indices) < len(b):
                                b = b[:len(indices)]
                            for i, ind in enumerate(indices[1:], start=1):
                                a.insert(ind+1, b[i])
                        -------------------------------------------
                        USE CASE 1:
                        mp = mulpoint2d(method='up2d_list',
                                        point_objects=[point2d(0, 0),
                                                       point2d(1, 0),
                                                       point2d(1, 1),
                                                       point2d(0, 1),
                                                       point2d(2, 2),
                                                       ])
                        obj = [point2d(0.651, 0.984),
                               point2d(4.046, 4.165),
                               point2d(9.136, 9.013),
                               point2d(12.13, 12.89)]

                        CASE 1A:
                        # Behind 1, behind 2, behind 3 and behind 4
                        indices = [1, 2, 3, 4]

                        # EXPECTED:
                        #    [point2d(0, 0),
                        #     point2d(0.651, 0.984)
                        #     point2d(1, 0),
                        #     point2d(4.046, 4.165),
                        #     point2d(1, 1),
                        #     point2d(9.136, 9.013),
                        #     point2d(0, 1),
                        #     point2d(12.13, 12.89)
                        #     point2d(2, 2),
                        #     ]

                        print(mp.points,'#',
                              mp.npoints,'#',
                              [mp.locx,mp.locy],'#',
                              mp.centroid)
                        mp.__add__(obj=obj, indices=indices)
                        print(mp.points,'#',
                              mp.npoints,'#',
                              [mp.locx,mp.locy],'#',
                              mp.centroid)

                        CASE 1B:
                        # Behind 1, behind 2, behind 3 and behind 4
                        indices = [1, 2, 3, 4]


                        ##################################
                              TO DO
                        BEHAVIOUR OF INDICES
                        CURRENTLY:
                            All elements in indices are integers
                            This wqill never allow insertion of adjacent points
                                in the list of user input points (i.e. in obj),
                                in an adjacent fashion

                        PROPOSED SOLUTION:
                            Some elements in indices couild be made lists.
                            When a list is encountered, the global indices of
                            input pouints will be incremented by unity.
                            This would ensure adjacent point insertion in the
                            mp.points.
                        ##################################
                        -------------------------------------------
                        '''
                        # Develop indices if not already provided
                        if obj:
                            if not indices:
                                indices = list(range(len(obj)))
                            if indices not in dth.dt.ITERABLES:
                                indices = [int(indices)]

                            # Sort the indices and points in order
                            ind_order = np.argsort(indices)
                            indices = [indices[i] for i in ind_order]
                            obj = [obj[i] for i in ind_order]

                            # Insert the point objects
                            #print('=================================')
                            #print(self.points)
                            self.points.insert(indices[0], obj[0])
                            #print(self.points)
                            #print('=================================')
                            if len(indices) > 1:
                                if len(indices) != len(obj):
                                    print('len(obj) MUST be same as indices')
                                else:
                                    for i, ind in enumerate(indices[1:],
                                                            start=1):
                                        self.points.insert(ind+i, obj[i])

                            self.recompute_from_mids()
                            ##############################
                            # testing if I can do without this.
                            # self.recompute_from_dist(tolerance_distance=0.0)
                            ##############################
                            self.npoints = len(self.locx)
                            self.recompute_centroid()
                            self.maketree(treeType='ckdtree',
                                          saa=True,
                                          throw=False)
                        else:
                            print('obj CANNOT be empty')
                    elif unique_datatypes[0] == "<class 'list'>":
                        if len(obj) == 1 and len(obj[0]) == 1:
                            # Single number
                            pass
                        if len(obj) == 1 and len(obj[0]) == 2:
                            self.points.append(point2d(x=obj[0][0],
                                                       y=obj[0][1],
                                                       lean='ignore'
                                                       )
                                               )
                        if len(obj) == 2 and len(obj[0]) > 2:
                            for _x, _y in zip(obj[0], obj[1]):
                                self.points.append(point2d(x=_x,
                                                           y=_y,
                                                           lean='ignore'
                                                           )
                                                   )
                        if len(obj) > 1:
                            # MIXED coordinates types
                            pass
            else:
                print('all elements of @obj@ must be of same dataype')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if obj.__class__.__name__ == 'point2d':
            '''
            Refer EXAMPLE - add single upxo point2d - 1
            '''
            self.points.append(obj)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if recomp:
            self.recompute(flag_mids=True, flag_dist=True,
                           flag_basics=True, flag_npoint=True,
                           flag_locxy=True, flag_centroid=True)

    def xadd(self, k):
        '''
        mp = mulpoint2d(method='xy_list', coordxy=np.random.rand(2, 10))
        print(mp.points, '___', mp.npoints, '___', mp.locx, '___', mp.centroid)
        mp.xadd(10)
        print(mp.points, '___', mp.npoints, '___', mp.locx, '___', mp.centroid)
        '''
        for i in range(0, self.npoints):
            self.points[i].xadd(k=k,
                                saa=True,
                                make_new=False,
                                lean='ignore',
                                throw=False)
        self.recompute(flag_mids=False, flag_dist=False, flag_basics=True,
                       flag_npoint=False, flag_locxy=True, flag_centroid=True)

    def yadd(self, k):
        '''
        mp = mulpoint2d(method='xy_list', coordxy=np.random.rand(2, 10))
        print(mp.points, '___', mp.npoints, '___', mp.locx, '___', mp.centroid)
        mp.yadd(10)
        print(mp.points, '___', mp.npoints, '___', mp.locx, '___', mp.centroid)
        '''
        for i in range(0, self.npoints):
            self.points[i].yadd(k=k,
                                saa=True,
                                make_new=False,
                                lean='ignore',
                                throw=False)
        self.recompute(flag_mids=False, flag_dist=False, flag_basics=True,
                       flag_npoint=False, flag_locxy=True, flag_centroid=True)

    def xmul(self, k):
        '''
        mp = mulpoint2d(method='xy_list', coordxy=np.random.rand(2, 10))
        print(mp.points, '___', mp.npoints, '___', mp.locx, '___', mp.centroid)
        print(id(mp.points[0]))
        mp.xmul(10)
        print(mp.points, '___', mp.npoints, '___', mp.locx, '___', mp.centroid)
        print(id(mp.points[0]))
        '''
        for i in range(0, self.npoints):
            self.points[i].xmul(k=k,
                                saa=True,
                                make_new=False,
                                lean='ignore',
                                throw=False)
        self.recompute(flag_mids=False, flag_dist=False, flag_basics=True,
                       flag_npoint=False, flag_locxy=True, flag_centroid=True)

    def ymul(self, k):
        '''
        mp = mulpoint2d(method='xy_list', coordxy=np.random.rand(2, 10))
        print(mp.points, '___', mp.npoints, '___', mp.locx, '___', mp.centroid)
        print(id(mp.points[0]))
        mp.ymul(10)
        print(mp.points, '___', mp.npoints, '___', mp.locx, '___', mp.centroid)
        print(id(mp.points[0]))
        '''
        for i in range(0, self.npoints):
            self.points[i].ymul(k=k,
                                saa=True,
                                make_new=False,
                                lean='ignore',
                                throw=False)
        self.recompute(flag_mids=False, flag_dist=False, flag_basics=True,
                       flag_npoint=False, flag_locxy=True, flag_centroid=True)

    def xdiv(self, k):
        '''
        mp = mulpoint2d(method='xy_list', coordxy=np.random.rand(2, 10))
        print(mp.points, '___', mp.npoints, '___', mp.locx, '___', mp.centroid)
        print(id(mp.points[0]))
        mp.xdiv(10)
        print(mp.points, '___', mp.npoints, '___', mp.locx, '___', mp.centroid)
        print(id(mp.points[0]))
        '''
        for i in range(0, self.npoints):
            self.points[i].xdiv(k=k,
                                saa=True,
                                make_new=False,
                                lean='ignore',
                                throw=False)
        self.recompute(flag_mids=False, flag_dist=False, flag_basics=True,
                       flag_npoint=False, flag_locxy=True, flag_centroid=True)

    def ydiv(self, k):
        '''
        mp = mulpoint2d(method='xy_list', coordxy=np.random.rand(2, 10))
        print(mp.points, '___', mp.npoints, '___', mp.locx, '___', mp.centroid)
        print(id(mp.points[0]))
        mp.ydiv(10)
        print(mp.points, '___', mp.npoints, '___', mp.locx, '___', mp.centroid)
        print(id(mp.points[0]))
        '''
        for i in range(0, self.npoints):
            self.points[i].ydiv(k=k,
                                saa=True,
                                make_new=False,
                                lean='ignore',
                                throw=False)
        self.recompute(flag_mids=False, flag_dist=False, flag_basics=True,
                       flag_npoint=False, flag_locxy=True, flag_centroid=True)

    def __abs__(self):
        '''
        mp = mulpoint2d(method='xy_list', coordxy=-np.random.rand(2, 100))
        print(mp.points, '___', mp.npoints, '___', mp.locx, '___', mp.centroid)
        print(id(mp.points[0]))
        abs(mp)
        print(mp.points, '___', mp.npoints, '___', mp.locx, '___', mp.centroid)
        print(id(mp.points[0]))
        '''
        for i in range(0, self.npoints):
            self.points[i].__abs__(saa=True,
                                   make_new=False,
                                   lean='ignore',
                                   throw=False)
        self.recompute(flag_mids=False, flag_dist=False, flag_basics=True,
                       flag_npoint=False, flag_locxy=True, flag_centroid=True)

    def xabs(self):
        '''
        mp = mulpoint2d(method='xy_list', coordxy=-np.random.rand(2, 10))
        print(mp.points, '___', mp.npoints, '___', mp.locx, '___', mp.centroid)
        print(id(mp.points[0]))
        mp.xabs()
        print(mp.points, '___', mp.npoints, '___', mp.locx, '___', mp.centroid)
        print(id(mp.points[0]))
        '''
        for i in range(0, self.npoints):
            self.points[i].xabs(saa=True,
                                make_new=False,
                                lean='ignore',
                                throw=False)
        self.recompute(flag_mids=False, flag_dist=False, flag_basics=True,
                       flag_npoint=False, flag_locxy=True, flag_centroid=True)

    def yabs(self):
        '''
        mp = mulpoint2d(method='xy_list', coordxy=-np.random.rand(2, 10))
        print(mp.points, '___', mp.npoints, '___', mp.locx, '___', mp.centroid)
        print(id(mp.points[0]))
        mp.yabs()
        print(mp.points, '___', mp.npoints, '___', mp.locx, '___', mp.centroid)
        print(id(mp.points[0]))
        '''
        for i in range(0, self.npoints):
            self.points[i].yabs(saa=True,
                                make_new=False,
                                lean='ignore',
                                throw=False)
        self.recompute(flag_mids=False, flag_dist=False, flag_basics=True,
                       flag_npoint=False, flag_locxy=True, flag_centroid=True)

    def intize(self):
        '''
        mp = mulpoint2d(method='xy_list', coordxy=-np.random.rand(2, 10))
        print(mp.points, '___', mp.npoints, '___', mp.locx, '___', mp.centroid)
        print(id(mp.points[0]))
        mp.intize()
        print(mp.points, '___', mp.npoints, '___', mp.locx, '___', mp.centroid)
        print(id(mp.points[0]))
        '''
        for i in range(0, self.npoints):
            self.points[i].intize(saa=True,
                                  make_new=False,
                                  lean='ignore',
                                  throw=False)
        self.recompute(flag_mids=True, flag_dist=True, flag_basics=True,
                       flag_npoint=True, flag_locxy=True, flag_centroid=True)

    def floatize(self):
        '''
        mp = mulpoint2d(method='xy_list', coordxy=-np.random.rand(2, 10))
        print(mp.points, '___', mp.npoints, '___', mp.locx, '___', mp.centroid)
        print(id(mp.points[0]))
        mp.floatize()
        print(mp.points, '___', mp.npoints, '___', mp.locx, '___', mp.centroid)
        print(id(mp.points[0]))
        '''
        for i in range(0, self.npoints):
            self.points[i].floatize(saa=True,
                                    make_new=False,
                                    lean='ignore',
                                    throw=False)
        self.recompute(flag_mids=False, flag_dist=False, flag_basics=True,
                       flag_npoint=False, flag_locxy=True, flag_centroid=True)

    def roundround(self, nd=5):
        '''
        mp = mulpoint2d(method='xy_list', coordxy=-np.random.rand(2, 10))
        print(mp.points, '___', mp.npoints, '___', mp.locx, '___', mp.centroid)
        print(id(mp.points[0]))
        mp.roundround()
        print(mp.points, '___', mp.npoints, '___', mp.locx, '___', mp.centroid)
        print(id(mp.points[0]))
        '''
        for i in range(0, self.npoints):
            self.points[i].roundround(nd=nd,
                                      saa=True,
                                      make_new=False,
                                      lean='ignore',
                                      throw=False)
        self.recompute(flag_mids=True, flag_dist=True, flag_basics=True,
                       flag_npoint=True, flag_locxy=True, flag_centroid=True)

    def xroundround(self, nd=5):
        '''
        mp = mulpoint2d(method='xy_list', coordxy=-np.random.rand(2, 10))
        print(mp.points, '___', mp.npoints, '___', mp.locx, '___', mp.centroid)
        print(id(mp.points[0]))
        mp.xroundround()
        print(mp.points, '___', mp.npoints, '___', mp.locx, '___', mp.centroid)
        print(id(mp.points[0]))
        '''
        for i in range(0, self.npoints):
            self.points[i].xroundround(nd=nd,
                                       saa=True,
                                       make_new=False,
                                       lean='ignore',
                                       throw=False)
        self.recompute(flag_mids=True, flag_dist=True, flag_basics=True,
                       flag_npoint=True, flag_locxy=True, flag_centroid=True)

    def yroundround(self, nd=5):
        '''
        mp = mulpoint2d(method='xy_list', coordxy=-np.random.rand(2, 10))
        print(mp.points, '___', mp.npoints, '___', mp.locx, '___', mp.centroid)
        print(id(mp.points[0]))
        mp.yroundround()
        print(mp.points, '___', mp.npoints, '___', mp.locx, '___', mp.centroid)
        print(id(mp.points[0]))
        '''
        for i in range(0, self.npoints):
            self.points[i].yroundround(nd=nd,
                                       saa=True,
                                       make_new=False,
                                       lean='ignore',
                                       throw=False)
        self.recompute(flag_mids=True, flag_dist=True, flag_basics=True,
                       flag_npoint=True, flag_locxy=True, flag_centroid=True)

    def roundceil(self, nd=5):
        '''
        mp = mulpoint2d(method='xy_list', coordxy=-np.random.rand(2, 10))
        print(mp.points, '___', mp.npoints, '___', mp.locx, '___', mp.centroid)
        print(id(mp.points[0]))
        mp.roundceil()
        print(mp.points, '___', mp.npoints, '___', mp.locx, '___', mp.centroid)
        print(id(mp.points[0]))
        '''
        for i in range(0, self.npoints):
            self.points[i].roundceil(nd=nd,
                                     saa=True,
                                     make_new=False,
                                     lean='ignore',
                                     throw=False)
        self.recompute(flag_mids=True, flag_dist=True, flag_basics=True,
                       flag_npoint=True, flag_locxy=True, flag_centroid=True)

    def xroundceil(self, nd=5):
        '''
        mp = mulpoint2d(method='xy_list', coordxy=-np.random.rand(2, 10))
        print(mp.points, '___', mp.npoints, '___', mp.locx, '___', mp.centroid)
        print(id(mp.points[0]))
        mp.xroundceil()
        print(mp.points, '___', mp.npoints, '___', mp.locx, '___', mp.centroid)
        print(id(mp.points[0]))
        '''
        for i in range(0, self.npoints):
            self.points[i].xroundceil(nd=nd,
                                      saa=True,
                                      make_new=False,
                                      lean='ignore',
                                      throw=False)
        self.recompute(flag_mids=True, flag_dist=True, flag_basics=True,
                       flag_npoint=True, flag_locxy=True, flag_centroid=True)

    def yroundceil(self, nd=5):
        '''
        mp = mulpoint2d(method='xy_list', coordxy=-np.random.rand(2, 10))
        print(mp.points, '___', mp.npoints, '___', mp.locx, '___', mp.centroid)
        print(id(mp.points[0]))
        mp.yroundceil()
        print(mp.points, '___', mp.npoints, '___', mp.locx, '___', mp.centroid)
        print(id(mp.points[0]))
        '''
        for i in range(0, self.npoints):
            self.points[i].yroundceil(nd=nd,
                                      saa=True,
                                      make_new=False,
                                      lean='ignore',
                                      throw=False)
        self.recompute(flag_mids=True, flag_dist=True, flag_basics=True,
                       flag_npoint=True, flag_locxy=True, flag_centroid=True)

    def roundfloor(self, nd=5):
        '''
        mp = mulpoint2d(method='xy_list', coordxy=-np.random.rand(2, 10))
        print(mp.points, '___', mp.npoints, '___', mp.locx, '___', mp.centroid)
        print(id(mp.points[0]))
        mp.roundfloor()
        print(mp.points, '___', mp.npoints, '___', mp.locx, '___', mp.centroid)
        print(id(mp.points[0]))
        '''
        for i in range(0, self.npoints):
            self.points[i].roundfloor(nd=nd,
                                      saa=True,
                                      make_new=False,
                                      lean='ignore',
                                      throw=False)
        self.recompute(flag_mids=True, flag_dist=True, flag_basics=True,
                       flag_npoint=True, flag_locxy=True, flag_centroid=True)

    def xroundfloor(self, nd=5):
        '''
        mp = mulpoint2d(method='xy_list', coordxy=-np.random.rand(2, 10))
        print(mp.points, '___', mp.npoints, '___', mp.locx, '___', mp.centroid)
        print(id(mp.points[0]))
        mp.xroundfloor()
        print(mp.points, '___', mp.npoints, '___', mp.locx, '___', mp.centroid)
        print(id(mp.points[0]))
        '''
        for i in range(0, self.npoints):
            self.points[i].xroundfloor(nd=nd,
                                       saa=True,
                                       make_new=False,
                                       lean='ignore',
                                       throw=False)
        self.recompute(flag_mids=True, flag_dist=True, flag_basics=True,
                       flag_npoint=True, flag_locxy=True, flag_centroid=True)

    def yroundfloor(self, nd=5):
        '''
        mp = mulpoint2d(method='xy_list', coordxy=-np.random.rand(2, 10))
        print(mp.points, '___', mp.npoints, '___', mp.locx, '___', mp.centroid)
        print(id(mp.points[0]))
        mp.yroundfloor()
        print(mp.points, '___', mp.npoints, '___', mp.locx, '___', mp.centroid)
        print(id(mp.points[0]))
        '''
        for i in range(0, self.npoints):
            self.points[i].yroundfloor(nd=nd,
                                       saa=True,
                                       make_new=False,
                                       lean='ignore',
                                       throw=False)
        self.recompute(flag_mids=True, flag_dist=True, flag_basics=True,
                       flag_npoint=True, flag_locxy=True, flag_centroid=True)

    def pop_points_indices(self,
                           indices=None, recomp_mids=True,
                           recomp_dist=True, recomp_tdist=0.0,
                           recomp_basics=True, recomp_npoints=True,
                           recomp_locxy=True, recomp_centroid=True,
                           throw_message=False
                           ):
        '''
        POP points in self using indices of the points

        mp = mulpoint2d(method='xy_list', coordxy=np.random.rand(2, 10))
        mp.points
        indices = [0, 2, 4]
        mp.pop_points_indices(indices=indices,
                              recomp_mids=True,
                              recomp_dist=True,
                              recomp_tdist=0.0,
                              recomp_basics=True,
                              recomp_npoints=True,
                              recomp_locxy=True,
                              recomp_centroid=True,
                              throw_message=False
                              )
        mp.points
        '''
        if not type(indices) in dth.dt.ITERABLES:
            indices = [indices]
        self.points = [i for j, i in enumerate(self.points)
                       if j not in indices]
        if recomp_mids:
            self.recompute_from_mids()
        if recomp_dist:
            self.recompute_from_dist(tolerance_distance=recomp_tdist)
        if recomp_basics:
            self.recompute_basics(npoints_flag=recomp_npoints,
                                  locx_locy_flag=recomp_locxy,
                                  centroid_flag=recomp_centroid,
                                  )
        if throw_message:
            if len(indices) > 0:
                print(f'{len(indices)} points removed')
            else:
                print('0 points removed')

    def find_points_within_radius(self,
                                  use_tree=True,
                                  centre=[0.0, 0.0],
                                  cor=1.0,
                                  nworkers=1
                                  ):
        '''
        # USE TREE = TRUE
        mp = mulpoint2d(method='xy_list', coordxy=np.random.rand(2, 10))
        print(mp.points, '#', mp.npoints, '#', mp.locx, '#', mp.centroid)
        mp.find_points_within_radius(use_tree=True, centre=[0,0], cor=0.5)

        # USE TREE = FALSE
        mp = mulpoint2d(method='xy_list', coordxy=np.random.rand(2, 10))
        print(mp.points, '#', mp.npoints, '#', mp.locx, '#', mp.centroid)
        mp.find_points_within_radius(use_tree=False, centre=[0,0], cor=0.5)
        '''
        if use_tree:
            indices = tuple(self.tree.query_ball_point(centre,
                                                       cor,
                                                       workers=nworkers
                                                       )
                            )
        else:
            dist = np.sqrt((self.locx-centre[0])**2 + (self.locy-centre[1])**2)
            indices = tuple(np.where(dist <= cor)[0])
        return indices

    def remove_within_radius(self,
                             centre=[0.0, 0.0],
                             cor=2.0,
                             recomp=True,
                             use_tree=False,
                             nworkers=1,
                             recomp_mids=True,
                             recomp_dist=True,
                             recomp_npoints=True,
                             recomp_locxy=True,
                             recomp_centroid=True
                             ):
        '''
        mp = mulpoint2d(method='xy_list', coordxy=np.random.rand(2, 10))
        print(mp.points, '#', mp.npoints, '#', mp.locx, '#', mp.centroid)
        mp.remove_within_radius(centre=[0,0], cor=0.5, use_tree=True)
        print(mp.points, '#', mp.npoints, '#', mp.locx, '#', mp.centroid)

        mp = mulpoint2d(method='xy_list', coordxy=np.random.rand(2, 10))
        print(mp.points, '#', mp.npoints, '#', mp.locx, '#', mp.centroid)
        mp.remove_within_radius(centre=[0,0], cor=0.75, use_tree=False)
        print(mp.points, '#', mp.npoints, '#', mp.locx, '#', mp.centroid)
        '''
        if use_tree:
            indices = self.find_points_within_radius(use_tree=True,
                                                     centre=centre,
                                                     cor=cor,
                                                     nworkers=nworkers
                                                     )
            indices = self.tree.query_ball_point(centre, cor, workers=1)
        else:
            dist = np.sqrt((self.locx-centre[0])**2 + (self.locy-centre[1])**2)
            indices = list(np.where(dist <= cor)[0])

        if len(indices) > 0:
            self.pop_points_indices(indices)
            print(f'{len(indices)} points removed')
        else:
            print('0 points removed')

        self.recompute(flag_mids=False, flag_dist=False, flag_basics=True,
                       flag_npoint=True, flag_locxy=True, flag_centroid=True)
        # if recomp:
        #     self.recompute_from_mids()
        #     self.recompute_from_dist(tolerance_distance=0.0)
        #     self.recompute_basics(npoints_flag=True,
        #                           locx_locy_flag=True,
        #                           centroid_flag=True,
        #                           )

    def remove_at_location(self,
                           xy_list=[[0.0, 0.0, 0.0], [1.0, 1.0, 2.0]],
                           cor=0.0,
                           recomp=True,
                           use_tree=False,
                           nworkers=1,
                           recomp_mids=True,
                           recomp_dist=True,
                           recomp_basics=True,
                           recomp_npoints=True,
                           recomp_locxy=True,
                           recomp_centroid=True,
                           throw_message=False
                           ):
        '''
        xy_list_for_mp = [[0, 0, 1, 2, 3], [0, 1, 2, 3, 4]]
        mp = mulpoint2d(method='xy_list', coordxy=xy_list_for_mp)
        print(mp.points, '___', mp.npoints, '___', mp.locx,
              '___', mp.centroid)

        mp.remove_at_location(xy_list=[[0.0, 0.0, 0.0], [1.0, 1.0, 2.0]],
                              cor=0.0, recomp=True, use_tree=False, nworkers=1,
                              recomp_mids=True, recomp_dist=True,
                              recomp_basics=True, recomp_npoints=True,
                              recomp_locxy=True, recomp_centroid=True,
                              throw_message=False)

        print(mp.points, '___', mp.npoints, '___', mp.locx,
              '___', mp.centroid)
        '''
        for x, y in zip(xy_list[0], xy_list[1]):
            dist = np.sqrt((self.locx-x)**2 + (self.locy-y)**2)
            indices = np.where(dist <= cor)[0]
            if len(indices) > 0:
                self.pop_points_indices(indices=indices,
                                        recomp_mids=recomp_mids,
                                        recomp_dist=recomp_dist,
                                        recomp_tdist=0.0,
                                        recomp_basics=recomp_basics,
                                        recomp_npoints=recomp_npoints,
                                        recomp_locxy=recomp_locxy,
                                        recomp_centroid=recomp_centroid,
                                        throw_message=throw_message
                                        )

    def __contains__(self, otype='xy_lists', obj=None,
                     prox=0.0, use_tree=False):
        '''
        BRANCH NO. 1: dth.opt.upxo_point2d_list: 'up2dlist'
        BRANCH NO. 2: dth.opt.coord_point2d_list: 'xy_list'
        BRANCH NO. 3: dth.opt.upxo_point2d_list_list: 'up2d_lists'
        BRANCH NO. 4: dth.opt.coord_point2d_list_list: 'xy_lists'
        '''
        if otype in dth.opt.upxo_point2d_list:
            # SINGLE LIST OF UPXO POINT2D OBJECTS
            # BRANCH: 'up2dlist'
            '''
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # DESCRIPTION
            Upon providing a single list of UPXO point2d objects, this branch
            returns the indices of points in the present multi-point which
            are within "prox" distance to the points querried

            # INPUTS
            1. otype = 'up2d_list'
            2. obj = list of objects of type dth.opt.upxo_point2d_list

            # OUTPUTS
            One list(a) of list(b) is returned

            # OUTPUT DESCRIPTION
            > size of list(a) = size of obj
            > output = [list0, list1, ..., listn,...]
            > listn = [n0, n1, ..., ni, ...]
            > listn contains indices of points in obj, which are contained
            inside the multi-point
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            #               EXAMPLE - 1
            # ~~ Prepare the self object
            from point2d import point2d
            from mulpoint2d import mulpoint2d
            x, y = [0, 1], [0, 1]
            mp = mulpoint2d(method='xlist_ylist', coordx=x, coordy=y)
            # -------------------------
            # ~~ Prepare the obj
            up2d_list = [point2d(x=0, y=0, lean='ignore'),
                         point2d(x=1, y=0, lean='ignore'),
                         point2d(x=0, y=0, lean='ignore'),
                         point2d(x=1, y=1.01, lean='ignore'),
                         point2d(x=2, y=2, lean='ignore'),]
            # -------------------------
            # ~~ Implementation
            a = mp.__contains__(otype='up2d_list', obj=up2d_list, prox=2.0)
            a
            # -------------------------
            # ~~ SAMPLE OUTPUT
            [[0, 1], [0, 1], [0, 1], [0, 1], [1]]
            # -------------------------
            # ~~ SAMPLE OUTPUT EXPLANATION
            > There are 5 lists, one for each queried point object
            > 1st list has 0 and 1.
            > The 0th of the 5 query points is at (0,0).
            > Search radius around this point is prox=2.0, as chosen
            in this example
            > mp-points in current mp are at (0,0) and (1, 1).
            > Both 0th and 1st mp-points fall inside the search radius.
            > Hence, [0, 1] for 0th query point
            > 5th list is the last one. It is [1]. For 5th
            querried point at (2, 2), only the 2nd mp-point (1,1) at
            position 1 in mp.locx falls inside or on the search circle.
            Hence [1].
            '''
            if not use_tree:
                obj = np.array([[p.x, p.y] for p in obj]).T
                contains = []
                for _x, _y in zip(obj[0], obj[1]):
                    distances = np.sqrt((_x-np.array(self.locx))**2 +
                                        (_x-np.array(self.locy))**2)
                    contains.append(list(np.where(distances <= prox)[0]))
                to_return = contains
            else:
                pass
        if otype in dth.opt.coord_point2d_list:
            # SINGLE LIST OF LIST OF X AND LIST OF Y COORDINATES
            # BRANCH: 'xy_list'
            '''
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # DESCRIPTION
            Upon providing a single xy_list, this branch
            returns the indices of points in the present multi-point which
            are within "prox" distance to the points querried

            # INPUTS
            1. otype = 'xy_list'
            2. obj = list of point coordinates of type
                     dth.opt.upxo_point2d_list

            # OUTPUTS
            One list(a) of lists(b) is returned

            # OUTPUT DESCRIPTION
            > size of list(a) = size of obj
            > output = [list0, list1, ..., listn,...]
            > listn = [n0, n1, ..., ni, ...]
            > listn contains indices of points in obj, which are contained
            inside the multi-point
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            #               EXAMPLE - 1
            # ~~ Prepare the self object
            from point2d import point2d
            from mulpoint2d import mulpoint2d
            x, y = [0, 1], [0, 1]
            mp = mulpoint2d(method='xlist_ylist', coordx=x, coordy=y)
            # -------------------------
            # ~~ Prepare the obj
            xy_list = [[0, 1, 0, 1, 2], [0, 0, 0, 1.01, 2]]
            # -------------------------
            # ~~ Implementation
            a = mp.__contains__(otype='xy_list', obj=xy_list, prox=2.0)
            a
            # -------------------------
            # ~~ SAMPLE OUTPUT
            [[0, 1], [0, 1], [0, 1], [0, 1], [1]]
            # -------------------------
            # ~~ SAMPLE OUTPUT EXPLANATION
            > There are 5 lists, one for each queried point
            > 1st list has 0 and 1.
            > The 0th of the 5 query points is at (0,0).
            > Search radius around this point is prox=2.0, as chosen
            in this example
            > mp-points in current mp are at (0,0) and (1, 1).
            > Both 0th and 1st mp-points fall inside the search radius.
            > Hence, [0, 1] for 0th query point
            > 5th list is the last one. It is [1]. For 5th
            querried point at (2, 2), only the 2nd mp-point (1,1) at
            position 1 in mp.locx falls inside or on the search circle.
            Hence [1].
            '''
            if not use_tree:
                contains = []
                for _x, _y in zip(obj[0], obj[1]):
                    distances = np.sqrt((_x-np.array(self.locx))**2 +
                                        (_x-np.array(self.locy))**2)
                    contains.append(list(np.where(distances <= prox)[0]))
                to_return = contains
            else:
                pass
        if otype in dth.opt.upxo_point2d_list_list:
            # LIST OF LISTS OF UPXO POINT2D OBJECTS
            # BRANCH: 'up2d_lists'
            '''
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # DESCRIPTION
            Upon providing a single list of lists of UPXO point2d objects,
            this branch returns the indices of points in the present
            multi-point which are within "prox" distance to the points
            querried

            # INPUTS
            1. otype = 'up2d_lists'
            2. obj = list of list of objects of
                     type dth.opt.coord_point2d_list_list

            # OUTPUTS
            One list(a) of lists(b) of lists(c) is returned

            # OUTPUT DESCRIPTION
            > size of list(a) = size of obj
            > output = [[list0, list1, ..., listn,...],
                        [list0, list1, ..., listn,...],...]
            > listn = [n0, n1, ..., ni, ...]
            > listn contains indices of points in obj, which are contained
            inside the multi-point
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            #               EXAMPLE - 1
            # ~~ Prepare the self object
            from point2d import point2d
            from mulpoint2d import mulpoint2d
            x, y = [0, 1], [0, 1]
            mp = mulpoint2d(method='xlist_ylist', coordx=x, coordy=y)
            # -------------------------
            # ~~ Prepare the obj
            up2d_list = [point2d(x=0, y=0, lean='ignore'),
                         point2d(x=1, y=0, lean='ignore'),
                         point2d(x=0, y=0, lean='ignore'),
                         point2d(x=1, y=1.01, lean='ignore'),
                         point2d(x=2, y=2, lean='ignore'),]
            up2d_lists = [up2d_list, up2d_list, up2d_list]
            # -------------------------
            # ~~ Implementation
            a = mp.__contains__(otype='up2d_lists', obj=up2d_lists, prox=2.0)
            a
            # -------------------------
            # ~~ SAMPLE OUTPUT
            [[[0, 1], [0, 1], [0, 1], [0, 1], [1]],
             [[0, 1], [0, 1], [0, 1], [0, 1], [1]],
             [[0, 1], [0, 1], [0, 1], [0, 1], [1]]]
            # -------------------------
            # ~~ SAMPLE OUTPUT EXPLANATION
            > There are 3 lists. One for each list of queried points
            > In each, there are 5 lists, one for each queried points
            > 1st list has 0 and 1.
            > The 0th of the 5 query points is at (0,0).
            > Search radius around this point is prox=2.0, as chosen
            in this example
            > mp-points in current mp are at (0,0) and (1, 1).
            > Both 0th and 1st mp-points fall inside the search radius.
            > Hence, [0, 1] for 0th query point
            > 5th list is the last one. It is [1]. For 5th
            querried point at (2, 2), only the 2nd mp-point (1,1) at
            position 1 in mp.locx falls inside or on the search circle.
            Hence [1].
            '''
            if not use_tree:
                locx, locy = self.locx, self.locy
                contains = []
                for _obj in obj:
                    _obj = np.array([[p.x, p.y] for p in _obj]).T
                    _contains = []
                    for _x, _y in zip(_obj[0], _obj[1]):
                        distances = np.sqrt((_x-np.array(locx))**2 +
                                            (_x-np.array(locy))**2)
                        _contains.append(list(np.where(distances <= prox)[0]))
                    contains.append(_contains)
                to_return = contains
            else:
                pass
        if otype in dth.opt.coord_point2d_list_list:
            # LIST OF LISTS OF LIST OF X AND LIST OF Y COORDINATES
            # BRANCH: 'xy_lists'
            '''
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # DESCRIPTION
            Upon providing a list of lists of x and y coordinates of points,
            this branch returns the indices of points in the present
            multi-point which are within "prox" distance to the points
            querried

            # INPUTS
            1. otype = 'xy_lists'
            2. obj = list of objects of type dth.opt.upxo_point2d_list

            # OUTPUTS
            One list(a) of list(b) is returned

            # OUTPUT DESCRIPTION
            > size of list(a) = size of obj
            > output = [list0, list1, ..., listn,...]
            > listn = [n0, n1, ..., ni, ...]
            > listn contains indices of points in obj, which are contained
            inside the multi-point
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            #               EXAMPLE - 1
            # ~~ Prepare the self object
            from point2d import point2d
            from mulpoint2d import mulpoint2d
            x, y = [0, 1], [0, 1]
            mp = mulpoint2d(method='xlist_ylist', coordx=x, coordy=y)
            # -------------------------
            # ~~ Prepare the obj
            xy_lists = [[[0, 1, 0, 1, 2], [0, 0, 0, 1.01, 2]],
                        [[0, 1, 0, 1, 2], [0, 0, 0, 1.01, 2]],
                        [[0, 1, 0, 1, 2], [0, 0, 0, 1.01, 2]]]
            # -------------------------
            # ~~ Implementation
            a = mp.__contains__(otype='xy_lists', obj=xy_lists, prox=2.0)
            a
            # -------------------------
            # ~~ SAMPLE OUTPUT
            [[[0, 1], [0, 1], [0, 1], [0, 1], [1]],
             [[0, 1], [0, 1], [0, 1], [0, 1], [1]],
             [[0, 1], [0, 1], [0, 1], [0, 1], [1]]]
            # -------------------------
            # ~~ SAMPLE OUTPUT EXPLANATION
            > There are 3 lists. One for each list of queried points
            > In each, there are 5 lists, one for each queried points
            > 1st list has 0 and 1.
            > The 0th of the 5 query points is at (0,0).
            > Search radius around this point is prox=2.0, as chosen
            in this example
            > mp-points in current mp are at (0,0) and (1, 1).
            > Both 0th and 1st mp-points fall inside the search radius.
            > Hence, [0, 1] for 0th query point
            > 5th list is the last one. It is [1]. For 5th
            querried point at (2, 2), only the 2nd mp-point (1,1) at
            position 1 in mp.locx falls inside or on the search circle.
            Hence [1].
            '''
            if not use_tree:
                locx, locy = self.locx, self.locy
                contains = []
                for _obj in obj:
                    _contains = []
                    for _x, _y in zip(_obj[0], _obj[1]):
                        distances = np.sqrt((_x-np.array(locx))**2 +
                                            (_x-np.array(locy))**2)
                        _contains.append(list(np.where(distances <= prox)[0]))
                    contains.append(_contains)
                to_return = contains
            else:
                pass
        return to_return

    def __pow__(self, power):
        '''
        mp = mulpoint2d(method='xy_list', coordxy=np.random.rand(2, 10))
        print(mp.points, '#', mp.npoints, '#', mp.locx, '#', mp.centroid)
        mp**2
        print(mp.points, '#', mp.npoints, '#', mp.locx, '#', mp.centroid)
        '''
        if self.lean in ('ignore', 'no', 'low', 'medium'):
            for _point in self.points:
                _point.x **= power
                _point.y **= power
        if self.lean in ('high', 'veryhigh'):
            self.locx **= power
            self.locy **= power
        self.recompute(flag_mids=False, flag_dist=False, flag_basics=True,
                       flag_npoint=False, flag_locxy=True, flag_centroid=True)

    def neg(self,
            axis: str = 'x',
            indices=None):
        '''
        '''
        if self.lean in ('no', 'low', 'medium'):
            if indices is None:
                _iterable = range(self.npoints)
            else:
                _iterable = indices

            if axis == 'x':
                for i in _iterable:
                    self.points[i].negx()
            elif axis == 'y':
                for i in _iterable:
                    self.points[i].negy()
            elif axis == 'xy':
                self.neg(axis='x', indices=indices)
                self.neg(axis='y', indices=indices)
            self.__state_change = True
            self.clean(depth=2)
        elif self.lean in ('high', 'veryhigh'):
            if axis == 'x':
                for i in indices:
                    self.locx[i] *= -1
            elif axis == 'y':
                for i in indices:
                    self.locy[i] *= -1
            elif axis == 'xy':
                self.neg(axis='x', indices=indices)
                self.neg(axis='y', indices=indices)
            self.__state_change = True
            self.clean(depth=2)
        self.recompute_basics(npoints_flag=True,
                              locx_locy_flag=False,
                              centroid_flag=True,)
        self.__state_change = False

    def translate(self, delx, dely):
        '''
        mp = mulpoint2d(method='xy_list', coordxy=np.random.rand(2, 10))
        print(mp.points, '#', mp.npoints, '#', mp.locx, '#', mp.centroid)
        mp.translate(-10, -10)
        print(mp.points, '#', mp.npoints, '#', mp.locx, '#', mp.centroid)
        '''
        if self.lean in ('ignore', 'no', 'low', 'medium'):
            if delx != 0:
                for pobj in self.points:
                    pobj.x += delx
            if dely != 0:
                for pobj in self.points:
                    pobj.y += dely
            self.recompute_basics(npoints_flag=False,
                                  locx_locy_flag=True,
                                  centroid_flag=True,
                                  )
        elif self.lean in ('high', 'veryhigh'):
            if delx != 0:
                self.locx += delx
            if dely != 0:
                self.locy += dely
            self.recompute_basics(npoints_flag=False,
                                  locx_locy_flag=False,
                                  centroid_flag=True,
                                  )

    def moveto(self, newx, newy):
        '''
        mp = mulpoint2d(method='xy_list', coordxy=np.random.rand(2, 10))
        print(mp.points, '#', mp.npoints, '#', mp.locx, '#', mp.centroid)
        mp.moveto(10, 10)
        print(mp.points, '#', mp.npoints, '#', mp.locx, '#', mp.centroid)
        '''
        self.translate(newx - self.centroid[0], newy - self.centroid[1])

    def DG(self,
           deformation='stretch',
           facx=None, facy=None,
           shear=None,
           F11=None, F12=None, F21=None, F22=None,
           angle=None
           ):
        '''
        Returns the deformation gradient.
        '''
        if deformation == 'stretch':
            return np.array([[facx, 0.0], [0.0, facy]])
        elif deformation == 'shearx':
            return np.array([[1.0, 0.0], [shear, 1.0]])
        elif deformation == 'sheary':
            return np.array([[1.0, shear], [0.0, 1.0]])
        elif deformation in ('pureshear', 'shear_xy', 'shearxy'):
            return np.array([[1.0, shear], [shear, 1.0]])
        elif deformation == 'rotation':
            θ = np.deg2rad(angle)
            c, s = np.cos(θ), np.sin(θ)
            # BELOW is Nothing but the rotation matrix
            return np.array([[c, -s], [s, c]])
        elif deformation == 'general':
            return np.array([[F11, F12], [F21, F22]])

    def rotate1(self, x, y, θ):
        '''
        x, y: coordinates of the rotation centre
        ang: angle of rotation.

        NOTE: Rotation will be counter-clockwise

        x1 = (x0 – xc)cos(θ) – (y0 – yc)sin(θ) + xc
        y1 = (x0 – xc)sin(θ) + (y0 – yc)cos(θ) + yc
        where:
            (x0, y0)	= Point to be rotated
            (xc, yc)	= Coordinates of center of rotation
            θ	        = Angle of rotation (positive counterclockwise)
            (x1, y1)	= Coordinates of point after rotation
        # Credit for above:
        https://danceswithcode.net/engineeringnotes/rotations_in_2d/rotations_in_2d.html

        Lets make this faster:
            In matrix form, this would become:
                |x1|  =  |x0-xc| x |cos(theta) -sin(theta)| + |xc|
                |y1|     |y0-yc|   |sin(theta)  cos(theta)|   |yc|
            Above can be done easily through numpy


        mp = mulpoint2d(method='xy_list', coordxy=np.random.rand(2, 10))
        print(mp.points, '___', mp.npoints, '___', mp.locx, '___', mp.centroid)
        mp.rotate1(0, 0, 180)
        print(mp.points, '___', mp.npoints, '___', mp.locx, '___', mp.centroid)


        '''
        # Form the rotation matrix
        # θ = np.deg2rad(θ)
        # c, s = np.cos(θ), np.sin(θ)
        # R = np.array([[c, -s], [s, c]])
        R = self.DG(deformation='rotation', angle=-θ)
        xyc = np.array([x, y])
        if self.lean in ('ignore', 'no', 'low', 'medium'):
            for i, (x0, y0) in enumerate(zip(self.locx, self.locy), start=0):
                self.points[i].x, self.points[i].y = np.matmul((np.array([x0, y0]) - xyc), R) + xyc

        if self.lean in ('high', 'veryhigh'):
            for i, (x0, y0) in enumerate(zip(self.locx, self.locy), start=0):
                self.locx[i], self.locy[i] = np.matmul((np.array([x0, y0]) - xyc), R) + xyc

        self.recompute(flag_mids=False, flag_dist=False, flag_basics=True,
                       flag_npoint=False, flag_locxy=True, flag_centroid=True)

    def explode(self, x, y, factorx, factory):
        '''
        1. Compute distances of all point objects to (x,y)
        2. delx and dely will be stretch_factorx and stretch_factory times the
        NOTES:
            1. Any negative factors will be made positive upon execution.


        mp = mulpoint2d(method='xy_list', coordxy=np.random.rand(2, 10))
        print(mp.points, '___', mp.npoints, '___', mp.locx, '___', mp.centroid)
        mp.explode(0, 0, 2, 2)
        print(mp.points, '___', mp.npoints, '___', mp.locx, '___', mp.centroid)
        '''
        d, θ = self.distance(x, y), self.angles(x, y)
        dcosang, dsinang = d*np.cos(θ)*abs(factorx), d*np.sin(θ)*abs(factory)
        for count in range(self.npoints):
            self.points[count].x = dcosang[count]
            self.points[count].y = dsinang[count]
        centroid_flag = True
        if factorx == factory:
            centroid_flag = False

        self.recompute(flag_mids=False, flag_dist=False, flag_basics=True,
                       flag_npoint=False, flag_locxy=True,
                       flag_centroid=centroid_flag)

    def shear(self, deformation='shearx', shear=None):
        '''
        shear deform the multi-point object
        OPTIONS:
            1. shearx: shear along x - axis. A +ve shear tilts left
                        vertical of fundamental rectangle right, about
                        bottom left corner
            2. sheary: shear along y - axis. A +ve shear tilts bottom
                       horizontal of fundamental rectangle up, about bottom
                       left corner
            3. pureshear/shearxy: shearx + sheary

        mp = mulpoint2d(method='xy_list', coordxy=np.random.rand(2, 10))
        print(mp.points, '___', mp.npoints, '___', mp.locx, '___', mp.centroid)
        mp.shear(deformation='shearxy', shear = 2.0)
        print(mp.points, '___', mp.npoints, '___', mp.locx, '___', mp.centroid)
        '''
        F = self.DG(deformation=deformation, shear=shear)
        if self.lean in ('ignore', 'no', 'low', 'medium'):
            for i, (x0, y0) in enumerate(zip(self.locx, self.locy), start=0):
                self.points[i].x, self.points[i].y = np.matmul(np.array([x0, y0]), F)
        elif self.lean in ('high', 'veryhigh'):
            for i, (x0, y0) in enumerate(zip(self.locx, self.locy), start=0):
                self.locx[i], self.locy[i] = np.matmul(np.array([x0, y0]), F)

        self.recompute(flag_mids=False, flag_dist=False, flag_basics=True,
                       flag_npoint=False, flag_locxy=True,
                       flag_centroid=True)

    def makegrid(self,
                 method='random',
                 gridding_technique='random',
                 sampling_technique='uniform',
                 randuni_calc='by_points',
                 nrndpnts=10,
                 char_length_mean=0.10,
                 char_length_min=0.05,
                 char_length_max=0.15,
                 n_trials=10,
                 n_iterations=10,
                 space='linear',
                 xbound=[0, 1],
                 ybound=[0, 1],
                 char_length=[0.25, 0.25],
                 angles=[0, 60],
                 bridson_sampling_k=30,
                 vis=True
                 ):
        if method == 'recgrid':
            self.__grid_rec(xbound,
                            ybound,
                            char_length,
                            space,
                            throw=False)
        elif method == 'hexgrid':
            pass
        elif method == 'trigrid1':
            self.__grid_tri(method,
                            angles,
                            xbound,
                            ybound,
                            char_length,
                            space,
                            throw=False)
        elif method == 'trigrid2':
            self.__grid_tri(method,
                            angles,
                            xbound,
                            ybound,
                            char_length,
                            space,
                            throw=False)
        elif method == 'random':
            if gridding_technique == 'random':
                if sampling_technique == 'uniform':
                    # uniformly distributed set of points
                    self.__grid_rnd(sampling_technique=sampling_technique,
                                    randuni_calc=randuni_calc,
                                    xbound=self.xbound,
                                    ybound=self.ybound,
                                    nrndpnts=nrndpnts,
                                    char_length_mean=char_length_mean,
                                    char_length_min=char_length_min,
                                    char_length_max=char_length_max,
                                    n_trials=n_trials,
                                    n_iterations=n_iterations,
                                    )
                elif sampling_technique == 'normal':
                    # points distributed with a normal distribution
                    pass
                elif sampling_technique == 'custom':
                    # points distributed with a custom distribution
                    pass
                elif sampling_technique == 'dart':
                    self.__grid_rnd(sampling_technique='dart',
                                    xbound=self.xbound,
                                    ybound=self.ybound,
                                    sampling_radius=char_length[0],
                                    sampling_k=bridson_sampling_k
                                    )
            elif gridding_technique == 'pds':
                self.__grid_rnd(sampling_technique=sampling_technique,
                                xbound=self.xbound,
                                ybound=self.ybound,
                                sampling_radius=char_length[0],
                                sampling_k=bridson_sampling_k,
                                )
        elif method == 'gradient':
            pass
        # ******************************
        # make constituent point objects as needed
        if self.lean in ('no', 'low', 'medium'):
            self.points = deque(point2d(x=_x, y=_y)
                                for (_x, _y) in zip(self.locx, self.locy))
        elif self.lean in ('high', 'veryhigh'):
            self.points = None

    def __grid_rec(self, xbound, ybound, char_length, space, throw):
        _x = np.linspace(xbound[0],
                         xbound[1],
                         int(abs(xbound[1]-xbound[0])/char_length[0]))
        _y = np.linspace(ybound[0],
                         ybound[1],
                         int(abs(ybound[1]-ybound[0])/char_length[1]))
        if space == 'linear':
            x, y = np.meshgrid(_x, _y)
        elif space == 'quadratic':
            _x *= np.sign(_x) * _x
            _y *= np.sign(_y) * _y
            # Treat left and right halves seperately
            _x = (_x < 0)*_x/abs(xbound[0]) + (_x >= 0)*_x/abs(xbound[1])
            _y = (_y < 0)*_y/abs(ybound[0]) + (_y >= 0)*_y/abs(ybound[1])
            # Make a grid
            x, y = np.meshgrid(_x, _y)
        elif space == 'logorithmic':
            # https://numpy.org/doc/stable/reference/generated/numpy.logspace.html
            pass
        elif space == 'euler':
            # TO AID ORIENTATION CLASS
            pass
        __xravel, __yravel = np.ravel(x), np.ravel(y)
        self.locx, self.locy = __xravel, __yravel
        if throw:
            return __xravel, __yravel

    def __grid_tri(self,
                   method,
                   angles,
                   xbound,
                   ybound,
                   char_length,
                   space,
                   throw=False):
        '''
        Make a triangular grid
        '''
        if method == 'trigrid1':
            _rad = np.radians
            # c, s  = np.cos(_rad(angles[0])), np.sin(_rad(angles[0]))
            x1 = np.arange(xbound[0], xbound[1], char_length[0])  # Row 1 - x
            x2 = x1 + char_length[0]*np.cos(_rad(angles[1]))  # Row 2 - x
            xi = np.vstack((x1, x2))  # Row 1 and Row 2 - x
            y1 = ybound[0]*np.ones(np.shape(x1))  # Row 1 - y
            y2 = y1 + char_length[1]*np.sin(_rad(angles[1]))  # Row 2 - y
            yi = np.vstack((y1, y2))  # Row 1 and Row 2 - y
            x, y = np.copy(xi), np.copy(yi)
            ncopies = int((ybound[1]-ybound[0])
                          / (0*char_length[1]
                             + 2*char_length[1]
                             * np.sin(np.radians(angles[1]))))
            # NEXT LINE: Just a constant to avoid being calculated in
            __k = 2*char_length[1]*np.sin(_rad(angles[1]))
            # every iteration below
            for count in range(ncopies):
                x = np.vstack((x, xi))
                y = np.vstack((y, yi+(count+1)*__k))
            __xravel, __yravel = np.ravel(x), np.ravel(y)
            # Pass it to attribute
            self.locx, self.locy = __xravel, __yravel
            if throw:
                return __xravel, __yravel
        elif method == 'trigrid2':
            # x = np.linspace(xmin, xmax, nx)
            # xincr = x[1] - x[0]
            # y = np.linspace(ymin, ymax, ny)
            # __xy = np.meshgrid(x, y)
            # x, y = __xy[0], __xy[1]
            # for incr_loc in np.arange(1, np.shape(x)[0], 2):
            #     x[incr_loc] += xincr*0.5
            # x_flat = x.flatten()
            # y_flat = y.flatten()
            pass

    def __grid_rnd(self,
                   sampling_technique='uniform',
                   xbound=[0, 1],
                   ybound=[0, 1],
                   randuni_calc='by_points',
                   nrndpnts=10,
                   char_length_mean=0.10,
                   char_length_min=0.05,
                   char_length_max=0.15,
                   n_trials=10,
                   n_iterations=10,
                   space='linear',
                   sampling_radius=0.025,
                   sampling_k=30,
                   throw=False
                   ):

        if sampling_technique == 'uniform':
            # ------------------------------
            # initiate the numpy>random>uniform method
            npruni = np.random.uniform
            # ------------------------------
            # Calculate the coordinate valeus
            _locx = list(npruni(low=xbound[0],
                                high=xbound[1],
                                size=(nrndpnts,)))
            _locy = list(npruni(low=ybound[0],
                                high=ybound[1],
                                size=(nrndpnts,)))
            # -----------------------------
            if randuni_calc == 'by_points':
                # nothing more to do here.
                # _locx and _locy have already been calculated
                pass
            # -----------------------------
            if randuni_calc == 'by_char_length':
                '''
                ITERATIVELY CALCULATE RANADOM UNIFORM NUMBER DISTRIBUTION
                UNTIL THE MEAN SEPERATION DISTANCE FALLS IN THE RANGE
                DEFINED BY THE 3 CHARACTERISTIC LENGTH VALUES, WHICH ARE,
                     1> char_length_mean - TO BE CONSIDERED LATER
                     2> char_length_min
                     3> char_length_max

                ALGORITHM:
                STEP 1: Calculate minimum, mean and maximum distances
                        from ckdtree of _locx and _locy
                STEP 2: If this
                '''
                # if hasattr(self, 'tree'):
                #     if len(self.tree.data) != self.npoints:
                #         self.maketree(treeType = 'ckdtree')
                # sparse_dist_matrix = m.tree.sparse_distance_matrix(m.tree, 0.0).toarray()

                # npmin = np.min
                # npmax = np.max
                npmean = np.mean
                # rng = np.random.default_rng()
                # points1 = rng.random((5, 2))
                # points2 = points1
                point_coords = np.vstack((_locx, _locy)).T
                from scipy.spatial import distance_matrix
                _distances = set(distance_matrix(point_coords,
                                                 point_coords).ravel())
                # INDLCUE EXCEPTION HANDLING HERE FOR VERY VERY
                # VERY SMALL VALUES
                _distances.remove(0.0)
                _distances = np.array(list(_distances), dtype=float)
                # _dmin = npmin(_distances)
                # _dmean = npmean(_distances)
                # _dmax = npmax(_distances)

                # _min_flag, _mean_flag, _max_flag = False, False, False
                # if self.is_within(input_value = _dmin,
                #                   reference_value = char_length_min,
                #                   percentage = 10,
                #                   method = '-+',
                #                   ):
                #     _min_flag = True
                # if self.is_within(input_value = _dmean,
                #                   reference_value = char_length_mean,
                #                   percentage = 10,
                #                   method = '-+',
                #                   ):
                #     _mean_flag = True
                # if self.is_within(input_value = _dmax,
                #                   reference_value = char_length_max,
                #                   percentage = 10,
                #                   method = '-+',
                #                   ):
                #     _max_flag = True
                print('==========================')
                print(_min_flag, _mean_flag, _max_flag)
                print('==========================')
                print('Qualification metric values:')
                print(char_length_min, char_length_mean, char_length_max)
                print('Metric value performance value:')
                print(_dmin)#, _dmean, _dmax)
                print('==========================')
            # -----------------------------
            self.locx, self.locy = _locx, _locy
            # -----------------------------
            if throw:
                return _locx, _locy

        if sampling_technique == 'bridson1':
            # pds: Poisson Disk Sampling
            xstart, ystart = xbound[0], ybound[0]
            # xend, yend = xbound[1], ybound[1]
            from stat_sampling01 import bridson1
            _points = bridson1(width=xbound[1]-xbound[0],
                               height=ybound[1]-ybound[0],
                               radius=sampling_radius,
                               k=sampling_k
                               )
            _locx = [_[0]+xstart for _ in _points]
            _locy = [_[1]+ystart for _ in _points]

            self.locx, self.locy = _locx, _locy

            if throw:
                return _locx, _locy

        if sampling_technique == 'dart':
            from stat_sampling01 import dart
            xstart, ystart = xbound[0], ybound[0]
            _points = dart(width=xbound[1]-xbound[0],
                           height=ybound[1]-ybound[0],
                           radius=sampling_radius,
                           k=sampling_k
                           )
            _locx = [_[0]+xstart for _ in _points]
            _locy = [_[1]+ystart for _ in _points]

            self.locx, self.locy = _locx, _locy

            if throw:
                return _locx, _locy

    def is_within(self,
                  input_value: float = None,
                  reference_value: float = None,
                  percentage: float = 10.0,
                  method: str = '-+',
                  ):
        # ************
        factor = percentage/100
        # ************
        if method == '-+':
            lowerbound = reference_value*(1-factor)
            upperbound = reference_value*(1+factor)
        elif method == '.+':
            lowerbound = reference_value
            upperbound = reference_value*(1+factor)
        elif method == '-.':
            lowerbound = reference_value*(1-factor)
            upperbound = reference_value
        # ************
        if input_value >= lowerbound and input_value <= upperbound:
            return True
        else:
            return False

    def assign_rid(self,
                   flag: bool = False,
                   idlength: int = 4):
        '''
        Assign a randomly generated ID.
        Unchanging ID for the instance's life.
        # CAUTION: Do not provide something like a change_rid function
                   down the lane.
        '''
        if flag:
            try:
                print(self.rid)
            except AttributeError:
                import string, random
                # string = string.ascii_uppercase +...
                # string.ascii_lowercase + string.digits +...
                # string.punctuation
                string = ''.join([string.ascii_uppercase,
                                  string.ascii_lowercase,
                                  string.digits,
                                  string.punctuation])
                self.rid = ''.join(random.choice(string)
                                   for _ in range(idlength))
        else:
            pass

    def build_global_rid(self):
        try:
            _ = self.rid
        except AttributeError:
            try:
                _ = self.points[0].rid
            except AttributeError:
                return [self.rid+p.rid for p in self.points]

    def unpack(self):
        '''
        Behaviour should change after bringing in the capability of this class
        being able to store many multi-point objects. In which case,
        this should return a single list of all points across all its
        point constituent objects.

        NOTE: "point constituent objects" could be either point2d objects or
               mulpoint2d objects

        DEVELOPMENT DECISION @ 12 October 2022:
            It may not be necessary to do this!, given the below restriction
            to be imposed
        RESTRICTION @ 12 October 2022:
            A mupoint2d object cannot and should not hold more than one sub-
            mulpoint2d object.
        '''
        return self.points

    def partition(self,
                  into: str = 'xtals_by_state',
                  ):
        '''
        Break up this multi-point object into multiple
        multi-point objects based on input conditions

        If "xtals_by_state": Useful for pixellated grain structure: either
            VTGS.2d or MCGS.2d
        '''
        pass

    def clean(self,
              depth=0
              ):
        '''
        OPERATIONS:
            (1) Remove duplicates
            (2) Removes all attributes except:
                    (a) self.locx, self.locy
        '''
        if depth == 0:
            # Remove duplicates
            if self.lean in ('no', 'low', 'medium'):
                self.recompute_from_mids()
                # =================================
                self.recompute_from_dist(method='points_normal',
                                         tolerance_distance=self.__POINT_TDIST__)
                # =================================
            if self.lean in ('ignore', 'high', 'veryhigh'):
                self.recompute_from_dist(method='coord_normal',
                                         tolerance_distance=self.__POINT_TDIST__)
        elif depth == 1:
            # clean at depth 0
            # Remove self.npoints and self.centroid
            pass
        elif depth == 2:
            # clean at depth 0
            # clean at depth 1
            # Remove self.scalars
            pass
        elif depth == 3:
            # clean at depth 0
            # clean at depth 1
            # clean at depth 2
            # Retain only self.locx and self.locy
            pass
    # -----------------------------------------------
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # END OF DUNDER METHODS
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # START OF CKD TREE METHODS
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    '''
    Use tree structure to deal with a very large system of points
    '''
    def maketree(self,
                 treeType: str = 'ckdtree',
                 saa: bool = True,
                 throw: bool = False
                 ):
        if treeType in ('ckdtree', 'kdtree'):
            # Scipy ckdtree
            from scipy.spatial import cKDTree as ckdt
            # Make the tree data-structure
            _tree = ckdt([[_x, _y] for (_x, _y) in zip(self.locx, self.locy)],
                         copy_data=False,
                         balanced_tree=True
                         )
            if saa:
                self.tree = _tree
            if throw:
                return _tree

    def neigh_pair_count(self,
                         other=None,
                         cor: float = 0.1,
                         ):
        '''
        Count how many nearby pairs can be formed.
        '''
        try:
            _ = self.tree
        except AttributeError:
            self.maketree(treeType='ckdtree')
        # ***********
        if other is None:
            return self.tree.count_neighbors(other=self.tree, r=cor, p=2)

    def neigh_points(self,
                     method: str = 'coord',
                     coord_list: list[list] = [[0, 0]],
                     point_object_list: list = None,
                     cor: list = [0.1],
                     mpnorm: int = 2,
                     workers: int = 1,
                     return_sorted: bool = False,
                     return_length: bool = False,
                     vis: bool = True
                     ):
        if method in ('coord', 'point'):
            try:
                _ = self.tree
            except AttributeError:
                self.maketree(treeType='ckdtree')
        # *********************************
        if method == 'coord':
            indices = [self.tree.query_ball_point(i,
                                                  r=_cor,
                                                  p=mpnorm,
                                                  workers=workers,
                                                  return_sorted=return_sorted,
                                                  return_length=return_length,
                                                  )
                       for (i, _cor) in zip(coord_list, cor)
                       ]
        elif method == 'point':
            indices = [self.tree.query_ball_point([point_object_list[i].x,
                                                   point_object_list[i].y],
                                                  r=cor,
                                                  p=mpnorm,
                                                  workers=workers,
                                                  return_sorted=return_sorted,
                                                  return_length=return_length,
                                                  )
                       for i in range(len(point_object_list))
                       ]
        # *********************************
        if vis:
            visopt = ('.', 0.5, 'k', 5, 'maroon', 1.0)
            fig = plt.figure(figsize=(1.6, 1.6),
                             dpi=100)
            left, right = min(self.locx), max(self.locx)
            down, up = min(self.locy), max(self.locy)
            ax = fig.add_axes([0, 0, right-left, up-down])
            ax.plot(self.locx,
                    self.locy,
                    marker=visopt[0],
                    color='none',
                    markeredgewidth=visopt[1],
                    markeredgecolor=visopt[2],
                    markersize=visopt[3],
                    markerfacecolor=visopt[4],
                    alpha=visopt[5]
                    )
            for ind in indices:
                for i in ind:
                    ax.plot(self.locx[i],
                            self.locy[i],
                            marker=visopt[0],
                            color='none',
                            markeredgewidth=visopt[1],
                            markeredgecolor=visopt[2],
                            markersize=2*visopt[3],
                            alpha=visopt[5]
                            )
            ax.set_xlim([left, right])
            ax.set_ylim([down, up])
        return indices

    def distances(self, x=0, y=0):
        '''
        xy_list = [[0,0,0,0,-1,-2,-2,-2,-1,0], [0,1,2,3,1,1,0,-1,-1,-1]]
        mp = mulpoint2d(method='xy_list', coordxy=xy_list)
        mp.points
        mp.distances(-1, 0)
        '''
        return np.sqrt((np.array(self.locx)-x)**2+(np.array(self.locy)-y)**2)

    def angles(self, x=0, y=0):
        '''
        xy_list = [[0,0,0,0,-1,-2,-2,-2,-1,0], [0,1,2,3,1,1,0,-1,-1,-1]]
        mp = mulpoint2d(method='xy_list', coordxy=xy_list)
        mp.points
        mp.angles(-1, 0)
        '''
        return np.rad2deg(np.arctan2(np.array(self.locy)-y,
                                     np.array(self.locx)-x)
                          )

    def merge(self,
              retain='first',
              ):
        '''
        #TODO
        Merge two or more points based on proximity or overlap condition
        '''
        pass

    def noise(self,
              perturb_flag=False,
              perturb_type='local_uniform',
              depth=None,
              ground=None,
              height=None,
              perturb_mag=None
              ):
        if perturb_type == 'dgh':
            # Depth, Ground and Height are lengths
            depth, ground, height = -abs(depth), abs(ground), abs(height)
            noise_x = ground + np.random.uniform(depth, height,
                                                 (self.npoints,))
            noise_y = ground + np.random.uniform(depth, height,
                                                 (self.npoints,))
            if self.lean in ('no', 'low', 'medium'):
                for count in range(self.npoints):
                    self.points[count].x += noise_x[count]
                    self.points[count].y += noise_y[count]
                self.recompute_basics(npoints_flag=False,
                                      locx_locy_flag=True,
                                      centroid_flag=True,
                                      )
            elif self.lean in ('high', 'veryhigh'):
                # for count in range(self.npoints):
                # self.locx[count] += noise_x[count]
                # self.locy[count] += noise_y[count]
                self.locx += noise_x
                self.locy += noise_y
                self.recompute_basics(npoints_flag=False,
                                      locx_locy_flag=False,
                                      centroid_flag=True,
                                      )
        if perturb_type == 'local_uniform':
            # TODO: CONSIDER NEGATIVE POINTS AS WELL.
            xpert = perturb_mag[0]*np.random.random(self.npoints)
            ypert = perturb_mag[1]*np.random.random(self.npoints)
            try:
                _ = self.xpert
            except AttributeError:
                self.xpert, self.ypert = xpert, ypert
            else:
                self.xpert += xpert
                self.ypert += ypert
            self.locx = np.array([_x+__x for (_x, __x) in zip(self.locx,
                                                              self.xpert)])
            self.locy = np.array([_y+__y for (_y, __y) in zip(self.locy,
                                                              self.ypert)])
            # Above two lines of code will replace the following two
            # commented lines. This is because, deque does not support
            # element wise addition.
            # self.locx += self.xpert
            # self.locy += self.ypert
            if self.lean in ('no', 'low', 'medium'):
                for count in range(self.npoints):
                    self.points[count].x = self.locx[count]
                    self.points[count].y = self.locy[count]
                self.recompute_basics(npoints_flag=False,
                                      locx_locy_flag=False,
                                      centroid_flag=True,
                                      )
            elif self.lean in ('high', 'veryhigh'):
                self.recompute_basics(npoints_flag=False,
                                      locx_locy_flag=False,
                                      centroid_flag=True,
                                      )
        if perturb_type == 'local_normal':
            self.xpert = perturb_mag[0][0]*np.random.normal(perturb_mag[0][1],
                                                            perturb_mag[0][2],
                                                            self.npoints
                                                            )
            self.ypert = perturb_mag[1][0]*np.random.normal(perturb_mag[1][1],
                                                            perturb_mag[1][2],
                                                            self.npoints)
            self.locx = np.array([_x+__x for (_x, __x) in zip(self.locx,
                                                              self.xpert)])
            self.locy = np.array([_y+__y for (_y, __y) in zip(self.locy,
                                                              self.ypert)])
            # Above two lines of code will replace the following two
            # commented lines. This is because, deque does not support
            # element wise addition.
            # self.locx += self.xpert
            # self.locy += self.ypert
            if self.lean in ('no', 'low', 'medium'):
                for count in range(self.npoints):
                    self.points[count].x = self.locx[count]
                    self.points[count].y = self.locy[count]
                self.recompute_basics(npoints_flag=False,
                                      locx_locy_flag=False,
                                      centroid_flag=True,
                                      )
            elif self.lean in ('high', 'veryhigh'):
                self.recompute_basics(npoints_flag=False,
                                      locx_locy_flag=True,
                                      centroid_flag=True,
                                      )

    def relax(self,
              iterations=2,
              bounds=None,
              options='Qbb Qc Qx',
              ):
        '''
        # TODO: MUST CLEAN THIS UP AND MAKE IT UPXO COMPATIBLE
        Credit: see below link
        https://vedo.embl.es/autodocs/content/vedo/pointcloud.html#vedo.pointcloud.Points.smoothLloyd2D
        '''
        # def smoothLloyd2D(self, iterations=2, bounds=None,
        #             options='Qbb Qc Qx'):
        """Lloyd relaxation of a 2D pointcloud."""
        # Credits: https://hatarilabs.com/ih-en/
        # tutorial-to-create-a-geospatial-voronoi-sh-mesh-with-python-scipy-and-geopandas
        from scipy.spatial import Voronoi as scipy_voronoi

        def _constrain_points(points):
            #Update any points that have drifted beyond the boundaries of this space
            if bounds is not None:
                for point in points:
                    if point[0] < bounds[0]:
                        point[0] = bounds[0]
                    if point[0] > bounds[1]:
                        point[0] = bounds[1]
                    if point[1] < bounds[2]:
                        point[1] = bounds[2]
                    if point[1] > bounds[3]:
                        point[1] = bounds[3]
            return points

        def _find_centroid(vertices):
            # The equation for the method used here to find the centroid of a
            # 2D polygon is given here:
            #       https://en.wikipedia.org/wiki/Centroid#Of_a_polygon
            area = 0
            centroid_x = 0
            centroid_y = 0
            for i in range(len(vertices)-1):
                step = (vertices[i, 0]*vertices[i+1, 1]) - \
                    (vertices[i+1, 0]*vertices[i, 1])
                centroid_x += (vertices[i, 0] + vertices[i+1, 0]) * step
                centroid_y += (vertices[i, 1] + vertices[i+1, 1]) * step
                area += step
            if area:
                centroid_x = (1.0/(3.0*area)) * centroid_x
                centroid_y = (1.0/(3.0*area)) * centroid_y
            # prevent centroids from escaping bounding box
            return _constrain_points([[centroid_x, centroid_y]])[0]

        def _relax(voronoi):
            # Moves each point to the centroid of its cell in the voronoi
            # map to "relax" the points (i.e. jitter the points so as
            # to spread them out within the space).
            centroids = []
            for idx in voronoi.point_region:
                # the region is a series of indices into voronoi.vertices
                # remove point at infinity, designated by index -1
                region = [i for i in voronoi.regions[idx] if i != -1]
                # enclose the polygon
                region = region + [region[0]]
                verts = voronoi.vertices[region]
                # find the centroid of those vertices
                centroids.append(_find_centroid(verts))
            return _constrain_points(centroids)
        if bounds is None:
            bounds = self.bounds()
        pts = self.points()[:, (0, 1)]
        for i in range(iterations):
            vor = scipy_voronoi(pts, qhull_options=options)
            _constrain_points(vor.vertices)
            pts = _relax(vor)
        # return Points(pts, c='k')
        return pts
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # END OF POINT REORDERING METHODS
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # START OF RECOMPUTE METHODS
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    def recompute_from_mids(self):
        '''
        mid: ID of the object's memory address'
        NOTE TO DEVELOPER: LEAVE THIS AS IS
        '''
        if self.lean in ('ignore', 'no', 'low', 'medium'):
            mids = [id(pobj) for pobj in self.points]  # object memory IDs
            if len(mids) > len(list(set(mids))):
                # make the mids unique and unsorted
                mids = list(dict.fromkeys(mids))
                self.unique_points_mids(mids)
                self.__state_change = True
            if self.__state_change:
                self.recompute_basics(npoints_flag=True,
                                      locx_locy_flag=True,
                                      centroid_flag=True,
                                      )
            # CAUTION: DO NOT CHANGE EXECUTION ORDER
            self.__state_change = False
        elif self.lean in ('high', 'veryhigh'):
            pass

    def recompute_from_dist(self,
                            method='points_normal',
                            tolerance_distance=0.0
                            ):
        '''
        DESCRIPTION: recompute() using point2d.__eq__(query_point_object)
        # WHEN TO USE:
            Use when system size is small.
            Exact size threshild YET to be determined
        '''
        # ============================
        self.__state_change = list(self.unique_points_dist(method=method,
                                                           tolerance_distance=tolerance_distance)
                                   )
        # ============================
        if self.__state_change[0]:
            self.recompute_basics(npoints_flag=True,
                                  locx_locy_flag=True,
                                  centroid_flag=True,
                                  )
        # CAUTION: DO NOT CHANGE EXECUTION ORDER
        self.__state_change[0] = False
        self.__state_change = tuple(self.__state_change)

    def unique_points_mids(self, mids):
        '''
        In this case, update op happens by comparing memory ID of point objects
        NOTE: For the forseable future, this method is to be used until its use
        been validated against.
        '''
        # mids = [id(pobj) for pobj in self.points] # object memory IDs
        # if len(mids) > len(list(set(mids))):
        # Choose numpy if number of points are more than 'npoints = 1000'
        if self.npoints > 1000:
            mids = np.array(mids)
        # initate _points as an empty list
        __points = []
        # Append it with unique points
        for key in mids:
            for pobj in self.points:
                if id(pobj) == key:
                    break
            __points.append(pobj)
        # Replace self.points with the new list
        self.points = __points

    def unique_points_dist(self,
                           method='points_normal',
                           tolerance_distance=0.0
                           ):
        '''
        Author: Dr. Sunil Anandatheertha, UKAEA
        Identify unique points as the first of the list of points which
        have a zero-distance in case of repeating points
            NOTE: zero-distance is decided by the tolerance_distance set by
                  the user

        # TODO: Enable multi-processing and speed up the process

        NOTE: LEAVE THIS METHOD AS IT IS
        '''
        if method == 'points_normal':
            __points = []
            repeated = [False for _ in self.points]
            count1 = 0
            for pobj_1 in self.points:
                if not repeated[count1]:
                    count2 = 0
                    for pobj_2 in self.points:
                        if count1 != count2:
                            if self.points[count1].__eq__(pobj_2,
                                                          tdist=tolerance_distance,
                                                          use_self_tdist=False):
                                repeated[count2] = True
                        count2 += 1
                count1 += 1
            if True in repeated:
                for (_repeat, _pobj) in zip(repeated, self.points):
                    if not _repeat:
                        __points.append(_pobj)
                self.points = __points
                return True, None, None
            else:
                return False, None, None
        elif method == 'coord_normal':
            __locx, __locy = [], []
            repeated = [False for _ in self.locx]
            for count1, (_x, _y) in enumerate(zip(self.locx,
                                                  self.locy), start=0):
                if not repeated[count1]:
                    for count2, (__x, __y) in enumerate(zip(self.locx,
                                                            self.locy),
                                                        start=0):
                        if count1 != count2:
                            distdiff = np.sqrt((_x - __x)**2 + (_y - __y)**2)
                            if distdiff <= tolerance_distance:
                                repeated[count2] = True
            if True in repeated:
                for (_repeat, _x, _y) in zip(repeated, self.locx, self.locy):
                    if not _repeat:
                        __locx.append(_x)
                        __locy.append(_y)
                from copy import deepcopy as dcopy
                self.locx, self.locy = dcopy(__locx), dcopy(__locy)
                return True, __locx, __locy
            else:
                return False, self.locx, self.locy
        elif method == 'points_fast':
            # TODO: patch development of the tree method once complete
            pass

    def recompute_basics(self,
                         npoints_flag: bool = True,
                         locx_locy_flag: bool = True,
                         centroid_flag: bool = True
                         ):
        '''
        On a need to basis, recompute the number of points,
        x and y coordinate lists (deques) and the centroid
        '''
        if locx_locy_flag:
            self.recompute_locx_locy()
        if npoints_flag:
            self.recompute_npoints()
        if centroid_flag:
            self.recompute_centroid()

    def recompute_npoints(self):
        '''
        Recompute the number of points in this collection
        '''
        if self.lean in ('no', 'low', 'medium'):
            try:
                self.npoints = len(self.points)
            except AttributeError:
                self.npoints = len(self.locx)
        if self.lean in ('ignore', 'high', 'veryhigh'):
            self.npoints = len(self.locx)

    def recompute_locx_locy(self):
        '''
        Recompute the coordinate location lists (deques)
        '''
        if self.lean in ('ignore', 'no', 'low', 'medium'):
            # TODO: Combine the following two lines into
            #     a single list comprehension
            self.locx = np.array([pobj.x for pobj in self.points])
            self.locy = np.array([pobj.y for pobj in self.points])
        if self.lean in ('high', 'veryhigh'):
            pass

    def recompute_centroid(self,
                           accuracy: str = 'exact'):
        '''
        Recompute the centroid of the point collection

        #TODO: In the case of multi-point collection, this is to return
        a list of centroids, in the order of the multi-point list
        '''
        _n = self.npoints
        _sum = np.sum
        self.centroid = [_sum(self.locx)/_n, _sum(self.locy)/_n]

    def qh(self,
           package: str = 'scipy',
           save_as_attribute: bool = True,
           throw: bool = True
           ):
        '''
        Compute the convex hull using various known packages
        # TODO: THIS REQUIRES IMMEDIATE FURTHER DEVELOPMENT
        '''
        if package == 'scipy':
            from scipy.spatial import ConvexHull
            # TODO:
            _points = [[_x, _y] for (_x, _y) in zip(self.locx, self.locy)]
            qh = ConvexHull(_points)
            if save_as_attribute:
                self.qh = qh
            if throw:
                return qh

    def qh_find_area(self):
        '''
        # TODO: THIS REQUIRES IMMEDIATE FURTHER DEVELOPMENT
        '''
        pass

    def qh_find_perimeter(self):
        '''
        # TODO: THIS REQUIRES IMMEDIATE FURTHER DEVELOPMENT
        '''
        pass

    def mapsf_area(self):
        '''
        Attach xtal area to the xtal point. The xtal point should be the xtal
        representative point.
        '''
        pass

    def mapsf_ori(self):
        '''
        Attach orientation distribution to the point distribution
        '''
        pass

    def mapsf_gnd(self):
        '''
        Attach GND field to the point distribution
        '''
        pass

    def _summary_(self,
                  print_summary=True,
                  make_point_objects=None,
                  make_ckdtree=False,
                  sep_n=10,
                  sep_string='-X'
                  ):
        if print_summary:
            from colorama import init as colorama_init
            from colorama import Fore, Back, Style
            colorama_init()
            sep_n, sep_string = sep_n, '-X-'
            if self.mulpoint_type == 'seed':
                _ = '(default)'
            else:
                _ = ''
            print(''.join([Fore.RED, sep_n*sep_string, '\n',
                           Fore.WHITE, f'Summary of: {self}', '\n\n',
                           Fore.WHITE, f'Multi-point type: {self.mulpoint_type} {_}', '\n',
                           Fore.WHITE, f'No. of points = {self.npoints}', '\n',
                           Fore.WHITE, f'Point objects exit: {make_point_objects}','\n\n',
                           Fore.WHITE, f'Centroid: {self.centroid[0], self.centroid[1]}',
                           ]
                          )
                  )
            if make_ckdtree:
                string = string.join([f'', '\n',
                                      ])
            # ----------------------
            # string = string.join([Fore.RED, sep_n*sep_string, '\n'])
            # ----------------------
            print(''.join([Fore.RED, sep_n*sep_string, '\n']))
            Style.RESET_ALL
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # START OF VIS METHODS
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def plot(self,
             flag: bool = True,
             dpi: int = 75,
             source: str = 'mulpoint_all',
             xbound: list = [0, 1],
             ybound: list = [0, 1]
             ):
        '''
        self.vizopt = ('o', 0.5, 'k', 10, 'w', 0.8)
        IN EXACT ORDER:
            o   : circle marker style
            0.5 : Marker edge line width
            k   : Marker edge color
            w   : marker face colour white
            10  : marker size
            0.8 : marker face colour alpha
        '''
        if flag:
            visopt = ('.', 0.5, 'k', 10, 'maroon', 1.0)
            fig = plt.figure(figsize=(1.6, 1.6), dpi=dpi)
            left, right = min(self.locx), max(self.locx)
            down, up = min(self.locy), max(self.locy)
            ax = fig.add_axes([0, 0, right-left, up-down])

            if source == 'mulpoint_all':
                ax.plot(self.locx,
                        self.locy,
                        marker=visopt[0],
                        color='none',
                        markeredgewidth=visopt[1],
                        markeredgecolor=visopt[2],
                        markersize=visopt[3],
                        markerfacecolor=visopt[4],
                        alpha=visopt[5]
                        )
            ax.set_xlim([left, right])
            ax.set_ylim([down, up])
            plt.show()

    def perthist(self):
        fig = plt.figure(figsize=(1.6, 1.6), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.hist(self.xpert, bins=25, label='x-pert',
                edgecolor='k', facecolor='teal', alpha=0.5)
        ax.hist(self.ypert, bins=25, label='y-pert',
                edgecolor='k', facecolor='orange', alpha=0.5)
        ax.legend(loc='upper left', shadow=False,
                  fontsize='small', edgecolor='k')
        # ax.set_xlim([-1, +1])
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # END OF VIS METHODS
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
