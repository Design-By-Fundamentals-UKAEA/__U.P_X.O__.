'''
UPXO is a python package to generate, analyze, manipulate, visualize, mesh and
export heirarchical multi-instanced spatially gradient multi-phase
poly-crystalline microstructures in 2 and 3 dimensions.

This module is a core to UPXO and has the following classes:
    point2d, point3d
    point2d_lean_highest, point3d_lean_highest
    point2d_lean_highest_mc0, point3d_lean_highest_mc0
    point2d_lean_highest_mc1, point3d_lean_highest_mc1
    point_q_space, point_be_space, point_gsm_stat_space

NOTE: NOT TO BE SHARED WITH ANYONE OTHER THAN:
    *@UKAEA: Vaasu Anandatheertha, Chris Hardie, Vikram Phalke
    *@UKAEA:  Ben Poole, Allan Harte, Cori Hamelin
    *@OX,UKAEA:  Eralp Demir, Ed Tarleton
'''

# from numpy.random import uniform as nprandu
from .._sup import dataTypeHandlers as dth
import matplotlib.pyplot as plt
import numpy as np
import math
from math import ceil, floor, radians, sin, cos
import cProfile
from ..geoEntities import pops
# import pops
# import random
__name__ = "UPXO-point"
__lead_developers__ = ["Dr. Vaasu Anandatheertha"]
__developers__ = ["Vaasu Anandatheertha (vaasu.anandatheertha@ukaea.uk)",
                  ]
__maintainers__ = ["Vaasu Anandatheertha (vaasu.anandatheertha@ukaea.uk)",
                   ]
__version__ = ["0.1.upto.271022.git-no", "0.2.from.281022.git-no",
               "0.3.from.031122.git-no", "0.4.from.111122.git-no",
               "0.5.from.021222.git-no", "0.6.from.081222.git-no",
               "0.7.from.221222.git-no", "0.8.from.200623.git-no"
               ]
__license__ = "GPL v3"

class point2d():
    '''Class representing point2d object in physical space.
    PROFILING DETAILS
    -----------------
    Number of runs = 10000
    Sl | Detail | Cum.Time | No. Func. Calls
    -- | ------ | -------- | ---------------
    01 | Instantiation-ignore  | 0.012 | 20001
    02 | Instantiation-low     | 0.012 | 20001
    03 | Instantiation-medium  | 0.010 | 20001
    04 | Instantiation-high    | 0.006 | 20001
    05 | Instantiation-highest | 0.005 | 20001

    '''
    """
    Decisions - point2d class
    1. ROUND_ZERO_DEC_PLACE - DONE. removed
    2. EPS_above - DONE. removed
    3. EPS_below - DONE. removed
    4. EPS_left - DONE. removed
    5. EPS_right - DONE. removed
    6. EPS_divisor - DONE. removed
    7. EPS_rotate - DONE. removed
    8. angle - DONE. removed
    9. angles - DONE. removed
    10. Put [loc, ptype, jn, phase_id, phase_name, tcname] inside a "state" dictionary - DONE.
    11. Put [_mulpoints_, _edges_, __muledges__, _xtals_, _polyxtals_, orientation_object] inside a "links" dictionary - DONE.
    12. vprop - can be removed? - DONE. removed
    13. Put [mesh_lc] inside a "msh" dictionary
    14. Put [sfv_pol_area, sfv_pol_perimeter, sfv_pol_ar, sfv_repr_ea, sfv_ea] inside a "sfv" dictionary slot
    15. Reduce functions calls
    16. Remove slot for pixels - DONE. removed
    17. Move pixellation method to a new pops module
    18. Remove dim - DONE. removed
    """
    EPS = 0.000000000001
    __slots__ = ('lean',  # Complexity branching
                 'mid',  # memory id of the point
                 'x', 'y',  # x and y - coordinates
                 'state',  # Dict of ptype, jn, phase_id, phase_name, tcname
                 'tdist',  # for use in comparison operations
                 'links',  # Dict of linked UPXO objects
                 '_original_location',  # original x and y
                 'data',  # Dict of Misc data
                 )

    def __init__(self,
                 x: float = 0.0,
                 y: float = 0.0,
                 lean: str = 'ignore',
                 tdist: float = 0.0000000000001,
                 dim: int = 2,
                 ptype: str = 'vt2dseed',
                 jn: int = 3,
                 loc: str = 'interior',
                 store_original_coord: bool = False,
                 phase_id: int = 1, phase_name: str = 'cucrzr',
                 tcname: str = 'B',
                 ea: list = [45, 35, 0],
                 orientation_object: object = None,
                 ):
        '''
        # TODO: Replace rare input arguments as kwargs

        point2d object defines a UPXO geometric location in 2D physical space
        Notes - 1
        ---------
        In all the below, the following holds:
            @user: something which user has to provide as an input
            @upxo: internally done by upxo. there is no need for
                   the user to remember
            @upxo, user: something which the user may also provide.
        '''
        self.x, self.y = float(x), float(y)
        self.lean = lean  # FOR ABOVE LINE: self.set_lean(lean)
        # --------------------------------------------------
        if lean == 'ignore':
            '''Do everything possible'''
            self.mid = id(self)  # FOR ABOVE LINE: self.make_mid()
            self.tdist = tdist  # FOR ABOVE LINE: self.set_tdist(tdist)
            self._original_location = (self.x, self.y)
            self.state = {'loc': loc,  # self.set_location(loc=loc)
                          'ptype': ptype,  # self.set_ptype(ptype=ptype)
                          'jn': jn,  # self.set_jn(jn=jn)
                          'angle': 0.0,  # Rotation about itself
                          'phase_id': phase_id,  # See next comment
                          'phase_name': phase_name,  # See next comment
                          'tcname': tcname  # self.set_tcname(tcname)
                          }
            # self.set_phase(phase_id=phase_id, phase_name=phase_name)
            self.links = {'me': [],  # Multi-edge objects
                          }
            self.data = {'mesh': {'lc': None,
                                  },
                         'sdv_gnd': [],  # Geo Necess Dislocation matrix
                         'sdv_rss': [],  # Resolved shear stress
                         'sdv_st': [],  # Schmidt tensor
                         }
        elif lean == 'low':
            '''
            Relative to lean = 'ignore', following holds:
                no angle in self.state
                no sdv related data im self.data
            '''
            self.mid = id(self)
            self.tdist = tdist
            self._original_location = (self.x, self.y)
            self.state = {'loc': loc,
                          'ptype': ptype,
                          'jn': jn,
                          'phase_id': phase_id,
                          'phase_name': phase_name,
                          'tcname': tcname
                          }
            self.links = {'me': [],
                          }
            self.data = {'mesh': {'lc': None,
                                  },
                         }
        elif lean == 'medium':
            '''
            Relative to lean = 'ignore', following holds:
                no angle in self.state
                no sdv related data im self.data
                no loc in self.state
                no phase_name in self.state
                no _original_location
            '''
            self.mid = id(self)
            self.tdist = tdist
            self.state = {'ptype': ptype,
                          'jn': jn,
                          'phase_id': phase_id,
                          'tcname': tcname
                          }
            self.links = {'me': [],
                          }
            self.data = {'mesh': {'lc': None,
                                  },
                         }
        elif lean == 'high':
            '''
            Relative to lean = 'ignore', following holds:
                no angle in self.state
                no sdv related data im self.data
                no loc in self.state
                no phase_name in self.state
                no _original_location
            '''
            self.tdist = tdist
            self.state = {'jn': jn,
                          'phase_id': phase_id,  # See next comment
                          'tcname': tcname
                          }
        elif lean == 'highest':
            '''
            Relative to lean = 'ignore', following holds:
                no other slots assigned valus other than x, y, lean and tdist
            '''
            self.tdist = tdist

    def __call__(self, x, y):
        '''Call point object instances like functions'''
        self.x, self.y = x, y

    def __eq__(self,
               point_objects: list = None,
               tdist: float = 0.0,
               use_self_tdist: bool = True,
               point_types: str = 'upxo',
               edge_types: str = ''
               ):
        '''Check if self is coincident with "point_objects" (see attr. below).

        Parameters
        ----------
        point_objects : list / tuple / deque
            A list / tuple / deque of point objects
        tdist : float
            User value for tolerance distance.
        use_self_tdist : bool
            If True, `self.tdist` will be used as tolerance distance, else
            user value for `tdist` will be used for tolerance distance

        Returns
        -------
        True if distance b/w self and points is <= tolerance distance,
        defined in "tdist"

        Examples
        --------
        >>> from point2d import point2d
        >>> p1 = point2d(x = 0.0, y = 0.0)
        '''
        if use_self_tdist:
            tdist = self.tdist
        if point_types == 'upxo':
            if isinstance(point_objects, point2d):
                if self.distance(otype='point2d',
                                 obj=point_objects) <= tdist:
                    return True
                else:
                    return False
            elif type(point_objects) in dth.dt.ITERABLES:
                to_return = [False for pobjcount in range(len(point_objects))]
                for pobjcount in range(len(point_objects)):
                    if self.distance(otype='point2d',
                                     obj=point_objects[pobjcount]) <= tdist:
                        to_return[pobjcount] = True
                return tuple(to_return)
            else:
                return 'ERROR: Only point2d instance or tuple of them allowed'
        elif point_types == 'shapely':
            pass
        elif point_types == 'vtk':
            pass
        elif point_types == 'pyvista':
            pass
        elif point_types == 'gmsh':
            pass

    def __ne__(self,
               point_objects=None,
               tdist=0.1,
               use_self_tdist=True,
               point_type: str = 'upxo'
               ):
        '''
        True if distance b/w self and point_object is GT tolerance distance
        '''
        if use_self_tdist:
            tdist = self.tdist
        if isinstance(point_objects, point2d):
            if self.distance(other_object_type='point2d',
                             point_data=point_objects) > tdist:
                return True
            else:
                return False
        elif isinstance(point_objects, list):
            to_return = [False for pobjcount in point_objects]
            for i, point in enumerate(point_objects):
                if self.distance(point) > tdist:
                    to_return[i] = True
            # to_return = [False for pobjcount in range(len(point_objects))]
            # for pobjcount in range(len(point_objects)):
            #    if self.distance(point_objects[pobjcount]) > tdist:
            #        to_return[pobjcount] = True
            return tuple(to_return)
        else:
            return 'ERROR: Only point2d instance or tuple of them allowed'

    def __add__(self, k=0.0, saa=False, make_new=True, lean='ignore',
                throw=True):
        if isinstance(k, point2d):
            if saa and make_new:  # \\\<---- 1
                self.x += k.x
                self.y += k.y
                to_return = self.make_new(x=self.x,
                                          y=self.y,
                                          lean=lean)
                to_return = (to_return, '[--parent also updated--]')
            if saa and not make_new:  # <---- 2
                self.x += k.x
                self.y += k.y
                to_return = '[--parent updated--]'
            if not saa and make_new:  # \\\<---- 3
                to_return = self.make_new(x=self.x+k.x,
                                          y=self.y+k.y,
                                          lean=lean)
            if not saa and not make_new:  # \\\<---- 4
                to_return = '[--saa FALSE make_nre FALSE--]'
        if type(k) in dth.dt.NUMBERS:
            if saa and make_new:  # \\\<---- 1
                self.x += k
                self.y += k
                to_return = self.make_new(x=self.x,
                                          y=self.y,
                                          lean=lean)
                to_return = (to_return, '[--parent also updated--]')
            if saa and not make_new:  # \\\<---- 2
                self.x += k
                self.y += k
                to_return = '[--parent updated--]'
            if not saa and make_new:  # \\\<---- 3
                to_return = self.make_new(x=self.x+k,
                                          y=self.y+k,
                                          lean=lean)
            if not saa and not make_new:  # \\\<---- 4
                to_return = '[--saa FALSE make_nre FALSE--]'
        if type(k) in dth.dt.ITERABLES:
            to_return = ()
            for k_ in k:
                if type(k_) in dth.dt.NUMBERS:
                    to_return += (self.make_new(self.x+k_,
                                                self.y+k_,
                                                lean=lean),)
                elif type(k_) in dth.dt.ITERABLES:
                    this = ()
                    for k__ in k_:
                        if type(k__) in dth.dt.NUMBERS:
                            this += (self.make_new(self.x+k__,
                                                   self.y+k__,
                                                   lean=lean),)
                        elif isinstance(k__, point2d):
                            this += (self.make_new(self.x+k__.x,
                                                   self.y+k__.y,
                                                   lean=lean),)
                    to_return += (this,)
                elif isinstance(k_, point2d):
                    to_return += (self.make_new(self.x+k_.x,
                                                self.y+k_.y,
                                                lean=lean),)
        if saa and self.__muledges__:
            for me in self.__muledges__:
                index = [*me.pmids].index(id(self))
                me.clist[index] = [self.x, self.y]
                me.mpoint.locx[index], me.mpoint.locy[index] = self.x, self.y

        if throw:
            return to_return

    def __sub__(self, k=0.0, saa=False, make_new=True, lean='ignore',
                throw=True):
        if isinstance(k, point2d):
            if saa and make_new:  # \\\<---- 1
                self.x -= k.x
                self.y -= k.y
                to_return = self.make_new(x=self.x,
                                          y=self.y,
                                          lean=lean)
                to_return = (to_return, '[--parent also updated--]')
            if saa and not make_new:  # <---- 2
                self.x -= k.x
                self.y -= k.y
                to_return = '[--parent updated--]'
            if not saa and make_new:  # \\\<---- 3
                to_return = self.make_new(x=self.x-k.x,
                                          y=self.y-k.y,
                                          lean=lean)
            if not saa and not make_new:  # \\\<---- 4
                to_return = '[--saa FALSE make_nre FALSE--]'
        if type(k) in dth.dt.NUMBERS:
            if saa and make_new:  # \\\<---- 1
                self.x -= k
                self.y -= k
                to_return = self.make_new(x=self.x,
                                          y=self.y,
                                          lean=lean)
                to_return = (to_return, '[--parent also updated--]')
            if saa and not make_new:  # \\\<---- 2
                self.x -= k
                self.y -= k
                to_return = '[--parent updated--]'
            if not saa and make_new:  # \\\<---- 3
                to_return = self.make_new(x=self.x-k,
                                          y=self.y-k,
                                          lean=lean)
            if not saa and not make_new:  # \\\<---- 4
                to_return = '[--saa FALSE make_nre FALSE--]'
        if type(k) in dth.dt.ITERABLES:
            to_return = ()
            for k_ in k:
                if type(k_) in dth.dt.NUMBERS:
                    to_return += (self.make_new(self.x-k_,
                                                self.y-k_,
                                                lean=lean),)
                elif type(k_) in dth.dt.ITERABLES:
                    this = ()
                    for k__ in k_:
                        if type(k__) in dth.dt.NUMBERS:
                            this += (self.make_new(self.x-k__,
                                                   self.y-k__,
                                                   lean=lean),)
                        elif isinstance(k__, point2d):
                            this += (self.make_new(self.x-k__.x,
                                                   self.y-k__.y,
                                                   lean=lean),)
                    to_return += (this,)
                elif isinstance(k_, point2d):
                    to_return += (self.make_new(self.x-k_.x,
                                                self.y-k_.y,
                                                lean=lean),)
        if throw:
            return to_return

    def __mul__(self, k=0.0, saa=False, make_new=True, lean='ignore',
                throw=True):
        if isinstance(k, point2d):
            if saa and make_new:  # \\\<---- 1
                self.x *= k.x
                self.y *= k.y
                to_return = self.make_new(x=self.x,
                                          y=self.y,
                                          lean=lean)
                to_return = (to_return, '[--parent also updated--]')
            if saa and not make_new:  # <---- 2
                self.x *= k.x
                self.y *= k.y
                to_return = '[--parent updated--]'
            if not saa and make_new:  # \\\<---- 3
                to_return = self.make_new(x=self.x*k.x,
                                          y=self.y*k.y,
                                          lean=lean)
            if not saa and not make_new:  # \\\<---- 4
                to_return = '[--saa FALSE make_nre FALSE--]'
        if type(k) in dth.dt.NUMBERS:
            if saa and make_new:  # \\\<---- 1
                self.x *= k
                self.y *= k
                to_return = self.make_new(x=self.x,
                                          y=self.y,
                                          lean=lean)
                to_return = (to_return, '[--parent also updated--]')
            if saa and not make_new:  # \\\<---- 2
                self.x *= k
                self.y *= k
                to_return = '[--parent updated--]'
            if not saa and make_new:  # \\\<---- 3
                to_return = self.make_new(x=self.x*k,
                                          y=self.y*k,
                                          lean=lean)
            if not saa and not make_new:  # \\\<---- 4
                to_return = '[--saa FALSE make_nre FALSE--]'
        if type(k) in dth.dt.ITERABLES:
            to_return = ()
            for k_ in k:
                if type(k_) in dth.dt.NUMBERS:
                    to_return += (self.make_new(self.x*k_,
                                                self.y*k_,
                                                lean=lean),)
                elif type(k_) in dth.dt.ITERABLES:
                    this = ()
                    for k__ in k_:
                        if type(k__) in dth.dt.NUMBERS:
                            this += (self.make_new(self.x*k__,
                                                   self.y*k__,
                                                   lean=lean),)
                        elif isinstance(k__, point2d):
                            this += (self.make_new(self.x*k__.x,
                                                   self.y*k__.y,
                                                   lean=lean),)
                    to_return += (this,)
                elif isinstance(k_, point2d):
                    to_return += (self.make_new(self.x*k_.x,
                                                self.y*k_.y,
                                                lean=lean),)
        if throw:
            return to_return

    def __truediv__(self, k=0.0, saa=False, make_new=True, lean='ignore',
                    throw=True):
        if isinstance(k, point2d):
            if abs(k.x) >= 0.000000000001 and abs(k.y) >= 0.000000000001:
                if saa and make_new:  # \\\<---- 1
                    self.x /= k.x
                    self.y /= k.y
                    to_return = self.make_new(x=self.x,
                                              y=self.y,
                                              lean=lean)
                    to_return = (to_return, '[--parent also updated--]')
                if saa and not make_new:  # <---- 2
                    self.x /= k.x
                    self.y /= k.y
                    to_return = '[--parent updated--]'
                if not saa and make_new:  # \\\<---- 3
                    to_return = self.make_new(x=self.x/k.x,
                                              y=self.y/k.y,
                                              lean=lean)
                if not saa and not make_new:  # \\\<---- 4
                    to_return = '[--saa FALSE make_new FALSE--]'
            else:
                to_return = '[--Zero k ERR--]'
        if type(k) in dth.dt.NUMBERS:
            if abs(k) >= 0.000000000001:
                if saa and make_new:  # \\\<---- 1
                    self.x /= k
                    self.y /= k
                    to_return = self.make_new(x=self.x,
                                              y=self.y,
                                              lean=lean)
                    to_return = (to_return, '[--parent also updated--]')
                if saa and not make_new:  # \\\<---- 2
                    self.x /= k
                    self.y /= k
                    to_return = '[--parent updated--]'
                if not saa and make_new:  # \\\<---- 3
                    to_return = self.make_new(x=self.x/k,
                                              y=self.y/k,
                                              lean=lean)
                if not saa and not make_new:  # \\\<---- 4
                    to_return = '[--saa FALSE make_nre FALSE--]'
            else:
                to_return = '[--Zero k ERR--]'
        if type(k) in dth.dt.ITERABLES:
            to_return = ()
            for k_ in k:
                if type(k_) in dth.dt.NUMBERS:
                    if abs(k_) >= 0.000000000001:
                        to_return += (self.make_new(self.x/k_,
                                                    self.y/k_,
                                                    lean=lean),)
                    else:
                        to_return += ('[--Zero k ERR--]',)
                elif type(k_) in dth.dt.ITERABLES:
                    this = ()
                    for k__ in k_:
                        if type(k__) in dth.dt.NUMBERS:
                            if abs(k__) >= 0.000000000001:
                                this += (self.make_new(self.x/k__,
                                                       self.y/k__,
                                                       lean=lean),)
                            else:
                                this += ('[--Zero k ERR--]',)
                        elif isinstance(k__, point2d):
                            if abs(k__.x) >= 0.000000000001 and abs(k__.y) >= 0.000000000001:
                                this += (self.make_new(self.x/k__.x,
                                                       self.y/k__.y,
                                                       lean=lean),)
                            else:
                                this += ('[--Zero k ERR--]',)
                    to_return += (this,)
                elif isinstance(k_, point2d):
                    if abs(k_.x) >= 0.000000000001 and abs(k_.y) >= 0.000000000001:
                        to_return += (self.make_new(self.x/k_.x,
                                                    self.y/k_.y,
                                                    lean=lean),)
                    else:
                        to_return += ('[--Zero k ERR--]',)
        if throw:
            return to_return

    def __abs__(self, saa=False, make_new=True, lean='ignore', throw=True):
        if saa or make_new:
            _x, _y = abs(self.x), abs(self.y)
        if saa and make_new:
            self.x, self.y = _x, _y
            to_return = (self.make_new(x=_x, y=_y, lean='ignore'),
                         '[--parent also updated--]')
        if not saa and make_new:
            to_return = self.make_new(x=_x, y=_y, lean='ignore')
        if saa and not make_new:
            self.x, self.y = _x, _y
            to_return = '[--parent updated--]'
        if not saa and not make_new:
            to_return = '#- saa FALSE make_new FALSE -#'
        if throw:
            return to_return


    def __repr__(self):
        """
        Instance representation.

        Returns
        -------
        str
            DESCRIPTION.

        """
        return f'upxo.p2d({round(self.x, 8)}, {round(self.y, 8)})'


    @property
    def _status_(self):
        '''
        Status update to developer. Not aimed at users
        '''
        if self.has('mid'):
            print('______________________________________')
            print('*~~* POINT2D STATUS INFORMATION *~~* ')
            print(' ')
            print(f'      MID: {self.mid}')
            print(f'      (X, Y): ({self.x}, {self.y})')
            print(f'      LEAN: {self.lean}')
            print('      \ \---------------------------')
            if self.has('lean'):
                if self.lean == 'ignore':
                    print(f'      POINT TYPE:     {self.ptype}')
                    print(f'      LOCATION:     {self.loc}')
                    print(f'      TOLERANCE DISTANCE:     {self.tdist}')
                    print(f'      JUNCTION ORDER:     {self.jn}')
                    print(f'      PHASE ID:     {self.phase_id}')
                    print('      \ \...........................')
                    print(f'      Polygonal AREA:     {self.sfv_pol_area}')
                    print(f'      Polygonal PERIMETER:     {self.sfv_pol_perimeter}')
                    print(f'      Polygonal ASPECT RATIO:     {self.sfv_pol_ar}')
                    print('      \ \...........................')
                    print(f'      EULER ANGLE REPR.:     {self.sfv_repr_ea}')
                    print(f'      EULER ANGLE:     {self.sfv_ea}')
                    print('      \ \...........................')
                    if self.has('orientation_object'):
                        print('      Orientation object: YES')
                    else:
                        print('      Orientation object: NO')
            print(' ')
            print('*~~* POINT2D STATUS INFORMATION *~~* ')
            print('______________________________________')
        else:
            print('mid does not exist. It will be made now.')
            # self.make_mid()
            self.mid= id(self)
            self._status_

    def borrow_tdist(self, point_object=None):
        """


        Parameters
        ----------
        point_object : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        if hasattr(point_object, 'tdist'):
            self.tdist = point_object.tdist

    def has(self, suffix):
        """


        Parameters
        ----------
        suffix : TYPE
            DESCRIPTION.

        Returns
        -------
        has_truth : TYPE
            DESCRIPTION.

        """
        suffix = suffix
        if suffix in dth.opt.point2d_tdist:
            suffix = 'tdist'
        elif suffix in dth.opt.point2d_jn:
            suffix = 'jn'
        elif suffix in dth.opt.point2d_ptype:
            suffix = 'ptype'
        elif suffix in dth.opt.point2d_loc:
            suffix = 'loc'
        elif suffix in dth.opt.point2d_rid:
            suffix = 'rid'
        elif suffix in dth.opt.point2d_mid:
            suffix = 'mid'
        elif suffix in dth.opt.point2d_dim:
            suffix = 'dim'
        elif suffix in dth.opt.point2d_pol_area:
            suffix = 'sfv_pol_area'
        elif suffix in dth.opt.point2d_pol_ar:
            suffix = 'sfv_pol_ar'
        elif suffix in dth.opt.point2d_pol_perimeter:
            suffix = 'sfv_pol_perimeter'
        elif suffix in dth.opt.point2d_pol_phaseid:
            suffix = 'phase_id'
        elif suffix in dth.opt.point2d_pol_phasename:
            suffix = 'phase_name'
        elif suffix in dth.opt.point2d_pol_tcname:
            suffix = 'tcname'
        elif suffix in dth.opt.point2d_pol_earepr:
            suffix = 'sfv_repr_ea'
        elif suffix in dth.opt.point2d_pol_eaunit:
            suffix = 'sfv_eaunit'
        elif suffix in dth.opt.point2d_pol_eangle:
            suffix = 'sfv_eangle'
        has_truth = False
        if hasattr(self, suffix):
            has_truth = True
        return has_truth

    def make_new(self, x=0.0, y=0.0, lean='low'):
        """
        Create a new point object.

        Parameters
        ----------
        x : TYPE, optional
            DESCRIPTION. The default is 0.0.
        y : TYPE, optional
            DESCRIPTION. The default is 0.0.
        lean : TYPE, optional
            DESCRIPTION. The default is 'low'.
        set_mid : TYPE, optional
            DESCRIPTION. The default is True.
        set_dim : TYPE, optional
            DESCRIPTION. The default is True.
        set_ptype : TYPE, optional
            DESCRIPTION. The default is False.
        store_original_coord : TYPE, optional
            DESCRIPTION. The default is True.
        set_tdist : TYPE, optional
            DESCRIPTION. The default is True.
        tdist : TYPE, optional
            DESCRIPTION. The default is 0.0000000000001.
         : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return point2d(x=x, y=y, lean=lean)

    def reset(self):
        """
        Resets location of self to UPXO default point2d location

        Returns
        -------
        None.

        """
        if hasattr(self, '_original_location'):
            self.x, self.y = self._original_location

    def LINK_mulpoint(self, mpo=None):
        """
        Attach this object to a multi-point object

        Parameters
        ----------
        mulpointObject : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        self._mulpoints_ = mpo

    def LINK_edge(self, eo=None):
        """
        Attach this object to an edge object

        Parameters
        ----------
        edgeObject : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        self._edges_ = eo

    def set_lean(self, lean):
        """
        Parameters
        ----------
        lean : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.lean = lean

    def set_dim(self, dim=2):
        """


        Parameters
        ----------
        dim : TYPE, optional
            DESCRIPTION. The default is 2.

        Returns
        -------
        None.

        """
        self.dim = dim

    def set_tdist(self, tdist=0.0):
        """


        Parameters
        ----------
        tdist : TYPE, optional
            DESCRIPTION. The default is 0.0.

        Returns
        -------
        None.

        """
        self.tdist = tdist

    def set_jn(self,
               jn=3,
               ):
        """
        jtype: number of l0e emanating from it / terminating in it.
        """
        self.jn = jn

    def set_ptype(self,
                  ptype='jp',
                  ):
        '''
        Assign which type of point it is going to be or it is.
        CASES INCLUDE THE FOLLOWING:
            1. jp: Junction point
            2. gbp: grain boundary point
            3. # TODO: write the many more that there are.
        '''
        self.ptype = ptype

    def attach_mulpoints(self, mulpoints):
        """


        Parameters
        ----------
        mulpoints : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self._mulpoints_ = mulpoints

    def attach_edges(self, edges):
        """


        Parameters
        ----------
        edges : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.edges = edges

    def attach_muledges(self, muledges):
        """

        :param muledges: DESCRIPTION
        :type muledges: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        self.muledges = muledges

    def attach_xtals(self, xtals):
        """

        :param xtals: DESCRIPTION
        :type xtals: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        self.xtals = xtals

    def attach_polyxtals(self, polyxtals):

        self.polyxtals = polyxtals

    def set_location(self, loc='pxtal_internal',):
        """


        Parameters
        ----------
        loc : TYPE, optional
            DESCRIPTION. The default is 'pxtal_internal'.

        Returns
        -------
        None.

        Suggested options for loc
        -------------------------
            1. pxtal_internal / pxtal_boundary
            2. on_grain_boundary
            3. particle_seed
            4. VTGS_seed
            5. voxel_centroid
            6. voxel_corners
        """
        self.loc = loc

    def set_sfv_pol_area(self, sfv_pol_area):
        """
        Set value for 'sfv_pol_area'.

        Parameters
        ----------
        sfv_pol_area : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.sfv_pol_area = sfv_pol_area

    def set_sfv_pol_perimeter(self, sfv_pol_perimeter):
        """


        Parameters
        ----------
        sfv_pol_perimeter : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.sfv_pol_perimeter = sfv_pol_perimeter

    def set_sfv_pol_ar(self, value):
        """


        Parameters
        ----------
        value : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.sfv_pol_ar = value

    def set_mesh_lc(self,
                    mesh_lc=1.0
                    ):
        '''
        Set value of mesh_lc for use in meshing using GMSH
        '''
        self.mesh_lc = mesh_lc

    def make_shapely(self, saa=True, make_and_throw=False):
        """
        Returs an equivalent shapely point object

        Returns
        -------
        shapely.geometry.point.Point object
            Shapely point object having the same x and y as self.x and self.y
        """
        from shapely.geometry import Point
        _ = Point(self.x, self.y)
        if saa:
            self.image_sh = _
        if make_and_throw:
            return _

    def make_vedo(self, saa=True, make_and_throw=False):
        """
        Make a vedo point object
        """
        pass

    def make_pyvtk(self, saa=True, make_and_throw=False):
        """
        make a pyvtk object
        """
        pass

    def make_paraview(self, saa=True, make_and_throw=False):
        """
        Make a paraview point object
        """
        pass

    def make_gmsh(self, saa=True, make_and_throw=False):
        """
        make a gmsh point object

        Returns
        -------
        None.

        """
        pass

    def make_pyvista(self, z=0.0):
        import pyvista
        return pyvista.PointSet(np.array([[self.x, self.y, z]],
                                         dtype=np.float64))

    def set_morph_curvature(self,
                            value: float = 0.0,
                            ):
        pass

    def set_gnd(self,
                value: float = 0.0,
                ):
        pass

    def set_lattice_curvature_(self,
                               value: list[list] = [[0.0, 0.0, 0.0],
                                                    [0.0, 0.0, 0.0],
                                                    [0.0, 0.0, 0.0]
                                                    ]
                               ):
        pass

    def set_phase(self,
                  phase_id: int = 0,
                  phase_name: str = 'cucrzr'
                  ):
        """
        Assign phase id and phase name to point

        Parameters
        ----------
        phase_id : int, optional
            DESCRIPTION. The default is 0.
        phase_name : str, optional
            DESCRIPTION. The default is 'cucrzr'.

        Returns
        -------
        None.

        """
        self.phase_id = phase_id
        self.phase_name = phase_name

    def set_tcname(self,
                   name: str = 'B'
                   ):
        """
        Assign name of the texture component to self

        Parameters
        ----------
        name : str, optional
            DESCRIPTION. The default is 'B'.

        Returns
        -------
        None.

        """
        self.tcname = name

    def set_sfv_repr_ea(self, sfv_repr_ea='Bunge'):
        """
        Assign the default representation of Euler angles

        Parameters
        ----------
        sfv_repr_ea : TYPE, optional
            DESCRIPTION. The default is 'Bunge'.

        Returns
        -------
        None.

        """
        self.sfv_repr_ea = sfv_repr_ea

    def set_sfv_ea1(self, ea1):
        """
        Assign 1st Euler angle to point object

        Parameters
        ----------
        ea1 : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.sfv_ea[0] = ea1

    def set_sfv_ea2(self, ea2):
        """
        Assign 2nd Euler angle to point object

        Parameters
        ----------
        ea2 : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.sfv_ea[1] = ea2

    def set_sfv_ea3(self, ea3):
        """
        Assign 3rd Euler angle to point object

        Parameters
        ----------
        ea3 : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.sfv_ea[2] = ea3

    def set_sfv_ea(self,
                   ea: list = [0, 0, 0]
                   ):
        """
        Assign the three Euler angles to point object

        Parameters
        ----------
        ea : list, optional
            DESCRIPTION. The default is [0, 0, 0].

        Returns
        -------
        None.

        """
        self.sfv_ea = ea

    def set_orientation_object(self, orientation_object):
        """
        Assign the UPXO orientation object to point object

        Parameters
        ----------
        orientation_object : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.orientation_object = orientation_object

    def plot(self, dpi=50, point: bool = True,
             buffer: bool = True, vprop=None,):
        # -------------------------
        if vprop is None:
            vprop = point.set_vis_prop()
        # -------------------------
        if point or buffer:
            fig = plt.figure(figsize=(1.6, 1.6), dpi=dpi)
            ax = fig.add_axes([0, 0, 1, 1])
        # -------------------------
        if point:
            ax.plot(self.x, self.y)
            plt.show()
        # -------------------------
        if buffer:
            pass

    def distance(self, otype='point2d', obj=None, cor=0.1, nworkers=1):
        """
        1. Single UPXO point2d object - DONE
        2. List of upxo point2d objects - DONE
        3. A single coordinate pair of point2d - DONE
        4. List of x coordinates and y coordinates - DONE
        """
        # ................
        # 1. Single UPXO point2d object - DONE
        # obj: UPXO point2d object
        if otype in dth.opt.upxo_point2d:
            '''
            Explanations:
                1. Use this when distances are to be computed against
                another point2d object
                2. INPUT TYPE of "obj": UPXO.point2d object

            Example 1: Point to point
                p1, p2 = point2d(x = 0, y = 0), point2d(x = 1, y = 1)
                p1.distance(otype = 'point2d', obj = p2)

            Example 2: Point to a list of points (method - 1)
                Not preferred, as there is a simpler method available under
                case 'point2d_list'
                x, y = list(range(0, 10, 2)), list(range(0, 20, 4))
                p = [point2d(x = _x, y = _y) for _x, _y in zip(x, y)]
                d = [p1.distance(otype = 'point2d',
                                 obj = pi) for pi in p]
            '''
            return np.sqrt((self.x-obj.x)**2 + (self.y-obj.y)**2)
        # ................
        # 2. List of upxo point2d objects - DONE
        # obj: list of point2d objects
        if otype in dth.opt.upxo_point2d_list:
            '''
            Explanations:
                1. Use this when distances are to be computed against a list
                of point2d objects
                2. INPUT TYPE of "obj": list

            Example 1: Point to list of points
                p1 = point2d(x = 0, y = 0)
                p2 = point2d(x = 1, y = 1)
                p1.distance(otype = 'up2dlist',
                            obj = [p1, p2])

            Example 2: Point to list of points
            '''
            x, y = zip(*[(_.x, _.y) for _ in obj])
            return np.sqrt((self.x-np.array(x))**2 + (self.y-np.array(y))**2)
        # ................
        # 3. Coordinate pair of a single point2d - DONE
        # obj: coordinate
        if otype in dth.opt.coord_point2d or otype in dth.opt.coord_pair_point2d:
            '''
            # INPUT: a list / tuple of two float / int coordinates
            # EXAMPLE INPUT: (x0, y0)

            p = point2d(x = 0, y = 0)
            print(p.distance(otype = 'coord2d', obj = (10, 10)))
            '''
            return np.sqrt((self.x-obj[0])**2 + (self.y-obj[1])**2)
        # ................
        # 4. List of x coordinates and y coordinates - DONE
        # obj: list of lists of x and y coordinates
        if otype in dth.opt.coord_point2d_list:
            '''
            INPUT: a list/tuple of two lists/tuples. Each of the two
            inner lists/tuples contain the list of coordinate values
            EXAMPLE INPUT: ((x0, x1, x2,...), (y0, y1, y2,...))

            Example 1: Point to list of x and y coordinate list
                p = point2d(x = -2, y = 0)
                obj = [[-2, -1, -0, 1, 2], [0, 0, 0, 0, 0]]
                d = p.distance(otype = 'xy_list', obj = obj)
            '''
            return np.sqrt((self.x-np.array(obj[0]))**2
                           + (self.y-np.array(obj[1]))**2)
        # ................
        # 5. List of coordinate pairs of point2d objects - DONE
        # obj: list of lists of x-y coordinate pairs
        if otype in dth.opt.coord_pairs_point2d_list:
            # INPUT: a list/tuple of numerous lists/tuples. Each of the
            # many lists/tuples contain coordinates of a point
            # EXAMPLE INPUT: ((x0, y0), (x1, y1), (x2, y2),....)
            obj = np.array(obj).T
            return np.sqrt((self.x-obj[0])**2 + (self.y-obj[1])**2)
        # ................
        # 6. A list of cKDTree objects - DONE
        # obj: list of ckdtree objects
        if otype in dth.opt.ckdtree_lists:
            _, distances, _, _ = dth.find_neighdata_ckdt_list(self.x,
                                                              self.y,
                                                              obj,
                                                              cor,
                                                              nworkers)
            return distances
        # ................
        # 7. A list of shapely point objects
        # ................
        # 8. A list of vtk point objects
        # ................
        # 9. A list of pyvista point objects
        # ................
        # 10. A single OR list of UPXO multi-point2d objects
        if otype in dth.opt.upxo_mp2d_list:
            '''
            Explanations:
                1. Use this when computimng distance sgainst a set of point2d
                   objects contained inside list of mulpoint2d objects
                2. INPUT TYPE of "obj": [upxo point object 1,
                                         upxo point object 2,
                                         ...,
                                         upxo point object n]

            Example 1:
                # Create the reference point2d object
                    p0 = point2d()

                # Create a mulpoint2d object
                    p1 = point2d(x = 2.0, y = 1.0)
                    p2 = p1 + 1
                    p3 = p2 * 0.6498
                    p4 = p1*0.468 + p3/p2
                    m1 = mulpoint2d(method = 'up2d_list',
                                    point_objects = [p1, p2, p3, p1 + p4*p1])

                # Calculate distance
                    d = p0.distance(otype = 'ump2d_list',
                                    obj = [m1, m1])
            '''
            _depack_ = False
            if str(obj.__class__.__name__) == 'mulpoint2d':
                # If user enters a single mul-point object, make list
                obj, _depack_ = [obj], True
            _x, _y, distances = self.x, self.y, []
            for mp in obj:
                distances.append(np.sqrt((_x-mp.locx)**2
                                         + (_y-mp.locy)**2)
                                 )
            if _depack_:
                distances = distances[0]

            return distances
        # ................
        # 11. A list of shapely mulobject objects (each made of points)
        # ...............
        # A single UPXO mulpoint3d objects
        if otype == 'upxo_mulpoint3d':
            pass
        # ................
        # A single edge2d object
        if otype == 'upxo_edge2d':
            pass
        # ................
        # A list of edge2d objects
        if otype == 'upxo_edge2d':
            pass
        # ................
        if otype in ('upxo_muledge2d', 'upxo_ring2d'):
            pass
        # ................
        if otype == 'upxo_edge3d':
            pass
        # ................
        if otype in ('upxo_muledge3d', 'upxo_ring3d'):
            pass
        # ................
        if otype == 'shapely_xtal2d_centroid':
            '''
            Explanations:
                1. Use this to find distance between self and centroid
                   of the shapely polygon object
                2. INPUT TYPE of "obj": a valid shapely polygon object
                3. Centroidal x and y of polygon object will be used as
                   obj = [[x], [y]]

            Example 1:
                from point2d_04 import point2d
                p0 = point2d()
                from shapely.geometry import Polygon
                shapelypol = Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])

                p0.distance(otype = 'shapely_xtal2d_centroid',
                            obj = shapelypol)
            '''
            centroid = obj.centroid
            return self.distance(otype='coord_list',
                                 obj=[[centroid.x], [centroid.y]])[0]
        if otype == 'shapely_xtal2dlist_centroid':
            '''
            Explanations:
                1. Use this to find distance between self and centroids of a
                   list of shapely polygon objects
                2. INPUT TYPE of "obj": list of valid shapely polygon
                   objects

            Example 1:
                from point2d_04 import point2d
                p0 = point2d()
                from shapely.geometry import Polygon
                shapelypol1 = Polygon([[0,0], [1,0], [1,1], [0,1], [0,0]])
                shapelypol2 = Polygon([[1,1], [2,1], [2,2], [1,2], [1,1]])

                obj = [shapelypol1, shapelypol2]
                p0.distance(otype = 'shapely_xtal2dlist_centroid',
                            obj = obj)
            '''
            centroids = [[_.centroid.x, _.centroid.y] for _ in obj]
            return self.distance(otype='coord_pairs',
                                 obj=centroids)

        if otype == 'shapely_xtal2d_reppoint':
            '''
            Explanations:
                1. Use this to find distance between self and reppoint of the
                   shapely polygon object
                2. INPUT TYPE of "obj": a valid shapely polygon object
                3. centroidal x and y of polygon object will be used as
                   obj = [[x], [y]]

            Example 1:
                from point2d_04 import point2d
                p0 = point2d()
                from shapely.geometry import Polygon
                shapelypol = Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])

                p0.distance(otype = 'shapely_xtal2d_reppoint',
                            obj = shapelypol)
            '''
            reppoint = obj.representative_point()
            return self.distance(otype='coord_list',
                                 obj=[[reppoint.x], [reppoint.y]])[0]
        if otype == 'shapely_xtal2dlist_reppoint':
            '''
            Explanations:
                1. Use this to find distance between self and reppoints of a
                   list of shapely polygon objects
                2. INPUT TYPE of "obj": list of valid shapely polygon
                   objects

            Example 1:
                from point2d_04 import point2d
                p0 = point2d()
                from shapely.geometry import Polygon
                shapelypol1 = Polygon([[0,0], [1,0], [1,1], [0,1], [0,0]])
                shapelypol2 = Polygon([[1,1], [2,1], [2,2], [1,2], [1,1]])

                obj = [shapelypol1, shapelypol2]
                p0.distance(otype = 'shapely_xtal2dlist_reppoint',
                            obj = obj)
            '''
            reppoints = [[_.representative_point().x,
                          _.representative_point().y] for _ in obj]
            return self.distance(otype='coord_pairs',
                                 obj=reppoints)
        # ................
        if otype == 'upxo_xtal2d_reppoint':
            # here obj will be the xtal containing the representative
            # point
            # representative point has to be UPXO point2d object
            return self.distance(otype='upxo_point2d',
                                 obj=obj.reppoint)
        if otype == 'upxo_xtal2dlist_reppoint':
            # This is to call self.distance operating on case
            # 'upxo_xtal2d_reppoint'
            return [self.distance(otype='upxo_xtal2d_reppoint',
                                  obj=_obj)
                    for _obj in obj]
        # ................
        if otype == 'upxo_xtal2d_vertices':
            # This is to call self.distance operating on case
            # 'upxo_xtal2d_reppoint'
            pass
        # ................
        if otype == 'upxo_xtal3d_centroid':
            pass
        # ................
        if otype == 'upxo_xtal3d_reppoint':
            pass
        # ................
        if otype == 'upxo_xtal3d_vertices':
            pass
        # ................
        if otype == 'shapely_point':
            pass
        # ................
        if otype == 'vtk_point':
            pass
        # ................
        if obj is None:
            print('Need other object to compute distance(s)')

    def neigh(self,
              otype='points2d', obj=None,
              cos='circle', cor=1.0, CMPOP='<',
              nckdt_workers=1
              ):
        """
        #######################################---------------
            PRE-REQUISITE DATA FOR EXAMPLES
        p1, p2 = point2d(x=-2, y=0), point2d(x=-1, y=0)
        p3, p4 = point2d(x=+0, y=0), point2d(x=+1, y=0)
        p5 = point2d(x=+2, y=0)
        #-------------------
        # example data - 1
        points_upxo= ([p1, p2, p3, p4, p5],
                      [p1.yadd(1),p2.yadd(1),p3.yadd(1),p4.yadd(1),p5.yadd(1)],
                      [p1.yadd(2),p2.yadd(2),p3.yadd(2),p4.yadd(2),p5.yadd(2)]
                     )
        #-------------------
        # example data - 2
        points_list_coord = ([[-2,-1,0,1,2], [0,0,0,0,0]],
                             [[-2,-1,0,1,2], [1,1,1,1,1]],
                             [[-2,-1,0,1,2], [2,2,2,2,2]]
                            )
        #-------------------
        # example data = 3
        points_coord_list = ([[-2,0],[-1,0],[0,0],[1,0],[2,0]],
                             [[-2,1],[-1,1],[0,1],[1,1],[2,1]],
                             [[-2,2],[-1,2],[0,2],[1,2],[2,2]],
                             )
        #-------------------
        # example data = 4
        n, nsets, = 5, 2
        points = [dth.make_upxo_point2d_RANDU(n) for _ in range(nsets)]
        cut_off_radii = [0.25, 0.50]
        #######################################---------------
        NOTE: both lists or tuples for obj will work
        #######################################---------------
                    LIST OF CASES
        CASE 01: list of list of UPXO point2d objects (see example data-1)
        CASE 02: list of list of coordinates of points (see example data-2)
        CASE 03: list of list of coordinate pairs (see example data-3)
        CASE 04: list of ckdtrees
        #######################################---------------
         OPTIONS FOR BRANCHING SEARCH STRINGS
        CASE 01: dth.opt.upxo_point2d_list
                 ['upxo_point2d_list', 'point2d_list', 'p2d_list',
                  'p2dlist', 'p2list', 'points2d']

        CASE 02: dth.opt.coord_point2d_list
                ['point_coord_2d_list', 'point_coord_2d_list',
                 'coord2d_list', 'xy_coord_list', 'xy_list',
                 'coord_lists', 'coord_list', 'clists', 'clists']

        CASE 03: dth.opt.coord_pairs_point2d_list
                ['xy_coord_pairs_list']
        #######################################---------------
        [Number ** 2 for list in Numbers for Number in list]
        #######################################---------------
        """
        limit_usetree = 5*10**3
        # -------------------------------------

        def _length_(data):
            return np.sum([len(_) for _ in obj])

        def _D_(otype, obj):
            # FIND DISTANCES
            # Use only when [_length_(obj) < limit_usetree]
            return [self.distance(otype=otype, obj=olist) for olist in obj]

        def _ind_co_circle_(D, R, CMPOP):
            # FIND INDICES OF POINTS INSIDE R
              # INSIDE, INSIDE AND ON, OUTSIDE, OR, OUTSIDE AND ON is decided
              # by beh
            # D: list of list of distances
            # R: cut-off radius
            # CMPOP: Comparison operator
            if CMPOP in ('lt', '<'):
                indices = [tuple(np.array(np.where(_ < R)[0]).tolist())
                           for _ in D]
            elif CMPOP in ('le', '<='):
                indices = [tuple(np.array(np.where(_ <= R)[0]).tolist())
                           for _ in D]
            elif CMPOP in ('gt', '>'):
                indices = [tuple(np.array(np.where(_ > R)[0]).tolist())
                           for _ in D]
            elif CMPOP in ('ge', '>='):
                indices = [tuple(np.array(np.where(_ >= R)[0]).tolist())
                           for _ in D]
            return indices
        # -------------------------------------
        #               CASE - 01
        # 1. List of list of upxo point2d objects - DONE
        '''
        p1, p2 = point2d(x=-2, y=0), point2d(x=-1, y=0)
        p3, p4 = point2d(x=+0, y=0), point2d(x=+1, y=0)
        p5 = point2d(x=+2, y=0)

        points_upxo= ([p1, p2, p3, p4, p5],
                      [p1.yadd(1),p2.yadd(1),p3.yadd(1),p4.yadd(1),p5.yadd(1)],
                      [p1.yadd(2),p2.yadd(2),p3.yadd(2),p4.yadd(2),p5.yadd(2)]
                     )

        p1.neigh(otype='points2d', obj=points_upxo,
                 cos='circle', cor=3, CMPOP = '<=')


        n, nsets, = 5001, 2
        points = [dth.make_upxo_point2d_RANDU(n) for _ in range(nsets)]

        locxy = [[point.x, point.y] for point in list_of_points]
        '''
        if otype in dth.opt.upxo_point2d_list:
            if _length_(obj) < limit_usetree:
                distances = _D_(otype, obj)
            else:
                pass
            to_return = (distances, _ind_co_circle_(distances, cor, CMPOP))
        # -------------------------------------
        #               CASE - 02
        # 2. List of list of xy coordiates - DONE
        '''
        points_list_coord = ([[-2,-1,0,1,2], [0,0,0,0,0]],
                             [[-2,-1,0,1,2], [1,1,1,1,1]],
                             [[-2,-1,0,1,2], [2,2,2,2,2]]
                            )

        p1.neigh(otype='xy_coord_list', obj=points_list_coord,
                 cos='circle', cor=3, CMPOP = '<=')
        '''
        if otype in dth.opt.coord_point2d_list:
            if _length_(obj) < limit_usetree:
                distances = _D_(otype, obj)
            else:
                pass
            to_return = (distances, _ind_co_circle_(distances, cor, CMPOP))
        # -------------------------------------
        #               CASE - 03
        # 3. List of list of x-y coordinate pairs - DONE
        '''
        points_coord_list = ([[-2,0],[-1,0],[0,0],[1,0],[2,0]],
                             [[-2,1],[-1,1],[0,1],[1,1],[2,1]],
                             [[-2,2],[-1,2],[0,2],[1,2],[2,2]],
                             )
        p1.neigh(otype='xy_coord_pairs_list', obj=points_coord_list,
                 cos='circle', cor=2, CMPOP = '<=')
        '''
        if otype in dth.opt.coord_pairs_point2d_list:
            if _length_(obj) < limit_usetree:
                distances = _D_(otype, obj)
            else:
                pass
            to_return = (distances, _ind_co_circle_(distances, cor, CMPOP))
        # -------------------------------------
        #               CASE - 04
        # 4. List of list of shapely objects
        # -------------------------------------
        #               CASE - 05
        # 5. List of list of cKDTress
        '''
        xy_pair_lists = [np.random.rand(2, 1000).T for _ in xy_lists]
        ckdtrees = [ckdt(cpairlist) for cpairlist in xy_pair_lists]
        p = point2d()
        _, _, _, d = p.neigh(otype='ckdt_list', obj=ckdtrees, cor=0.5, CMPOP='<=')
        print(d)
        '''
        if otype in dth.opt.ckdtree_list:
            xself, yself = self.x, self.y
            indices, distances, neigh, nneigh = [], [], [], []
            for _tree_ in obj:
                # Find indices of shortlisted points from the original dataset
                ind = _tree_.query_ball_point([xself, yself], cor, workers=1)
                # coordinates of shortlisted points
                _locxy_ = _tree_.data[ind].T
                # actual distances of shortlisted points
                dist = np.sqrt((xself-_locxy_[0])**2+(yself-_locxy_[1])**2)
                # Indices to sort distancesa in ascending order
                ascend_ind = np.argsort(dist)
                # Distances in ascending order
                dist_ascend = dist[ascend_ind]
                # coordinate list of points in ascending distances
                locxy_ascend = _locxy_.T[ascend_ind].T
                # BUILD DATA LISTS
                distances.append(dist_ascend)
                neigh.append(locxy_ascend)
                nneigh.append(len(locxy_ascend[0]))
                indices.append(ind)
            to_return = (indices, distances, neigh, nneigh)
        # -------------------------------------
        # N. List of Multi-points -- point wise check
        '''
        Returns True and indices of multi-points and the points- within
        which fall inside and on the circle of cut-off-radius centred at
        current point object.

        p = point2d(x=0, y=0, lean='ignore')
        xy_lists = [[np.random.rand(10)+1, np.random.rand(10)+0],
                    [np.random.rand(10)+1, np.random.rand(10)+1],
                    [np.random.rand(10)+0, np.random.rand(10)+1],
                    [np.random.rand(10)-1, np.random.rand(10)+1],
                    [np.random.rand(10)-1, np.random.rand(10)+0],
                    [np.random.rand(10)-1, np.random.rand(10)-1],
                    [np.random.rand(10)+0, np.random.rand(10)-1],
                    [np.random.rand(10)+1, np.random.rand(10)-1]
                    ]
        mp_list = [mulpoint2d(method='xy_list',
                              coordxy=xy_list) for xy_list in xy_lists]

        # TO FIND THE DISTANCES AND DOING MANUALLY
        distances = p.distance(otype='ump2d_list', obj=mp_list)
        neighbour_points = []
        neighbour_mp = []
        for i, dist in enumerate(distances):
            neigh_points = list(np.where(dist <= 0.75)[0])
            neighbour_points.append(neigh_points)
            if len(neigh_points) > 0:
                neighbour_mp.append(i)

        # ALTERNATIVE TO ABOVE MANUAL METHOD
        p.neigh(otype='ump2d_list', obj=mp_list, cor=1.75, CMPOP='<=',
                nckdt_workers=1)
        '''
        if otype in dth.opt.upxo_mp2d_list:
            if str(obj.__class__.__name__) == 'mulpoint2d':
                obj = [obj]
            dist = _D_('ump2d_list', obj)
            to_return = tuple(_ind_co_circle_(dist, cor, CMPOP))
        # -------------------------------------
        return to_return

    def find_parent_mulpoints(self,
                              mulpoints: list
                              ):
        """
        find the mul-point from the list of mulpoints which contains this
        point with tol_dist = 0.0

        Parameters
        ----------
        mulpoints : list
            DESCRIPTION.

        Returns
        -------
        None.

        """
        pass

    def find_containing_mulpoints(self,
                                  mulpoints: list
                                  ):
        """
        use convex hull polygon of the mulpoint

        Parameters
        ----------
        mulpoints : list
            DESCRIPTION.

        Returns
        -------
        None.

        """
        pass

    def find_nearest_point_cloud(self,
                                 cor_start=0.1,
                                 cor_end=1.0,
                                 search_method='bisection'
                                 ):
        pass

    def find_nearest_mulpoints(self,
                               mulpoints: list
                               ):
        pass

    def find_neigh_mulpoints(self,
                             mulpoints: list
                             ):
        pass

    def find_neigh_edges(self,
                         edges: list
                         ):
        """
        Find edges with normal distance less than or equal to tol_dist

        Parameters
        ----------
        edges : list
            DESCRIPTION.

        Returns
        -------
        None.

        """
        pass

    def find_parent_edges(self,
                          edges: list
                          ):
        """
        Find parent edges which contain this point

        Parameters
        ----------
        edges : list
            DESCRIPTION.

        Returns
        -------
        None.

        """
        pass

    def find_nearest_edges(self,
                           edges: list
                           ):
        """
        Find the edge nearest to this point

        Parameters
        ----------
        edges : list
            DESCRIPTION.

        Returns
        -------
        None.

        """
        pass

    def find_containing_muledges(self,
                                 muledges: list
                                 ):
        """
        Find the muledge, on one of whos edge, the point lies within
        the tolerance distance

        Parameters
        ----------
        muledges : list
            DESCRIPTION.

        Returns
        -------
        None.

        """
        pass

    def find_bounding_muledges(self,
                               muledges: list
                               ):
        """
        Find the muledge, who's convex hull contains the point object

        Parameters
        ----------
        muledges : list
            DESCRIPTION.

        Returns
        -------
        None.

        """
        pass

    def find_nearest_muledges(self,
                              muledges: list
                              ):
        """
        Use the hauseldroff distaince from the point and rthe vertices
        of the muledge

        Parameters
        ----------
        muledges : list
            DESCRIPTION.

        Returns
        -------
        None.

        """
        pass

    def find_parent_partitions(self,
                               partitions: list
                               ):
        pass

    def find_neigh_partitions(self,
                              partitions: list
                              ):
        pass

    def find_nearest_partitions(self,
                                partitions: list
                                ):
        pass

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


class point2d_lean_highest():
    '''
    UPXO core class.
    Represents a point in 2D Cartesian space.
    Leanness is highest, meaning only stores the coordinates x and y

    NOTES TO USER:
        1. x and y must be floats
    '''
    __slots__ = ('x', 'y')

    def __init__(self, x=0.0, y=0.0):
        self.x, self.y = x, y
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


class point2d_lean_highest_mc0():
    '''
    UPXO core class.
    Represents a point in 2D Cartesian space, made for Monte-Carlo simulation
    Leanness is highest, meaning only stores the coordinates x and y and the
    integer state, s

    NOTES TO USER:
        1. x and y must be floats
        2. s must be an int
    '''
    __slots__ = ('x', 'y', 's')

    def __init__(self,
                 x: float = 0.0,
                 y: float = 0.0,
                 s: int = 1
                 ):
        self.x, self.y, self.s = x, y, s
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


class point2d_lean_highest_mc1():
    '''
    UPXO core class.
    Represents a point in 2D Cartesian space, made for Monte-Carlo simulation
    Leanness is highest, meaning only stores the coordinates x and y and the
    integer state, s

    NOTES TO USER:
        1. x and y must be floats
        2. s must be an int
        3. h. no input necessary. Taken internally
    '''
    __slots__ = ('x', 'y', 's', 'h')

    def __init__(self,
                 x: float = 0.0,
                 y: float = 0.0,
                 s: int = 1,
                 h: float = 0.0
                 ):
        self.x, self.y, self.s, self.h = x, y, s, h
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


class point3d():
    '''
    UPXO core class.
    Represents a point in 2D Cartesian space.
    '''
    # -----------------------------------------------------
    """Number of decimal places to round off the very small number to zero"""
    ROUND_ZERO_DEC_PLACE = 10
    # -----------------------------------------------------
    __slots__ = ('dim',  # Dimension of the model
                 'lean',  # Leanness of the point object
                 'rid',  # random id of the point
                 'mid',  # memory id of the point
                 'x',  # x-coordinate
                 'y',  # y-coordinate
                 'z',  # z-coordinate
                 'lean',  # Complexity branching
                 '_mulpoints_',  # list of multi-point object
                 '_edges_',  # list of edge object
                 '_muledges_',  # list of multi-edge object
                 '_surfaces_',  # list of surface object
                 '_mulsurfaces_',  # list of multi-surface object
                 '_xtals_',  # list of xtal object
                 '_polyxtals_',  # list of poly-xtal object
                 'ptype',  # point type
                 'jn',  # Junction order
                 'loc',  # Location of the point in pxtal
                 'store_original_coord',
                 '_original_location',  # original x and y
                 'phase_id',  # Phase id
                 'phase_name',  # Phase name. Defaults to CuCrZr
                 'tcname',  # texture component name
                 'sfv_pol_area',  # Xtal area
                 'sfv_repr_ea',  # Euler angle representation (ex: 'Bunge')
                 'ea',  # Euler angle in degrees
                 'oo',  # UPXO Orientation object
                 'tdist',  # for use in comparison operations
                 'partition',  # Dataclass instance to store buffer polygon objects
                 'vprop'  # Visialuzation properties
                 )
    # -----------------------------------------------------
    # OPTIONS for some above slotted variables: SEE END OF FILE
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # START OF DUNDERS
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    def __init__(self,
                 x: float = 0.0,
                 y: float = 0.0,
                 z: float = 0.0,
                 lean: str = 'lowest',
                 set_rid: bool = False, rid_length: int = 4,
                 set_mid: bool = False,
                 set_dim: bool = False, dim: int = 2,
                 set_ptype: bool = False, ptype: str = 'vt2dseed',
                 set_jn: bool = False, jn: int = 3,
                 set_loc: bool = False, loc: str = 'interior',
                 store_original_coord: bool = False,
                 attach_mulpoints: bool = False,
                 attach_edges: bool = False,
                 attach_muledges: bool = False,
                 attach_xtals: bool = False,
                 attach_polyxtals: bool = False,
                 set_phase: bool = False, phase_id: int = 1, phase_name: str = 'cucrzr',
                 set_sfv_pol_area: bool = True, sfv_pol_area: int = 0,
                 set_tcname: bool = False, tcname: str = 'B',
                 set_ea: bool = True, sfv_repr_ea='Bunge', ea: list = [45, 35, 0],
                 set_oo: bool = True, oo: object = None,
                 set_tdist: bool = True, tdist: float = 0.0000000000001,
                 store_vis_prop: bool = False,
                 make_partition: bool = False,
                 partition_n=4,
                 partition_char_lengths=[1.0],
                 partition_char_length_type='radius',
                 partition_make_polygon=True,
                 partition_tool='mpl',
                 partition_rotation=0,
                 feed_partition=False,
                 feed_partition_name='grain',
                 feed_partition_tool='shapely',
                 feed_partition_polygon=None
                 ):
        '''
        Exaplantions:
            1.
            2.
        '''
        if lean in ('ignore'):
            self.x, self.y, self.z = float(x), float(y), float(z)
            self.lean = lean  # TODO: MAKE SETTER
            self.make_mid()
            self.dim = dim
            self.set_ptype(ptype=ptype)
            self.set_jn(jn=jn)
            self.set_location(loc=loc)
            self.store_original_coord = store_original_coord
            self._original_location = (self.x, self.y)
            self.set_phase(phase_id=phase_id, phase_name=phase_name)
            self.sfv_pol_area = sfv_pol_area
            self.tcname = tcname
            self.sfv_repr_ea = sfv_repr_ea
            self.ea = [45, 35, 0]
            self.oo = oo
            self.tdist = tdist
            # self.make_partition(restart = False,
            #                    saa = True,
            #                    n = partition_n,
            #                    char_lengths = partition_char_lengths,
            #                    char_length_type = partition_char_length_type,
            #                    make_polygon = partition_make_polygon,
            #                    tool = partition_tool,
            #                    rotation = partition_rotation,
            #                    feed_partition = False,
            #                    feed_partition_name = 'grain',
            #                    feed_partition_tool = 'shapely',
            #                    feed_partition_polygon = None
            #                    )
        if lean in ('lowest'):
            self.x, self.y, self.z = float(x), float(y), float(z)
            self.lean = lean
            if set_rid:
                self.make_rid(idlength=rid_length)
            if set_mid:
                self.make_mid()
            if set_dim:
                self.dim = dim
            if set_ptype:
                self.set_ptype(ptype=ptype)
            if set_jn:
                self.set_jn(jn=jn)
            if set_loc:
                self.set_location(loc=loc)
            if store_original_coord:
                self.store_original_coord = store_original_coord
                self._original_location = (self.x, self.y)
            if attach_mulpoints:
                pass
            if attach_edges:
                pass
            if attach_muledges:
                pass
            if attach_xtals:
                pass
            if attach_polyxtals:
                pass
            if attach_muledges:
                pass
            if attach_xtals:
                pass
            if attach_polyxtals:
                pass
            if set_phase:
                self.set_phase(phase_id=phase_id, phase_name=phase_name)
            if set_sfv_pol_area:
                self.sfv_pol_area = sfv_pol_area
            if set_tcname:
                self.tcname = tcname
            if set_ea:
                self.sfv_repr_ea = sfv_repr_ea
                self.ea = [45, 35, 0]
            if set_oo:
                self.oo = oo
            if set_tdist:
                self.tdist = tdist
            if make_partition:
                self.make_partition(restart=False,
                                    saa=True,
                                    n=partition_n,
                                    char_lengths=partition_char_lengths,
                                    char_length_type=partition_char_length_type,
                                    make_polygon=partition_make_polygon,
                                    tool=partition_tool,
                                    rotation=partition_rotation,
                                    feed_partition=False,
                                    feed_partition_name='grain',
                                    feed_partition_tool='shapely',
                                    feed_partition_polygon=None
                                    )
        if lean in ('low'):
            self.x, self.y, self.z = float(x), float(y), float(z)
            self.lean = lean
            if set_rid:
                self.make_rid(idlength=rid_length)
            if set_mid:
                self.make_mid()
            if set_dim:
                self.dim = dim
            if set_ptype:
                self.set_ptype(ptype=ptype)
            if set_jn:
                self.set_jn(jn=jn)
            if set_loc:
                self.set_location(loc=loc)
            if store_original_coord:
                self.store_original_coord = store_original_coord
                self._original_location = (self.x, self.y)
            if attach_mulpoints:
                pass
            if attach_edges:
                pass
            if attach_muledges:
                pass
            if attach_xtals:
                pass
            if attach_polyxtals:
                pass
            if attach_muledges:
                pass
            if attach_xtals:
                pass
            if attach_polyxtals:
                pass
            if set_phase:
                self.set_phase(phase_id=phase_id, phase_name=phase_name)
            if set_sfv_pol_area:
                self.sfv_pol_area = sfv_pol_area
            if set_tcname:
                self.tcname = tcname
            if set_ea:
                self.sfv_repr_ea = sfv_repr_ea
                self.ea = [45, 35, 0]
            if set_oo:
                self.oo = oo
            if set_tdist:
                self.tdist = tdist
        if lean in ('highest'):
            self.x, self.y = float(x), float(y)
    # -----------------------------------------------------

    def __call__(self, x, y):
        '''Call point object instances like functions'''
        self.x, self.y, self.z = x, y, z
    # -----------------------------------------------------

    def __eq__(self,
               point_objects=None,
               tdist=0.0,
               use_self_tdist=True
               ):
        '''
        True if distance b/w self and point_objects is LE tolerance distance
        '''
        if use_self_tdist:
            tdist = self.tdist
        if isinstance(point_objects, point2d):
            if self.distance(other_object_type='point3d',
                             point_data=point_objects) <= tdist:
                return True
            else:
                return False
        elif isinstance(point_objects, list):
            to_return = [False for pobjcount in range(len(point_objects))]
            for pobjcount in range(len(point_objects)):
                if self.distance(point_objects[pobjcount]) <= tdist:
                    to_return[pobjcount] = True
            return tuple(to_return)
        else:
            return f'ERROR: Only point2d instance or tuple of them allowed'
    # -----------------------------------------------------

    def __ne__(self,
               point_objects=None,
               tdist=0.0,
               use_self_tdist=True
               ):
        '''
        True if distance b/w self and point_object is GT tolerance distance
        '''
        if use_self_tdist:
            tdist = self.tdist
        if isinstance(point_objects, point2d):
            if self.distance(other_object_type='point3d',
                             point_data=point_objects) > tdist:
                return True
            else:
                return False
        elif isinstance(point_objects, list):
            to_return = [False for pobjcount in range(len(point_objects))]
            for pobjcount in range(len(point_objects)):
                if self.distance(point_objects[pobjcount]) > tdist:
                    to_return[pobjcount] = True
            return tuple(to_return)
        else:
            return f'ERROR: Only point2d instance or tuple of them allowed'
    # -----------------------------------------------------

    def __add__(self,
                toadd=0.0,
                copy_=True
                ):
        '''
        CASE-1: if toadd is instance of point2d:
            addition over fundamental axes lengths
        CASE-2: if toadd is single scalar:
            scalar added to both fundamental axes lengths
        CASE-3: if toadd is two scalar element tuple:
        CASE-3: if toadd is tuple of scalars or point2d objects:
            scalar added to both fundamental axes lengths and a
            tuple of point2d objects is returned. Element wise check
            of scalar or point2d is carried out.
        CASE-4: if toadd is a tuple of tuples, then the inner tuple,
        in the form (x, y) is added to the x and y of self and the
        corresponding point2d object is made and appended to tuple to
        be returned.

        copy_ applies only to cases 1,2 and 3 of above 4 cases

        RESTRICTIONS:
            toadd: if not cases 1 and 2, toadd MUST be in tuple format
        '''
        if isinstance(toadd, point3d):  # CASE-1
            if copy_:
                return self.make_new(self.x+toadd.x, self.y+toadd.y, self.z+toadd.z)
            else:
                self.x += toadd.x
                self.y += toadd.y
        if isinstance(toadd, int) or isinstance(toadd, float):
            if copy_:
                return self.make_new(self.x+toadd, self.y+toadd)
            else:
                self.x += toadd
                self.y += toadd
        if isinstance(toadd, tuple):
            to_return = ()
            # count = 0
            for toadd_ in toadd:
                if isinstance(toadd_, int) or isinstance(toadd_, float):
                    to_return += (self.make_new(self.x+toadd_,
                                  self.y+toadd_, self.z+toadd_),)
                elif isinstance(toadd_, tuple):
                    to_return += (self.make_new(self.x +
                                  toadd_[0], self.y+toadd_[1], self.z+toadd_[2]),)
                elif isinstance(toadd_, point2d):
                    to_return += (self.make_new(self.x+toadd_.x,
                                  self.y+toadd_.y, self.z+toadd_.z),)
            #    count+=1
            return to_return
    # -----------------------------------------------------

    def __sub__(self,
                tosub=0.0,
                copy_=True
                ):
        '''
        CASE-1: if tosub is instance of point2d:
            addition over fundamental axes lengths
        CASE-2: if tosub is single scalar:
            scalar added to both fundamental axes lengths
        CASE-3: if tosub is two scalar element tuple:
        CASE-3: if tosub is tuple of scalars or point2d objects:
            scalar added to both fundamental axes lengths and a
            tuple of point2d objects is returned. Element wise check
            of scalar or point2d is carried out.
        CASE-4: if tosub is a tuple of tuples, then the inner tuple,
        in the form (x, y) is added to the x and y of self and the
        corresponding point2d object is made and appended to tuple to
        be returned.

        copy_ applies only to cases 1,2 and 3 of above 4 cases

        RESTRICTIONS:
            tosub: if not cases 1 and 2, tosub MUST be in tuple format
        '''
        if isinstance(tosub, point3d):  # CASE-1
            if copy_:
                return self.make_new(self.x-tosub.x, self.y-tosub.y, self.z-tosub.z)
            else:
                self.x -= tosub.x
                self.y -= tosub.y
                self.z -= tosub.z
        if isinstance(tosub, int) or isinstance(tosub, float):
            if copy_:
                return self.make_new(self.x-tosub, self.y-tosub, self.z-tosub)
            else:
                self.x -= tosub
                self.y -= tosub
                self.z -= tosub
        if isinstance(tosub, tuple):
            to_return = ()
            count = 0
            for tosub_ in tosub:
                if isinstance(tosub_, int) or isinstance(tosub_, float):
                    to_return += (self.make_new(self.x-tosub_,
                                  self.y-tosub_, self.z-tosub_),)
                elif isinstance(tosub_, tuple):
                    to_return += (self.make_new(self.x -
                                  tosub_[0], self.y-tosub_[1], self.z-tosub_[2]),)
                elif isinstance(tosub_, point3d):
                    to_return += (self.make_new(self.x-tosub_.x,
                                  self.y-tosub_.y, self.z-tosub_.z),)
                count += 1
                if count == len(tosub):
                    return to_return
    # -----------------------------------------------------

    def __mul__(self,
                multiplier=1.0
                ):
        ''''''
        if isinstance(multiplier, float) or isinstance(multiplier, int):
            return self.make_new(self.x*multiplier, self.y*multiplier, self.z*multiplier)
        elif isinstance(multiplier, point3d):
            return self.make_new(self.x*multiplier.x, self.y*multiplier.y, self.z*multiplier.z)
    # -----------------------------------------------------

    def __truediv__(self,
                    divisor=1.0
                    ):
        ''''''
        if isinstance(divisor, float) or isinstance(divisor, int):
            if not divisor == 0:
                return self.make_new(self.x/divisor, self.y/divisor, self.z/divisor)
            else:
                return f'ERROR: Division by zero'
        elif isinstance(divisor, point3d):
            return self.make_new(self.x/divisor.x, self.y/divisor.y, self.z/divisor.z)
    # -----------------------------------------------------
    # -----------------------------------------------------

    def __repr__(self):
        '''
        instance representation.
        '''
        return f'upxo.p3d({round(self.x, 5)}, {round(self.y, 5)}, {round(self.z, 5)})'
    # -----------------------------------------------------

    def make_mid(self):
        '''
        assign_mid Assign the memory address id of self to self.mid
        '''
        self.mid = id(self)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # START OF STATUS METHODS
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    @property
    def _status_(self):
        '''
        Status update to developer. Not aimed at users
        '''
        if self.has('mid'):
            print(f'mid = {self.rid}')
        else:
            print(f'mid does not exist. It will be made now.')
            self.make_mid()
            self._status_
        print('___coordinates___')
        print('(x, y) = ({self.x}, {self.y})')
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # END OF STATUS METHODS
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # END OF ID AND POINT TYPE ASSIGNMENT METHODS
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # START OF BORROWERS
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    def borrow_tdist(self,
                     point_object=None
                     ):
        if hasattr(point_object, 'tdist'):
            self.tdist = point_object.tdist
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # END OF BORROWERS
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # START OF HASSERS
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    def has(self, suffix):
        suffix = suffix
        if suffix in ('tdist', 'toldist', 'tolerance'):
            suffix = 'tdist'
        elif suffix in ('jn', 'bjn', 'bj_n', 'j_n', 'jporder', 'xvo', 'xtal_vertex_order'):
            suffix = 'jn'
        elif suffix in ('ptype', 'point_type'):
            suffix = 'ptype'
        elif suffix in ('loc', 'pxtal_loc'):
            suffix = 'loc'
        elif suffix in ('rid', 'randid', 'randomid', 'random_id'):
            suffix = 'rid'
        elif suffix in ('mid', 'omid', 'memory_id', 'mem_id', 'memid', 'object_memory_id'):
            suffix = 'mid'
        elif suffix in ('dim', 'dimensionality'):
            suffix = 'dim'
        elif suffix in ('sfv_pol_area', 'area'):
            suffix = 'sfv_pol_area'
        elif suffix in ('ar', 'aspect_ratio'):
            suffix = 'ar'
        elif suffix in ('phase_id', 'phase_id', 'phase_id_number'):
            suffix = 'phase_id'
        elif suffix in ('phase_name', 'phasename', 'phase_name', 'phase'):
            suffix = 'phase_name'
        elif suffix in ('tcname', 'texcomp', 'texture_component', 'ori_name', ''):
            suffix = 'tcname'
        elif suffix in ('eaunit', 'euler_angle_unit', 'eaunits', 'euler_angle_units'):
            suffix = 'eaunit'
        elif suffix in ('sfv_repr_ea', 'eatype', 'ea_repr', 'ea_type', 'ea_representation'):
            suffix = 'sfv_repr_ea'
        elif suffix in ('ea', 'ea_val', 'eaval', 'ea_value', 'eulerangle', 'orientation'):
            suffix = 'ea'
        # ...................
        has_truth = False
        if hasattr(self, suffix):
            has_truth = True
        return has_truth
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # END OF HASSERS
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # START OF OBJECT RE-CREATION CALCULATION METHODS
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    def make_new(self,
                 x=0.0,
                 y=0.0,
                 z=0.0,
                 lean='low',
                 set_mid=True,
                 set_dim=True,
                 set_ptype=False,
                 store_original_coord=True,
                 set_tdist=True,
                 tdist=0.0000000000001,
                 ):
        '''
        Create a new point object.
        '''
        return point3d(x=x,
                       y=y,
                       z=z,
                       lean=lean,
                       set_mid=set_mid,
                       set_dim=set_dim,
                       set_ptype=set_ptype,
                       store_original_coord=store_original_coord,
                       set_tdist=set_tdist,
                       tdist=tdist,
                       )
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # END OF OBJECT RE-CREATION CALCULATION METHODS
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # START OF KINEMATICS METHODS
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    def negx(self):
        ''' Mirror about y-axis '''
        self.x = -self.x
        return self

    def negy(self):
        ''' Mirror about x-axis '''
        self.y = -self.y
        return self

    def negz(self):
        ''' Mirror about x-axis '''
        self.z = -self.z
        return self

    def negxy(self):
        ''' Mirror about y = -x axis '''
        self.x, self.y = -self.x, -self.y
        return self

    def negyz(self):
        ''' Mirror about y = -x axis '''
        self.y, self.z = -self.y, -self.z
        return self

    def negzx(self):
        ''' Mirror about y = -x axis '''
        self.z, self.x = -self.z, -self.x
        return self

    def negxyz(self):
        ''' Mirror about y = -x axis '''
        self.x, self.y, self.z = -self.x, -self.y, -self.z
        return self
    # -----------------------------------------------------

    def displace_by(self,
                    delx: float = 0.0,
                    dely: float = 0.0,
                    delz: float = 0.0
                    ):
        '''
        Move self by xdisp and ydisp
        '''
        self.x, self.y, self.z = self.x+delx, self.y+dely, self.z+delz
    # -----------------------------------------------------

    def move_to(self,
                xlocation: float = 0.0,
                ylocation: float = 0.0,
                zlocation: float = 0.0
                ):
        '''
        Move self to (locx, locy)
        '''
        self.x, self.y, self.z = xlocation, ylocation, zlocation
    # -----------------------------------------------------

    def align_to(self,
                 method='point2d',
                 ref_point_object=None,
                 xlocation: float = None,
                 ylocation: float = None,
                 zlocation: float = None
                 ):
        '''
        Align self to a new point

        NOTE: This is akin to alignTo of vedo. See below.
        https://vedo.embl.es/autodocs/content/vedo/pointcloud.html#vedo.pointcloud.Points
        '''
        if method == 'point2d':
            self.x, self.y, self.z = ref_point_object.x, ref_point_object.y, ref_point_object.z
        elif method == 'coord':
            self.x, self.y, self.z = xlocation, ylocation, zlocation
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # END OF KINEMATICS METHODS
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#################################################################################
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # START OF RESET METHODS
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    def reset(self):
        '''
        Reset's location of self to UPXO default point2d location'
        '''
        if self.store_original_coord:
            if hasattr(self, '_original_location'):
                self.x, self.y, self.z = self._original_location
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # END OF RESET METHODS
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # START OF ATTACH METHODS
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    def attach_mulpoint(self,
                        mulpointObject=None
                        ):
        '''
        Attach this object to a multi-point object
        '''
        self._mulpoints_ = mulpointObject
    # -----------------------------------------------------

    def attach_edge(self,
                    edgeObject=None
                    ):
        '''
        Attach this object to an edge object
        '''
        self._edges_ = edgeObject
    # -----------------------------------------------------

    def attach_surface(self,
                       surfaceObject=None
                       ):
        '''
        Attach this object to a surface object
        '''
        self._surfaces_ = surfaceObject
    # -----------------------------------------------------

    def attach_surface(self,
                       volume_object=None
                       ):
        '''
        Attach this object to the volume object
        '''
        pass
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # END OF ATTACH METHODS
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # START OF SETTERS
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    def set_dim(self,
                dim=3):
        self.dim = dim
    # -----------------------------------------------------

    def set_tdist(self,
                  tdist=0.0):
        self.tdist = tdist
    # -----------------------------------------------------

    def set_jn(self,
               jn=3
               ):
        """
        jtype: number of l0e emanating from it / terminating in it.
        """
        self.jn = jn
    # -----------------------------------------------------

    def set_ptype(self,
                  ptype='jp'
                  ):
        '''
        Assign which type of point it is going to be or it is.
        CASES INCLUDE THE FOLLOWING:
            1. jp: Junction poinzt
            2. gbp: grain boundary point
            3. # TODO: write the many more that there are.
        '''
        self.ptype = ptype
    # -----------------------------------------------------

    def set_location(self,
                     loc='pxtal_internal'
                     ):
        '''
        assign a desctiptive 1-word string.
        Suggested:
            1. pxtal_internal / pxtal_boundary
            2. on_grain_boundary
            3. particle_seed
            4. VTGS_seed
            5. voxel_centroid
            6. voxel_corners
        '''
        self.loc = loc
    # -----------------------------------------------------

    def set_vis_prop(self,
                     mtype='o',
                     mew=1.0,
                     mec='k',
                     msz=10,
                     mfill='w',
                     malpha=1.0,
                     bfill='teal'
                     ):
        """
        set_marker_prop Set some properties of markers

        Args:
            mtype (str): marker type. Defaults to 'o'.
            mew (float): marker edge width. Defaults to 1.0.
            mec (str): marker edge colour. Defaults to 'k'.
            msz (int): marker size. Defaults to 10.
            mfill (str): marker fill. Defaults to 'w'.
            malpha (float): marker alpha. Defaults to 1.0.
            bfill (float): marker buffer fill. Defaults to 'teal'.

        Returns:
            dict: A dictionary of visualization parameters
        """
        vprop = {'mtype': mtype,
                 'mew': mew,
                 'mec': mec,
                 'msz': msz,
                 'mfill': mfill,
                 'malpha': malpha,
                 'bfill': bfill
                 }
        return vprop
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # START OF SCALAR ASSIGNMENT METHODS
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    def set_area(self,
                 value: float = 0.0
                 ):
        self.area = area
    # ---------------------------------------

    def set_ar(self,
               value: float = 0.0,
               ):
        self.ar = ar
    # ---------------------------------------

    def set_morph_curvature(self,
                            value: float = 0.0,
                            ):
        pass
    # ---------------------------------------

    def set_gnd(self,
                value: float = 0.0,
                ):
        pass
    # ---------------------------------------

    def set_lattice_curvature_(self,
                               value: list[list] = [[0.0, 0.0, 0.0],
                                                    [0.0, 0.0, 0.0],
                                                    [0.0, 0.0, 0.0]
                                                    ]
                               ):
        pass
    # ---------------------------------------

    def set_phase(self,
                  phase_id: int = 0,
                  phase_name: str = 'cucrzr'
                  ):
        """
        set_phase Assign phase id and phase name to point

        Args:
            pid (int, optional): _description_. Defaults to 0.
            pname (str, optional): _description_. Defaults to 'cucrzr'.
        """
        self.phase_id = phase_id
        self.phase_name = phase_name
    # ---------------------------------------

    def set_tcname(self,
                   name: str = 'B'
                   ):
        """
        set_tcname Assign texture component name to point

        Args:
            name (str, optional): _description_. Defaults to 'B'.
        """
        self.tcname = name
    # ---------------------------------------

    def set_sfv_repr_ea(self,
                        sfv_repr_ea='Bunge'
                        ):
        """
        sfv_repr_ea Assign the default representation of Euler angles

        Args:
            repr (str, optional): _description_. Defaults to 'Bunge'.
        """
        self.sfv_repr_ea = sfv_repr_ea
    # ---------------------------------------

    def set_sfv_ea1(self,
                    ea1: float = 0,
                    ):
        """
        ea1 assign 1st Euler angle to point object

        Args:
            ea1 (float, optional): _description_. Defaults to 0.
        """
        self.ea[0] = ea1
    # ---------------------------------------

    def set_sfv_ea2(self,
                    ea2: float = 0,
                    ):
        """
        ea2 assign 2nd Euler angle to point object

        Args:
            ea2 (float, optional): _description_. Defaults to 0.
        """
        self.ea[1] = ea2
    # ---------------------------------------

    def set_sfv_ea3(self,
                    ea3: float = 0,
                    ):
        """
        ea3 assign 3rd Euler angle to point object

        Args:
            ea2 (float, optional): _description_. Defaults to 0.
        """
        self.ea[2] = ea3
    # ---------------------------------------

    def set_sfv_ea(self,
                   ea: list = [0, 0, 0]
                   ):
        """
        ea Assign the three Euler angles to point object

        Args:
            ea (list, optional): _description_. Defaults to [0, 0, 0].
        """
        self.ea = ea
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # END OF SCALAR ASSIGNMENT METHODS
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


class point3d_lean_highest():
    """
    UPXO core class.
    Represents a point in 3D Cartesian space.
    Leanness is highest, meaning only stores the coordinates x, y and z

    NOTES TO USER:
        1. x, y and z must be floats
    """
    __slots__ = ('x', 'y', 'z')

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


class point3d_lean_highest_mc0():
    """
    UPXO core class.
    Represents a point in 3D Cartesian space, made for Monte-Carlo simulation
    Leanness is highest, meaning only stores the coordinates x and y and the
    integer state, s

    NOTES TO USER:
        1. x, y and z must be floats
        2. s must be an int
    """
    __slots__ = ('x', 'y', 'z', 's')

    def __init__(self,
                 x: float = 0.0,
                 y: float = 0.0,
                 z: float = 0.0,
                 s: int = 1
                 ):
        self.x, self.y, self.z, self.s = x, y, z, s
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


class point3d_lean_highest_mc1():
    """
    UPXO core class.
    Represents a point in 3D Cartesian space, made for Monte-Carlo simulation
    Leanness is highest, meaning only stores the coordinates x and y and the
    integer state, s

    NOTES TO USER:
        1. x, y and z must be floats
        2. s must be an int
        3. h. no input necessary. Taken internally
    """
    __slots__ = ('x', 'y', 'z', 's', 'h')

    def __init__(self,
                 x: float = 0.0,
                 y: float = 0.0,
                 z: float = 0.0,
                 s: int = 1,
                 h: float = 0.0
                 ):
        self.x, self.y, self.z = x, y, z
        self.s, self.h = s, h
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


class point_q_space():
    """
    Class representing Quaternion space
    """
    __slots__ = ('q1', 'q2', 'q3', 'q4')

    def __init__(self,
                 q1=0.0,
                 q2=0.0,
                 q3=0.0,
                 q4=0.0
                 ):
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3
        self.q4 = q4
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


class point_be_space():
    """
    Class representing Bunge's Euler space
    """
    __slots__ = ('phi1', 'psi', 'phi2')

    def __init__(self,
                 phi1=0.0,
                 psi=0.0,
                 phi2=0.0
                 ):
        self.phi1 = phi1
        self.psi = psi
        self.phi2 = phi2
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


class point_gsm_stat_space():
    """
    Class representing a sub-space of statistics in grain structure morphology
    space

    The sub-space can be used to represent grain structure parameters such as:
        area, perimeters, junction point order, grain boundary tortuosity, etc..
    """
    __slots__ = ('amin', 'amax', 'astd', 'avar',
                 'aper05', 'aper25', 'aper50', 'aper75', 'aper95',
                 'amodality', 'amodes', 'akurt', 'askew'
                 )

    def __init__(self,
                 minn=0.0,
                 maxx=100.0,
                 std=50.0,
                 var=10.0,
                 per05=5.0,
                 per25=25.0,
                 per50=50.0,
                 per75=75.0,
                 per95=95.0,
                 modality=2.0,
                 modes=[15.0, 60.0],
                 widths=[5.0, 20.0],
                 kurt=0.0,
                 skew=0.0,
                 ):
        self.minn = minn
        self.maxx = maxx
        self.std = std
        self.var = var
        self.per05 = per05
        self.per25 = per25
        self.per50 = per50
        self.per75 = per75
        self.per95 = per95
        self.modality = modality
        self.modes = modes
        self.widths = widths
        self.kurt = kurt
        self.skew = skew
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


class point_tc_stat_space():
    __slots__ = ('name',
                 'ghw',
                 'vfraction',
                 'intensity'
                 )

    def __init__(self,
                 name='',
                 ghw=5.0,
                 vfraction=0.1,
                 intensity=2.0,
                 ):
        self.name = name
        self.ghw = ghw
        self.vfraction = vfraction
        self.intensity = intensity
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
