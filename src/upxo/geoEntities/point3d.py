import numpy as np
import upxo._sup.dataTypeHandlers as dth

class point_ops2d():
    __slots__ = ('x', 'y')
    def __init__(*self):
        self.x = 0
        self.y = 0
    def __repr__(self):
        return f"({self.x}, {self.y})"

class p2d_leanest():
    """
    Leanest redefinition of 2d point class. Intended for private use only.

    Author
    ------
    Dr. Sunil Anandatheertha

    @dev
    ----
    Restrict any further development

    Examples
    --------
    from upxo.geoEntities.point3d import p2d_leanest
    a = [p2d_leanest(1, 2), p2d_leanest(1, 2)]

    # Extension: Check if all of the above list belong to the same type
    all_isinstance(p2d_leanest, a)
    """

    __slots__ = ('_x', '_y')

    def __init__(self, x, y):
        self._x, self._y = x, y

    def __repr__(self):
        """Return string representation of self."""
        return f'Lean 2D point at ({self._x}, {self._y})'


class p3d_leanest():
    """
    Leanest redefinition of 3d point class. Intended for private use only.

    Author
    ------
    Dr. Sunil Anandatheertha

    @dev
    ----
    Restrict any further development

    Examples
    --------
    from upxo.geoEntities.point3d import p3d_leanest
    a = [p3d_leanest(1, 2, 0), p3d_leanest(1, 2, 0)]
    """

    __slots__ = ('_x', '_y', '_z')

    def __init__(self, x, y, z):
        self._x, self._y, self._z = x, y, z

    def __repr__(self):
        """Return string representation of self."""
        return f'Lean 3D point at ({self._x}, {self._y}, {self._z})'


def isinstance_many(tocheck, dtype):
    """
    Check if all elements of tocheck belongs to a valid dtype.

    Arguments
    ---------
    tocheck: An iterable of data.
    dtype: Valid datatype, in dth.dt.ITERABLES

    Return
    ------
    list of bools. True indicates element belonging to dtype

    Example
    -------
    from upxo.geoEntities.point3d import p2d_leanest, p3d_leanest
    a = [p2d_leanest(1, 2), p3d_leanest(1, 2, 1)]
    isinstance_many(a, p3d_leanest)

    Author
    ------
    Dr. Sunil Anandatheertha
    """
    if type(tocheck) not in dth.dt.ITERABLES:
        tocheck = (tocheck, )
    return [isinstance(tc, dtype) for tc in tocheck]

def get_upxo_p2d(p2d):
    """
    Example
    -------
    1.
    p2d = [[1, 2], [3, 4]]
    get_upxo_p2d(p2d)

    2.
    p2d = [[1, 2, 3, 4], [3, 2, 1, 4]]
    get_upxo_p2d(p2d)
    """
    from upxo.geoEntities.point3d import p2d_leanest
    if dth.DEEPCHECK_is_coord2d_list(p2d):
        '''Then, p2d must like [[1, 2], [3, 4]]'''
        return [p2d_leanest(x=xy[0], y=xy[1]) for xy in p2d]
    elif dth.DEEPCHECK_is_xy2d_list(p2d):
        '''Then, p2d must like [[1, 2, 3, 4], [3, 2, 1, 4]]'''
        return [p2d_leanest(x=_x, y=_y) for _x, _y in zip(p2d[0], p2d[1])]

def get_upxo_p2d(p2d):
    from upxo.geoEntities.point3d import p2d_leanest as p
    all_isinstance(np.ndarray, p2d)


def all_isinstance(dtype, *args):
    if len(args) > 0:
        print(args)
        return all(isinstance(arg, dtype) for arg in args)


class point3d():
    ε = 0.000000000001
    __slots__ = ('x', 'y', 'z', 'mp', 'gmsh')

    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z
    def __eq__(self, p3dlist):
        pass
    def __ne__(self, p3dlist):
        pass
    def __repr__(self):
        pass
    def from_p2d(self, p2dlist=None, *, plane=[None, None, 0]):
        pass
    def distance(self, *, p3dlist=None):
        pass
    def translate_by(self, *, vector=None, dist=None, make_new=False):
        pass
    def translate_to(self, *, upxo_p2d=None, make_new=False):
        pass
    def snap_to(self, *, p3d=None):
        pass
    def rotate_about(self, *, p2d=None, axis=None):
        pass
    def attach_mp(self, *, mp=None, name=None):
        self.mp[name] = mp
    def attach_xtal(self, *, xtals=None):
        pass
    def make_edges(self, *, p3dlist=None, saa=False, throw=True):
        # VAIDATE p3dlist
        if not throw and not saa:
            # Nothing to do here
            pass
        if saa and not throw:
            # MAKE EDGE3D OBJECT
            pass
        if saa and throw:
            # MAKE EDGE3D OBJECTS
            # STORE EDGE3D BJECTS
            pass
    def front(self, p3dlist):
        pass
    def back(self, p3dlist):
        pass
    def left(self, p3dlist):
        pass
    def right(self, p3dlist):
        pass
    def top(self, p3dlist):
        pass
    def bottom(self, p3dlist):
        pass
    def find_nearest_point_neigh_by_distance(self, *, p3dlist=None):
        pass
    def find_nearest_point_neigh_by_count(self, *, p3dlist=None):
        pass
    def find_nearest_mulpoint_neigh_by_distance(self, *, e3dlist=None):
        pass
    def find_nearest_mulpoint_neigh_by_count(self, *, e3dlist=None):
        pass
    def find_nearest_edge_neigh_by_distance(self, *, e3dlist=None):
        pass
    def find_nearest_edge_neigh_by_count(self, *, e3dlist=None):
        pass
    def find_nearest_muledge_neigh_by_distance(self, *, me3dlist=None):
        pass
    def find_nearest_muledge_neigh_by_count(self, *, me3dlist=None):
        pass
    def find_nearest_xtal_neigh_by_distance(self, *, xlist=None):
        pass
    def find_nearest_xtal_neigh_by_count(self, *, xlist=None):
        pass
    def find_nearby_edges(self, *, e3dlist=None):
        pass
    def set_gmsh_props(self, prop_dict):
        pass
    def become_voxel(self, l=1):
        pass
    def become_c3d8(self, l=1):
        pass
    def become_c2d20(self, l=1):
        pass
