import numpy as np
from abc import ABC, abstractmethod
import upxo._sup.dataTypeHandlers as dth


class Point(ABC):
    """Template base class for point object. Expands to both 2D and 3D."""

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __repr__(self):
        """docstring."""
        pass

    @abstractmethod
    def __eq__(self, p3dlist):
        """Check if the two points are coincident."""
        pass

    @abstractmethod
    def __ne__(self, p3dlist):
        """Check if the two points are not coincident."""
        pass

    @abstractmethod
    def distance(self, *, plist=None):
        """Calculate the EUclidean distance between self and list of points."""
        pass

    @abstractmethod
    def translate_by(self, *, vector=None, dist=None,
                     update=False, throw=True):
        """
        Translate the point by a Euclidean distance.

        Translate the ppoint along the vector by a given distance. If
        update is True, then coords of the self will be updated.

        If throw is True and update is False, a new point of the new
        coordinates shall be returned. If throw is True and update is True, a
        deepcopy of the self shall be returned.

        Parameters
        ----------
        vector: Direction of translation. Two specifications allowed are:
             Specification 1: [vector start point coords,
                               vector end point coords]
             Specification 2: 'x+', 'z-'

        dist: Euclidean distance

        update: Update the current point if True, do not update if False.

        throw: Return a point if True, else return nothing if False.

        Return
        ------
        UPXO point object: Conditional, depending on input throw (refer to
                                                                  description).
        """
        pass

    @abstractmethod
    def translate_to(self, *, point=None, update=False, throw=True):
        """
        Translate self to the specified location.

        New location is specified by point object. POint object could be
        specified by an another UPXO point object or an Iterable of coords.

        If throw is True and update is False, a new point of the new
        coordinates shall be reyturned. If throw is True and update is True, a
        deepcopy of the self shall be returned.

        Parameters
        ----------
        point: New position. UPXO / direct point specification.

        update: Update the current point if True, do not update if False.

        throw: Return a point if True, else return nothing if False.

        Return
        ------
        UPXO point object: Conditional, depending on input throw (refer to
                                                                  description).
        """
        pass

    @abstractmethod
    def rotate_about(self, *, axis=None, angle=0, degree=True,
                     update=False, throw=True):
        """
        Rotate point about the specified axis by the specifi3ed angle.

        New location is specified by point object. Point object could be
        specified by an another UPXO point object or an Iterable of coords.

        If throw is True and update is False, a new point of the new
        coordinates shall be reyturned. If throw is True and update is True, a
        deepcopy of the self shall be returned.

        Parameters
        ----------
        axis: Axis of rotation. Two specifications allowed are:
             Specification 1: [axis start point coords,
                               axis end point coords]
             Specification 2: 'x+', 'z-'

        angle: Counter-Clockwise positive, angle of rotation

        degree: angle considered in degrees if True, radians if False.

        update: Update the current point if True, do not update if False.

        throw: Return a point if True, else return nothing if False.

        Return
        ------
        UPXO point object: Conditional, depending on input throw (refer to
                                                                  description).
        """
        pass

    @abstractmethod
    def attach_mp(self, *, mp=None, name=None):
        """Attach a UPXO multi-poiont object and a name."""
        self.mp[name] = mp

    @abstractmethod
    def attach_xtal(self, *, xtals=None):
        """Attach a list of UPXO xtal objects."""
        pass

    @abstractmethod
    def find_neigh_point_by_distance(self, *, plist=None, plane='xy', r=0):
        """
        Find the nearest neighbouring point(s) in specified list of points.

        Parameters
        ----------
        plist: Elements of plist must contain the coordinates either in direct
        Iterable form  (such a list of [x, y] or a nparray np.array([x, y]))
        OR a 2D/3D UPXO point object.

        plane: Specify the plane of the self point. Only used if self is a 2D
        point object. Defaults to 'xy'.

        r: Euclidean radius of search.
           If 0, the closest point will be looked out for.
           If > 0, all points which fall in or on a circle of radius r will be
           looked out for.

        Return
        ------
        Indices in plist. Empty list if no points are inside r.
        """
        pass

    @abstractmethod
    def find_neigh_point_by_count(self, *, plist=None, n=None,
                                  plane='xy'):
        """
        Find n nearest neighbouring points in a specified list of points.

        Parameters
        ----------
        plist:  Elements of plist must contain the coordinates either in direct
        Iterable form  (such a list of [x, y] or a nparray np.array([x, y]))
        OR a 2D/3D UPXO point object.

        n: Number of nearest neighbours to return. If not entered, a single
        point shall be returned.

        plane: Specify the plane of the self point. Only used if self is a 2D
        point object. Defaults to 'xy'.

        Return
        ------
        Indices in plist.
        """
        pass

    @abstractmethod
    def find_neigh_mulpoint_by_distance(self, *, mplist=None,
                                        plane='xy', r=0, tolf=-1):
        """
        Find the nearest UPXO multi-point in specified list of UPXO mulpoints.

        If tolerance factor, tolf is provided to be -1, then, even if a single
        point in a mp falls in  or on r, the index of mp in mplist will be
        added to the list to be returned. If 0 < tolf <= 1, then even if tolf
        factor of total number of points in a mp falls inside r, then the index
        of this mp in mplist will be added to the list to be returned.

        Parameters
        ----------
        mplist: Elements of plist must contain the coordinates either in direct
        Iterable form  (such a list of [x, y] or a nparray np.array([x, y]))
        OR a 2D/3D UPXO point object.

        plane: Specify the plane of the self point. Only used if self is a 2D
        point object. Defaults to 'xy'.

        r: Euclidean radius of search.
           If 0, the closest point will be looked out for.
           If > 0, all points which fall in or on a circle of radius r will be
           looked out for.

        Return
        ------
        Indices in mplist.
        """
        pass

    @abstractmethod
    def find_neigh_mulpoint_count(self, *, mplist=None):
        pass

    @abstractmethod
    def find_neigh_edge_by_distance(self, *, elist=None,
                                    plane='xy', refloc='starting', r=0):
        """
        Find the nearest UPXO edge in specified list of UPXO edges.

        Parameters
        ----------
        elist: Elements of elist must contain edges in either of the following
            two formats:
                1. 2D/3D UPXO edge objects.
                2. Iterable object with elements starting point: [x0, y0, z0]
                   and ending point: [x1, y1, z1]

        plane: Specify the plane of the self point. Only used if self is a 2D
        point object. Defaults to 'xy'.

        refloc: Specify the location on th edge which is to be used for
        calculating the distance form the self point itself and the edge. It
        can have the folloqwing optrions:
            * 'starting'. Starting point of the edge. Alternative use: start
            * 'ending'. Ending point of the edge. Alternative use: end
            * 'middle'. Mid point of the edge.  Alternative use: mid
            * 'any'. Any point of the edge.  No altewrnate.
            * 'all'. Both start and end points of the edge.  No alterate.

        r: Euclidean radius of search.
           If 0, the closest point will be looked out for.
           If > 0, all points which fall in or on a circle of radius r will be
           looked out for.

        Return
        ------
        Indices in mplist.
        """
        pass

    @abstractmethod
    def find_neigh_edge_by_count(self, *, elist=None):
        pass

    @abstractmethod
    def find_neigh_muledge_by_distance(self, *, melist=None,
                                       plane='xy', refloc='starting', r=0):
        """
        Find the nearest UPXO muledge in specified list of UPXO muledges.

        Parameters
        ----------
        melist: Elements of melist must contain medges in either of the following
            single format:
                1. 2D/3D UPXO medge objects.

        plane: Specify the plane of the self point. Only used if self is a 2D
        point object. Defaults to 'xy'.

        refloc: Specify the location on th edge which is to be used for
        calculating the distance form the self point itself and the edge. It
        can have the folloqwing optrions:
            * 'starting'. Starting point of the edge. Alternative use: start
            * 'ending'. Ending point of the edge. Alternative use: end
            * 'middle'. Mid point of the edge.  Alternative use: mid
            * 'any'. Any point of the edge.  No altewrnate.
            * 'all'. Both start and end points of the edge.  No alterate.

        r: Euclidean radius of search.
           If 0, the closest point will be looked out for.
           If > 0, all points which fall in or on a circle of radius r will be
           looked out for.

        Return
        ------
        Indices in mplist.
        """
        pass

    @abstractmethod
    def find_neigh_muledge_by_count(self, *, me3dlist=None):
        pass

    @abstractmethod
    def find_neigh_xtal_by_distance(self, *, xlist=None,
                                    plane='xy', refloc='starting', r=0):
        """
        Find the nearest UPXO xtal in specified list of UPXO xtals.

        Parameters
        ----------
        xlist: Elements of xlist must contain xtals in either of the
        following three formats:
                1. 2D/3D UPXO xtal objects.
                2. Shapely polygon object.
                3. GMSH closed region.
                4. VTK polyhedra object.

        plane: Specify the plane of the self point. Only used if self is a 2D
        point object. Defaults to 'xy'.

        refloc: Specify the location on th edge which is to be used for
        calculating the distance form the self point itself and the edge. It
        can have the folloqwing optrions:
            * 'starting'. Starting point of the edge. Alternative use: start
            * 'ending'. Ending point of the edge. Alternative use: end
            * 'middle'. Mid point of the edge.  Alternative use: mid
            * 'any'. Any point of the edge.  No altewrnate.
            * 'all'. Both start and end points of the edge.  No alterate.

        r: Euclidean radius of search.
           If 0, the closest point will be looked out for.
           If > 0, all points which fall in or on a circle of radius r will be
           looked out for.

        Return
        ------
        Indices in mplist.
        """
        pass

    @abstractmethod
    def find_neigh_xtal_by_count(self, *, xlist=None):
        pass

    @abstractmethod
    def set_gmsh_props(self, prop_dict):
        pass

    @abstractmethod
    def make_shapely(self):
        """Return shapely point object. Only valid for 2D."""
        pass

    @abstractmethod
    def make_vtk(self):
        """Make VTK point object."""
        pass

    @property
    @abstractmethod
    def coords(self):
        """Return coordinate array"""
        return np.array([self.x, self.y])

    @abstractmethod
    def array_translation(self, *,
                          ncopies=10,
                          vector=[[0, 0, 0], [0, 0, 1]],
                          spacing='constant'):
        """
        Make an array of points by repeat-translating self.

        Parameters
        ----------
        ncopies:
        vector:
        spacing:

        Return
        ------
        list of point objects
        """

    @abstractmethod
    def lies_on_which_edge(self, *, elist=None, consider_ends=True):
        """
        Get indices from a list of edges, which contain the self point.

        The point could lie on the end points of an edge or in-between the
        two end points of an edge.

        Parameters
        ----------
        elist: list of edge objects

        consider_ends: If True, index of the edge containing the selfpoint
        coordinates at one of its end points will also be returned. If False,
        the index will be included only if the point is not on the end
        points but completely inside the edge's end points.

        Return
        ------
        List of indices of points which satisfy the condition.
        """
        pass

    @abstractmethod
    def lies_in_which_xtal(self, *, xlist=None,
                           cosider_boundary=True,
                           consider_boundary_ends=True):
        """
        Get indices from a list of xtals, which contain the self point.

        The point could lie inside the xtal, on the boundaries of the xtal
        or on one of the end points of the many edges of the xtal.

        Parameter
        ---------
        xlist: list of edge objects

        consider_boundary: If True, search will be carried out to see if the
        self point lies on one of the boundary edges of the xtal. How this
        search behaves is decided by consider_boundary_ends.

        consider_boundary_ends: If True, index of the xtal containing the
        selfpoint coordinates at one of the end points of its the xtal's
        many boundary edges will be returned. If False, the index will be
        included only if the point is not on the end points but completely
        inside the edge's end points, provided that the cosider_boundary is
        True.

        Return
        ------
        List of indices of points which satisfy the condition.
        """
        pass


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
    def find_neigh_point_neigh_by_distance(self, *, p3dlist=None):
        pass
    def find_neigh_point_neigh_by_count(self, *, p3dlist=None):
        pass
    def find_neigh_mulpoint_neigh_by_distance(self, *, e3dlist=None):
        pass
    def find_neigh_mulpoint_neigh_by_count(self, *, e3dlist=None):
        pass
    def find_neigh_edge_neigh_by_distance(self, *, e3dlist=None):
        pass
    def find_neigh_edge_neigh_by_count(self, *, e3dlist=None):
        pass
    def find_neigh_muledge_neigh_by_distance(self, *, me3dlist=None):
        pass
    def find_neigh_muledge_neigh_by_count(self, *, me3dlist=None):
        pass
    def find_neigh_xtal_neigh_by_distance(self, *, xlist=None):
        pass
    def find_neigh_xtal_neigh_by_count(self, *, xlist=None):
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
