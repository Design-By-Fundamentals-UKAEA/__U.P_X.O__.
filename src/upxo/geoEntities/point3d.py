"""
Core module of UKAEA Poly-XTAL Operations.

Authors
-------
Dr. Sunil Anandatheertha
"""

import math
from copy import deepcopy
import numpy as np
from abc import ABC, abstractmethod
import upxo._sup.dataTypeHandlers as dth
from scipy.spatial import cKDTree

class Point(ABC):
    """Template base class for point object. Expands to both 2D and 3D."""

    __slots__ = ('x', 'y', 'f')

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __repr__(self):
        """docstring."""
        pass

    @abstractmethod
    def __eq__(self, plist, *, use_tol=True):
        """Check if the two points are coincident."""
        pass

    @abstractmethod
    def __ne__(self, plist, *, use_tol=True):
        """Check if the two points are not coincident."""
        pass

    @abstractmethod
    def __add__(self, distances, update=True, throw=False):
        pass

    @abstractmethod
    def __sub__(self, distances, update=True, throw=False):
        pass

    @abstractmethod
    def __mul__(self, factors, update=True, throw=False):
        pass

    @abstractmethod
    def distance(self, plist=None):
        """Calculate the EUclidean distance between self and list of points."""
        pass

    def __hash__(self):
        '''return hash((self.x, self.y))'''
        pass
    def __lt__(self):
        '''return (self.x, self.y) < (other.x, other.y)'''
        pass
    def __gt__(self):
        '''return (self.x, self.y) > (other.x, other.y)'''
        pass
    def __le__(self):
        '''return (self.x, self.y) <= (other.x, other.y)'''
        pass
    def __ge__(self):
        '''return (self.x, self.y) >= (other.x, other.y)'''
        pass
    def above(self, plist, plane='xy'):
        pass
    def below(self, plist, plane='xy'):
        pass
    def left(self, plist, plane='xy'):
        pass
    def right(self, plist, plane='xy'):
        pass
    def top(self, plist, plane='xy'):
        pass
    def bottom(self, plist, plane='xy'):
        pass
    def dot(self, plist, plane='xy'):
        pass
    def cross(self, plist, vlist):
        pass
    def norm(self):
        pass
    def collinear(self, pi, pj):
        pass
    def angle_with(self, plist):
        """
        def angle_with(self, other):
            dot_product = self.x * other.x + self.y * other.y
            magnitude_product = self.magnitude() * other.magnitude()
            angle_radians = math.acos(dot_product / magnitude_product)
            return math.degrees(angle_radians)
        """
        pass
    def project_onto(self, vector, plane):
        pass
    @classmethod
    def from_array(cls, arr):
        pass
    @classmethod
    def from_centroid(cls, feature):
        """
        @classmethod
        def centroid(cls, points):
            x_sum = sum(point.x for point in points)
            y_sum = sum(point.y for point in points)
            return cls(x_sum / len(points), y_sum / len(points))
        """
        pass
    @classmethod
    def from_dict(cls, d):
        '''return cls(d['x'], d['y'])'''
        pass
    def to_dict(self):
        '''Serialize the point to a dictionary.'''
        pass
    def lerp(self, point, t):
        """Linearly interpolate between this point and another point by a
        parameter t (0 <= t <= 1)
        return Point2D(self.x + (other.x - self.x) * t, self.y + (other.y - self.y) * t)
        """
        pass
    def reflect_over_axis(self, axis):
        """
        def mirror_across_line(self, line_start, line_end):
            # Formula to calculate the mirrored point across a line
            dx = line_end.x - line_start.x
            dy = line_end.y - line_start.y
            a = (dx * dx - dy * dy) / (dx * dx + dy * dy)
            b = 2 * dx * dy / (dx * dx + dy * dy)
            x_mirror = a * (self.x - line_start.x) + b * (self.y - line_start.y) + line_start.x
            y_mirror = b * (self.x - line_start.x) - a * (self.y - line_start.y) + line_start.y
            return Point2D(x_mirror, y_mirror)
        """
        pass
    def is_on_line(self, start, end):
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

        Author
        ------
        Dr. Sunil Anandatheertha
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
        Author
        ------
        Dr. Sunil Anandatheertha
        """
        pass

    @abstractmethod
    def rotate_about(self, *, point=None, axis=None, angle=0, degree=True,
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
        point: POint about which the axis is considered.

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
        Author
        ------
        Dr. Sunil Anandatheertha
        """
        pass

    @abstractmethod
    def attach_feature(self, *, feature=None, feature_id=None):
        """Attach a feature object and a name."""
        pass

    @abstractmethod
    def find_neigh_point_by_distance(self, *, plist=None, plane='xy', r=0,
                                     on_boundary=True):
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

        Author
        ------
        Dr. Sunil Anandatheertha
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

        Author
        ------
        Dr. Sunil Anandatheertha
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

        Author
        ------
        Dr. Sunil Anandatheertha
        """
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

        Author
        ------
        Dr. Sunil Anandatheertha
        """
        pass

    @abstractmethod
    def find_neigh_muledge_by_distance(self, *, melist=None,
                                       plane='xy', refloc='starting', r=0):
        """
        Find the nearest UPXO muledge in specified list of UPXO muledges.

        Parameters
        ----------
        melist: Elements of melist must contain medges in either of the
        following single format:
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

        Author
        ------
        Dr. Sunil Anandatheertha
        """
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

        Author
        ------
        Dr. Sunil Anandatheertha
        """
        pass

    @abstractmethod
    def set_gmsh_props(self, prop_dict):
        """Set dictionary of gmsh properties."""
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
        """Return coordinate array."""
        pass

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

        Author
        ------
        Dr. Sunil Anandatheertha
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

        Author
        ------
        Dr. Sunil Anandatheertha
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

        Author
        ------
        Dr. Sunil Anandatheertha
        """
        pass


class Edge(ABC):
    """
    Template base class for edge object. Expands to both 2D and 3D.

    Attributes
    ----------
    i: starting point coordinates
    j: ending point coordinates
    """

    __slots__ = ('i', 'j', )

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __repr__(self):
        """docstring."""
        pass

    @abstractmethod
    def __eq__(self, elist):
        """Check if the two edges are coincident."""
        pass

    @abstractmethod
    def __ne__(self, elist):
        """Check if the two edges are not coincident."""
        pass

    @property
    @abstractmethod
    def mid(self):
        """Return the mid point."""
        pass

    @property
    @abstractmethod
    def ang(self):
        """Return the ccw + angle in radians."""
        pass

    @classmethod
    def by_coord(self, start_point, end_point):
        """Create edge by specifying end coordinates."""
        pass

    @classmethod
    def by_loc_len_ang(self, *, ref='i', loc=[0, 0, 0],
                       length=1, ang=0, degree=True):
        """
        Create edge by specifying location, length and angle.

        Parameters
        ----------
        ref: Specifies which point on the edge is used to spcify the edge.

        loc: Specifies the ref point.

        length: Length of the edge to be made.

        ang: Angle(s) of inclination of the edge. If 2D, single value. If 3D,
        this specifies a list of three angles. First angle

        degree: ang considered in degree if degree is True, in radians if
        otherwise.
        """
        pass

    @abstractmethod
    def distance_to_points(self, *, plist=None):
        """Calculate the EUclidean distance between self and list of points."""
        pass

    @abstractmethod
    def distance_to_edges(self, *, elist=None,
                          method='ref', refi='mid', refj='mid'):
        """
        Calculate the Baudhāyana distance between self and list of edges.

        Parameters
        ----------
        elist: List of UPXO edge objects
        method: Indicate the method of calculation. Can take following values:
            * 'ref': Use the referece location specifiers to calculate
            disatance. That is, use refi and refj.
            * 'min': Minimum Baudhāyana distance
            * 'max': MAximum Baudhāyana distance
            * 'mean': Mean Baudhāyana distance
        refi: reference location for self i.e. i
        refj: referecne location for the other edge i.e. j

        Return
        ------
        List of distances to all edges.

        Author
        ------
        Dr. Sunil Anandatheertha
        """
        pass

    @abstractmethod
    def translate_by(self, *, vector=None, dist=None,
                     update=False, throw=True):
        """
        Translate the Edge by a Euclidean distance.

        Translate the Edge along the vector by a given distance. If
        update is True, then coords of the self will be updated.

        If throw is True and update is False, a new edge of the new
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

        throw: Return a edge if True, else return nothing if False.

        Return
        ------
        UPXO edge object: Conditional, depending on input throw (refer to
                                                                 description).

        Author
        ------
        Dr. Sunil Anandatheertha
        """
        pass

    @abstractmethod
    def translate_to(self, *, ref='i', point=None, update=False, throw=True):
        """
        Translate self to the specified location.

        New location is specified by point object. POint object could be
        specified by an another UPXO point object or an Iterable of coords.

        If throw is True and update is False, a new point of the new
        coordinates shall be reyturned. If throw is True and update is True, a
        deepcopy of the self shall be returned.

        Parameters
        ----------
        ref: Location on the edge which translates to new point. Values of ref
        could be:
            * 'i': Starting point of the edge
            * 'j': Ending point of the edge
            * 'mid': Middle point of the edge
            * [x, y, (z)]: Coordinate value

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
    def find_neigh_muledge_by_distance(self, *, melist=None,
                                       plane='xy', refloc='starting', r=0):
        """
        Find the nearest UPXO muledge in specified list of UPXO muledges.

        Parameters
        ----------
        melist: Elements of melist must contain medges in either of the
        following single format:
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
    def set_gmsh_props(self, prop_dict):
        """Set dictionary of gmsh properties."""
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
        """Return coordinate array."""
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


class Point2d(Point):
    ε = 1E-8
    __slots__ = Point.__slots__ + ('f', )

    def __init__(self, x, y):
        super().__init__(x, y)

    def __repr__(self):
        return f"uxpo point.2d: ({self.x}, {self.y})"

    def __eq__(self, plist, *, use_tol=True):
        if not plist:
            raise ValueError("plist is empty.")
        if dth.DEEPCHECK_is_coord2d_list(plist):
            '''Then, plist must be like [[1, 2], [3, 4]]'''
            if use_tol:
                # TODO: Add tol checking functionality
                return [self.x == p[0] and self.y == p[1]
                        for p in plist]
            else:
                return [self.x == p[0] and self.y == p[1]
                        for p in plist]
        elif dth.DEEPCHECK_is_xy2d_list(plist):
            '''Then, plist must be like [[1, 2, 3, 4], [3, 2, 1, 4]]'''
            if use_tol:
                # TODO: Add tol checking functionality
                return [self.x == _x and self.y == _y
                        for _x, _y in zip(plist[0], plist[1])]
            else:
                return [self.x == _x and self.y == _y
                        for _x, _y in zip(plist[0], plist[1])]
        else:
            '''UPXO point objects'''
            # Lets assume for now, plist is a UPXO point object.
            # TODO: Further validations to be done later on.
            if use_tol:
                # TODO: Add tol checking functionality
                return [self.x == p.x and self.y == p.y
                        for p in plist]
            else:
                return [self.x == p.x and self.y == p.y
                        for p in plist]

    def __ne__(self, plist, *, use_tol=True):
        self.__eq__(plist, use_tol=use_tol)

    def __add__(self, distances, update=True, throw=False):
        if type(distances) in dth.dt.NUMBERS:
            # add distances to both x and y
            addtox, addtoy = distances, distances
        if type(distances) in dth.dt.ITERABLES:
            # add distances contents seperatrely to x and y
            # There is a chance that distances[0] coudl result in error.
            # TODO: Include validation to deal wth above line.
            addtox, addtoy = distances[0], distances[1]
        else:
            raise ValueError('Invalid distances type')
        if update:
            self.x += addtox
            self.y += addtoy
        if update and throw:
            return deepcopy(self)
        if not update and throw:
            return Point2D(self.x+addtox, self.y+addtoy)

    def distance(self, plist=None):
        if not plist:
            raise ValueError("plist is empty.")
        if type(plist) not in dth.dt.ITERABLES:
            plist = list(plist)
        if dth.DEEPCHECK_is_coord2d_list(plist):
            '''Then, plist must be like [[1, 2], [3, 4]]'''
            dist = self.distance(plist=np.array(plist).T)
        elif dth.DEEPCHECK_is_xy2d_list(plist):
            '''Then, plist must be like [[1, 2, 3, 4], [3, 2, 1, 4]]'''
            plist = np.array(plist)
            dist = np.sqrt((self.x-plist[0])**2 + (self.y-plist[1])**2)
        elif isinstance(plist, cKDTree):
            # TODO: include statem,ents if plist is a ckdtree
            pass
        else:
            '''UPXO point objects'''
            # Lets assume for now, plist is a UPXO point object.
            # TODO: validations to be done.
            plist = np.array([[p.x, p.y] for p in plist]).T
            dist = self.distance(plist=plist)
        return dist

    def translate_by(self, *, vector=None, dist=None,
                     update=False, throw=True):
        # TODO: Include validations
        distances = (np.array(vector) / np.linalg.norm(vector)) * dist
        if update:
            self.__add__(distances, update=update, throw=throw)
        if update and throw:
            return deepcopy(self)
        if not update and throw:
            newloc = self + (vector / np.linalg.norm(vector)) * dist
            return Point2D(self.x+distances[0], self.y+distances[1])

    @staticmethod
    def validate_single_point_input(point):
        if isinstance(point, Point2d):
            xpoint, ypoint = point.x, point.y
        if dth.DEEPCHECK_is_coord2d_list(point):
            raise ValueError(f'Invalid point input.')
        if dth.DEEPCHECK_is_xy2d_list(point):
            raise ValueError(f'Invalid point input.')
        if type(point) in dth.dt.ITERABLES:
            if (type(point[0]) in dth.dt.NUMBERS) and type(point[1]) in dth.dt.NUMBERS:
                xpoint, ypoint = point[0], point[1]
        return xpoint, ypoint

    def translate_to(self, *, point=None, update=False, throw=True):
        if not point:
            raise ValueError('Must provide point object. Could also be coord.')
        xloc, yloc = Point2d.validate_single_point_input(point)
        if update:
            self.x, self.y = xloc, yloc
        if update and throw:
            # Retain this here, as the behaviour may change later on
            return deepcopy(self)
        if not update and throw:
            return Point2D(xloc, yloc)

    def rotate_about(self, *, point=None, axis=None, angle=0, degree=True,
                     update=False, throw=True):
        if not point:
            raise ValueError('Must provide point object. Could also be coord.')
        xpoint, ypoint = Point2d.validate_single_point_input(point)
        if type(angle) not in dth.dt.NUMBERS:
            raise ValueError('Invalid angle.')
        if degree:
            A = np.radians(angle)
        newloc = np.dot(np.array([[np.cos(A), -np.sin(A)],
                                  [np.sin(A), np.cos(A)]]),
                        np.array([self.x, self.y]))
        if update:
            self.x, self.y = xpoint, ypoint
        if update and throw:
            # Retain this here, as the behaviour may change later on
            return deepcopy(self)
        if not update and throw:
            return Point2D(xpoint, ypoint)

    def attach_feature(self, *, feature=None, feature_id=None):
        if not feature:
            raise ValueError('feature cannot be empty.')
        if not feature_id:
            raise ValueError('feature_id cannot be empty.')
        fname = feature.__class__.__name__
        if not hasattr(self, 'f'):
            self.f = {}
        if fname not in self.f:
            self.f[fname] = {}
        if feature_id in self.f[fname].keys():
            raise KeyError('Cannot attach feature. feature_id: '
                           f'{feature_id} already in dict {self.f[fname]}.')
        self.f[fname][feature_id] = fname

    def find_neigh_point_by_distance(self, *, plist=None, plane='xy', r=0,
                                     on_boundary=True):
        if type(r) not in dth.dt.NUMBERS:
            raise TypeError('Invalid r type.')
        if r <= self.ε:
            return self.distance(plist)[0]
        else:
            if on_boundary:
                return np.argwhere(self.distance(plist) <= r)
            else:
                return np.argwhere(self.distance(plist) < r)

    def find_neigh_point_by_count(self, *, plist=None, n=None,
                                  plane='xy'):
        if not isinstance(n, int):
            raise TypeError('n must be an int type.')
        if n > len(plist):
            raise ValueError('n is greater than len(plist).')
        return self.distance(plist)[:n]

    def find_neigh_mulpoint_by_distance(self, *, mplist=None,
                                        plane='xy', r=0, tolf=-1):
        # Use the ckdtree option.
        pass

    def find_neigh_edge_by_distance(self, *, elist=None,
                                    plane='xy', refloc='starting', r=0):
        pass

    def find_neigh_muledge_by_distance(self, *, melist=None,
                                       plane='xy', refloc='starting', r=0):
        pass

    def find_neigh_xtal_by_distance(self, *, xlist=None,
                                    plane='xy', refloc='starting', r=0):
        pass

    def set_gmsh_props(self, prop_dict):
        pass

    def make_shapely(self):
        pass

    def make_shape(self):
        pass

    def make_vtk(self):
        pass

    @property
    def coords(self):
        return np.array([self.x, self.y])

    def array_translation(self, *,
                          ncopies=10,
                          vector=[[0, 0, 0], [0, 0, 1]],
                          spacing='constant'):
        pass

    def lies_on_which_edge(self, *, elist=None, consider_ends=True):
        pass

    def lies_in_which_xtal(self, *, xlist=None,
                           cosider_boundary=True,
                           consider_boundary_ends=True):
        pass


class Edge(ABC):
    """
    Template base class for edge object. Expands to both 2D and 3D.

    Attributes
    ----------
    i: starting point coordinates
    j: ending point coordinates
    """

    __slots__ = ('i', 'j', )

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __repr__(self):
        """docstring."""
        pass

    @abstractmethod
    def __eq__(self, elist):
        """Check if the two edges are coincident."""
        pass

    @abstractmethod
    def __ne__(self, elist):
        """Check if the two edges are not coincident."""
        pass

    @property
    @abstractmethod
    def mid(self):
        """Return the mid point."""
        pass

    @property
    @abstractmethod
    def ang(self):
        """Return the ccw + angle in radians."""
        pass

    @classmethod
    def by_coord(self, start_point, end_point):
        """Create edge by specifying end coordinates."""
        pass

    @classmethod
    def by_loc_len_ang(self, *, ref='i', loc=[0, 0, 0],
                       length=1, ang=0, degree=True):
        """
        Create edge by specifying location, length and angle.

        Parameters
        ----------
        ref: Specifies which point on the edge is used to spcify the edge.

        loc: Specifies the ref point.

        length: Length of the edge to be made.

        ang: Angle(s) of inclination of the edge. If 2D, single value. If 3D,
        this specifies a list of three angles. First angle

        degree: ang considered in degree if degree is True, in radians if
        otherwise.
        """
        pass

    @abstractmethod
    def distance_to_points(self, *, plist=None):
        """Calculate the EUclidean distance between self and list of points."""
        pass

    @abstractmethod
    def distance_to_edges(self, *, elist=None,
                          method='ref', refi='mid', refj='mid'):
        """
        Calculate the Baudhāyana distance between self and list of edges.

        Parameters
        ----------
        elist: List of UPXO edge objects
        method: Indicate the method of calculation. Can take following values:
            * 'ref': Use the referece location specifiers to calculate
            disatance. That is, use refi and refj.
            * 'min': Minimum Baudhāyana distance
            * 'max': MAximum Baudhāyana distance
            * 'mean': Mean Baudhāyana distance
        refi: reference location for self i.e. i
        refj: referecne location for the other edge i.e. j

        Return
        ------
        List of distances to all edges.

        Author
        ------
        Dr. Sunil Anandatheertha
        """
        pass

    @abstractmethod
    def translate_by(self, *, vector=None, dist=None,
                     update=False, throw=True):
        """
        Translate the Edge by a Euclidean distance.

        Translate the Edge along the vector by a given distance. If
        update is True, then coords of the self will be updated.

        If throw is True and update is False, a new edge of the new
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

        throw: Return a edge if True, else return nothing if False.

        Return
        ------
        UPXO edge object: Conditional, depending on input throw (refer to
                                                                 description).

        Author
        ------
        Dr. Sunil Anandatheertha
        """
        pass

    @abstractmethod
    def translate_to(self, *, ref='i', point=None, update=False, throw=True):
        """
        Translate self to the specified location.

        New location is specified by point object. POint object could be
        specified by an another UPXO point object or an Iterable of coords.

        If throw is True and update is False, a new point of the new
        coordinates shall be reyturned. If throw is True and update is True, a
        deepcopy of the self shall be returned.

        Parameters
        ----------
        ref: Location on the edge which translates to new point. Values of ref
        could be:
            * 'i': Starting point of the edge
            * 'j': Ending point of the edge
            * 'mid': Middle point of the edge
            * [x, y, (z)]: Coordinate value

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
    def find_neigh_muledge_by_distance(self, *, melist=None,
                                       plane='xy', refloc='starting', r=0):
        """
        Find the nearest UPXO muledge in specified list of UPXO muledges.

        Parameters
        ----------
        melist: Elements of melist must contain medges in either of the
        following single format:
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
    def set_gmsh_props(self, prop_dict):
        """Set dictionary of gmsh properties."""
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
        """Return coordinate array."""
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



# =============================================================================

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
    Return a list of UPXO point2d objects from p2d.

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
    else:
        raise ValueError('Invalid input.')

def get_upxo_p2d(p2d):
    from upxo.geoEntities.point3d import p2d_leanest as p
    all_isinstance(np.ndarray, p2d)


def all_isinstance(dtype, *args):
    if len(args) > 0:
        print(args)
        return all(isinstance(arg, dtype) for arg in args)

class Point2D(Point):
    ε = 0.000000000001
    __slots__ = ('x', 'y', 'z', 'mp', 'gmsh')
    def __init__(self):
        pass

    def __repr__(self):
        """docstring."""
        pass

    def __eq__(self, p3dlist):
        """Check if the two points are coincident."""
        pass

    def __ne__(self, p3dlist):
        """Check if the two points are not coincident."""
        pass

    def distance(self, *, plist=None):
        """Calculate the EUclidean distance between self and list of points."""
        pass

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

    def attach_mp(self, *, mp=None, name=None):
        """Attach a UPXO multi-poiont object and a name."""
        self.mp[name] = mp

    def attach_xtal(self, *, xtals=None):
        """Attach a list of UPXO xtal objects."""
        pass

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

    def find_neigh_muledge_by_distance(self, *, melist=None,
                                       plane='xy', refloc='starting', r=0):
        """
        Find the nearest UPXO muledge in specified list of UPXO muledges.

        Parameters
        ----------
        melist: Elements of melist must contain medges in either of the
        following single format:
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

    def set_gmsh_props(self, prop_dict):
        """Set dictionary of gmsh properties."""
        pass

    def make_shapely(self):
        """Return shapely point object. Only valid for 2D."""
        pass

    def make_vtk(self):
        """Make VTK point object."""
        pass

    @property
    def coords(self):
        """Return coordinate array."""
        return np.array([self.x, self.y])

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
    def become_pixel(self, l=1):
        pass
    def become_c2d4(self, l=1):
        pass
    def become_c2d8(self, l=1):
        pass


class Point3D(Point):
    ε = 0.000000000001
    __slots__ = ('x', 'y', 'z', 'mp', 'gmsh')
    def __init__(self):
        pass

    def __repr__(self):
        """docstring."""
        pass

    def __eq__(self, p3dlist):
        """Check if the two points are coincident."""
        pass

    def __ne__(self, p3dlist):
        """Check if the two points are not coincident."""
        pass

    def distance(self, *, plist=None):
        """Calculate the EUclidean distance between self and list of points."""
        pass

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

    def attach_mp(self, *, mp=None, name=None):
        """Attach a UPXO multi-poiont object and a name."""
        self.mp[name] = mp

    def attach_xtal(self, *, xtals=None):
        """Attach a list of UPXO xtal objects."""
        pass

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

    def find_neigh_muledge_by_distance(self, *, melist=None,
                                       plane='xy', refloc='starting', r=0):
        """
        Find the nearest UPXO muledge in specified list of UPXO muledges.

        Parameters
        ----------
        melist: Elements of melist must contain medges in either of the
        following single format:
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

    def set_gmsh_props(self, prop_dict):
        """Set dictionary of gmsh properties."""
        pass

    def make_shapely(self):
        """Return shapely point object. Only valid for 2D."""
        pass

    def make_vtk(self):
        """Make VTK point object."""
        pass

    @property
    def coords(self):
        """Return coordinate array."""
        return np.array([self.x, self.y])

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
    def become_voxel(self, l=1):
        pass
    def become_c3d8(self, l=1):
        pass
    def become_c3d20(self, l=1):
        pass
