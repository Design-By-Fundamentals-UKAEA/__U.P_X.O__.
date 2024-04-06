"""
Core module of UKAEA Poly-XTAL Operations.

Authors
-------
Dr. Sunil Anandatheertha

@Developer notes
----------------
We could benefit a lot from the below links.
https://docs.sympy.org/latest/modules/geometry/points.html
https://www.geeksforgeeks.org/python-sympy-segment-perpendicular_bisector-method/?ref=next_article
https://www.geeksforgeeks.org/python-sympy-line-is_parallel-method/
https://www.geeksforgeeks.org/python-sympy-line-smallest_angle_between-method/
https://www.geeksforgeeks.org/python-sympy-line-parallel_line-method/
https://www.geeksforgeeks.org/python-sympy-line-are_concurrent-method/
https://www.geeksforgeeks.org/python-sympy-ellipse-equation-method/
https://www.geeksforgeeks.org/python-sympy-ellipse-method/
https://www.geeksforgeeks.org/python-sympy-plane-equation-method/?ref=ml_lbp
https://www.geeksforgeeks.org/python-sympy-polygon-cut_section-method/?ref=ml_lbp
https://www.geeksforgeeks.org/python-sympy-plane-is_coplanar-method/?ref=ml_lbp
https://www.geeksforgeeks.org/python-sympy-plane-perpendicular_plane-method/?ref=ml_lbp
https://www.geeksforgeeks.org/python-sympy-plane-projection-method/?ref=ml_lbp
https://www.geeksforgeeks.org/python-sympy-line-intersection-method/?ref=ml_lbp
https://www.geeksforgeeks.org/python-sympy-curve-translate-method/?ref=ml_lbp
https://www.geeksforgeeks.org/python-sympy-triangle-is_right-method/?ref=ml_lbp
https://www.geeksforgeeks.org/python-sympy-triangle-is_isosceles-method/?ref=ml_lbp
"""

import math
import numpy as np
from copy import deepcopy
from abc import ABC, abstractmethod
from scipy.spatial import cKDTree
import vtk
import upxo._sup.dataTypeHandlers as dth


class _coord_():
    __slots__ = ('x', 'y', 'z')
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


def find_pnt_spec_type_2d(data):
    """
    from upxo.geoEntities.point3d import Point2d as p2d
    from upxo.geoEntities.point3d import p2d_leanest

    find_pnt_spec_type_2d(p2d(1, 2))
    find_pnt_spec_type_2d([p2d(1, 2), p2d(3, 3)])
    find_pnt_spec_type_2d(p2d_leanest(1, 2))
    find_pnt_spec_type_2d([p2d_leanest(1, 2), p2d_leanest(1, 2)])
    find_pnt_spec_type_2d([1, 2])
    find_pnt_spec_type_2d([[1, 2]])
    find_pnt_spec_type_2d([[1,2],[3,4],[5,6]])
    find_pnt_spec_type_2d([[2,1,1,2],[3,4,5,6]])
    """
    NUMBERS, ITERABLES, known = dth.dt.NUMBERS, dth.dt.ITERABLES, False
    # -------------------------------------
    if data.__class__.__name__ == 'p2d_leanest':
        known = True
        return 'p2d_leanest'
    # -------------------------------------
    if isinstance(data, ITERABLES) and all(_.__class__.__name__ == 'p2d_leanest' for _ in data):
        known = True
        return '[p2d_leanest]'
    # -------------------------------------
    if data.__class__.__name__ == 'Point2d':
        known = True
        return 'Point2d'
    # -------------------------------------
    if isinstance(data, ITERABLES) and all(_.__class__.__name__ == 'Point2d' for _ in data):
        known = True
        return '[Point2d]'
    # -------------------------------------
    if isinstance(data, ITERABLES) and all(isinstance_many(data, NUMBERS)) and len(data) == 2:
        # p = [1, 2]
        known = True
        return 'type-[1,2]'
    # -------------------------------------
    if isinstance(data, ITERABLES) and len(data) == 1 and isinstance(data[0], ITERABLES) and len(data[0]) == 2 and all(isinstance_many(data[0], NUMBERS)):
        # p = [[1, 2]]
        known = True
        return 'type-[[1,2]]'
    # -------------------------------------
    if isinstance(data, ITERABLES) and all(isinstance_many(data, ITERABLES)) and all(len(_) == 2 for _ in data):
        # p = [[2, 1], [10, 20], [31, 49]]
        known = True
        return 'type-[[1,2],[3,4],[5,6]]'
    # -------------------------------------
    if isinstance(data, ITERABLES) and len(data) == 2 and all(isinstance_many(data, ITERABLES)):
        if (all(len(_) == 1 for _ in data) or all(len(_) >= 2 for _ in data)):
            if all(isinstance_many(data[0], NUMBERS)) and all(isinstance_many(data[1], NUMBERS)):
                known = True
                # p = [[2, 1, 1, 2], [3, 4, 5, 6]]
                return 'type-[[2,1,1,2],[3,4,5,6]]'
    # -------------------------------------
    if not known:
        return 'unknown'


def make_p2d(p, option='leanest'):
    """
    DEVELOPMENT TARGETS AND PROGRESS
    --------------------------------
    PHASE - 1: LEANEST. DONE
    PHASE - 2: 3D ALTERNATIVE. # TODO

    EXAMPLES
    --------
    make_p2d(p2d(1, 2))
    make_p2d([p2d(1, 2), p2d(3, 3)])
    make_p2d(p2d_leanest(1, 2))
    make_p2d([p2d_leanest(1, 2), p2d_leanest(1, 2)])
    make_p2d([1, 2])  # Valid
    make_p2d([1, 2, 3])  # Invalid: Produces no output !!!
    make_p2d([[1, 2]])  # Valid
    make_p2d([[2, 1], [1, 2], [3, 4]])  # Valid
    make_p2d([[2, 1, 1, 2], [3, 4, 5, 6]])  # Valid
    """
    NUMBERS, ITERABLES = dth.dt.NUMBERS, dth.dt.ITERABLES
    # ============================================================
    if option == 'leanest':
        if find_pnt_spec_type_2d(p) == 'p2d_leanest':
            return [p]
        # -------------------------------------
        if find_pnt_spec_type_2d(p) == '[p2d_leanest]':
            return p
        # -------------------------------------
        if find_pnt_spec_type_2d(p) == 'Point2d':
            return [p2d_leanest(p.x, p.y)]
        # -------------------------------------
        if find_pnt_spec_type_2d(p) == '[Point2d]':
            return [p2d_leanest(_.x, _.y) for _ in p]
        # -------------------------------------
        if find_pnt_spec_type_2d(p) == 'type-[1,2]':
            # p = [1, 2]
            return [p2d_leanest(p[0], p[1])]
        # -------------------------------------
        if find_pnt_spec_type_2d(p) == 'type-[[1,2]]':
            # p = [[1, 2]]
            return [p2d_leanest(p[0][0], p[0][1])]
        # -------------------------------------
        if find_pnt_spec_type_2d(p) == 'type-[[1,2],[3,4],[5,6]]':
            # p = [[2, 1], [10, 20], [31, 49]]
            return [p2d_leanest(_[0], _[1]) for _ in p]
        # -------------------------------------
        if find_pnt_spec_type_2d(p) == 'type-[[2,1,1,2],[3,4,5,6]]':
            # p = [[2, 1, 1, 2], [3, 4, 5, 6]]
            return [p2d_leanest(x, y) for x, y in zip(p[0], p[1])]
        else:
            raise ValueError('Invalid point specification')


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


class UPXO_Point(ABC):
    """Template base class for point object. Expands to both 2D and 3D."""

    __slots__ = ('x', 'y', 'pln', 'f')

    @abstractmethod
    def __init__(self, x=.0, y=.0, pln='ij'):
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
    def add(self, distances, update=True, throw=False,
            mydecatlen2NUM='taxx'):
        pass

    @abstractmethod
    def __mul__(self, f, update=True, throw=False):
        """
        Multiple f to point coord & update self or return new point objects.

        All descriptions in parameters below, naturally extend to 3D.

        Parameters
        ----------
        f: list of multiplication factors. Depending on d, functionaliy changes
        as below.
            * [1, 2, 3, 4]: Each entry is multipled to both x and y. 4 new
            point objects gets created.
            * [[1, 2], [3, 4]]: [1, 2] denote first set of x and y distances.
            They get multipled with self.x and self.y to make a new point.
            Similar operation extewnds to [3, 4]. Two new points are created.
            * [[1, 2, 3, 4], [5, 6, 7, 8]]: These are X and Y arrays. Each x
            and y in X and Y, gets multipled with self.x and self.y to make n
            points, where n = len(d[0]).
            * [po1, po2, po3]: List of point objects. Point objects could be
            2D or 3D. UPXO, GMSH, VTK, PyVista, Shapely types are allowed.

        update: If True and if f is either K or Iterable(P, Q), where, K, P and
            Q are dth.dt.NUMBERS, self will be updated as self.x*K and self.y*K
            or self.x*P and self.y*Q.

        throw: If True and if additional conditions provided in update are
            atisfied, then the deepcopy of the point will be returned. If,
            however, update is False, a new point with coordiates self.x*K and
            self.y*K or self.x*P and self.y*Q, shall be created and returned.
        """
        pass

    @abstractmethod
    def distance(self, plist=None):
        """Calculate the EUclidean distance between self and list of points."""
        pass

    @abstractmethod
    def distance(self, plist=None):
        """Calculate the EUclidean distance between self and list of points."""
        pass


class Point2d(UPXO_Point):
    """
    UPXO Point2d object, new version.

    DEVELOPMENTAL PHASES AND PROGRESS
    ---------------------------------
    __eq__: DONE
    __ne__: DONE


    Parameters
    ----------
    pln: Denotes plane which contains the self point.
    i: 1st coordinate of the point.
    j: 2nd coordinate of the point.
    f: Feature dictionary containsing features attached to the point.

    Explanations
    ------------
    If pln is 'ij' or 'ji': x, y = x_, y_: True representation
    If pln is 'jk' or 'kj': x, y = y_, z_: False representation
    If pln is 'ki' or 'ik': x, y = x_, z_: False representation
    Where, x_, y_ and z_ are actual coordinate axes.

    Notes to users
    --------------
    @user: Please refer to examples and Jupyter notebook demos before use.

    Notes to developers and maintainers
    -----------------------------------
    @dev: Inherits from ABC: Point.
    @dev: Lets not use pydantic in the interest of maintaining speed of
        instantiation and reducing memory overhead.

    Import statement
    ----------------
    from upxo.geoEntities.point3d import Point2d as p2d
    Example 1: Creation
    -------------------
    A, B, C = p2d(10, 12), p2d(10, 12), p2d(11, 12)
    print(A, B, C)

    Example 2: equality check
    -------------------------
    print(A == B, A != B, A == C, A != C)

    Example 3: Addition and subtraction
    -----------------------------------
    A + 10
    print(A)
    A - 10
    print(A)
    A + [10, 20, 30]
    """

    ε = 1E-8
    __slots__ = UPXO_Point.__slots__ + ('add_new_slot_if_needed', )

    def __init__(self, x, y):
        super().__init__(x, y)
        self.x = x
        self.y = y

    def __repr__(self):
        return f"uxpo-p2d ({self.x},{self.y})"

    def __eq__(self, plist, *, use_tol=True):
        """
        from upxo.geoEntities.point3d import Point2d as p2d
        point = p2d(3, 4)
        plist1 = [[1, 2], [3, 4], [5, 6]]
        plist2 = [[1, 3, 5], [2, 4, 6]]
        plist3 = [p2d(1, 2), p2d(3, 4)]
        plist4 = [[3, 4]]
        plist5 = [[3], [4]]

        print(point == plist1)  # True
        print(point == plist2)  # True
        print(point == plist3)  # True
        print(point == plist4)  # True
        print(point == plist5)  # True
        """
        # HUGE RESTRUCTURING NEEDED.
        if not plist:
            raise ValueError("plist is empty.")
        if find_pnt_spec_type_2d(plist) == 'type-[[1,2],[3,4],[5,6]]':
            '''Then, plist must be like [[1, 2], [3, 4]]'''
            if use_tol:
                # TODO: Add tol checking functionality
                return [self.x == p[0] and self.y == p[1]
                        for p in plist]
            else:
                return [self.x == p[0] and self.y == p[1]
                        for p in plist]
        if find_pnt_spec_type_2d(plist) == 'type-[[2,1,1,2],[3,4,5,6]]':
            '''Then, plist must be like [[1, 2, 3, 4], [3, 2, 1, 4]]'''
            if use_tol:
                # TODO: Add tol checking functionality
                return [self.x == _x and self.y == _y
                        for _x, _y in zip(plist[0], plist[1])]
            else:
                return [self.x == _x and self.y == _y
                        for _x, _y in zip(plist[0], plist[1])]
        if find_pnt_spec_type_2d(plist) == 'Point2d':
            return (self.x, self.y) == (plist.x, plist.y)
        if find_pnt_spec_type_2d(plist) == '[Point2d]':
            return [(self.x, self.y) == (p.x, p.y) for p in plist]
        if find_pnt_spec_type_2d(plist) == '[p2d_leanest]':
            return [(self.x, self.y) == (p.x, p.y) for p in plist]
        else:
            '''UPXO point objects'''
            # Lets assume for now, plist is a UPXO point object.
            # TODO: Further validations to be done later on.
            plist = [plist]
            if use_tol:
                # TODO: Add tol checking functionality
                equality = [self.x == p.x and self.y == p.y
                            for p in plist]
            else:
                equality = [self.x == p.x and self.y == p.y
                            for p in plist]
            if len(plist) == 1:
                return equality[0]
            else:
                return equality

    def __ne__(self, plist, *, use_tol=True):
        return not self.__eq__(plist, use_tol=use_tol)

    def add(self, d, update=True, throw=False, mydecatlen2NUM='b'):
        """
        Add distances to point coord & update self or return new point objects.

        All descriptions in parameters below, naturally extend to 3D.

        Parameters
        ----------
        d: list of distances. Depending on distances, functionaliy changes as
        below.
            * [1, 2, 3, 4]: Each entry is added to both x and y. 4 new point
            objects gets created.
            * [[1, 2], [3, 4]]: [1, 2] denote first set of x and y distances.
            They get added with self.x and self.y to make a new point. Similar
            operation extewnds to [3, 4]. Two new points are created.
            * [[1, 2, 3, 4], [5, 6, 7, 8]]: These are X and Y arrays. Each x
            and y in X and Y, gets added with self.x and self.y to make n
            points, where n = len(distances[0]).
            * [po1, po2, po3]: List of point objects. Point objects could be
            2D or 3D. UPXO, GMSH, VTK, PyVista, Shapely types are allowed.

        update: If True and if distances is either K or Iterable(P, Q), where,
            K, P and Q are dth.dt.NUMBERS, self will be updated as self.x+K and
            self.y+K or self.x+P and self.y+P.

        throw: If True and if additional conditions provided in update are
            atisfied, then the deepcopy of the point will be returned. If,
            however, update is False, a new point with coordiates self.x+K and
            self.y+K or self.x+P and self.y*Q, shall be created and returned.

        mydecatlen2NUM: My Decision At len(d)=2 when all in d are NUMBERS.
        Options include the following:
            * 'a' OR 'ta2sd': Treat as two seperate distances. In this case,
            d[0] and d[1] will be added seperately and two pint objects shall
            be made.
            * 'b' OR 'atxy': Add d[0] to x and d[1] to y. Self point may
            update and/or new point may be returned.
        """
        NUMBERS, ITERABLES = dth.dt.NUMBERS, dth.dt.ITERABLES
        if type(d) in NUMBERS:
            '''
            Ex. @ d --> 10

            from upxo.geoEntities.point3d import Point2d as p2d
            d = 10

            Case A
            ------
            A = p2d(10, 12)
            A.add(d)
            print(A)

            Case B
            ------
            A = p2d(10, 12)
            A.add(d, update=True, throw=False)  # Same as case A
            print(A)

            Case C
            ------
            A = p2d(10, 12)
            A.add(d, update=True, throw=True)
            print(A)

            Case D
            ------
            A = p2d(10, 12)
            A.add(d, update=False, throw=True)
            print(A)
            '''
            print('-------1--------')
            # add d to both x and y
            if update:
                self.x += d
                self.y += d
            if update and throw:
                return deepcopy(self)
            if not update and throw:
                return Point2d(self.x+d, self.y+d)
        # =======================================================
        if type(d) in ITERABLES:
            # add d contents seperatrely to x and y
            # ................
            # CASE - 1
            print('-------2--------')
            if len(d) == 1 and type(d[0]) in NUMBERS:
                '''
                Ex. @ d --> [10]

                from upxo.geoEntities.point3d import Point2d as p2d
                d = [10]

                Case A
                ------
                A = p2d(10, 12)
                A.add(d)
                print(A)

                Case B
                ------
                A = p2d(10, 12)
                A.add(d, update=True, throw=False)  # Same as case A
                print(A)

                Case C
                ------
                A = p2d(10, 12)
                A.add(d, update=True, throw=True)
                print(A)

                Case D
                ------
                A = p2d(10, 12)
                A.add(d, update=False, throw=True)
                print(A)
                '''
                print('-------2A--------')
                if update:
                    self.x += d[0]
                    self.y += d[0]
                if update and throw:
                    return deepcopy(self)
                if not update and throw:
                    return Point2d(self.x+d[0], self.y+d[0])
            # ................
            # CASE - 2
            if len(d) == 1 and type(d[0]) in ITERABLES and len(d[0]) == 2:
                """
                from upxo.geoEntities.point3d import Point2d as p2d
                d = [[10, 12]]

                Case A1
                -------
                A = p2d(10, 12)
                A.add(d)
                print(A)

                Case B1 # Same as Case - a
                -------
                A = p2d(10, 12)
                A.add(d, update=True, throw=False, mydecatlen2NUM='b')

                print(A)

                Case C1
                -------
                A = p2d(10, 12)
                A.add(d, update=True, throw=True, mydecatlen2NUM='b')
                print(A)

                Case D1
                -------
                A = p2d(10, 12)
                A.add(d, update=False, throw=True, mydecatlen2NUM='b')
                print(A)

                Case A2
                -------
                A = p2d(10, 12)
                A.add(d, update=True, throw=False, mydecatlen2NUM='a')
                # throw and update ignored.
                # results will be returned by default.
                print(A)  # Remains unaltere3d as update is ignored
                """
                print('-------3--------')
                if mydecatlen2NUM in ('a', 'ta2sd'):
                    return [Point2d(self.x+_, self.y+_) for _ in d[0]]
                if mydecatlen2NUM in ('b', 'atxy'):
                    if update:
                        self.x += d[0][0]
                        self.y += d[0][1]
                    if update and throw:
                        return deepcopy(self)
                    if not update and throw:
                        return Point2d(self.x+d[0][0], self.y+d[0][1])

            # ................
            # CASE - 3
            if len(d) == 2 and all(isinstance_many(d, NUMBERS)):
                """
                from upxo.geoEntities.point3d import Point2d as p2d
                d = [10, 12]

                Case A1
                -------
                A = p2d(10, 12)
                A.add(d)
                print(A)

                Case B1 # Same as Case - a
                -------
                A = p2d(10, 12)
                A.add(d, update=True, throw=False, mydecatlen2NUM='b')

                print(A)

                Case C1
                -------
                A = p2d(10, 12)
                A.add(d, update=True, throw=True, mydecatlen2NUM='b')
                print(A)

                Case D1
                -------
                A = p2d(10, 12)
                A.add(d, update=False, throw=True, mydecatlen2NUM='b')
                print(A)

                Case A2
                -------

                A = p2d(10, 12)
                A.add(d, update=True, throw=False, mydecatlen2NUM='a')
                # throw and update ignored.
                # results will be returned by default.
                print(A)  # Remains unaltere3d as update is ignored
                """
                print('-------4--------')
                if mydecatlen2NUM in ('a', 'ta2sd'):
                    return [Point2d(self.x+_, self.y+_) for _ in d]
                if mydecatlen2NUM in ('b', 'atxy'):
                    if update:
                        self.x += d[0]
                        self.y += d[1]
                    if update and throw:
                        return deepcopy(self)
                    if not update and throw:
                        return Point2d(self.x+d[0], self.y+d[1])
            # ................
            if all(_.__class__.__name__ == 'p2d_leanest' for _ in d):
                """
                from upxo.geoEntities.point3d import Point2d as p2d
                from upxo.geoEntities.point3d import p2d_leanest
                P = [p2d_leanest(-10, -12), p2d_leanest(-2, 2)]

                EXAMPLE CASES
                -------------
                # Only case possible
                # update and throw input arguments will be ignored.

                Case A
                ------
                A = p2d(10, 12)
                A.add(P, update=True, throw=False)
                print(A)
                """
                print('-------6--------')
                return [Point2d(self.x+_._x, self.y+_._y) for _ in d]
            # ................
            if all(_.__class__.__name__ == 'Point2d' for _ in d):
                """
                from upxo.geoEntities.point3d import Point2d as p2d
                P = [p2d(-10, -12), p2d(-2, 2)]

                EXAMPLE CASES
                -------------
                # Only case possible
                # update and throw input arguments will be ignored.

                Case A
                ------
                A = p2d(10, 12)
                A.add(P, update=True, throw=False)
                print(A)
                """
                print('-------7--------')
                return [Point2d(self.x+_.x, self.y+_.y) for _ in d]
            # ................
            if len(d) == 2 and all(isinstance_many(d, ITERABLES)):
                print('-------8--------')
                if len(d[0]) == 1 and len(d[1]) == 1:
                    """
                    from upxo.geoEntities.point3d import Point2d as p2d
                    d = [[10], [12]]

                    Case A
                    ------
                    A = p2d(10, 12)
                    A.add(d)
                    print(A)

                    Case A1 # Same as case A
                    ------
                    A = p2d(10, 12)
                    A.add(d, update=True, throw=False, mydecatlen2NUM='b')
                    print(A)

                    Case B
                    ------
                    A = p2d(10, 12)
                    A.add(d, update=True, throw=True, mydecatlen2NUM='b')
                    print(A)

                    Case C
                    ------
                    A = p2d(10, 12)
                    A.add(d, update=False, throw=True, mydecatlen2NUM='b')
                    print(A)

                    Case D  # NOTHING CHANGES!
                    ------
                    A = p2d(10, 12)
                    A.add(d, update=False, throw=False, mydecatlen2NUM='b')
                    print(A)
                    """
                    print('-------8A--------')
                    if update:
                        self.x += d[0][0]
                        self.y += d[1][0]
                    if update and throw:
                        return deepcopy(self)
                    if not update and throw:
                        return Point2d(self.x+d[0][0], self.y+d[1][0])
                elif len(d[0]) > 1 and (len(d[0]) == len(d[1])):
                    """
                    from upxo.geoEntities.point3d import Point2d as p2d
                    d = [[10, 11, 12, 13], [12, 13, 14, 15]]

                    EXAMPLE CASES
                    -------------
                    # Only case possible
                    # update and throw input arguments will be ignored.

                    Case A
                    ------
                    A = p2d(10, 12)
                    A.add(d)
                    print(A)
                    """
                    print('-------8b--------')
                    return [Point2d(self.x+_x, self.y+_y)
                            for _x, _y in zip(d[0], d[1])]
            # ................
            # CASE - 4
            if len(d) > 2 and all(isinstance_many(d, NUMBERS)):
                """
                from upxo.geoEntities.point3d import Point2d as p2d
                d = [10, 11, 12, 13]

                EXAMPLE CASES
                -------------
                # Only case possible
                # update and throw input arguments will be ignored.

                Case A
                ------
                A = p2d(10, 12)
                A.add(d)
                print(A)
                """
                print('-------10--------')
                return [Point2d(self.x+_d, self.y+_d) for _d in d]
            # ................
            # CASE - 5
            if len(d) > 2 and all(isinstance_many(d, ITERABLES)):
                print('-------11--------')
                if all(len(_) == 2 for _ in d):
                    """
                    from upxo.geoEntities.point3d import Point2d as p2d
                    d = [[2, 3], [4, 5], [5, 6], [0, 10]]

                    EXAMPLE CASES
                    -------------
                    # Only case possible
                    # update and throw input arguments will be ignored.

                    Case A
                    ------
                    A = p2d(10, 12)
                    A.add(d)
                    print(A)
                    """
                    print('-------11A--------')
                    return [Point2d(self.x+_d[0], self.y+_d[1]) for _d in d]
                else:
                    '''Ex. @ d --> [[2, 3, 5], [4, 5], [5, 6], [0, 10]]'''
                    '''Ex. @ d --> [[2, 3, 6], [4], [5, 6], [0, 10]]'''
                    '''Ex. @ d --> [[2, 3, 6], [4, 5, 6], [0, 5, 10]]'''
                    print('-------7B--------')
                    raise ValueError('Invalid distances.')
        # =======================================================
        if d.__class__.__name__ == 'Point2d':
            """
            from upxo.geoEntities.point3d import Point2d as p2d
            P = p2d(-10, -12)

            Case A
            ------
            A = p2d(10, 12)
            A.add(P, update=True, throw=False)
            print(A)

            Case B
            ------
            A = p2d(10, 12)
            A.add(P, update=True, throw=True)
            print(A)

            Case C
            ------
            A = p2d(10, 12)
            A.add(P, update=False, throw=True)
            print(A)

            Case D
            ------
            A = p2d(10, 12)
            A.add(P, update=False, throw=False)
            print(A)
            """
            print('-------5--------')
            if update:
                self.x += d.x
                self.y += d.y
            if update and throw:
                return deepcopy(self)
            if not update and throw:
                return Point2d(self.x+d.x, self.y+d.y)
        # =======================================================
        if d.__class__.__name__ == 'p2d_leanest':
            """
            from upxo.geoEntities.point3d import Point2d as p2d
            from upxo.geoEntities.point3d import p2d_leanest
            P = p2d_leanest(-10, -12)

            Case A
            ------
            A = p2d(10, 12)
            A.add(P, update=True, throw=False)
            print(A)

            Case B
            ------
            A = p2d(10, 12)
            A.add(P, update=True, throw=True)
            print(A)

            Case C
            ------
            A = p2d(10, 12)
            A.add(P, update=False, throw=True)
            print(A)

            Case D
            ------
            A = p2d(10, 12)
            A.add(P, update=False, throw=False)
            print(A)
            """
            print('-------5--------')
            if update:
                self.x += d._x
                self.y += d._y
            if update and throw:
                return deepcopy(self)
            if not update and throw:
                return Point2d(self.x+d._x, self.y+d._y)

    def __mul__(self, f=1.0, update=True, throw=False):
        # Validate f
        # ----------------------------------
        # DEVELOPMENT STAGE - 1
        # TARGET: succeffull working when f is a single number
        if not isinstance(f, dth.dt.NUMBERS):
            raise TypeError('Invald factor')
        if update:
            self.x *= f
            self.y *= f
        if update and throw:
            return deepcopy(self)
        if not update and throw:
            return Point2d(self.x*f, self.y*f)

    def squared_distance(self, plist=None):
        """
        Calculate the squared distances between self point and plist.

        DEVELOPMENT PHASES AND PROGRESS
        -------------------------------
        PHASE 1: 2D case: DONE
        PHASE 2: 3D case: # TODO

        USE CASES
        ---------
        from upxo.geoEntities.point3d import Point2d as p2d
        A = p2d(0, 0)

        EXAMPLE 1
        ---------
        A.squared_distance( [1, 2] )
        A.squared_distance( [[1, 2]] )
        A.squared_distance( plist=[[1, 2], [10, 12]] )
        A.squared_distance( plist=[[1, 2], [10, 12], [0, -5]] )
        A.squared_distance( plist=[[1, 2, -1, -3], [4, 5, 5, 6]] )
        """
        plist = make_p2d(plist, option='leanest')
        if not plist:
            raise ValueError('Invalid list of points')
        X, Y = np.array([[p._x, p._y] for p in plist]).T
        return (self.x-X)**2 + (self.y-Y)**2

    def distance(self, plist=None):
        """
        Calculate the distances between self point and plist.

        DEVELOPMENT PHASES AND PROGRESS
        -------------------------------
        PHASE 1: 2D case: DONE
        PHASE 2: 3D case: # TODO

        USE CASES
        ---------
        from upxo.geoEntities.point3d import Point2d as p2d
        A = p2d(0, 0)

        EXAMPLE 1
        ---------
        A.distance( [1, 2] )
        A.distance( [[1, 2]] )
        A.distance( plist=[[1, 2], [10, 12]] )
        A.distance( plist=[[1, 2], [10, 12], [0, -5]] )
        A.distance( plist=[[1, 2, -1, -3], [4, 5, 5, 6]] )
        """
        return np.sqrt(self.squared_distance(plist))

    def translate(self, *, vector=None, dist=None, update=False,
                  throw=True):
        """
        Translate the self along the vector by dist.

        Development phases
        ------------------
        PHASE 1:
        PHASE 2: Validation for dist
        PHASE 3: Validation for vector

        Examples
        --------
        from upxo.geoEntities.point3d import Point2d as p2d
        A = p2d(0, 0)

        Example-1
        ---------
        A.translate(vector=[1,1], dist=5, update=True, throw=False)
        """
        distances = (np.array(vector) / np.linalg.norm(vector)) * dist
        print(distances)
        if update:
            self.x += distances[0]
            self.y += distances[1]
            # self.add(distances, update=update, throw=throw)
        if update and throw:
            return deepcopy(self)
        if not update and throw:
            return Point2d(self.x+distances[0], self.y+distances[1])

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
            return Point2d(xloc, yloc)

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
            return Point2d(xpoint, ypoint)

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
        self.f[fname][feature_id] = feature

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

    def set_z(self, z=0):
        """
        from upxo.geoEntities.point3d import Point2d as p2d
        A, z = p2d(10, 12), 100
        A.set_z(z=100)
        A.f['_coord_'][-1].z
        """
        from upxo.geoEntities.point3d import _coord_
        self.attach_feature(feature=_coord_(self.x, self.y, z),
                            feature_id=-1)

    def make_vtk_point(self, z=0):
        """
        from upxo.geoEntities.point3d import Point2d as p2d
        A, z = p2d(10, 12), 100
        vtkobj = A.make_vtk_point(z=100)

        # Accessing data in the vtk_point
        x, y, z = vtkobj['pd'].GetPoint(vtkobj['id'])
        print(x, y, z)
        """
        if not hasattr(self, 'f'):
            self.set_z(z=z)
        points = vtk.vtkPoints()
        point_id = points.InsertNextPoint(self.x,
                                          self.y,
                                          self.f['_coord_'][-1].z)
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        return {'id': point_id,
                'pd': poly_data,
                'help': "return['pd'].GetPoint(return['id'])"}

    def make_shapely(self):
        pass

    def make_shape(self):
        pass


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


# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////
#                           BEWGINNING OF EDGE CLASSES
# /////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////


class Edge(ABC):

    __slots__ = ('i', 'j', )

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def __eq__(self, elist):
        pass

    @abstractmethod
    def __ne__(self, elist):
        pass

    @property
    @abstractmethod
    def mid(self):
        pass

    @property
    @abstractmethod
    def ang(self):
        pass

    @property
    @abstractmethod
    def length(self):
        """Calculate and return self length"""
        pass

    @classmethod
    def by_coord(cls, start_point, end_point):
        pass

    @classmethod
    def by_loc_len_ang(cls, *, ref='i', loc=[0, 0, 0],
                       length=1, ang=0, degree=True):
        pass

    @abstractmethod
    def distance_to_points(self, *, plist=None):
        pass

    @abstractmethod
    def distance_to_edges(self, *, elist=None,
                          method='ref', refi='mid', refj='mid'):
        pass

    @abstractmethod
    def translate_by(self, *, vector=None, dist=None,
                     update=False, throw=True):
        pass

    @abstractmethod
    def translate_to(self, *, ref='i', point=None, update=False, throw=True):
        pass

    @abstractmethod
    def rotate_about(self, *, axis=None, angle=0, degree=True,
                     update=False, throw=True):
        pass

    @abstractmethod
    def attach_mp(self, *, mp=None, name=None):
        self.mp[name] = mp

    @abstractmethod
    def attach_xtal(self, *, xtals=None):
        pass

    @abstractmethod
    def find_neigh_point_by_distance(self, *, plist=None, plane='xy', r=0):
        pass

    @abstractmethod
    def find_neigh_point_by_count(self, *, plist=None, n=None,
                                  plane='xy'):
        pass

    @abstractmethod
    def find_neigh_mulpoint_by_distance(self, *, mplist=None,
                                        plane='xy', r=0, tolf=-1):
        pass

    @abstractmethod
    def find_neigh_edge_by_distance(self, *, elist=None,
                                    plane='xy', refloc='starting', r=0):
        pass

    @abstractmethod
    def find_neigh_muledge_by_distance(self, *, melist=None,
                                       plane='xy', refloc='starting', r=0):
        pass

    @abstractmethod
    def find_neigh_xtal_by_distance(self, *, xlist=None,
                                    plane='xy', refloc='starting', r=0):
        pass

    @abstractmethod
    def set_gmsh_props(self, prop_dict):
        pass

    @abstractmethod
    def make_shapely(self):
        pass

    @abstractmethod
    def make_vtk(self):
        pass

    @property
    @abstractmethod
    def coords(self):
        return np.array([self.x, self.y])

    @abstractmethod
    def array_translation(self, *,
                          ncopies=10,
                          vector=[[0, 0, 0], [0, 0, 1]],
                          spacing='constant'):
        pass

    @abstractmethod
    def lies_on_which_edge(self, *, elist=None, consider_ends=True):
        pass

    @abstractmethod
    def lies_in_which_xtal(self, *, xlist=None,
                           cosider_boundary=True,
                           consider_boundary_ends=True):
        pass


class edge2d_leanest():
    """
    IMPORT
    ------
    from upxo.geoEntities.point3d import edge2d_leanest

    EXAMPLES
    --------
    # Example-1
    e = edge2d_leanest(-2, 3, 4, 5)

    # Example-2
    for coord in e:
        print(coord)

    # Example-3
    print(e[1])
    """

    __slots__ = ('x0', 'y0', 'x1', 'y1')

    def __init__(self, x0, y0, x1, y1):
        self.x0, self.y0 = x0, y0
        self.x1, self.y1 = x1, y1

    def __repr__(self):
        """Repr function."""
        return f'UPXO-e2d-lean ({self.x0},{self.y0})-({self.x1},{self.y1})'

    def __iter__(self):
        """Make self an iterable over its two points."""
        return (i for i in ((self.x0, self.y0), (self.x1, self.y1)))

    def __getitem__(self, index):
        """Make self indexable. 0: 1st point, 1: 2nd point, other: Error."""
        return ((self.x0, self.y0), (self.x1, self.y1))[index]

    def length(self):
        """Return length of self."""
        return math.sqrt((self.x0-self.x1) ^ 2 + (self.y0-self.y1) ^ 2)


class Edge2d(Edge):
    """
    Docstring.

    Xamples
    -------
    Xample-1
    """

    ε = 1E-8
    __slots__ = ('x0', 'y0', 'x1', 'y1', 'f')

    def __init__(self, x0, y0, x1, y1):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1

    def __repr__(self):
        """Repr function."""
        return f'UPXO-e2d ({self.x0},{self.y0})-({self.x1},{self.y1})'

    def __iter__(self):
        """Make self an iterable over its two points."""
        return (i for i in ((self.x0, self.y0), (self.x1, self.y1)))

    def __getitem__(self, i):
        """Make self indexable. 0: 1st point, 1: 2nd point, other: Error."""
        return ((self.x0, self.y0), (self.x1, self.y1))[i]

    def __eq__(self, elist):
        """Check if the two edges are coincident."""
        pass

    def __ne__(self, elist):
        """Check if the two edges are not coincident."""
        pass

    @classmethod
    def by_coord(cls, start, end):
        """
        Create edge by specifying end coordinates.

        Parameters
        ----------
        start: Starting point coordinate [x0, y0]
        end: Ending point coordinate [x1, y1]

        Example
        -------
        from upxo.geoEntities.point3d import Edge2d as e2d
        A = e2d.by_coord([-1, 2], [3, 4])
        """
        return cls(start[0], start[1], end[0], end[1])

    @classmethod
    def by_perp_bisector(cls, e, p):
        """
        Calculate and make the perpendicular bisector edge b/w edge and point.

        Parameters
        ----------
        e: Edge specification. Preferred: UPXO edge2d_leanest
        p: POint specificaiton. Preferred: UPXO point2d_leanest

        Examples
        --------
        from upxo.geoEntities.point3d import Edge2d as e2d
        from upxo.geoEntities.point3d import edge2d_leanest
        from upxo.geoEntities.point3d import p2d_leanest

        e = edge2d_leanest(-2, 3, 4, 5)
        e[1]
        p = p2d_leanest(1, 2)

        from sympy import Point, Segment
        s = Segment(Point(e[0], e[1]), Point(e[2], e[3]))
        """
        pass

    @classmethod
    def by_transform(cls, refedge=None, shiftxy=[0, 1],
                     rot=+45, degree=True, rot_pnt_f=0.5):
        """
        Calculate and make the new edge by transofrming self.

        Parameters
        ----------
        refedge: Reference egde which is to be transformed. Preferably, provide
            UPXO edge2d object.
        shiftxy: Translation along x and y axes.
        rot: Rotation angle about a point. This point location is determined
            using input rot_pnt_f. Positive value indicates CCW from x+ axis.
            Negative value indicated CW from x+ axis. Domain is [0, 180] and
            [-(180-eps), 0-eps], eps is very small number in python (value?).
        degree: If True, provided rot value will vbe considered in degrees,
            else, in radians.
        rot_pnt_f: Valid domain [0.0, 1.0]. This is a factor of length. For
            example, if rot_pnt_f = 0.25, then the point about which the edge
            shall be rotated by rot angle will be located at 25% length away
            from start (i.e, (x0, y0)) point of the refedge.
        """
        pass

    @classmethod
    def by_loc_len_ang(cls, *, ref='i', loc=[0, 0, 0],
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

    @classmethod
    def by_dist_bw_points(cls, *, refpoint=None, points=None, f):
        """
        Create edge at a calcuated distance between a refpoint and points.

        Parameters
        ----------
        refpoint: Single point specirication

        points: A list of points

        f: Length factor with valid domain (0., 1.). If 0.5, then an edge2d
            willbe created between refpoint and that p in points, which has
            distance to refpoint closest to 0.5 of the maximum distance.
        """
        pass

    @property
    def mid(self):
        """Return the mid point."""
        return ((self.x0+self.x1)/2, (self.y0+self.y1)/2)

    @property
    def ang(self):
        """Return the ccw + angle in radians."""
        return math.atan2(self.y1-self.y0, self.x1-self.x0)

    @property
    def angdeg(self):
        """Return the ccw + angle in radians."""
        return math.degrees(self.ang)

    @property
    def length(self):
        """Calculate and return self length."""
        return math.sqrt((self.x0-self.x1)**2 + (self.y0-self.y1)**2)

    def distance_to_points(self, *, plist=None):
        """Calculate the EUclidean distance between self and list of points."""
        pass

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

    def translate_by(self, *, vector=None, refloc = 'i', dist=None,
                     update=False, throw=True):
        """
        Translate the Edge by a Euclidean distance.

        Translate the Edge along the vector by a given distance. If
        update is True, then coords of the self will be updated.

        If throw is True and update is False, a new edge of the new
        coordinates shall be returned. If throw is True and update is True, a
        deepcopy of the self shall be returned.

        Development phases
        ------------------
        Phase - 1: Without any validations

        Parameters
        ----------
        vector: Direction of translation. Two specifications allowed are:
             Specification 1: [vector start point coords,
                               vector end point coords]
             Specification 2: 'x+', 'z-'

        dist: Euclidean distance. If None and not a Number, then the
            translation distrance will be the lenght of the vector. If a
            number, then dist will be the translation distance, in which case
            the vector will only be used to know the translation direction.

        update: Update the current point if True, do not update if False.

        throw: Return a edge if True, else return nothing if False.

        Return
        ------
        UPXO edge object: Conditional, depending on input throw (refer to
                                                                 description).

        Author
        ------
        Dr. Sunil Anandatheertha

        Examples
        --------
        from upxo.geoEntities.point3d import Edge2d as e2d
        from upxo.geoEntities.point3d import edge2d_leanest
        from upxo.geoEntities.point3d import p2d_leanest

        e = edge2d_leanest(-2, 3, 4, 5)
        e.x0
        p = p2d_leanest(1, 2)

        """
        pass

    def pixelise(self, pixel_size=None):
        pass

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
