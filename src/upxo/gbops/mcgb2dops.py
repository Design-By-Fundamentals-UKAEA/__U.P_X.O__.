import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries


def circular_sort(coords=None,
                  order='ccw',
                  origin_spec='tl',
                  origin_point=(0, 0)
                  ):
    """
    This sorts the grain boundary points in circular order. The grain boundary
    points are specified by coords. Circular sorting order is specified by
    order. Point to start sorting from is specified by the origin.

    Parameters
    ----------
    coords : np.array, optional
        np.vstack of (x, y) corrdinates of points. The default is None.
    order : str, optional
        Specifies circular sorting order. Options include (cw, clockwise) and
        (ccw, acw, counterclockwise, anticlockwise). The default is 'ccw'.
    origin_spec : str, optional
        Specifies the starting point location of the sorted point array.
        Options include 'tl', 'tr', 'br', 'bl', 'l', 't', 'r', 'b', 'closest'
        and 'ignore'. The default is 'tl'. They stand for 'tl': top-left,
        'tr': top-right, 'br': bottom-right, 'bl': bottom-left, 'l': leftmost,
        't': topmost, 'r': right-most, 'b': bottom-most, 'closest': the point
        in the given coords closest to origin_point. If 'ignore': the
        origin_point will be used.
    origin_point : tuple/list, optional
        Helps determine the first point of the sorted list. If origin_spec is
        set to 'ignore', origin_point will be used as the first point. It must
        be a member of coords input point. If the origin_spec is set to
        'closest', distances between origin_point and all coords points will
        be first calculated and the coord point closest to the origin_point
        will be used as the starting point. If more than one coord point has
        the  same distance to the origin_point, then the top-left point of
        these sub-set will be used as the starting point.
    method: int, optional
        Specifies the method of sorting. Three methods are offered, 1, 2 & 3.
    """
    # STEP-1: UNPACK jp_grainwise
    circular_sort_method1(coords)

    if method==2:
        pass

    if method==3:
        pass

def circular_sort_method1(jp_grainwise,
                            sort_order,
                            method='method1',
                            debug_mode=True):
    """
    DOC-STRING HERE

    Author
    ------
    Dr. Sunil Anandatheertha

    Example pre-requisites
    ----------------------
    from upxo.pxtal.mcgs import monte_carlo_grain_structure as mcgs
    PXGS = mcgs()
    PXGS.simulate()
    PXGS.detect_grains()
    from upxo.gbops.mcgb2dops import sort_gb_junction_points

    Examples
    --------
    sort_gb_junction_points(PXGS.gs[8], sort_order)
    """
    """ VALIDATE coords """
    """ VALIDATE sort_order """
    _default_sorting_method = 'method1'
    # BUild the return variable structure
    _jp_sorted = {gid: {'coords': None,
                        'indices': None} for gid in jp_grainwise.keys()}
    jp_sorted = {
        'method': method,
        'sortorder': sort_order if method == 'method2'
        else 'cw' if method == 'method1' else _default_sorting_method,
        'sorted': _jp_sorted}
    # Sort the coordinates clockwise or anti-clockwise
    for gid, coords in jp_grainwise.items():
        if coords[0].size == 1:  # No sorting for a single point !!
            jp_sorted['values'][gid]['coords'] = coords
            jp_sorted['values'][gid]['indices'] = 0
        elif coords[0].size > 1:
            if debug_mode:
                print('\n', 10*'-', '\n', 'x:', coords[0], 'y:', coords[1])
            if method == 'method1':
                if coords[0].size == 2:
                    sorted_coords = coords
                    sorted_indices = np.array([0, 1])
                elif coords[0].size > 2:
                    sorted_coords, sorted_indices = method1(coords)
            elif method == 'method2':
                sorted_coords, sorted_indices = method2(coords)
            if debug_mode:
                print('Sorted coords: X.',
                      sorted_coords[0],
                      'Y:', sorted_coords[1])
            jp_sorted['values'][gid]['coords'] = sorted_coords
            jp_sorted['values'][gid]['indices'] = sorted_indices
    return jp_sorted


def method1(coords):
    """
    This method is borrowed heavily from the below source
    Source: https://gist.github.com/flashlib/e8261539915426866ae910d55a3f9959
    """
    pts = coords.T
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # if use Euclidean distance, it will run in error when the object
    # is trapezoid. So we should use the same simple y-coordinates order
    # method.

    # now, sort the right-most coordinates according to their
    # y-coordinates so we can grab the top-right and bottom-right
    # points, respectively
    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    print(rightMost)
    (tr, br) = rightMost

    sorted_indices = None
    sorted_coords = np.array([tl, tr, br, bl])
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return sorted_coords, sorted_indices


def method2(coords):
    # Step 1: Find the leftmost (and then topmost) point
    leftmost_point = coords[np.lexsort((coords[:, 1],
                                        coords[:, 0]))][0]
    # Step 2: Center the points around the leftmost point
    centered_coords = coords - leftmost_point
    # Step 3: Calculate angles
    angles = np.arctan2(centered_coords[:, 1],
                        centered_coords[:, 0])
    # Step 4: Sort by angles
    # Note: np.arctan2 returns angles in [-pi, pi], negative angles
    # are in the 3rd and 4th quadrants
    # Sorting in ascending order effectively gives us a
    # counterclockwise direction, so we might need to adjust based
    # on clockwise requirement.
    sorted_indices = np.argsort(angles)
    sorted_coords = coords[sorted_indices]
    return sorted_coords, sorted_indices


def method3_2d(coords=None,
               order='ccw',
               origin_spec='tl',
               origin_point=(0, 0),
               method=3
               ):
    """
    This sorts the grain boundary points in circular order. The grain boundary
    points are specified by coords. Circular sorting order is specified by
    order. Point to start sorting from is specified by the origin.

    Parameters
    ----------
    coords : np.array, optional
        np.vstack of (x, y) corrdinates of points. The default is None.
    order : str, optional
        Specifies circular sorting order. Options include (cw, clockwise) and
        (ccw, acw, counterclockwise, anticlockwise). The default is 'ccw'.
    origin_spec : str, optional
        Specifies the starting point location of the sorted point array.
        Options include 'tl', 'tr', 'br', 'bl', 'l', 't', 'r', 'b', 'closest'
        and 'ignore'. The default is 'tl'. They stand for 'tl': top-left,
        'tr': top-right, 'br': bottom-right, 'bl': bottom-left, 'l': leftmost,
        't': topmost, 'r': right-most, 'b': bottom-most, 'closest': the point
        in the given coords closest to origin_point. If 'ignore': the
        origin_point will be used.
    origin_point : tuple/list, optional
        Helps determine the first point of the sorted list. If origin_spec is
        set to 'ignore', origin_point will be used as the first point. It must
        be a member of coords input point. If the origin_spec is set to
        'closest', distances between origin_point and all coords points will
        be first calculated and the coord point closest to the origin_point
        will be used as the starting point. If more than one coord point has
        the  same distance to the origin_point, then the top-left point of
        these sub-set will be used as the starting point.
    method: int, optional
        Specifies the method of sorting. Three methods are offered, 1, 2 & 3.
    """
    # Calculate the centroid of the points
    centroid = np.mean(coords, axis=0)

    # Function to calculate angle and distance from centroid
    def angle_and_distance(point, centroid, clockwise=False):
        angle = np.arctan2(point[1] - centroid[1], point[0] - centroid[0])
        if clockwise:
            # To sort clockwise, we can invert the angle
            angle = -angle
        distance = np.linalg.norm(point - centroid)
        return angle, distance

    # Function to sort points with an option for direction
    def sort_points(points, clockwise=False):
        # Sorting indices based on angles (and distances for collinear points)
        sorted_indices = sorted(range(len(points)), key=lambda i: angle_and_distance(points[i], centroid, clockwise))
        # Sorted points using the sorted indices
        sorted_points = points[sorted_indices]
        return sorted_points, sorted_indices

    # Example usage
    clockwise_sorted_points, clockwise_sorted_indices = sort_points(coords, clockwise=True)
    counterclockwise_sorted_points, counterclockwise_sorted_indices = sort_points(coords, clockwise=False)

def calculate_junction_order(matrix, junction_point):
    """
    DOC-STRING HERE

    Author
    ------
    Dr. Sunil Anandatheertha
    """
    y, x = junction_point
    # Extract the 3x3 neighborhood around the junction point
    neighborhood = matrix[max(0, y-1):y+2, max(0, x-1):x+2]
    # Count unique grains in the neighborhood
    unique_grains = np.unique(neighborhood)
    # Exclude background identifier if necessary (assuming background/grain identifiers are > 0)
    order = len(unique_grains[unique_grains > 0])  # Modify condition based on your background identifier
    return order
