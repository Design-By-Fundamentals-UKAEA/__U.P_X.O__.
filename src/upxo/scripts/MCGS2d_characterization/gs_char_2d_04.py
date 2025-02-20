# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 10:45:20 2024

@author: Dr. Sunil Anandatheertha

This converts the non-geometric MCGS2d to geometric grain strcture.
"""

import cv2
import numpy as np
import gmsh
import pyvista as pv
from copy import deepcopy
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter
from meshpy.triangle import MeshInfo, build
from upxo.ggrowth.mcgs import mcgs
# ---------------------------
pxt = mcgs()
pxt.simulate()
pxt.detect_grains()
tslice = 49
pxt.char_morph_2d(tslice)
gstslice = pxt.gs[tslice]
gstslice.neigh_gid
gstslice.find_grain_boundary_junction_points()
folder, fileName = r'D:\export_folder', 'sunil'
gstslice.export_ctf(folder, fileName, factor=1, method='nearest')
# ---------------------------
fname = r'D:\export_folder\sunil'
gstslice.set_pxtal(instance_no=1, path_filename_noext=fname)
# ---------------------------
gstslice.pxtal[1].find_gbseg1()
gstslice.pxtal[1].gbseg1
gstslice.pxtal[1].extract_gb_discrete(retrieval_method='external',
                                      chain_approximation='simple')
gstslice.pxtal[1].plot_gb_discrete(bjp_kwargs={'marker': 'o', 'mfc': 'yellow',
                                               'mec': 'black', 'ms': 2.5},
                                   simple_all_preference='simple')
# ---------------------------
gstslice.pxtal[1].gb_discrete.keys()

# =====================================================================
def sort_clockwise(points, center=None):
    """Sorts a list of 2D points in clockwise order around a given center.

    Args:
        points: A list of tuples or lists representing 2D points [(x1, y1), (x2, y2), ...].
        center (optional): A tuple or list representing the center point. If not provided,
                            the centroid of the points is used.

    Returns:
        The sorted list of points in clockwise order.
    """

    if center is None:
        center = np.mean(points, axis=0)  # Calculate centroid if not given

    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    sort_indices = np.argsort(angles)

    # Adjust for angles in the range [0, pi) being sorted before angles in [-pi, 0)
    sort_indices = np.roll(sort_indices, -np.where(sort_indices == np.argmin(angles))[0][0] - 1)

    return points[sort_indices]


gid = 10
mesh_algorithm = 4
element_size = 1.0
'''
mesh_algorithm options

1: MeshAdapt
2: Automatic (default)
3: Delaunay <----------------
4: Frontal-Delaunay <---------------- <----------------
5: BAMG (Bidimensional Anisotropic Mesh Generator)
6: Frontal-Delaunay for quads <----------------
7: Packing of Parallelograms
'''
# =====================================================================
'''gstslice.pxtal[1].plot_grains_gids([gid], gclr='color', title="user grains",
                     cmap_name='coolwarm', plot_centroids=True,
                     add_gid_text=True, plot_gbseg=True)'''
# =====================================================================
xy = gstslice.pxtal[1].gb_discrete[gid]['gb_points']['simple']

gstslice.pxtal[1].n

import numpy as np
import rasterio
from shapely.geometry import shape, Polygon, MultiPolygon

def polygonize_voronoi_grid(lgi, gids):
    """
    Polygonizes grains in self.lgi and returns a shapely MultiPolygon.

    Parameters
    ----------
    lgi: A NumPy array representing the integer grid values.

    Returns
    -------
    rioshapes
    polygons: list of shapely polygon objects of each grain
    multi_polygon: shapely multi-polygon object
    """
    lgi, gids = gstslice.pxtal[1].lgi, gstslice.pxtal[1].gid


    import rasterio
    from rasterio.features import shapes
    from shapely.geometry import shape as ShShape
    from shapely.geometry import MultiPolygon
    rioshapes = rasterio.features.shapes
    # Create a raster dataset from the lgi array

    with rasterio.Env():
        profile = rasterio.profiles.DefaultGTiffProfile()
        profile.update(width=lgi.shape[1],
                       height=lgi.shape[0], count=1,
                       dtype=lgi.dtype,
                       transform=rasterio.transform.Affine.identity())
        with rasterio.MemoryFile() as memfile:
            with memfile.open(**profile) as dataset:
                dataset.write(lgi, 1)
                # Find unique cell IDs; same as self.gid
                # gids = np.unique(lgi)
                # Polygonize each unique cell
                polygons = []
                RESULTS = []
                for gid in gids:
                    mask = (lgi == gid).astype(np.uint8)
                    rs = list(rioshapes(mask, mask=mask, transform=dataset.transform))
                    coordinates = np.array(rs[0][0]['coordinates']).squeeze()
                    #print(gid,
                    #      ', Min of mask:', mask.min(),
                    #      rs[0][0]['type'],
                    #      coordinates.size)
                    #polygons.append(Polygon(coordinates))
                    #list(rioshapes(mask, mask=mask,
                    #                         transform=dataset.transform))


                    results = list(rioshapes(mask, mask=mask,
                                             transform=dataset.transform))
                    if results:
                        RESULTS.append(results)
                        # Convert to Shapely polygons and append
                        polygons.extend([ShShape(geom[0])
                                         for geom in results])

    # Create a MultiPolygon from the collected polygons
    multi_polygon = MultiPolygon(polygons)

    return RESULTS, polygons, multi_polygon

from shapely.strtree import STRtree
def find_polygon_neighbors(polygons):
    """Calculates neighboring polygon IDs for a list of polygons.

    Args:
        polygons: A list of Shapely Polygon objects.

    Returns:
        A dictionary where keys are polygon IDs (0-based index) and values are lists of
        neighboring polygon IDs.
    """

    if not isinstance(polygons, MultiPolygon):
        polygons = MultiPolygon(polygons)

    neighbors = {}
    for i, polygon in enumerate(polygons, start=1):
        neighbors[i] = []
        for j, other_polygon in enumerate(polygons, start=1):
            if i != j and polygon.touches(other_polygon):  # Check if polygons touch
                neighbors[i].append(j)

    return neighbors


def plot_voronoi_cells(multi_polygon, lgi=None):
    """Plots a MultiPolygon representing Voronoi cells and optionally the underlying grid.

    Args:
        multi_polygon: A Shapely MultiPolygon object.
        grid_array (optional): The original NumPy array of grid values for background visualization.
    """

    fig, ax = plt.subplots()

    if lgi is not None:
        # Plot the grid array as a background image
        ax.imshow(lgi, cmap='viridis', origin='lower')

    # Plot each polygon in the MultiPolygon
    for i, polygon in enumerate(multi_polygon.geoms, start=1):
        x, y = polygon.exterior.xy
        x, y = np.array(list(x))-0.5, np.array(list(y))-0.5
        ax.plot(x, y, color='black', lw=1, ls='-', marker='.')
        pcx, pcy = polygon.centroid.coords.xy
        pcx, pcy = pcx[0], pcy[0]
        ax.plot(pcx, pcy, 'ko')
        ax.text(pcx, pcy, str(i), color='white', fontsize=12, fontweight='bold')

    ax.set_aspect('equal')
    plt.show()

from shapely.geometry import Polygon, LineString, Point, MultiLineString
def get_polygon_touch_lines(poly1, poly2):
    """Calculates the lines or points where two Shapely polygons touch.

    Args:
        poly1, poly2: The Shapely Polygon objects.

    Returns:
        A list of Shapely LineString objects (if the polygons share lines)
        or a list of Shapely Point objects (if they touch at a single point).
    """
    intersection = poly1.intersection(poly2)

    if intersection.is_empty:
        return []  # No intersection

    if isinstance(intersection, Point):
        return [intersection]  # Single point contact
    elif isinstance(intersection, LineString):
        return [intersection]  # Single line contact
    else:  # MultiLineString or GeometryCollection
        return list(intersection.geoms)  # Multiple lines or points


import matplotlib.pyplot as plt
from shapely.geometry import LineString, MultiLineString

def plot_linestrings(linestrings, ax=None, color='blue', linewidth=1, **kwargs):
    """Plots a list of Shapely LineStrings and returns the axis.

    Args:
        linestrings: A list of LineString or MultiLineString objects.
        ax (optional): A Matplotlib Axes object to plot on. If None, a new figure and axis will be created.
        color (optional): Color of the lines (default: 'blue').
        linewidth (optional): Width of the lines (default: 1).
        **kwargs: Additional keyword arguments to pass to plt.plot().

    Returns:
        The Matplotlib Axes object on which the lines were plotted.
    """

    if ax is None:
        fig, ax = plt.subplots()  # Create a figure and axis if not provided

    for linestring in linestrings:
        if isinstance(linestring, MultiLineString):
            for geom in linestring.geoms:  # Plot each LineString in a MultiLineString
                x, y = geom.xy
                ax.plot(x, y, color=color, linewidth=linewidth, **kwargs)
        else:
            x, y = linestring.xy
            ax.plot(x, y, color=color, linewidth=linewidth, **kwargs)

    ax.set_aspect('equal')  # Ensure equal aspect ratio for accurate representation
    #plt.title('Plot of LineStrings')  # Optional title

    plt.show()  # Show the plot (optional)

    return ax  # Return the axis object


def get_multilinestring_touch_points(mls1, mls2):
    """Calculates the point(s) where two Shapely MultiLineStrings touch.

    Args:
        mls1, mls2: The Shapely MultiLineString objects.

    Returns:
        A list of Shapely Point objects representing the touch points, or an empty list if there are no touch points.
    """

    touch_points = []
    for line1 in mls1.geoms:  # Iterate through linestrings in mls1
        for line2 in mls2.geoms:  # Iterate through linestrings in mls2
            intersection = line1.intersection(line2)  # Find intersection
            if isinstance(intersection, Point):
                touch_points.append(intersection)

    return touch_points

gstslice.pxtal[1].n
mp = polygonize_voronoi_grid(gstslice.pxtal[1].lgi, gstslice.pxtal[1].gid)
len(mp[1])
gstslice.pxtal[1].n
polygons = mp[1]

find_polygon_neighbors(polygons)
gstslice.pxtal[1].neigh_gid
plot_voronoi_cells(MultiPolygon(polygons), gstslice.pxtal[1].lgi)

plt.imshow(gstslice.pxtal[1].lgi)
centroids = []
for polygon in polygons:
    x, y = polygon.exterior.xy
    x, y = np.array(list(x))-0.5, np.array(list(y))-0.5
    plt.plot(x, y, color='black', lw=1, ls='-', marker='.')
    pcx, pcy = polygon.centroid.coords.xy
    pcx, pcy = pcx[0]-0.5, pcy[0]-0.5
    centroids.append([pcx, pcy])
centroids = np.array(centroids)
plt.plot(centroids[:, 0], centroids[:, 1], 'ko')

NEIGHS1 = find_polygon_neighbors(polygons)
gstslice.pxtal[1].neigh_gid
for n1, n2 in zip(gstslice.pxtal[1].neigh_gid.values(), NEIGHS1.values()):
    print(len(n1)-len(n2))



mps1 = []
for pid, neigh in NEIGHS1.items():
    neigh = [n-1 for n in NEIGHS1[1]]
    mps1.append(MultiPolygon(list(np.array(polygons)[neigh])))
mps2 = []

plot_voronoi_cells(mps1[23], gstslice.pxtal[1].lgi)


for _mp_ in mps:
    plot_voronoi_cells(_mp_, grid_array=gstslice.pxtal[1].lgi)

neigh_pols = {pid: [polygons[i-1] for i in NEIGHS1[pid]] for pid in gstslice.pxtal[1].gid}

gbsegments = {}
for pid in gstslice.pxtal[1].gid:
    gbsegments[pid] = [get_polygon_touch_lines(polygons[pid-1],
                                               np) for np in neigh_pols[pid]]

pid = 2
for i, n in enumerate(gbsegments[pid], start=1):
    if i == 1:
        ax = plot_linestrings(n, ax=None, color='green', linewidth=i)
    else:
        ax = plot_linestrings(n, ax=ax, color='green', linewidth=i)

gbsegments_mls = {}
for pid in gstslice.pxtal[1].gid:
    gbsegments_mls[pid] = [MultiLineString(gbseglines) for gbseglines in gbsegments[pid]]



from shapely.strtree import STRtree
def find_junction_points(polygon_A, polygons_B):
    """Finds junction points and their associated information in a polygon arrangement.

    Args:
        polygon_A: The central Shapely Polygon object.
        polygons_B: A list of Shapely Polygon objects surrounding polygon_A.

    Returns:
        A dictionary where:
            - Keys are junction Point objects.
            - Values are dictionaries containing:
                - 'polygons': A list of polygon indices (0 for polygon_A, 1+ for polygons_B) that meet at this junction.
                - 'count': The number of polygons meeting at this junction.
    """

    all_polygons = [polygon_A] + polygons_B
    tree = STRtree(all_polygons)

    junction_points = {}

    for i, poly in enumerate(all_polygons):
        # Get candidate junctions from intersection points
        for other_poly in tree.query(poly):
            if poly == other_poly:
                continue  # Skip self-intersection
            intersection = poly.intersection(other_poly)

            if isinstance(intersection, Point):
                # It's a junction point if it touches at least 3 polygons
                touching_polygons = tree.query(intersection.buffer(1e-8))
                if len(touching_polygons) >= 3:
                    junction_points.setdefault(intersection, {'polygons': [], 'count': 0})
                    junction_points[intersection]['polygons'].append(i)  # Add polygon index
                    junction_points[intersection]['count'] += 1

    return junction_points


def calculate_mean_multilinestring(mls, window_size=3):
    """Calculates a mean MultiLineString by smoothing the input using a moving average.

    Args:
        mls: A Shapely MultiLineString object.
        window_size: The size of the moving average window (odd number, default=3).

    Returns:
        A Shapely MultiLineString object representing the smoothed mean lines.
    """
    mean_lines = []
    for line in mls.geoms:
        coords = np.array(line.coords)
x
        # Handle case of single-point LineString
        if coords.shape[0] == 1:
            mean_lines.append(line)  # No smoothing needed
            continue

        # Pad coordinates to ensure enough points for moving average at the start and end
        padding = [(coords[0] - (coords[1] - coords[0]))] * (window_size // 2)
        padded_coords = np.vstack([padding, coords, padding])

        # Calculate moving average along each axis
        mean_x = np.convolve(padded_coords[:, 0], np.ones(window_size) / window_size, mode='valid')
        mean_y = np.convolve(padded_coords[:, 1], np.ones(window_size) / window_size, mode='valid')

        # Create new LineString from mean coordinates
        mean_line = LineString(zip(mean_x, mean_y))

        mean_lines.append(mean_line)

    return MultiLineString(mean_lines)

def moving_average(data, window_size):
    """Compute the moving average of the given data with the specified window size."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def extract_coordinates_multiline(multiline: MultiLineString):
    """
    Extract coordinates from a MultiLineString.

    Parameters:
    multiline (MultiLineString): The MultiLineString object.

    Returns:
    numpy.ndarray: An array of coordinates.
    """
    coords = []
    for line in multiline:
        coords.extend(line.coords)
    return np.array(coords)


def extract_coordinates(multiline: MultiLineString):
    """
    Extract coordinates from a MultiLineString.

    Parameters:
    multiline (MultiLineString): The MultiLineString object.

    Returns:
    list: A list of coordinates for each LineString.
    """
    return [list(line.coords) for line in multiline]

def mean_coordinates_multiline(multiline: MultiLineString, window_size: int) -> LineString:
    # Extract the coordinates from the MultiLineString
    coords = [point for line in multiline for point in line.coords]

    if len(coords) < window_size:
       return multiline

    # Convert coordinates to numpy array for easier manipulation
    coords_array = np.array(coords)

    # Separate the coordinates into x and y components
    x = coords_array[:, 0]
    y = coords_array[:, 1]

    # Apply moving average to the x and y components separately
    x_smooth = moving_average(x, window_size)
    y_smooth = moving_average(y, window_size)

    # Calculate the number of points removed by the moving average
    num_removed = window_size - 1

    # Add the original end points to the smoothed coordinates
    smoothed_coords = np.vstack([
        [x[0], y[0]],  # Start point
        np.column_stack([x_smooth, y_smooth]),
        [x[-1], y[-1]]  # End point
    ])

    # Create a new LineString with the smoothed coordinates
    smoothed_line = LineString(smoothed_coords)

    return smoothed_line

def plot_multilines(multiline: MultiLineString, ax):
    # Plot original MultiLineString
    xy = extract_coordinates_multiline(multiline)
    ax.plot(xy[:, 0], xy[:, 1], '-')

fig, ax = plt.subplots()
for i, segment in enumerate(gbsegments_mls[pid]):
    xy = extract_coordinates_multiline(segment)
    print(i, len(xy))
    # plotting
    ax.plot(xy[:, 0], xy[:, 1], '-', linewidth=2)

mean_coordinates_multiline(gbsegments_mls[pid][0], 4)

fig, ax = plt.subplots()
for i, segment in enumerate(gbsegments_mls[pid]):
    smoothed_line = mean_coordinates_multiline(segment, 3)
    xy = extract_coordinates_multiline(smoothed_line)
    ax.plot(xy[:, 0], xy[:, 1], ':', linewidth=1)

fig, ax = plt.subplots(nrows=1, ncols=15)
for spn, pid in enumerate(gstslice.pxtal[1].gid[0:15], start=0):

    # pid = 1
    smoothed_segments_l1 = []
    for i, segment in enumerate(gbsegments_mls[pid]):
        # ------------------'
        # orieginal grain boundary segmewt
        xy = extract_coordinates_multiline(segment)
        ax[spn].plot(xy[:, 0], xy[:, 1], ':', linewidth=1)
        # ------------------'
        # smoothed grain boundary segmewt
        smoothed_line = mean_coordinates_multiline(segment, 3)
        smoothed_segments_l1.append(smoothed_line)
        xy = extract_coordinates_multiline(smoothed_line)
        ax[spn].plot(xy[:, 0], xy[:, 1], '--', linewidth=1)

    smoothed_segments_l2 = []
    for sgl1 in smoothed_segments_l1:
        sgl2 = mean_coordinates_multiline(sgl1, 3)
        smoothed_segments_l2.append(sgl2)
        xy = extract_coordinates_multiline(sgl2)
        ax[spn].plot(xy[:, 0], xy[:, 1], '-', linewidth=4)






def extract_coordinates(multiline: MultiLineString):
    """
    Extract coordinates from a MultiLineString.

    Parameters:
    multiline (MultiLineString): The MultiLineString object.

    Returns:
    list: A list of coordinates for each LineString.
    """
    return [list(line.coords) for line in multiline]

def mean_coordinates_multiline(multiline: MultiLineString, window_size: int) -> MultiLineString:
    # Extract coordinates for each LineString in the MultiLineString
    all_coords = extract_coordinates(multiline)

    # Flatten the list of coordinates
    flattened_coords = [coord for line_coords in all_coords for coord in line_coords]

    # Check if there are enough points for the moving average
    if len(flattened_coords) < window_size:
        return multiline  # Return the original MultiLineString if not enough points

    # Convert coordinates to numpy array for easier manipulation
    coords_array = np.array(flattened_coords)

    # Separate the coordinates into x and y components
    x = coords_array[:, 0]
    y = coords_array[:, 1]

    # Apply moving average to the x and y components separately
    x_smooth = moving_average(x, window_size)
    y_smooth = moving_average(y, window_size)

    # Calculate the number of points removed by the moving average
    num_removed = window_size - 1

    # Add the original end points to the smoothed coordinates if there are enough points
    if len(x_smooth) > 0 and len(y_smooth) > 0:
        smoothed_coords = np.vstack([
            [x[0], y[0]],  # Start point
            np.column_stack([x_smooth, y_smooth]),
            [x[-1], y[-1]]  # End point
        ])
    else:
        smoothed_coords = coords_array  # If not enough points, use original coordinates

    # Split the smoothed coordinates back into the original structure
    smoothed_lines = []
    index = 0
    for line_coords in all_coords:
        num_points = len(line_coords)
        smoothed_line_coords = smoothed_coords[index:index + num_points]

        # Ensure we have at least 2 points for the LineString
        if len(smoothed_line_coords) >= 2:
            smoothed_lines.append(LineString(smoothed_line_coords))
        else:
            smoothed_lines.append(LineString(line_coords))  # Use original if smoothed has less than 2 points

        index += num_points

    return MultiLineString(smoothed_lines)

# =========================================================================
pid_A = 1
pids_B = NEIGHS1[pid]
polygon_A = polygons[pid-1]
polygons_B = [polygons[i-1] for i in pids_B]

all_polygons = [polygon_A] + polygons_B
tree = STRtree(all_polygons)
junction_points = {}
for i, poly in enumerate(all_polygons):
    # Get candidate junctions from intersection points
    for other_poly in tree.query(poly):
        if poly == other_poly:
            continue  # Skip self-intersection
        intersection = poly.intersection(other_poly)
# =========================================================================


grid_array = gstslice.pxtal[1].lgi
plt.imshow(grid_array)
mp = polygonize_voronoi_grid(gstslice.pxtal[1].lgi, gstslice.pxtal[1].gid)
mp[1]

plot_voronoi_cells(mp[2], grid_array)

polygons = mp[1]

x, y = np.array(mp[0][45][0][0]['coordinates']).squeeze().T

np.vstack(mp[1][45].exterior.coords.xy).T

type(mp[1][45].intersection(mp[1][190]))
# mp[1][45].intersection(mp[1][190]).coords

# lines = [g for g in mp[1][45].intersection(mp[1][81]).geoms]



# type(mp[1][45].intersection(mp[1][76]))
# inter_geom = mp[1][45].intersection(mp[1][76])
# [g for g in mp[1][45].intersection(mp[1][76]).geoms]


plt.imshow(gstslice.pxtal[1].bbox_ex[46])

ax = gstslice.pxtal[1].plot_grains_gids(gstslice.pxtal[1].neigh_gid[46], cmap_name='coolwarm')
x, y = np.vstack(mp[1][45].exterior.coords.xy)
ax.plot(x-0.5, y-0.5, '-k.')

for i, line in enumerate(lines, start=1):
    x, y = line.xy
    x, y = np.array(list(x))-0.5, np.array(list(y))-0.5
    ax.plot(x, y, '-g', linewidth=i+0.25, alpha=0.2)


from shapely.geometry import MultiLineString, LineString, Point
from shapely.geometry.collection import GeometryCollection

CENTRE_GRAIN = 21

for gid in gstslice.pxtal[1].neigh_gid[CENTRE_GRAIN]:
    ax = gstslice.pxtal[1].plot_grains_gids(gstslice.pxtal[1].neigh_gid[CENTRE_GRAIN], cmap_name='coolwarm')
    x, y = np.vstack(mp[1][CENTRE_GRAIN-1].exterior.coords.xy)
    #ax.plot(x-0.5, y-0.5, '-k.')
    inter_geom = mp[1][CENTRE_GRAIN-1].intersection(mp[1][gid-1])
    if type(inter_geom) == MultiLineString:
        lines = [g for g in inter_geom.geoms]
        for i, line in enumerate(lines, start=1):
            x, y = line.xy
            x, y = np.array(list(x))-0.5, np.array(list(y))-0.5
            ax.plot(x, y, '-k.', linewidth=1, alpha=0.8)
    if type(inter_geom) == LineString:
        x, y = inter_geom.xy
        x, y = np.array(list(x))-0.5, np.array(list(y))-0.5
        ax.plot(x, y, '-k.', linewidth=1, alpha=0.8)
    if type(inter_geom) == GeometryCollection:
        gobjs = [g for g in inter_geom]
        for gobj in gobjs:
            if type(gobj) == LineString:
                x, y = gobj.xy
                x, y = np.array(list(x))-0.5, np.array(list(y))-0.5
                ax.plot(x, y, '-k.', linewidth=1, alpha=0.8)
            elif type(gobj) == Point:
                # We will ignore the point and move on
                pass
            elif type(gobj) == MultiLineString:
                gobjs_lines = [g for g in gobjs.geoms]
                for i, gobjs_line in enumerate(gobjs_lines, start=1):
                    x, y = gobjs_line.xy
                    x, y = np.array(list(x))-0.5, np.array(list(y))-0.5
                    ax.plot(x, y, '-k.', linewidth=1, alpha=0.8)


CENTRE_GRAIN = 21
plt.imshow(gstslice.pxtal[1].lgi)

for CENTRE_GRAIN in gstslice.pxtal[1].gid:
    for gid in gstslice.pxtal[1].neigh_gid[CENTRE_GRAIN]:
        x, y = np.vstack(mp[1][CENTRE_GRAIN-1].exterior.coords.xy)
        inter_geom = mp[1][CENTRE_GRAIN-1].intersection(mp[1][gid-1])
        if type(inter_geom) == MultiLineString:
            lines = [g for g in inter_geom.geoms]
            for i, line in enumerate(lines, start=1):
                x, y = line.xy
                x, y = np.array(list(x))-0.5, np.array(list(y))-0.5
                plt.plot(x, y, '.', linewidth=1, alpha=0.8)
        if type(inter_geom) == LineString:
            x, y = inter_geom.xy
            x, y = np.array(list(x))-0.5, np.array(list(y))-0.5
            plt.plot(x, y, '.', linewidth=1, alpha=0.8)
        if type(inter_geom) == GeometryCollection:
            gobjs = [g for g in inter_geom]
            for gobj in gobjs:
                if type(gobj) == LineString:
                    x, y = gobj.xy
                    x, y = np.array(list(x))-0.5, np.array(list(y))-0.5
                    plt.plot(x, y, '.', linewidth=1, alpha=0.8)
                elif type(gobj) == Point:
                    # We will ignore the point and move on
                    pass
                elif type(gobj) == MultiLineString:
                    gobjs_lines = [g for g in gobjs.geoms]
                    for i, gobjs_line in enumerate(gobjs_lines, start=1):
                        x, y = gobjs_line.xy
                        x, y = np.array(list(x))-0.5, np.array(list(y))-0.5
                        plt.plot(x, y, '.', linewidth=1, alpha=0.8)

len(mp[0])
len(gstslice.pxtal[1].gid)
gstslice.pxtal[1].n

# Collect all the points
X, Y = [], []
grains_with_holes = {gid: {'flag': None, 'nholes': None, 'coords': None} for gid in gstslice.pxtal[1].gid}
bpoints = {gid: {'x': None, 'y': None} for gid in gstslice.pxtal[1].gid}
for CENTRE_GRAIN in gstslice.pxtal[1].gid:
    if len(mp[0][CENTRE_GRAIN-1][0][0]['coordinates']) > 1:
        # When there are holes in a grain.
        grains_with_holes[gid]['flag'] = True
        grains_with_holes[gid]['nholes'] = len(mp[0][CENTRE_GRAIN-1][0][0]['coordinates']) - 1
        coords = mp[0][CENTRE_GRAIN-1][0][0]['coordinates'][0]
        bpoints_x, bpoints_y = np.array(coords).squeeze().T
        grains_with_holes[gid]['coords'] = deepcopy(mp[0][CENTRE_GRAIN-1][0][0]['coordinates'][1:])
    else:
        coords = mp[0][CENTRE_GRAIN-1][0][0]['coordinates']
        bpoints_x, bpoints_y = np.array(coords).squeeze().T
    #bpoints_x = np.array(list(mp[1][CENTRE_GRAIN-1].exterior.coords.xy[0]))
    #bpoints_y = np.array(list(mp[1][CENTRE_GRAIN-1].exterior.coords.xy[1]))
    bpoints[CENTRE_GRAIN]['x'] = bpoints_x-0.5
    bpoints[CENTRE_GRAIN]['y'] = bpoints_y-0.5
    for gid in gstslice.pxtal[1].neigh_gid[CENTRE_GRAIN]:
        inter_geom = mp[1][CENTRE_GRAIN-1].intersection(mp[1][gid-1])
        if type(inter_geom) == MultiLineString:
            lines = [g for g in inter_geom.geoms]
            for i, line in enumerate(lines, start=1):
                x, y = line.xy
                X.extend(list(np.array(list(x))-0.5))
                Y.extend(list(np.array(list(y))-0.5))
        if type(inter_geom) == LineString:
            x, y = inter_geom.xy
            X.extend(list(np.array(list(x))-0.5))
            Y.extend(list(np.array(list(y))-0.5))
        if type(inter_geom) == GeometryCollection:
            gobjs = [g for g in inter_geom]
            for gobj in gobjs:
                if type(gobj) == LineString:
                    x, y = gobj.xy
                    X.extend(list(np.array(list(x))-0.5))
                    Y.extend(list(np.array(list(y))-0.5))
                elif type(gobj) == Point:
                    # We will ignore the point and move on
                    pass
                elif type(gobj) == MultiLineString:
                    gobjs_lines = [g for g in gobjs.geoms]
                    for i, gobjs_line in enumerate(gobjs_lines, start=1):
                        x, y = gobjs_line.xy
                        X.extend(list(np.array(list(x))-0.5))
                        Y.extend(list(np.array(list(y))-0.5))

XY = np.vstack((X, Y)).T
XY.shape
XY = np.unique(XY, axis=0)
XY.shape
plt.imshow(gstslice.pxtal[1].lgi)
plt.plot(XY.T[0], XY.T[1], 'k.')

from upxo.geoEntities.point2d import Point2d
from upxo.geoEntities.mulpoint2d import MPoint2d
XY_UPXO_points = [Point2d(xy[0], xy[1]) for xy in XY]
XY_UPXO_mulpoint = MPoint2d.from_upxo_points2d(XY_UPXO_points)
id(XY_UPXO_points[0])
id(XY_UPXO_mulpoint.points[0])

plt.imshow(gstslice.pxtal[1].lgi)
for gid in gstslice.pxtal[1].gid:
    plt.plot(bpoints[gid]['x'], bpoints[gid]['y'], '-k')

bounds = {gid: [bpoints[gid]['x'].min(), bpoints[gid]['x'].max(),
                bpoints[gid]['y'].min(), bpoints[gid]['y'].max()]
          for gid in gstslice.pxtal[1].gid}

bpcounts = {i: [] for i, xy in enumerate(XY)}
for i, xy in enumerate(XY):
    for gid in gstslice.pxtal[1].gid:
        if (xy[0] >= bounds[gid][0]) and (xy[0] <= bounds[gid][1]):
            if (xy[1] >= bounds[gid][2]) and (xy[1] <= bounds[gid][3]):
                # This means xy is a boundary point of gid.
                bpcounts[i].append(gid)

xminlocs = np.argwhere(XY[:, 0] == XY[:, 0].min()).squeeze().T
xmaxlocs = np.argwhere(XY[:, 0] == XY[:, 0].max()).squeeze().T
yminlocs = np.argwhere(XY[:, 1] == XY[:, 1].min()).squeeze().T
ymaxlocs = np.argwhere(XY[:, 1] == XY[:, 1].max()).squeeze().T

boundary_bpoints_ids = np.unique(np.hstack((xminlocs, xmaxlocs, yminlocs, ymaxlocs)))
if boundary_bpoints_ids.size > 0:
    boundary_bpoints = XY[boundary_bpoints_ids]

plt.imshow(gstslice.pxtal[1].lgi)
plt.plot(boundary_bpoints[:, 0], boundary_bpoints[:, 1], 'ro')

boundary_gids = []
for bbp in boundary_bpoints:  # correct
    for gid in gstslice.pxtal[1].gid:
        boundary_point_coords = np.vstack((bpoints[gid]['x'], bpoints[gid]['y'])).T
        if np.any((boundary_point_coords == bbp).all(axis=1)):
            boundary_gids.append(gid)

gstslice.pxtal[1].plot_grains_gids(boundary_gids, gclr='color', title="user grains",
                     cmap_name='viridis', plot_centroids=True,
                     add_gid_text=True, plot_gbseg=False,
                     bjp_kwargs={'marker': 'o', 'mfc': 'yellow',
                                 'mec': 'black', 'ms': 2.5}
                     )

# junction point amongst boundary points
jp_bp = {i: {'flag': None, 'jo': None, 'gsboundary': None} for i, xy in enumerate(XY)}
for bp_id, grain_no in bpcounts.items():
    # ----------------------------------
    # Calculate the junction order
    jo = len(bpcounts[bp_id])
    jp_bp[bp_id]['jo'] = len(bpcounts[bp_id])
    # ----------------------------------
    # Store flag: True if indeed a junction point. Else False.
    if bp_id not in boundary_bpoints_ids:
        # If the boundary pont is not on the grain strucure boundary, then it
        # can be a junction point only if it connects threee or more grains.
        jp_bp[bp_id]['gsboundary'] = False
        if jo >= 3:
            jp_bp[bp_id]['flag'] = True
        else:
            jp_bp[bp_id]['flag'] = False
    else:
        # If boundary point is on grain structure boundary, the it
        # automatically qualifies to be a junction point.
        jp_bp[bp_id]['flag'] = True
        jp_bp[bp_id]['gsboundary'] = True
    # ----------------------------------

jpids = [bp_id for bp_id in jp_bp.keys() if jp_bp[bp_id]['flag']]

plt.imshow(gstslice.pxtal[1].lgi)
coords = XY[jpids]
for jpid in jpids:
    jpcoord = XY[jpid]
    if jp_bp[jpid]['gsboundary']:
        plt.plot(jpcoord[0], jpcoord[1], 'kx')
    else:
        if jp_bp[jpid]['jo'] == 3:
            plt.plot(jpcoord[0], jpcoord[1], 'bo')
        elif jp_bp[jpid]['jo'] == 4:
            plt.plot(jpcoord[0], jpcoord[1], 'ro')
        elif jp_bp[jpid]['jo'] == 5:
            plt.plot(jpcoord[0], jpcoord[1], 'rs')
remaining_points_ids = list(set(bpcounts) - set(jpids))
if remaining_points_ids:
    for pid in remaining_points_ids:
        plt.plot(XY[pid][0], XY[pid][1], 'w.')




bpoints_ids = {bpid: [] for bpid in bpoints.keys()}
for gid in gstslice.pxtal[1].gid:
    for coord in np.vstack((bpoints[gid]['x'], bpoints[gid]['y'])).T:
        loc = np.where((XY == coord).all(axis=1))[0]
        if len(loc) > 0:
            bpoints_ids[gid].append(np.where((XY == coord).all(axis=1))[0][0])


bpoints_ids
jpids
gb_junction_partial_masks = deepcopy(bpoints_ids)
for gid in gstslice.pxtal[1].gid:
    for i, bpid in enumerate(gb_junction_partial_masks[gid], start=0):
        if bpid in jpids:
            gb_junction_partial_masks[gid][i] = True
        else:
            gb_junction_partial_masks[gid][i] = False

gid = 208
gb_junction_partial_masks[gid]


gb_junction_point_coords_only = {gid: None for gid in gstslice.pxtal[1].gid}
for gid in gstslice.pxtal[1].gid:
    _ = list(np.argwhere(gb_junction_partial_masks[gid]).T.squeeze())
    coords = np.array([list(XY[j]) for j in [bpoints_ids[gid][i] for i in _]])
    # gb_junction_point_coords_only[gid] = sort_clockwise(coords)
    gb_junction_point_coords_only[gid] = coords


plt.imshow(gstslice.pxtal[1].lgi)
for gid in gstslice.pxtal[1].gid:
    plt.plot(gb_junction_point_coords_only[gid][:, 0],
             gb_junction_point_coords_only[gid][:, 1], '-ko')

gid = 4
# plt.imshow(gstslice.pxtal[1].lgi)
plt.plot(gb_junction_point_coords_only[gid][:, 0],
         gb_junction_point_coords_only[gid][:, 1], '-ko')

# Example usage
points = np.array(gb_junction_point_coords_only[195])
sorted_points = sort_clockwise(points)
print(sorted_points)

Polygon(points)
Polygon(sorted_points)


gids_with_atleast_one_practical_gbsegment = []
for gid in gstslice.pxtal[1].gid:
    if any(gb_junction_partial_masks[gid]):
        gids_with_atleast_one_practical_gbsegment.append(gid)
np.array(gids_with_atleast_one_practical_gbsegment)

gstslice.pxtal[1].plot_grains_gids(gids_with_atleast_one_practical_gbsegment,
                                   points=XY,
                                   title="grains with atleast one practical grain boundary segment !",
                                   cmap_name='viridis', plot_centroids=True,
                                   add_gid_text=False, plot_gbseg=True,
                                   bjp_kwargs={'marker': 'o', 'mfc': 'yellow',
                                               'mec': 'black', 'ms': 2.5})










multi_polygon = mp[2]
mp[0][0]

plt.imshow(grid_array)
plot_voronoi_cells(mp[2], grid_array=grid_array)



import pygmsh
geometry = pygmsh.geo.Geometry()
# ---------------------------------------------------
# Fetch model we would like to add data to
model = geometry.__enter__()
# ---------------------------------------------------
# Make pygmsh polygon objects from each upxo xtal vertices
areas = np.array([g.area for g in mp.geoms])
eqdia = np.sqrt(4*areas/np.pi)
factor = 5
elsizes = eqdia/factor
elsizes = np.full(elsizes.shape, 0.3)

for i, g in enumerate(mp.geoms, start=0):
    coords = np.vstack((g.exterior.coords.xy[0][:-1],
                        g.exterior.coords.xy[1][:-1])).T
    model.add_polygon(coords, mesh_size=elsizes[i])

dir(model)

model.synchronize()
dim=2
elshape = 'quad'
elorder = 1
algorithm = 2



mesh = geometry.generate_mesh(dim=dim, order=elorder,
                              algorithm=algorithm)



mesh.write(r"D:\export_folder\sunil1.vtk")


grid = pv.read(r"D:\export_folder\sunil1.vtk")
pv.global_theme.background='maroon'
plotter = pv.Plotter(window_size = (1400, 800))
_ = plotter.add_axes_at_origin(x_color = 'red', y_color = 'green', z_color = 'blue',
                                line_width = 1,
                               xlabel = 'x', ylabel = 'y', zlabel = 'z',
                                labels_off = True)
_ = plotter.add_points(np.array([0,0,0]),
                        render_points_as_spheres = True,
                        point_size = 25)
_ = plotter.add_points(grid.points,
                        render_points_as_spheres = True,
                        point_size = 2)
#_ = plotter.add_bounding_box(line_width=2, color='black')
_ = plotter.add_mesh(grid,
                      show_edges = True,
                      edge_color = 'black',
                      line_width = 2,
                      render_points_as_spheres = True,
                      point_size = 10,
                      style = 'surface', opacity=0.5)
plotter.view_xy()
#plotter.set_viewup([0, 0, 1])
plotter.add_axes(interactive=True)
plotter.camera.zoom(1.5)
plotter.show()

angle = 180.  # Example: rotate by 45 degrees
plotter.camera.elevation += angle


plt.imshow(gstslice.pxtal[1].lgi)









import pyvista as pv

clr_background = 'white'
wnd_size = (800, 800)
triad_show = True
triad_par = {'line_width': 4,
             'ambient': 0.0,
             'x_color': 'red',
             'y_color': 'green',
             'z_color': 'blue',
             'cone_radius': 0.3,
             'shaft_length': 0.9,
             'tip_length': 0.1,
             'xlabel': 'X Axis',
             'ylabel': 'Y Axis',
             'zlabel': 'Z Axis',
             'label_size': (0.08, 0.08)
             }
xtal_vert_point_size = 10
el_edge_width = 2
el_point_size = 2
mesh_opacity = 1.0
mesh_display_style = 'wireframe'
mesh_qual_fields = None
mesh_qual_field_vis_par = {'mesh_quality_measure': 'Quality Measure name',
                           'cpos': 'xy',
                           'scalars': 'CellQuality',
                           'show_edges': False,
                           'cmap': 'jet',
                           'clim': [-1, 1],
                           'below_color': 'white',
                           'above_color': 'black',
                           }
stat_distr_data = None
stat_distr_data_par = {'mesh_quality_measure': 'Quality Measure name',
                       'density': True,
                       'figsize': (1.6, 1.6),
                       'dpi': 200,
                       'nbins': 100,
                       'xlabel_text': 'use_from_data',
                       'ylabel_text': 'Count', # stat_distr_data_par['ylabel_text']
                       'xlabel_fontsize': 8,
                       'ylabel_fontsize': 8,
                       'hist_xlim': [1, 2.5],
                       'hist_ylim': [0, 100]
                       }
throw_hist = False

pv.set_plot_theme('document')
#....................
if rff:
    grid = pv.read(f'{filename}')
if rfia:
    grid = grid
#....................
pv.global_theme.background = clr_background
#....................
plotter = pv.Plotter(window_size = wnd_size,
                     border = True,
                     line_smoothing = True,
                     lighting = 'three_lights'
                     )
#....................
if triad_show:
    marker = pv.create_axes_marker(line_width = triad_par['line_width'],
                                   ambient = triad_par['ambient'],
                                   x_color = triad_par['x_color'],
                                   y_color = triad_par['y_color'],
                                   z_color = triad_par['z_color'],
                                   cone_radius = triad_par['cone_radius'],
                                   shaft_length = triad_par['shaft_length'],
                                   tip_length = triad_par['tip_length'],
                                   xlabel = triad_par['xlabel'],
                                   ylabel = triad_par['ylabel'],
                                   zlabel = triad_par['zlabel'],
                                   label_size = triad_par['label_size'],
                                   )
    #....................
_ = plotter.add_actor(marker)
#....................
# import numpy as np
#_ = plotter.add_points(np.array([0,0,0]),
#                        render_points_as_spheres = True,
#                        point_size = 20, color = 'cyan')
#....................
# for xtal in xtals:
#    _ = plotter.add_lines(add each of the pxtal edge objects)
#....................
_ = plotter.add_points(grid.points,
                       render_points_as_spheres = False,
                       point_size = el_point_size,
                       color = 'cyan'
                       )
#....................
#_ = plotter.add_bounding_box(line_width=2, color='black')
#....................
_ = plotter.add_mesh(grid,
                     #color = 'black',
                     show_edges = True,
                     edge_color = 'black',
                     line_width = el_edge_width,
                     render_points_as_spheres = True,
                     point_size = xtal_vert_point_size,
                     style = mesh_display_style,
                     opacity = mesh_opacity,
                     )
#....................
plotter.view_xy()
plotter.camera.zoom(1.0)
plotter.show()
