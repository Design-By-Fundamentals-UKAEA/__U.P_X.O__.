# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 01:53:05 2024

@author: Dr. Sunil Anandatheertha
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
# ---------------------------
gstslice.pxtal[1].set_geom()

from upxo.pxtal.geometrification import polygonised_grain_structure as pgs

geom = pgs(gstslice.pxtal[1].lgi, gstslice.pxtal[1].gid, gstslice.pxtal[1].neigh_gid)

geom.n
gstslice.pxtal[1].n
geom.polygonize()
geom.raster_img_polygonisation_results
geom.make_polygonal_grains_raw()
geom.polygons_raw_exteriors
geom.polygons_raw_holes
geom.polygons
geom.expol
geom.holes_exist
geom.hlpol

geom.set_polygons()
geom.polygons

id(geom.polygons[0])
id(geom.polygons_raw_exteriors[1])

geom.make_gsmp()
geom.gsmp
geom.plot_gsmp(raw=True, overlay_on_lgi=True, xoffset=0.5, yoffset=0.5)
geom.find_neighbors()
geom.neigh_gid
gstslice.pxtal[1].neigh_gid


geom.val_neigh_gid(gstslice.pxtal[1].neigh_gid)
geom.make_neighpols()
geom.neigh_pols


geom.neigh_pols[1].intersect(geom.neigh_pols[1][0])


geom.extract_gbsegments_raw()
geom.gbsegments_raw


geom.extract_gbsegments_mls_raw()
geom.gbsegments_mls[1]

geom.neigh_gid[24]
geom.gbsegments_mls[24]
geom.gbsegments_mls[20]


np.array(geom.raster_img_polygonisation_results[20][0][0]['coordinates'][0]).mean(axis=0)
geom.gbsegments_mls[20]

geom.set_grain_centroids_raw()


geom.neigh_gid[24]




plot_point_numbers = True
for pid in [11]:
    coords = geom.raster_img_polygonisation_results[pid-1][0][0]['coordinates'][0]
    fig, ax = plt.subplots()
    centroids = []
    for i, segment in enumerate(geom.gbsegments_mls[pid]):
        xy = extract_coordinates_multiline(segment)
        print(i, len(xy))
        ax.plot(xy[:, 0], xy[:, 1], '-', marker='.', linewidth=2)
        cen = xy[:, 0].mean(), xy[:, 1].mean()
        ax.text(cen[0], cen[1], str(i), fontweight='bold', color='black',
                fontsize=12)
        centroids.append(cen)
    C = np.mean(centroids, axis=0)
    plt.text(C[0], C[1], pid, fontsize=12, fontweight='bold', color = 'red')
    if plot_point_numbers:
        for pn, c in enumerate(coords, start=0):
            ax.text(c[0], c[1], pn, color='blue')


for pid in [1, 9, 5, 15, 11]:
    fig, ax = plt.subplots()
    centroids = []
    for i, segment in enumerate(geom.gbsegments_mls[pid]):
        xy = extract_coordinates_multiline(segment)
        print(i, len(xy))
        ax.plot(xy[:, 0], xy[:, 1], '-', linewidth=2)
        cen = xy[:, 0].mean(), xy[:, 1].mean()
        ax.text(cen[0], cen[1], str(i), fontweight='bold', color='black',
                fontsize=12)
        centroids.append(cen)
    C = np.mean(centroids, axis=0)
    plt.text(C[0], C[1], pid, fontsize=12, fontweight='bold', color = 'red')


pid = 9
i = 3
coords = geom.raster_img_polygonisation_results[pid-1][0][0]['coordinates'][0]
segment = geom.gbsegments_mls[pid][i]
xy = extract_coordinates_multiline(segment)
plt.plot(xy[:, 0], xy[:, 1], '-', linewidth=2)
cen = xy[:, 0].mean(), xy[:, 1].mean()
plt.text(cen[0], cen[1], f'{pid},{i}', fontweight='bold', color='black', fontsize=12)

plt.axis('off')

len(xy)
len(np.unique(xy, axis=1))

pid = 9
i = 0
# All grain boundary coordinates
allgbcoords = geom.raster_img_polygonisation_results[pid-1][0][0]['coordinates'][0]
allgbcoords = np.array(allgbcoords)
# multi line string coordinates
getcoords = extract_coordinates_multiline
mlscoords = [getcoords(mls) for mls in geom.gbsegments_mls[pid]]

plt.plot(allgbcoords[:, 0], allgbcoords[:, 1], '-k.')
for mlscoord in mlscoords:
    plt.plot(mlscoord[:, 0], mlscoord[:, 1], 'o')


pid = 9

# All grain boundary points of pid grain
gbp = np.array(geom.raster_img_polygonisation_results[pid-1][0][0]['coordinates'][0])
# All grain boiundary multilinestring segments of pid grain
mlsxy = extract_coordinates_multiline(geom.gbsegments_mls[pid][0])
fpmls = mlsxy[0]  # Firsty point
lpmls = mlsxy[-1]  # Last point


pid = 11
extract_coordinates_multiline(geom.gbsegments_mls[pid][0])
extract_coordinates_multiline(geom.gbsegments_mls[pid][1])
extract_coordinates_multiline(geom.gbsegments_mls[pid][2])
extract_coordinates_multiline(geom.gbsegments_mls[pid][3])
extract_coordinates_multiline(geom.gbsegments_mls[pid][4])








def extract_coordinates_multiline(multiline):
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

for pid in geom.gid:
    fig, ax = plt.subplots()
    centroids = []
    for i, segment in enumerate(geom.gbsegments_mls[pid]):
        xy = extract_coordinates_multiline(segment)
        print(i, len(xy))
        ax.plot(xy[:, 0], xy[:, 1], '-', linewidth=2)
        cen = xy[:, 0].mean(), xy[:, 1].mean()
        ax.text(cen[0], cen[1], str(i), fontweight='bold', color='black',
                fontsize=12)
        centroids.append(cen)
    C = np.mean(centroids, axis=0)
    plt.plot(C[0], C[1], 'ks')

xy
geom.raster_img_polygonisation_results[27-1]















import math

def sort_neighbors_clockwise(neighbors, centroid):
    """Sorts neighboring grains clockwise around a centroid."""

    translated_neighbors = [(x - centroid[0], y - centroid[1]) for x, y in neighbors]
    polar_neighbors = [(math.hypot(x, y), math.atan2(y, x)) for x, y in translated_neighbors]

    sorted_neighbors = sorted(polar_neighbors, key=lambda x: x[1])  # Sort by angle (theta)

    # Optional: Adjust angle range to 0 to 2pi
    sorted_neighbors = [(r, t if t >= 0 else t + 2 * math.pi) for r, t in sorted_neighbors]

    #return to cartesian
    sorted_neighbors = [(r*math.cos(t) + centroid[0],r*math.sin(t)+centroid[1]) for r, t in sorted_neighbors]

    return sorted_neighbors




from shapely.geometry import MultiPolygon, Point
from collections import Counter

def extract_internal_junctions(multipolygon, neigh_giod):
    """Extracts junction points where 3+ internal polygons meet in a MultiPolygon,
       using neighbor information from a dictionary."""

    # 1. Break down into polygons
    polygons = list(multipolygon.geoms)
    num_grains = len(polygons)

    # 2. Extract vertices & 3. Count occurrences using neighbor information
    vertex_counts = Counter()
    for grain_id, neighbors in neigh_giod.items():
        for neighbor_id in neighbors:
            if neighbor_id > grain_id:  # Avoid double-counting (symmetric relationship)
                # Assuming you have a function to get the centroid coordinates of a grain
                vertex = get_centroid_coordinates(grain_id)
                vertex_counts[vertex] += 1  # Increment count for this junction point

    # 4. Identify junctions (where 3 or more polygons meet)
    junction_points = [point for point, count in vertex_counts.items() if count >= 3]

    # 5. Filter by interior polygons
    interior_junctions = []
    for point in junction_points:
        is_interior = True
        for i in range(num_grains):  # Iterate over all grains (polygons)
            if polygons[i].contains(Point(point)) and i not in neigh_giod:  # Check if point is inside and grain is on the boundary
                is_interior = False
                break  # If on boundary, not an interior junction
        if is_interior:
            interior_junctions.append(point)

    return interior_junctions

# --- Example Usage ---
# (Assume 'my_multipolygon' is your MultiPolygon object)
neigh_giod = {
    1: [12, 14], 2: [18, 3, 14],  # ... your neighbor dictionary data ...
}
junctions = extract_internal_junctions(my_multipolygon, neigh_giod)
print(junctions)
