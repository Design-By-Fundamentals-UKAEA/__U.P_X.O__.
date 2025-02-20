# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 00:06:29 2024

@author: rg5749
"""
import cv2
import numpy as np
import gmsh
import pyvista as pv
from copy import deepcopy
from shapely.geometry import Polygon, MultiPoint
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

geom.set_grain_centroids_raw()

geom.val_neigh_gid(gstslice.pxtal[1].neigh_gid)
geom.make_neighpols()
geom.neigh_pols


gid = 1
A = np.vstack(geom.polygons[gid-1].exterior.coords.xy).T
B = np.vstack(geom.neigh_pols[gid][0].exterior.coords.xy).T

from upxo._sup.data_ops import find_common_coordinates

find_common_coordinates(A, B)


np.array(geom.raster_img_polygonisation_results[gid-1][0][0]['coordinates'][0])


relloc_flag = [is_boundary_polygon(geom.gsmp['raw'], pol) for pol in geom.polygons]



from shapely.geometry import Point
def is_boundary_polygon(multipolygon, polygon):
    """
    Checks if a given polygon lies on the boundary of a MultiPolygon.

    Args:
        multipolygon: A Shapely MultiPolygon object.
        polygon: A Shapely Polygon object to check.

    Returns:
        True if the polygon is on the boundary, False otherwise.
    """
    for coord in polygon.exterior.coords:
        # Iterate over each polygon in the MultiPolygon
        for poly in multipolygon.geoms:
            if poly.exterior.contains(Point(coord)):  # Check against each polygon's exterior
                return True

    return False

# ====================================================
def assemble_all_gbpoints(self):
    for gid in geom.gid:
        if gid == 1:
            gbpoints = np.array(geom.raster_img_polygonisation_results[gid-1][0][0]['coordinates'][0][:-1])-0.5
        else:
            new_gbpoints = list(np.array(geom.raster_img_polygonisation_results[gid-1][0][0]['coordinates'][0][:-1])-0.5)
            gbpoints = np.array(list(gbpoints) + new_gbpoints)

gbpoints = np.unique(gbpoints, axis=1)
len(gbpoints)

xmin, xmax = gbpoints[:, 0].min(), gbpoints[:, 0].max()
ymin, ymax = gbpoints[:, 1].min(), gbpoints[:, 1].max()

plt.imshow(geom.lgi)
plt.plot(gbpoints[:, 0], gbpoints[:, 1], 'k.')



# #########################################################################
'''LABEL GRAINS AS PER THEIR LOCATION IN THR GRAIN STRUCTUER'''
border_grain = [False for gid in geom.gid]
internal_grain = [False for gid in geom.gid]
corner_grain = [False for gid in geom.gid]
bl_grain = [False for gid in geom.gid]
tl_grain = [False for gid in geom.gid]
br_grain = [False for gid in geom.gid]
tr_grain = [False for gid in geom.gid]
for gid in geom.gid:
    coords = np.array(geom.raster_img_polygonisation_results[gid-1][0][0]['coordinates'][0])-0.5
    c1 = xmin in coords[:, 0]
    c2 = xmax in coords[:, 0]
    c3 = ymin in coords[:, 1]
    c4 = ymax in coords[:, 1]
    if any((c1, c2, c3, c4)):
        border_grain[gid-1] = True
        if any((coords[:, 0] == xmin) & (coords[:, 1] == ymin)):
            corner_grain[gid-1] = True
            bl_grain[gid-1] = True  # Bottom left
        elif any((coords[:, 0] == xmin) & (coords[:, 1] == ymax)):
            corner_grain[gid-1] = True
            tl_grain[gid-1] = True  # Top left
        elif any((coords[:, 0] == xmax) & (coords[:, 1] == ymin)):
            corner_grain[gid-1] = True
            br_grain[gid-1] = True  # Bottom right
        elif any((coords[:, 0] == xmax) & (coords[:, 1] == ymax)):
            corner_grain[gid-1] = True
            tr_grain[gid-1] = True  # Top right
    else:
        internal_grain[gid-1] = True
# #########################################################################
# Approach 1
gbpon = [0 for gbp in gbpoints]  # Order number

for i, gbp in enumerate(gbpoints, start=1):
    if not i % 500:
        print(f'Processing gb point {i}/{len(gbpoints)}.')
    for gid in geom.gid:
        coords = np.array(geom.raster_img_polygonisation_results[gid-1][0][0]['coordinates'][0][:-1])-0.5
        if any((coords[:, 0] == gbp[0]) & (coords[:, 1] == gbp[1])):
            gbpon[i-1] += 1
        #if np.any(np.all((coords == gbp), axis=1)):
        #    gbpon[i-1] += 1

gid = 1
coords = np.array(geom.raster_img_polygonisation_results[gid-1][0][0]['coordinates'][0][:-1])-0.5


np.array(geom.raster_img_polygonisation_results[12-1][0][0]['coordinates'][0])
np.array(geom.raster_img_polygonisation_results[19-1][0][0]['coordinates'][0])
# #########################################################################
# Approach 2
from scipy.spatial import cKDTree
gbp_trees = []
for gid in geom.gid:
    gbp_trees.append(cKDTree(np.array(geom.raster_img_polygonisation_results[gid-1][0][0]['coordinates'][0][:-1])-0.5))

presence = np.zeros((np.unique(gbpoints, axis=1).shape[0], geom.n))
for i, tree in enumerate(gbp_trees, 0):
    tree = gbp_trees[i]
    distances, indices = tree.query(gbpoints, k=1)
    zero_distance_indices = np.where(distances == 0)[0]
    presence[:, i][zero_distance_indices] += 1

presence.sum(axis=1)
# #########################################################################
# Approach 3
geom.polygons
geom.neigh_pols
geom.neigh_gid

xy = []
for gid1 in geom.gid:
    for gid2 in geom.gid:
        if gid2 <= gid1:
            continue
        # print(gid1, gid2)
        intersec = geom.polygons[gid1-1].intersection(geom.polygons[gid2-1])
        # intersec = geom.polygons[gid1-1].boundary.intersection(geom.polygons[gid2-1].boundary)

        # print(type(intersec))
        # print(isinstance(intersec, MultiLineString))
        # print(isinstance(intersec, LineString))
        if isinstance(intersec, LineString):
            print(intersec.coords.xy)
            # xy.append([intersec.coords.xy[0][0], intersec.coords.xy[1][0]])
            pass
        elif isinstance(intersec, MultiLineString):
            # print(dir(intersec))
            print(intersec.boundary)#.coords.xy)
            if isinstance(intersec.boundary, MultiPoint):
                for pnt in intersec.boundary.geoms:
                    xy.append([pnt.x, pnt.y])
            if isinstance(intersec.boundary, tuple):
                xy.append([intersec.boundary[0][0], intersec.boundary[1][0]])
                xy.append([intersec.boundary[0][1], intersec.boundary[1][1]])
            #for line in intersec.geoms:
            #    xy.append([line.coords.xy[0][0], line.coords.xy[1][0]])
            #    xy.append([line.coords.xy[0][1], line.coords.xy[1][1]])
        elif isinstance(intersec, Point):
            xy.append([intersec.x, intersec.y])
xy = np.unique(xy, axis = 0)


plt.imshow(geom.lgi)
plt.plot(xy[:, 0]-0.5, xy[:, 1]-0.5, 'ko', markersize=10, alpha=0.8)
plt.plot(gbpoints[:, 0], gbpoints[:, 1], 'r.')
# #########################################################################
plt.imshow(geom.lgi)
plt.plot(gbpoints[:, 0], gbpoints[:, 1], 'k.')
for bg, cen in zip(border_grain, geom.centroids_raw):
    text = 'BG' if bg else None
    plt.text(cen[0], cen[1], text, fontweight='bold')


mask = deepcopy(geom.lgi)
for gid in geom.gid:
    if not tr_grain[gid-1]:
        mask[mask == gid] = 0
plt.imshow(mask)


mask = deepcopy(geom.lgi)
for gid in geom.gid:
    plt.subplots()
    plt.imshow(mask == gid)
