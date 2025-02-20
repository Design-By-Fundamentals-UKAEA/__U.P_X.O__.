# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 23:25:49 2024

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
from shapely.geometry import Point
from shapely.geometry import MultiPolygon, Point
from scipy.spatial import cKDTree
from upxo.geoEntities.mulsline2d import ring2d
from upxo._sup.data_ops import find_common_coordinates
from shapely.geometry import LineString, MultiLineString
# ---------------------------
AUTO_PLOT_EVERYTHING = False
# ---------------------------
pxt = mcgs()
pxt.simulate()
pxt.detect_grains()
tslice = 18
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
geom.set_polygons()
geom.make_gsmp()
geom.find_neighbors()
geom.set_grain_centroids_raw()
geom.plot_gsmp(raw=True, overlay_on_lgi=True, xoffset=0.5, yoffset=0.5)
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
for gid in geom.gid:
    if gid == 1:
        gbpoints = np.array(geom.raster_img_polygonisation_results[gid-1][0][0]['coordinates'][0][:-1])-0.5
    else:
        new_gbpoints = list(np.array(geom.raster_img_polygonisation_results[gid-1][0][0]['coordinates'][0][:-1])-0.5)
        gbpoints = np.array(list(gbpoints) + new_gbpoints)


gbpoints = np.unique(gbpoints, axis=1)

xmin, xmax = gbpoints[:, 0].min(), gbpoints[:, 0].max()
ymin, ymax = gbpoints[:, 1].min(), gbpoints[:, 1].max()

# #########################################################################
'''LABEL GRAINS AS PER THEIR LOCATION IN THR GRAIN STRUCTUER'''
border_grain_flags = [False for gid in geom.gid]
internal_grain_flags = [False for gid in geom.gid]
corner_grain_flags = [False for gid in geom.gid]
bl_grain_flags = [False for gid in geom.gid]
tl_grain_flags = [False for gid in geom.gid]
br_grain_flags = [False for gid in geom.gid]
tr_grain_flags = [False for gid in geom.gid]
for gid in geom.gid:
    coords = np.array(geom.raster_img_polygonisation_results[gid-1][0][0]['coordinates'][0])-0.5
    c1 = xmin in coords[:, 0]
    c2 = xmax in coords[:, 0]
    c3 = ymin in coords[:, 1]
    c4 = ymax in coords[:, 1]
    if any((c1, c2, c3, c4)):
        border_grain_flags[gid-1] = True
        if any((coords[:, 0] == xmin) & (coords[:, 1] == ymin)):
            corner_grain_flags[gid-1] = True
            bl_grain_flags[gid-1] = True  # Bottom left
        elif any((coords[:, 0] == xmin) & (coords[:, 1] == ymax)):
            corner_grain_flags[gid-1] = True
            tl_grain_flags[gid-1] = True  # Top left
        elif any((coords[:, 0] == xmax) & (coords[:, 1] == ymin)):
            corner_grain_flags[gid-1] = True
            br_grain_flags[gid-1] = True  # Bottom right
        elif any((coords[:, 0] == xmax) & (coords[:, 1] == ymax)):
            corner_grain_flags[gid-1] = True
            tr_grain_flags[gid-1] = True  # Top right
    else:
        internal_grain_flags[gid-1] = True

border_grain_gids = list(np.argwhere(border_grain_flags).squeeze()+1)
internal_grain_gids = list(np.argwhere(internal_grain_flags).squeeze()+1)
corner_grain_gids = list(np.argwhere(corner_grain_flags).squeeze()+1)
bl_grain_gids = np.argwhere(bl_grain_flags).squeeze()+1
tl_grain_gids = np.argwhere(tl_grain_flags).squeeze()+1
br_grain_gids = np.argwhere(br_grain_flags).squeeze()+1
tr_grain_gids = np.argwhere(tr_grain_flags).squeeze()+1
# Identify grains which are pure edge grains. These grains are not on corner.
l_grain_flags = [False for gid in geom.gid]
r_grain_flags = [False for gid in geom.gid]
b_grain_flags = [False for gid in geom.gid]
t_grain_flags = [False for gid in geom.gid]
for gid in geom.gid:
    coords = np.array(geom.raster_img_polygonisation_results[gid-1][0][0]['coordinates'][0])-0.5
    if any(coords[:, 0] == xmin):
           l_grain_flags[gid-1] = True
    if any(coords[:, 0] == xmax):
           r_grain_flags[gid-1] = True
    if any(coords[:, 1] == ymin):
           b_grain_flags[gid-1] = True
    if any(coords[:, 1] == ymax):
           t_grain_flags[gid-1] = True
l_grain_gids = list(np.argwhere(l_grain_flags).squeeze()+1)
r_grain_gids = list(np.argwhere(r_grain_flags).squeeze()+1)
b_grain_gids = list(np.argwhere(b_grain_flags).squeeze()+1)
t_grain_gids = list(np.argwhere(t_grain_flags).squeeze()+1)

pl_grain_gids = list(set(l_grain_gids) - set([bl_grain_gids]) - set([tl_grain_gids]))
pr_grain_gids = list(set(r_grain_gids) - set([br_grain_gids]) - set([tr_grain_gids]))
pb_grain_gids = list(set(b_grain_gids) - set([bl_grain_gids]) - set([br_grain_gids]))
pt_grain_gids = list(set(t_grain_gids) - set([tl_grain_gids]) - set([tr_grain_gids]))

grain_loc_ids = {'internal': internal_grain_gids,
                 'boundary': border_grain_gids,
                 'left': l_grain_gids,
                 'bottom': b_grain_gids,
                 'right': r_grain_gids,
                 'top': t_grain_gids,
                 'pure_left': pl_grain_gids,
                 'pure_bottom': pb_grain_gids,
                 'pure_right': pr_grain_gids,
                 'pure_top': pt_grain_gids,
                 'corner': corner_grain_gids,
                 'bottom_left_corner': bl_grain_gids,
                 'bottom_right_corner': br_grain_gids,
                 'top_right_corner': tr_grain_gids,
                 'top_left_corner': tl_grain_gids,
                 }
# #########################################################################
# #########################################################################
# Approach 3
xy = []
for gid1 in geom.gid:
    gid1_xy = []
    if gid1 % 100 == 0:
        print(f'Extracting Junction points {np.round(gid1*100/geom.n, 2)} % complete.')
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
            # print(intersec.coords.xy)
            xy.append([intersec.coords.xy[0][0], intersec.coords.xy[1][0]])
            pass
        elif isinstance(intersec, MultiLineString):
            # print(dir(intersec))
            # print(intersec.boundary)#.coords.xy)
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
xy = np.unique(xy, axis = 0) - 0.5
JNP = deepcopy(xy)

# #########################################################################
# Check if all coordinates are closed and forms a ring.
postpolcoords = geom.raster_img_polygonisation_results
A = []
for gid in geom.gid:
    A.append(np.array(postpolcoords[gid-1][0][0]['coordinates'][0][0])-np.array(postpolcoords[gid-1][0][0]['coordinates'][0][-1]))
A = np.array(A)
all(A[:, 0] == A[:, 1])  # True means all coords form a ring.
# #########################################################################
# Assemble all grain boundary points
postpolcoords = geom.raster_img_polygonisation_results
GBP = []
for gid in geom.gid:
    GBP.extend(np.array(postpolcoords[gid-1][0][0]['coordinates'][0][:-1]).tolist())
GBP = np.array(GBP) - 0.5
# #########################################################################
# Calculate the bounds of the po;lygonized grain structure.
xmin = GBP[:, 0].min()
xmax = GBP[:, 0].max()
ymin = GBP[:, 1].min()
ymax = GBP[:, 1].max()
# #########################################################################
GBPx = GBP[:, 0]
GBPy = GBP[:, 1]
GBP_left = GBP[GBPx == xmin]
GBP_right = GBP[GBPx == xmax]
GBP_bot = GBP[GBPy == ymin]
GBP_top = GBP[GBPy == ymax]
GBP_at_boundary = np.unique(np.vstack((GBP_left, GBP_bot, GBP_right, GBP_top)), axis=0)
# #########################################################################
# Add missing points to the junction points.
GBP_at_boundary_flag_to_remove = [False for _ in GBP_at_boundary]
for i, gbpboundary in enumerate(GBP_at_boundary):
    x, y = gbpboundary
    if not any((xy[:, 0] == x) & (xy[:, 1] == y)):
        GBP_at_boundary_flag_to_remove[i] = True

GBP_at_boundary[GBP_at_boundary_flag_to_remove]
JNP = np.append(JNP, GBP_at_boundary, axis=0)
JNP = np.unique(JNP, axis=0)
# #########################################################################
if AUTO_PLOT_EVERYTHING:
    geom.plot_gsmp(raw=True, overlay_on_lgi=False, xoffset=0.5, yoffset=0.5)
    plt.imshow(geom.lgi)
    plt.plot(JNP[:, 0], JNP[:, 1], 'ro', markersize=5, alpha=1.0)
    for i, jnp in enumerate(JNP):
        plt.text(jnp[0], jnp[1], i, color = 'black')
# #########################################################################
from upxo.geoEntities.point2d import Point2d
from upxo.geoEntities.mulpoint2d import MPoint2d
from shapely.geometry import Point as ShPoint2d
"""Build coordinates of all junction points"""
jnp_all_coords = JNP
"""Build Point2d of all junction points"""
jnp_all_upxo = np.array([Point2d(jnp[0], jnp[1]) for jnp in JNP])
"""BUild shapely point objects of all junction points"""
jnp_all_shapely = [ShPoint2d(jnp[0], jnp[1]) for jnp in JNP]
jnp_all_upxo_mp = MPoint2d.from_xy(jnp_all_coords.T)
jnp_all_upxo_mp = MPoint2d.from_upxo_points2d(jnp_all_upxo, zloc=0.0)
# -----------------------------------------
"""Build coordinates of all grain boundary points. Global."""
gbp_all_coords = []
len(gbp_all_coords)
for gid in geom.gid:
    _ = np.array(postpolcoords[gid-1][0][0]['coordinates'][0][:-1])-0.5
    gbp_all_coords.extend(_.tolist())
gbp_all_coords = np.unique(gbp_all_coords, axis=0)
"""Build UPXO points from gbp_all_coords data """
gbp_all_upxo = np.array([Point2d(gbp[0], gbp[1]) for gbp in gbp_all_coords])
gbp_all_shapely = [ShPoint2d(gbp[0], gbp[1]) for gbp in gbp_all_coords]
gbp_all_upxo_mp = MPoint2d.from_upxo_points2d(gbp_all_upxo, zloc=0.0)
# -----------------------------------------
"""Build coordinates of all points of grian boundary points, grain wese."""
gbp_grain_wise_coords = {}
for gid in geom.gid:
    _ = np.array(postpolcoords[gid-1][0][0]['coordinates'][0][:-1])-0.5
    gbp_grain_wise_coords[gid] = _
# -----------------------------------------
"""
Build indices of jnp for every grain. Indices will be from jnp_all_coords.
These indices relate to:
    * jnp_all_coords
    * jnp_all_upxo
    * jnp_all_shapely
    * jnp_all_upxo_mp.points
    * jnp_all_upxo_mp.coortds
"""
from shapely import affinity
jnp_grain_wise_indices = {}
for gid in geom.gid:
    jnp_grain_wise_indices[gid] = []
    pol_shapely = affinity.translate(geom.polygons[gid-1], xoff=-0.5, yoff=-0.5)
    for i, jnp_shapely in enumerate(jnp_all_shapely, start=0):
        if pol_shapely.touches(jnp_shapely):
            jnp_grain_wise_indices[gid].append(i)
'''
Use
---
Data access:
    jnp_all_coords[jnp_grain_wise_indices[gid]]
    jnp_all_upxo[jnp_grain_wise_indices[gid]]
    jnp_all_upxo_mp.points[jnp_grain_wise_indices[gid]]

Verification
------------
gid = 10
plt.imshow(geom.lgi)
coord = jnp_all_coords[jnp_grain_wise_indices[gid]]
plt.plot(coord[:, 0], coord[:, 1], 'ko')

Note
----
mid1 = id(jnp_all_upxo[jnp_grain_wise_indices[gid]][0])
mid2 = id(jnp_all_upxo_mp.points[jnp_grain_wise_indices[gid]][0])
mid1 == mid2

Note@dev
--------
gid = 1
jnp_all_coords[jnp_grain_wise_indices[gid]]
jnp_all_upxo[jnp_grain_wise_indices[gid]]
gbp_grain_wise_coords[gid]
'''
# -----------------------------------------
"""
Build indices of gbp for every grain. Indices will be from gbp_all_coords.
These indices relate to:
    * gbp_all_coords
    * gbp_all_upxo
    * gbp_all_shapely
    * gbp_all_upxo_mp.points
    * gbp_all_upxo_mp.coortds

id(gbp_all_upxo[0])
id(gbp_all_upxo_mp.points[0])

"""
gbp_grain_wise_indices = {}
for gid in geom.gid:
    gbp_grain_wise_indices[gid] = []
    pol_shapely = affinity.translate(geom.polygons[gid-1], xoff=-0.5, yoff=-0.5)
    for i, gbp_shapely in enumerate(gbp_all_shapely, start=0):
        if pol_shapely.touches(gbp_shapely):
            gbp_grain_wise_indices[gid].append(i)
# ---------------------------------------------------------
def is_a_in_b(a, b):
    return any((b[:, 0] == a[0]) & (b[:, 1] == a[1]))

def insert_coorda_into_coordarrayb(coorda, coordarrayb):
    pass

def find_coorda_loc_in_coords_arrayb(a, b):
    # find_coorda_loc_in_coords_arrayb(neigh_points[1], sinkarray)
    return np.argwhere((b[:, 0] == a[0]) & (b[:, 1] == a[1]))[0][0]

# #############################################################################
# #############################################################################
# #############################################################################
# #############################################################################
findloc = find_coorda_loc_in_coords_arrayb

gbp_grain_wise_indices = {gid: None for gid in geom.gid}
gbp_grain_wise_points = {gid: None for gid in geom.gid}
for gid in geom.gid:
    plist = gbp_grain_wise_coords[gid]
    gbp_grain_wise_indices[gid] = [findloc(p, gbp_all_coords) for p in plist]
    gbp_grain_wise_points[gid] = gbp_all_upxo[gbp_grain_wise_indices[gid]]

from upxo.geoEntities.mulsline2d import MSline2d
'''We will start by first creating UPXO line objects for grain boundaries.'''
# Build lines from gbp_grain_wise_coordinates.
gbmullines_grain_wise = {gid: [] for gid in geom.gid}
for gid in geom.gid:
    nodes = gbp_grain_wise_points[gid].tolist()
    gbmullines_grain_wise[gid] = MSline2d.by_nodes(nodes, close=False)
    # ax = gbmullines_grain_wise[gid].plot()

for gid in geom.gid:
    gbmullines_grain_wise[gid].close(reclose=False)

for gid in geom.gid:
    # Build the jnp coordinates which are not there in gbp array.
    jnp_indices_toinsert = []
    for i, jnp in enumerate(jnp_all_coords[jnp_grain_wise_indices[gid]]):
        if not is_a_in_b(jnp, gbp_grain_wise_coords[gid]):
            '''This means jnp should be inserted into gbp_grain_wise_coords[gid]
            at the right place.'''
            jnp_indices_toinsert.append(i)
    jnps_to_insert = jnp_all_upxo[jnp_grain_wise_indices[gid]][jnp_indices_toinsert]
    # --------------------------
    #if len(jnps_to_insert) > 0:
    gbmullines_grain_wise[gid].add_nodes(jnps_to_insert)

if AUTO_PLOT_EVERYTHING:
    plt.figure()
    plt.imshow(geom.lgi)
    for gid in geom.gid:
        coord = gbmullines_grain_wise[gid].get_node_coords()
        plt.plot(coord[:, 0], coord[:, 1], '-k.')
        jnp = jnp_all_coords[jnp_grain_wise_indices[gid]]
        plt.plot(jnp[:, 0], jnp[:, 1], 'ro', mfc='c', alpha=0.5)


    for gid in geom.gid:
        plt.figure()
        coord = gbmullines_grain_wise[gid].get_node_coords()
        plt.plot(coord[:, 0], coord[:, 1], '-k.')
        jnp = jnp_all_coords[jnp_grain_wise_indices[gid]]
        plt.plot(jnp[:, 0], jnp[:, 1], 'ro', mfc='c', alpha=0.5)
        plt.plot(coord[:, 0][0], coord[:, 1][0], 'kx')
        plt.plot(coord[:, 0][-2], coord[:, 1][-2], 'bx')

    gid = 3
    coord = gbmullines_grain_wise[gid].get_node_coords()
    plt.plot(coord[:, 0], coord[:, 1], '-k.')
    for i, c in enumerate(coord[:-1,:], 0):
        plt.text(c[0]+0.15, c[1]+0.15, i)
    jnp = jnp_all_coords[jnp_grain_wise_indices[gid]]
    plt.plot(jnp[:, 0], jnp[:, 1], 'ro', mfc='c', alpha=0.5)
    for i, j in enumerate(jnp):
        plt.text(j[0]+0.15, j[1], i, color='red')
# #############################################################################
def arrange_junction_point_coords(gbcoords_thisgrain, junction_points_coord):
    # Create a dictionary to map coordinates to their indices in gbcoords_thisgrain
    coord_index_map = {tuple(coord): idx for idx, coord in enumerate(gbcoords_thisgrain)}
    # Sort junction_points_coord based on their indices in gbcoords_thisgrain
    sorted_junction_points = sorted(junction_points_coord, key=lambda x: coord_index_map[tuple(x)])
    return np.array(sorted_junction_points)

def arrange_junction_point_coords_new(gbcoords_thisgrain, junction_points_coord):
    # Create a dictionary to map coordinates to their indices in gbcoords_thisgrain
    coord_index_map = {tuple(coord): idx for idx, coord in enumerate(gbcoords_thisgrain)}
    # Generate a list of indices for sorting
    sorted_indices = sorted(range(len(junction_points_coord)), key=lambda i: coord_index_map[tuple(junction_points_coord[i])])
    # Sort junction_points_coord based on the sorted indices
    sorted_junction_points = np.array([junction_points_coord[i] for i in sorted_indices])
    return sorted_junction_points, sorted_indices

def arrange_junction_points_upxo(gbpoints_thisgrain, junction_points_points):
    # Create a dictionary to map coordinates to their indices in gbpoints_thisgrain
    coord_index_map = {(point.x, point.y): idx for idx, point in enumerate(gbpoints_thisgrain)}
    # Generate a list of indices for sorting
    sorted_indices = sorted(range(len(junction_points_points)), key=lambda i: coord_index_map[(junction_points_points[i].x, junction_points_points[i].y)])
    # Sort junction_points_points based on the sorted indices
    sorted_junction_points = [junction_points_points[i] for i in sorted_indices]
    return sorted_junction_points, sorted_indices


"""Check if nodes allways start with a junction point"""
find_coord_loc = find_coorda_loc_in_coords_arrayb
jnp_all_sorted_coords = {gid: None for gid in geom.gid}
jnp_all_sorted_upxo = {gid: None for gid in geom.gid}
for gid in geom.gid:
    gbpoints_thisgrain = gbmullines_grain_wise[gid].nodes
    gbcoords_thisgrain = gbmullines_grain_wise[gid].get_node_coords()
    junction_points_upxo = []
    junction_points_coord = []
    for i, jnpcoord in enumerate(jnp_all_coords):
        if is_a_in_b(jnpcoord, gbcoords_thisgrain):
            junction_points_coord.append(jnpcoord)
            junction_points_upxo.append(jnp_all_upxo[i])
    # junction_points_coord = arrange_junction_point_coords(gbcoords_thisgrain, junction_points_coord)
    junction_points_coord, _ = arrange_junction_point_coords_new(gbcoords_thisgrain, junction_points_coord)
    junction_points_upxo = list(np.array(junction_points_upxo)[_])
    # junction_points_upxo, _ = arrange_junction_points_upxo(gbpoints_thisgrain, junction_points_upxo)
    jnp_all_sorted_coords[gid] = junction_points_coord
    jnp_all_sorted_upxo[gid] = junction_points_upxo


if AUTO_PLOT_EVERYTHING:
    for gid in geom.gid:
        plt.figure()
        coord = gbmullines_grain_wise[gid].get_node_coords()
        plt.plot(coord[:, 0], coord[:, 1], '-k.')
        for i, c in enumerate(coord[:-1,:], 0):
            plt.text(c[0]+0.15, c[1]+0.15, i)

        jnp = jnp_all_sorted_coords[gid]
        plt.plot(jnp[:, 0], jnp[:, 1], 'ro', mfc='c', alpha=0.5)
        for i, j in enumerate(jnp):
            plt.text(j[0]+0.15, j[1], i, color='red')

for gid in geom.gid:
    roll_distance = find_coorda_loc_in_coords_arrayb(jnp_all_sorted_coords[gid][0],
                                                     gbmullines_grain_wise[gid].get_node_coords())
    gbmullines_grain_wise[gid].roll(roll_distance)


if AUTO_PLOT_EVERYTHING:
    for gid in geom.gid:
        plt.figure()
        coord = gbmullines_grain_wise[gid].get_node_coords()
        plt.plot(coord[:, 0], coord[:, 1], '-k.')
        for i, c in enumerate(coord[:-1,:], 0):
            plt.text(c[0]+0.15, c[1]+0.15, i)

        jnp = jnp_all_sorted_coords[gid]
        plt.plot(jnp[:, 0], jnp[:, 1], 'ro', mfc='c', alpha=0.5)
        for i, j in enumerate(jnp):
            plt.text(j[0]+0.15, j[1], i, color='red')

# ##################################################################
"""
Splice the grain boundary into grain boundary segments using jnp point data
"""
from upxo.geoEntities.mulsline2d import MSline2d
gbsegments = {gid: [] for gid in geom.gid}
for gid in geom.gid:
    if len(jnp_all_sorted_upxo[gid]) == 1:
        segment = gbmullines_grain_wise[gid].lines
    elif len(jnp_all_sorted_upxo[gid]) > 1:
        choplocs = []
        for point in jnp_all_sorted_upxo[gid]:
            location = point.eq_fast(gbmullines_grain_wise[gid].nodes[:-1], point_spec=2)
            choplocs.append(np.argwhere(location).squeeze().tolist())

        if choplocs[0] != 0:
            choplocs = [0] + choplocs
        if choplocs[-1] == len(gbmullines_grain_wise[gid].lines):
            choplocs = choplocs[:-1]

        ranges = []
        for i in range(1, len(choplocs)):
            ranges.append([choplocs[i-1], choplocs[i]])

        for r in ranges:
            lines = gbmullines_grain_wise[gid].lines[r[0]:r[1]]
            gbsegments[gid].append(MSline2d.from_lines(lines, close=False))
        rem_lines = gbmullines_grain_wise[gid].lines[r[1]:len(gbmullines_grain_wise[gid].lines)]
        gbsegments[gid].append(MSline2d.from_lines(rem_lines, close=False))

if AUTO_PLOT_EVERYTHING:
    fig, ax = plt.subplots()
    ax.imshow(geom.lgi)
    for gid in geom.gid:
        for gbseg in gbsegments[gid]:
            coords = gbseg.get_node_coords()
            ax.plot(coords[:, 0], coords[:, 1], '-o', ms=5)

        ax.plot(jnp_all_sorted_coords[gid][:, 0], jnp_all_sorted_coords[gid][:, 1], 'k*',
                ms = 7)
        centroid = gbmullines_grain_wise[gid].get_node_coords()[:-1].mean(axis=0)
        plt.text(centroid[0], centroid[1], gid, color='white', fontsize=12)
# ##################################################################
#for seg in gbsegments[11]:
#    plt.plot(seg.coords[:, 0], seg.coords[:, 1])
# ##################################################################
"""
Now, we will collect all grain boundary segments.
"""
GBSEG = []
gbseg_map_indices = {gid: [] for gid in geom.gid}
i = 0
for gid in geom.gid:
    for gbseg in gbsegments[gid]:
        GBSEG.append(gbseg)
        gbseg_map_indices[gid].append(i)
        i += 1

quality = []
for gid in geom.gid:
    quality.append(int(len(gbseg_map_indices[gid]) == len(geom.neigh_gid)))
quality = (geom.n-sum(quality))*100/geom.n
print(f'Grain boundary segmentation quality measure 1: {quality} %')

def create_pair_ids(neigh_gid):
    """Creates a dictionary mapping unique grain pairs to integer IDs.

    Args:
        neigh_gid (dict): A dictionary where keys are grain IDs and values are lists
                          of neighboring grain IDs.

    Returns:
        dict: A dictionary where keys are integer pair IDs and values are tuples of grain IDs.
    """

    pair_ids = {}
    pair_id = 1  # Start with pair ID 1

    for gid, neighbors in neigh_gid.items():
        for neighbor in neighbors:
            # Create a sorted tuple of the pair (ensures uniqueness)
            pair = tuple(sorted((gid, neighbor)))

            # Assign a new pair ID if not seen before
            if pair not in pair_ids:
                pair_ids[pair_id] = list(pair)
                pair_id += 1

    pair_ids_unique_lr = np.unique(np.array(list(pair_ids.values())), axis=0)
    pair_ids_unique_rl = np.flip(pair_ids_unique_lr, axis=1)

    return pair_ids, pair_ids_unique_lr, pair_ids_unique_rl
# -----------------------------------------------------------------
pair_ids, pair_ids_unique_lr, pair_ids_unique_rl = create_pair_ids(geom.neigh_gid)
# #############################################################################
geom.neigh_gid
gbs_mid_dict = {gid: [id(seg) for seg in gbsegments[gid]] for gid in geom.gid}
gbsegments[gid]

def get_random_gbpoint_between_jnpoints(gbcoords, jnpcoords):
    """
    jnpcoords = jnp_all_sorted_coords[46][0:2]
    gbcoords = gbmullines_grain_wise[46].get_node_coords()
    get_random_gbpoint_between_jnpoints(gbcoords, jnpcoords)
    """
    first = find_coorda_loc_in_coords_arrayb(jnpcoords[0], gbcoords)
    last = find_coorda_loc_in_coords_arrayb(jnpcoords[1], gbcoords)
    if last-first >= 2:
        return gbcoords[np.random.randint(first+1, last)]
    else:
        return None

def get_random_gbpoints_between_jnpoints(gbcoords, seg_ends):
    """
    jnpcoords = jnp_all_sorted_coords[46][0:2]
    gbcoords = gbmullines_grain_wise[46].get_node_coords()
    get_random_gbpoint_between_jnpoints(gbcoords, jnpcoords)
    """
    random_gb_points = {i: None for i in seg_ends.keys()}
    for key, seg_end in seg_ends.items():
        # -----------------------
        first = find_coorda_loc_in_coords_arrayb(seg_end[0], gbcoords)
        if sum(seg_end[1] - seg_ends[0][0]) == 0.0:
            last = len(gbcoords)
        else:
            last = find_coorda_loc_in_coords_arrayb(seg_end[1], gbcoords)

        if last-first >= 2:
            random_gb_points[key] = gbcoords[np.random.randint(first+1, last)]
        else:
            random_gb_points[key] = None
    return random_gb_points


#def extract_gbseg_ends(gid):
    seg_ends = {}
    for count in range(len(jnp_all_sorted_coords[gid])):
        seg_ends[count] = jnp_all_sorted_coords[gid][count:count+2]
    seg_ends[count] = np.vstack((seg_ends[count],
                                 gbmullines_grain_wise[gid].get_node_coords()[-1]))
#    return seg_ends

def neigh_connectivity_flags(unique_pair_ids,
                                         field_names=['gbseg']):
    return {tuple(neighpair): {fn: [] for fn in field_names}
            for neighpair in unique_pair_ids}


#def check_if_gbseg_exists_on_gb(gid, gbseg):
    """
    Example inputs
    --------------
    gid: int number in geom.gid
    gbseg: gbsegments[gid][0]

    Exaplanations
    -------------
    grain_boundary is represented as gbsegments[gid]
    """
    geom.gid
    gid = 20
    # ------------------------
    # Extract segment end coordinates
    seg_ends = {}
    for count in range(len(jnp_all_sorted_coords[gid])):
        seg_ends[count] = jnp_all_sorted_coords[gid][count:count+2]
    seg_ends[count] = np.vstack((seg_ends[count],
                                 gbmullines_grain_wise[gid].get_node_coords()[-1]))
    seg_ends
    # ------------------------
    # ------------------------
    # ------------------------
    # geom.neigh_gid[gid], geom.neigh_gid[gid][0]
    # ------------------------
    jnpcoords = jnp_all_sorted_coords[gid]
    jnpcoords
    # ------------------------
    gbcoords = gbmullines_grain_wise[gid].get_node_coords()
    gbcoords
    # ------------------------
    gbsegments[gid]
    gbsegments[gid][0].nodes
    # ------------------------
    gbp_markers_random = get_random_gbpoints_between_jnpoints(gbcoords, seg_ends)
    gbp_markers_random
    # ------------------------


nconn = neigh_connectivity_flags(pair_ids_unique_lr,
                                 field_names=['gbseg',
                                              'nnodes_eq',
                                              'length_eq',
                                              'n',
                                              'areas_raw',
                                              'uniquified',
                                              ])

gbseg_unique = {gid: None for gid in geom.gid}


pair = pair_ids_unique_lr[0]
pair

gbsegments[pair[0]]
gbsegments[pair[1]]

gbsegments[pair[0]][0].centroid_p2dl
pair_flags = [False for _ in pair_ids_unique_lr]

# ###########################################################################
EPSILON = 1E-8
for i, pair in enumerate(pair_ids_unique_lr):
    # flag = nconn[tuple(pair)]['gbseg']
    pair_rl = tuple((pair[1], pair[0]))
    nconn[tuple(pair)]['areas_raw'].append(geom.area_gid(pair[0], gsrepr='raw'))
    nconn[tuple(pair)]['areas_raw'].append(geom.area_gid(pair[1], gsrepr='raw'))
    # ====================================
    for gbseg1 in gbsegments[pair[0]]:
        """iterating through all grian boundary segments of the centre grain,
        which is pair[0].
        """
        # gbseg1 = gbsegments[pair[0]][0]
        # Number of nodes
        gbseg1_nnodes = gbseg1.nnodes
        # Centroidal point object
        gbseg1_centroid = gbseg1.centroid_p2dl
        # Total lemngth
        gbseg1_length = gbseg1.length
        # ====================================
        for gbseg2 in gbsegments[pair[1]]:
            """iterating through all grian boundary segments current neighbour
            grain, which is pair[1].
            """
            gbseg2_nnodes = gbseg2.nnodes
            proceed = False
            '''Prepare for the nnodes equality test.'''
            nnodes_equality = gbseg1_nnodes == gbseg2_nnodes
            if nnodes_equality:
                '''nnodes equality test passed.'''
                '''Prepare for the next test.'''
                _fx_ = gbseg1_centroid.is_p2dl_within_cor
                centroid_equality = _fx_(gbseg2.centroid_p2dl, EPSILON)
            else:
                continue
            # ----------------------------
            if centroid_equality:
                '''centroid equality test passed.'''
                '''Prepare for the next test.'''
                ldiff = abs(gbseg1_length - gbseg2.length)
                length_equality = ldiff <= EPSILON
            else:
                continue
            # ----------------------------
            nconn[tuple(pair)]['gbseg'].append(gbseg1)
            nconn[tuple(pair)]['gbseg'].append(gbseg2)
            # ----------------------------
            nconn[tuple(pair)]['nnodes_eq'].append(gbseg1.nnodes == gbseg2.nnodes)
            # ----------------------------
            nconn[tuple(pair)]['length_eq'].append(gbseg1.length == gbseg2.length)
            # ----------------------------
            nconn[tuple(pair)]['n'].append(len(nconn[tuple(pair)]['gbseg']))
            nconn[tuple(pair)]['uniquified'] = False


def get_unique_object_indices(nnodes, lengths, centroids, tol=1e-8):
    """
    Finds indices of unique objects based on nnodes, lengths, and centroids.

    Args:
        nnodes (np.ndarray): 1D array of nnode values.
        lengths (np.ndarray): 1D array of length values.
        centroids (np.ndarray): 2D array of centroid coordinates.
        tol (float, optional): Tolerance for centroid coordinate comparison. Defaults to 1e-6.

    Returns:
        np.ndarray: 1D array of indices corresponding to unique objects.
    """

    # Create a structured array for combined properties
    dtype = [('nnodes', nnodes.dtype),
             ('lengths', lengths.dtype),
             ('centroids', centroids.dtype, (2,))]
    data = np.empty(nnodes.shape[0], dtype=dtype)
    data['nnodes'] = nnodes
    data['lengths'] = lengths
    data['centroids'] = centroids

    # Round centroids to handle floating-point errors
    data['centroids'] = np.round(data['centroids'],
                                 decimals=int(-np.log10(tol)))

    # Find unique entries
    _, unique_indices = np.unique(data, return_index=True)

    return unique_indices






# Gather grain boundary segments of all pairs
GBSEGMENTS = {tuple(pair): None for pair in pair_ids_unique_lr}
for pair in pair_ids_unique_lr:
    _gbseg_ = nconn[tuple(pair)]['gbseg']
    if len(_gbseg_) > 0:
        ###########################################
        # MAKE UNIQUE THE LIST OF gbsegments in _gbseg_.
        nnodes = np.array([seg.nnodes for seg in nconn[tuple(pair)]['gbseg']])
        lengths = np.array([seg.length for seg in nconn[tuple(pair)]['gbseg']])
        centroids = np.array([seg.centroid for seg in nconn[tuple(pair)]['gbseg']])
        ui = get_unique_object_indices(nnodes, lengths, centroids, tol=1e-8)
        ###########################################
        GBSEGMENTS[tuple(pair)] = [nconn[tuple(pair)]['gbseg'][_ui_] for _ui_ in ui]
        '''As we no longer need any duplicates, we will override the repeated
        ones in ncoon as well.'''
        nconn[tuple(pair)]['gbseg'] = [nconn[tuple(pair)]['gbseg'][_ui_] for _ui_ in ui]
        ###########################################
        nconn[tuple(pair)]['uniquified'] = True
        # GBSEGMENTS[tuple(pair)] = _gbseg_[0]

# [nconn[tuple((32, 36))]['gbseg'][_ui_] for _ui_ in ui]

def consolidate_gbsegments(GBSEGMENTS, squeeze_segment_data_structure=False):

    """Consolidates grain boundary segments by grain ID.
    Args:
        GBSEGMENTS (dict): A dictionary where keys are tuples of grain IDs (gid1, gid2)
                           and values are grain boundary segments.
    Returns:
        dict: A dictionary where keys are grain IDs and values are lists of
              grain boundary segments associated with that grain.
    """
    grain_segments = {}  # To store the consolidated segments
    for pair, segment in GBSEGMENTS.items():
        gid1, gid2 = pair
        grain_segments.setdefault(gid1, []).append(segment)  # Add to gid1
        grain_segments.setdefault(gid2, []).append(segment)  # Add to gid2
    # -------------------------------------
    grain_segments_reordered = {}
    for index in np.unique(list(grain_segments.keys())):
        grain_segments_reordered[index] = grain_segments[index]
    # -------------------------------------
    if squeeze_segment_data_structure:
        for index in np.unique(list(grain_segments.keys())):
            squeezed = []
            if grain_segments_reordered[index] is not None:
                for a in grain_segments_reordered[index]:
                    if a is not None:
                        squeezed.extend(a)
            grain_segments_reordered[index] = squeezed
    # -------------------------------------
    return grain_segments_reordered

consolidated_segments = consolidate_gbsegments(GBSEGMENTS, squeeze_segment_data_structure=True)

'''
consolidated_segments[49]
gbsegments[49]
for seg in gbsegments[49]:
    seg.plot(ax)
for cseg in consolidated_segments[49]:
    cseg.plot(ax)
'''
# Assess the performance of unique grain boundary segment calculation.
areas, nidentical_seg, nnodes_eq, length_eq = [], [], [], []
for nconn_val in nconn.values():
    areas.append(nconn_val['areas_raw'])
    nidentical_seg.append(nconn_val['n'])
    nnodes_eq.append(nconn_val['nnodes_eq'])
    length_eq.append(nconn_val['length_eq'])

areas = np.array(areas)
# Identify the problematic grains
areas_min = np.array(areas).min(axis=1)
areas_min_pair_locations = np.argwhere(areas_min == 1)
pairids = pair_ids_unique_lr[areas_min_pair_locations].squeeze()
pairids_areas = areas[areas_min_pair_locations.squeeze()]

problematic_grains = {gid: geom.polygons[gid-1] for gid in
                      np.unique(np.hstack((pairids[:, 0][pairids_areas[:, 0] == 1],
                                           pairids[:, 1][pairids_areas[:, 1] == 1])))}


if AUTO_PLOT_EVERYTHING:
    fig, ax = plt.subplots()
    ax.imshow(geom.lgi)
    for prbgrain in problematic_grains.values():
        cenx, ceny = prbgrain.centroid.xy
        cenx, ceny = cenx[0]-0.5, ceny[0]-0.5
        ax.plot(cenx, ceny, 'kx')
    problematic_grains

if AUTO_PLOT_EVERYTHING:
    fig, ax = plt.subplots()
    ax = geom.plot_gsmp(raw=True, overlay_on_lgi=True, xoffset=0.5, yoffset=0.5, ax=ax)
    ax.plot(JNP[:, 0], JNP[:, 1], 'ro', markersize=5, alpha=0.25)

if AUTO_PLOT_EVERYTHING:
    fig, ax = plt.subplots()
    ax.imshow(geom.lgi)
    for pair in pair_ids_unique_lr:
        if GBSEGMENTS[tuple(pair)] is not None:
            for gbseg in GBSEGMENTS[tuple(pair)]:
                ax = gbseg.plot(ax=ax)



# =========================================================================
"""
We see that there are segments missing from consolidated_segments[grain_loc_ids['bottom_left_corner']]
HOwever, gbsegments is complete. So the task is to identify wbhich segments
to transfer from gbsegments data structure into consolidated segments data
struct8ure.

This needs to be done for all the boundary grains !!!
"""
if AUTO_PLOT_EVERYTHING:
    fig, ax = plt.subplots()
    #ax.imshow(geom.lgi)
    for seg in gbsegments[grain_loc_ids['bottom_left_corner']]:
        seg.plot(ax)

if AUTO_PLOT_EVERYTHING:
    fig, ax = plt.subplots()
    #ax.imshow(geom.lgi)
    for seg in consolidated_segments[grain_loc_ids['bottom_left_corner']]:
        seg.plot(ax)

'''
gnum = grain_loc_ids['bottom_right_corner']
consolidated_segments[gnum]
gbsegments[gnum]
for seg in gbsegments[gnum]:
    seg.plot(ax)
for cseg in consolidated_segments[gnum]:
    cseg.plot(ax)
'''
grain_loc_ids.keys()
#'pure_left'
# 'pure_bottom'
# 'pure_right'
# 'pure_top'
# 'corner'
# 'bottom_left_corner'
# 'bottom_right_corner'
# 'top_right_corner'
# 'top_left_corner'

for i, seg in enumerate(gbsegments[grain_loc_ids['bottom_left_corner']]):
    if seg.has_coord([xmin, ymin]):
        seg = gbsegments[grain_loc_ids['bottom_left_corner']][i]
        consolidated_segments[grain_loc_ids['bottom_left_corner']].append(seg)

for i, seg in enumerate(gbsegments[grain_loc_ids['bottom_right_corner']]):
    if seg.has_coord([xmax, ymin]):
        seg = gbsegments[grain_loc_ids['bottom_right_corner']][i]
        consolidated_segments[grain_loc_ids['bottom_right_corner']].append(seg)

for i, seg in enumerate(gbsegments[grain_loc_ids['top_right_corner']]):
    if seg.has_coord([xmax, ymax]):
        seg = gbsegments[grain_loc_ids['top_right_corner']][i]
        consolidated_segments[grain_loc_ids['top_right_corner']].append(seg)

for i, seg in enumerate(gbsegments[grain_loc_ids['top_left_corner']]):
    if seg.has_coord([xmin, ymax]):
        seg = gbsegments[grain_loc_ids['top_left_corner']][i]
        consolidated_segments[grain_loc_ids['top_left_corner']].append(seg)

def find_segs_at_loc(gbsegs, axis='y', location=-0.5):
    gb_indices, gbsegs_at_location = [], []
    column = 0 if axis == 'x' else 1 if axis == 'y' else None
    for i, gbseg in enumerate(gbsegs, start=0):
        if all(gbseg.get_node_coords()[:, column] == location):
            gb_indices.append(i)
            gbsegs_at_location.append(gbseg)
    return gb_indices, gbsegs_at_location

for gid in grain_loc_ids['pure_bottom']:
    gbind, gbs = find_segs_at_loc(gbsegments[gid], axis='y', location=ymin)
    if len(gbs) > 0:
        for _gbs_ in gbs:
            consolidated_segments[gid].append(_gbs_)

for gid in grain_loc_ids['pure_right']:
    gbind, gbs = find_segs_at_loc(gbsegments[gid], axis='x', location=xmax)
    if len(gbs) > 0:
        for _gbs_ in gbs:
            consolidated_segments[gid].append(_gbs_)

for gid in grain_loc_ids['pure_top']:
    gbind, gbs = find_segs_at_loc(gbsegments[gid], axis='y', location=ymax)
    if len(gbs) > 0:
        for _gbs_ in gbs:
            consolidated_segments[gid].append(_gbs_)

for gid in grain_loc_ids['pure_left']:
    gbind, gbs = find_segs_at_loc(gbsegments[gid], axis='x', location=xmin)
    if len(gbs) > 0:
        for _gbs_ in gbs:
            consolidated_segments[gid].append(_gbs_)

if AUTO_PLOT_EVERYTHING:
    fig, ax = plt.subplots()
    ax.imshow(geom.lgi)
    for gid in geom.gid:
        for _ in consolidated_segments[gid]:
            _.plot(ax=ax)
        coord = gbmullines_grain_wise[gid].get_node_coords()
        plt.plot(coord[:, 0], coord[:, 1], 'k.')
        jnp = jnp_all_sorted_coords[gid]
        plt.plot(jnp[:, 0], jnp[:, 1], 'ks', mfc='c', ms=10, alpha=0.25)


def check_if_all_gbsegs_can_form_closed_rings(GBSEGS,
                                              _print_individual_excoord_order_=True,
                                              _print_statement_=True
                                              ):
    jnp_unique_counts_grain_wise = []
    for gid in GBSEGS.keys():
        extreme_coords_unique = []
        for seg in GBSEGS[gid]:
            extreme_coords_unique.extend(seg.get_node_coords()[[0, -1], :])
        extreme_coords_unique = np.unique(extreme_coords_unique, axis=0)
        extreme_coords_unique_count = [0 for _ in extreme_coords_unique]
        for i, excoord in enumerate(extreme_coords_unique):
            for seg in GBSEGS[gid]:
                # seg = GBSEGS[gid][0]
                segcoords = seg.get_node_coords()
                if is_a_in_b(excoord, segcoords):
                    extreme_coords_unique_count[i] += 1
        if _print_individual_excoord_order_:
            print(extreme_coords_unique_count)
        jnp_unique_counts_grain_wise.append(all(np.array(extreme_coords_unique_count) == 2))
    if all(jnp_unique_counts_grain_wise):
        if _print_statement_:
            print(40*'-', '\n All gid mapped gbsegs can form closed ring structure. \n', 40*'-')
        return True, jnp_unique_counts_grain_wise
    else:
        if _print_statement_:
            print(40*'-', '\n Some gbsegs cannot form closed ring structure. \n', 40*'-')
        return False, jnp_unique_counts_grain_wise

gbsegs_can_form_rings, _ = check_if_all_gbsegs_can_form_closed_rings(consolidated_segments,
                                                                     _print_individual_excoord_order_=False,
                                                                     _print_statement_=True)

if not gbsegs_can_form_rings:
    raise ValueError('Some gid mapped gbsegs cannot form closed ring structure.')
# ====================================================================
reordering_needed = []
for gid in geom.gid:
    gbsegs = consolidated_segments[gid]
    if not gbsegs[0].nodes[0].eq_fast(gbsegs[-1].nodes[-1])[0]:
        reordering_needed.append(gid)
# ====================================================================
""" Useful Definitions of MulSline2D
do_i_precede(multisline2d)
do_i_proceed(multisline2d)
is_adjacent(multisline2d)
find_spatially_next_multisline2d(self, multislines2d)
"""
GB = {gid: None for gid in geom.gid}
GBS_reordering_success = {gid: None for gid in geom.gid}
niterations = {gid: 0 for gid in geom.gid}
plot_each_grain_details = False
from upxo.geoEntities.mulsline2d import ring2d
for gid in geom.gid:
    # gid = 33
    gbsegs = consolidated_segments[gid]
    print(40*'-')
    _gbseg_ring_ = ring2d(segments=gbsegs,
                          segids=list(range(len(gbsegs))),
                          segflips=[False for _ in gbsegs])
    # segments=None, segids=None, segflips=None
    # _gbseg_ring_.segments
    if plot_each_grain_details:
        _gbseg_ring_.plot_segs(plot_centroid=True, centroid_text=gid,
                               plot_coord_order=True, visualize_flip_req=False)
    # _gbseg_ring_.create_polygon_from_coords()

    continuity, flip_needed, i_precede_chain = _gbseg_ring_.assess_spatial_continuity()
    print(f'gid={gid}: ', 'gbsegs continuous' if continuity else 'gbsegs not continuous. Attempting reorder.')
    # print(40*'-', '\n Extreme coordinates of gbsegs are:\n')
    # for i, gbseg in enumerate(gbsegs, start = 0):
        # print(f'Segment {i}: ', gbseg.nodes[0], gbseg.nodes[-1])
    # print(40*'-')

    # plt.imshow(geom.lgi==gid)

    if continuity:
        '''If the segments are indeed continous, ring formation is
        stright-forward.'''
        # -------------
        """segments=None, segids=None, segflips=None"""
        # -------------
        NSEG_rng = range(len(gbsegs))
        GB[gid] = ring2d(gbsegs, list(NSEG_rng), [False for _ in NSEG_rng])
        # GB[gid].segments
        # GB[gid].segids
        # GB[gid].segflips
        GBS_reordering_success[gid] = True
        if plot_each_grain_details:
            GB[gid].plot_segs(plot_centroid=True, centroid_text=gid,
                               plot_coord_order=True, visualize_flip_req=True)
    if not continuity:
        '''If the segments are not found to be continous, ring formation
        requires the calculation of exact segids and segflip values. segids provide
        the spatial order of segments in ring.segments and segflips provide the
        boolean value indicating the need to flip a segment to ensure spatial
        continuity of ending nodes of adjacent multi-line-segments.
        '''
        GBS_reordering_success[gid] = False
        # ------------------------
        segids = list(range(len(gbsegs)))
        '''segstart_flip to be set to True if current gbsegs[0] goes
        counter-clockwise. Setting to False for now. Needs seperate assessment.
        '''
        segstart_flip = False
        GB[gid] = ring2d([gbsegs[0]], [0], [segstart_flip])
        '''
        GB[gid].segments
        GB[gid].segids
        GB[gid].segflips
        '''
        seg_num = 0
        used_segids = [seg_num]
        search_segids = set(segids)
        max_iterations = 10*len(gbsegs)
        itcount = 1  # iteration_count
        while len(search_segids) > 0:
            current_seg = GB[gid].segments[-1]
            flip_req_previous = GB[gid].segflips[-1]
            # print(40*'-', '\n Current segment end nodes:\n')
            # print(current_seg.nodes[0], current_seg.nodes[-1], '\n', 40*'-')
            search_segids = set(segids) - set(used_segids)
            adj = []
            for candidate_segid in search_segids:
                # candidate_segid = 2
                # print(40*'-', '\n Current segment end nodes:\n')
                # print(current_seg.nodes[0], current_seg.nodes[-1])
                candidate_seg = gbsegs[candidate_segid]
                if flip_req_previous:
                    adjacency = current_seg.do_i_proceed(candidate_seg)
                else:
                    adjacency = current_seg.do_i_precede(candidate_seg)
                adj.append(adjacency)
                # print(40*'-', f'\n Cand. seg. {candidate_segid} end nodes:')
                # print(candidate_seg.nodes[0], candidate_seg.nodes[-1])
                # print(40*'-', f'\n Current precede candidate: {adjacency}\n', 40*'-')
                # adjacency = current_seg.do_i_proceed(candidate_seg)
                # print(adjacency)
                if adjacency[0]:
                    used_segids.append(candidate_segid)
                    GB[gid].add_segment_unsafe(candidate_seg)
                    GB[gid].add_segid(candidate_segid)
                    GB[gid].add_segflip(adjacency[1])
                    '''
                    GB[gid].segments
                    GB[gid].segids
                    GB[gid].segflips
                    GB[gid].plot_segs()

                    for i, gbseg in enumerate(GB[gid].segments, start = 0):
                        print(f'Segment {i}: ', gbseg.nodes[0], gbseg.nodes[-1])
                    '''
                    break
            if itcount >= max_iterations:
                'Prevent infinite loop.'
                break
            itcount += 1
        if len(search_segids) == 0:
            # --------------------------------
            # Ensure complete connectivity
            if not GB[gid].segments[0].nodes[0].eq_fast(GB[gid].segments[-1].nodes[-1]):
                GB[gid].segflips[-1] = True
            # --------------------------------
            GBS_reordering_success[gid] = True
            niterations[gid] = itcount
            print(f'Re-ordering success. gbsegs are continous. N.Segs={len(gbsegs)}. N.Iterations={itcount}')
        else:
            # GBS_reordering_success[gid] = False < -- BY default.
            # So, nothinhg more to do here.
            pass
        if plot_each_grain_details:
            GB[gid].plot_segs(plot_centroid=True, centroid_text=gid,
                                   plot_coord_order=True, visualize_flip_req=True)
        # GB[gid].segflips
print(40*'-', f'\nTotal number of iterations: {sum(list(niterations.values()))}')
# Re-assessment - segflips of the last segment.
for gid in geom.gid:
    start = GB[gid].segments[0].nodes[0]
    end0 = GB[gid].segments[-1].nodes[0]
    end1 = GB[gid].segments[-1].nodes[-1]
    condition1 = start.eq_fast(end0)[0]
    condition2 = start.eq_fast(end1)[0]
    if condition1:
        GB[gid].segflips[-1] = True



# GB[gid].create_polygon_from_coords()
GBCoords = {gid: None for gid in geom.gid}
for gid in geom.gid:
    if gid in grain_loc_ids['boundary']:
        coord = GB[gid].create_coords_from_segments(force_close=True)
    else:
        coord = GB[gid].create_coords_from_segments(force_close=False)
    GBCoords[gid] = coord


gid = 43
GB[gid].segments
GB[gid].segflips
coords = GB[gid].segments[0].get_node_coords()
for i, seg in enumerate(self.segments[1:], start=1):
    if self.segflips[i]:
        thissegcoords = np.flip(seg.get_node_coords(), axis=0)
        coords = np.vstack((coords, thissegcoords[1:]))
    else:
        coords = np.vstack((coords, seg.get_node_coords()[1:]))
if force_close:
    coords = self.force_close_coordinates(coords, assess_first=True)



plt.figure()
for gid in geom.gid:
    c = GBCoords[gid]
    plt.plot(c[:, 0], c[:, 1])
# ------------------------------------------
GBCoords[43]
GB[43].plot_segs()
# ------------------------------------------

GRAINS = {gid: Polygon(GBCoords[gid]) for gid in geom.gid}
POLYXTAL = MultiPolygon(GRAINS.values())

grain = deepcopy(GRAINS[1])
#grain_core = grain.buffer(-2.0, resolution=0, cap_style=1, join_style=1,
#                          mitre_limit=1)
# grain_core = grain_core.convex_hull
# grain_core = grain_core.simplify(tolerance=1)
# gbz = grain - grain_core
gbz

# plot_multipolygon(POLYXTAL, invert_y=True)
fig, ax = plt.subplots()
ax.imshow(geom.lgi)


def AssembleGBSEGS(geom, GB):
    mids = []
    for gb in GB.values():
        mids.extend([id(seg) for seg in gb.segments])
    mids = np.unique(mids)
    sgseg_list = [None for mid in mids]
    for i, mid in enumerate(mids, start=0):
        for gid in geom.gid:
            for gb in GB[gid].segments:
                if mid == id(gb):
                    sgseg_list[i] = gb
    return mids, np.array(sgseg_list)

def get_mids_gbsegs(GB, gid):
    return [id(seg) for seg in GB[gid].segments]

all_mids, sgseg_list = AssembleGBSEGS(geom, GB)

def get_gbmid_indices_at_gid(GB, gid, all_mids):
    segmids = get_mids_gbsegs(GB, gid)
    locs = []
    for segmid in segmids:
        locs.append(np.argwhere(all_mids == segmid)[0][0])
    return locs

gid = 2
midlocs = {gid: get_gbmid_indices_at_gid(GB, gid, all_mids)
           for gid in geom.gid}
sgseg_list[midlocs[gid]].tolist()
GB[gid].segments

"""
Now that the individual gbsegs have been compiled into a list and the
mapping has been established between this list and the individual gbsegs in
GB data structure, we can now proceed with smoothing.
"""
GB_smooth = deepcopy(GB)

all_mids, sgseg_list = AssembleGBSEGS(geom, GB_smooth)

for seg in sgseg_list:
    seg.smooth(max_smooth_level=3)

for gid in geom.gid:
    GB_smooth[gid].plot_segs()

for gid in geom.gid:
    GB[gid].plot_segs()

GBCoords_smoothed = {gid: None for gid in geom.gid}
for gid in geom.gid:
    if gid in grain_loc_ids['boundary']:
        coord = GB_smooth[gid].create_coords_from_segments(force_close=True)
    else:
        coord = GB_smooth[gid].create_coords_from_segments(force_close=False)
    GBCoords_smoothed[gid] = coord

plt.figure()
for gid in geom.gid:
    c = GBCoords_smoothed[gid]
    plt.plot(c[:, 0], c[:, 1])

GRAINS = {gid: Polygon(GBCoords[gid]) for gid in geom.gid}
POLYXTAL = MultiPolygon(GRAINS.values())
plot_multipolygon(POLYXTAL, invert_y=True)

GRAINS_SMOOTHED = {gid: Polygon(GBCoords_smoothed[gid]) for gid in geom.gid}
POLYXTAL_SMOOTHED = MultiPolygon(GRAINS_SMOOTHED.values())
plot_multipolygon(POLYXTAL_SMOOTHED, invert_y=True)
# ===============================================================
"""
Calculate grain boundary zones.
"""
grains = [deepcopy(g) for g in GRAINS_SMOOTHED.values()]
areas = np.array([g.area for g in GRAINS_SMOOTHED.values()])
diameters = np.sqrt(4*areas/math.pi)
# plt.hist(diameters)
diameter_threshold = np.quantile(diameters, 0.5)
grains_for_gbz = diameters >= diameter_threshold
# quantiles = np.hstack((np.arange(0, 1, 0.1), np.arange(0.905, 1, 0.005)))
# quantile_values = [np.quantile(areas, i) for i in quantiles]
# plt.plot(quantiles, quantile_values)
Grains = []
grains_details = {gid: {'grain': None, 'core': None, 'gbz': None} for gid in geom.gid}
for i, (grain, go) in enumerate(zip(grains, grains_for_gbz), start=0):
    if not go:
        Grains.append(grain)
        grains_details[i+1]['grain'] = grain
        continue
    gbz_thickness = 10*diameters[i]/100  # 10%
    grain_core = grain.buffer(-gbz_thickness,
                              resolution=0,
                              cap_style=1,
                              join_style=1,
                              mitre_limit=1)
    if isinstance(grain_core, MultiPolygon):
        cores = list(grain_core.geoms)
        for i, core in enumerate(cores, start=0):
            # core = core.simplify(tolerance=10)
            # core = core.convex_hull
            Grains.append(core)
            if i == 0:
                gbz = grain - core
            else:
                gbz = gbz - core
        Grains.append(gbz)
    else:
        gbz = grain - grain_core
        Grains.append(grain_core)
        Grains.append(gbz)
    grains_details[i+1]['grain'] = grain
    grains_details[i+1]['core'] = grain_core
    grains_details[i+1]['gbz'] = gbz
    Grains.extend([gbz, grain_core])

POLYXTAL_SMOOTHED_gbz = MultiPolygon(Grains)
plot_multipolygon(POLYXTAL_SMOOTHED_gbz, invert_y=True)
# ===============================================================
import pygmsh
geometry = pygmsh.geo.Geometry()
model = geometry.__enter__()
for i, g in enumerate(POLYXTAL_SMOOTHED, start=0):
    coords = np.vstack((g.exterior.coords.xy[0][:-1],
                        g.exterior.coords.xy[1][:-1])).T
    model.add_polygon(coords, mesh_size=1)
model.synchronize()
dim=2
elshape = 'quad'
elorder = 1
algorithm = 8
mesh = geometry.generate_mesh(dim=dim, order=elorder, algorithm=algorithm)
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
# [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
# [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
# [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
# [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
# [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
gid = 1
seg = GB[gid].segments[0]
seg.get_node_coords()
seg.coords

smoothed_gbsegs = {gid: [] for gid in geom.gid}
smoothed_coords = {gid: [] for gid in geom.gid}
for gid in geom.gid:
    for seg in GB[gid].segments:
        smoothed_seg, _smoothed_coords_ = seg.smooth(max_smooth_level=4)
        smoothed_gbsegs[gid].append(smoothed_seg)
        smoothed_coords[gid].append(_smoothed_coords_))

from upxo.geoEntities.mulsline2d import ring2d
smoothed_rings = {gid: ring2d(smoothed_gbsegs[gid]) for gid in geom.gid}
for gid in geom.gid:
    smoothed_rings[gid].segflips = GB[gid].segflips



gid = 1
smoothed_rings[gid].segments
GB[gid].segments
coordinates = []
for gbsegcount, gbseg in enumerate(smoothed_rings[gid].segments, start=0):
    coords = gbseg.get_node_coords()
    coordinates.append(coords)



smoothed_rings[gid].get_coords_newdef()

for gid in geom.gid:
    smoothed_rings[gid].plot_segs()

GBCoords = {gid: None for gid in geom.gid}
GBCoords_assembled = {gid: None for gid in geom.gid}

for gid in geom.gid:
    # ---------------------------------
    COORD = []
    for i, seg in enumerate(smoothed_rings[gid].segments, start=0):
        COORD.append(seg.get_node_coords())
    GBCoords[gid] = COORD
    # ---------------------------------
    coord_assembled = []
    for coord in COORD:
        coord_assembled.append(coord[:-1])
    coord_assembled = np.vstack((coord_assembled))
    GBCoords_assembled[gid] = coord_assembled
    # ---------------------------------

GRAINS = {gid: Polygon(GBCoords[gid]) for gid in geom.gid}

POLYXTAL = MultiPolygon(GRAINS.values())



def moving_average(data, window_size):
    """Compute the moving average of the given data with the specified window size."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def mean_coordinates(coords, window_size):
    """
    Smooths the given 2D numpy array of coordinates using a moving average.

    Parameters:
    coords (numpy.ndarray): A 2D numpy array of shape (n, 2) where n is the number of points.
    window_size (int): The window size for the moving average.

    Returns:
    numpy.ndarray: A 2D numpy array of the smoothed coordinates.
    """
    # Check if there are enough points for the moving average
    if len(coords) < window_size:
        return coords  # Return the original coordinates if not enough points

    # Separate the coordinates into x and y components
    x = coords[:, 0]
    y = coords[:, 1]

    # Apply moving average to the x and y components separately
    x_smooth = moving_average(x, window_size)
    y_smooth = moving_average(y, window_size)

    # Add the original end points to the smoothed coordinates if there are enough points
    if len(x_smooth) > 0 and len(y_smooth) > 0:
        smoothed_coords = np.vstack([
            [x[0], y[0]],  # Start point
            np.column_stack([x_smooth, y_smooth]),
            [x[-1], y[-1]]  # End point
        ])
    else:
        smoothed_coords = coords  # If not enough points, use original coordinates

    return smoothed_coords


smoothing_level = 4

plt.imshow(geom.lgi)
smoothed_segments_coords = {gid: [] for gid in geom.gid}
for gid in geom.gid:
    for gbs in consolidated_segments[gid]:
        gbs_coords = gbs.get_node_coords()
        if gbs.nnodes > 4:
            smoothed_coords = mean_coordinates(gbs_coords, smoothing_level)
            plt.plot(smoothed_coords[:, 0], smoothed_coords[:, 1], '-k')
        elif gbs.nnodes == 4:
            smoothed_coords = mean_coordinates(gbs_coords, 3)
            plt.plot(smoothed_coords[:, 0], smoothed_coords[:, 1], '-k')
        elif gbs.nnodes == 3:
            smoothed_coords = mean_coordinates(gbs_coords, 2)
            plt.plot(smoothed_coords[:, 0], smoothed_coords[:, 1], '-k')
        else:
            plt.plot(gbs_coords[:, 0], gbs_coords[:, 1], '-k')
        smoothed_segments_coords[gid].append(smoothed_coords)
ymin, ymax = ax.get_ylim()
ax.set_ylim(ymax, ymin)




smoothed_segments_coords = {gid: [] for gid in geom.gid}
for gid in geom.gid:
    for gbs in consolidated_segments[gid]:
        gbs_coords = gbs.get_node_coords()
        if gbs.nnodes > 4:
            smoothed_coords = mean_coordinates(gbs_coords, smoothing_level)
            smoothed_segments_coords[gid].append(smoothed_coords)
        else:
            smoothed_segments_coords[gid].append(gbs_coords)

smoothed_segments = {gid: [] for gid in geom.gid}
for gid in geom.gid:
    for gbs in consolidated_segments[gid]:
        gbs_coords = gbs.get_node_coords()
        if gbs.nnodes > 4:
            smoothed_coords = mean_coordinates(gbs_coords, smoothing_level)
            segment = MSline2d.by_nodes([Point2d(sc[0], sc[1]) for sc in smoothed_coords],
                                         close=False)
            smoothed_segments[gid].append(segment)
        else:
            smoothed_segments[gid].append(gbs)

fig, ax = plt.subplots()
for gid in geom.gid:
    for seg in smoothed_segments[gid]:
        seg.plot(ax)

# ############################################################################
# ############################################################################
# ############################################################################
# ############################################################################
# ############################################################################

GRAINS[29]

GBCoords[53]
GB[53].create_coords_from_segments()

GB[53].plot_segs(visualize_flip_req=True)
GB[53].segflips
GB[53].create_coords_from_segments()
GB[53].connectivity0()
GB[53].segments

gid = 1
GB[gid].plot_segs()

print(40*'-', '\n Extreme coordinates of gbsegs are:\n')
for i, gbseg in enumerate(gbsegs, start = 0):
    print(f'Segment {i}: ', gbseg.nodes[0], gbseg.nodes[-1])
print(40*'-')
print(40*'-', '\n Extreme coordinates of gbsegs are:\n')
for i, gbseg in enumerate(growing_ring.segments, start = 0):
    print(f'Segment {i}: ', gbseg.nodes[0], gbseg.nodes[-1])
print(40*'-')


    current_seg = growing_ring.segments[-1]
    print(current_seg.nodes[0], current_seg.nodes[-1])
    search_segids = set(segids) - set(used_segids)
    for candidate_segid in search_segids:
        # candidate_segid = 2
        candidate_seg = gbsegs[candidate_segid]
        print(candidate_seg.nodes[0], candidate_seg.nodes[-1])
        precedence = current_seg.do_i_precede(candidate_seg)
        # candidate_seg.do_i_precede(current_seg)
        print(precedence)
        if precedence[0]:
            used_segids.append(candidate_segid)
            growing_ring.add_segment_unsafe(candidate_seg)
            growing_ring.add_segid(candidate_segid)
            growing_ring.add_segflip(precedence[1])
            '''
            growing_ring.segments
            growing_ring.segids
            growing_ring.segflips
            '''
            break

continuity, flip_needed, i_precede_chain = growing_ring.assess_spatial_continuity()
print(f'gid={gid}: ', 'gbsegs continuous' if continuity else 'gbsegs not continuous. Re-order needed')
print(40*'-', '\n Extreme coordinates are:\n')
for i, gbseg in enumerate(growing_ring.segments, start = 0):
    print(f'Segment {i}: ', gbseg.nodes[0], gbseg.nodes[-1])
print(40*'-')


for seg in growing_ring.segments:
    print(seg.nodes[0], seg.nodes[-1])

print(consolidated_segments[gid])
for roll_count, seg in enumerate(consolidated_segments[gid], start=0):
    print('===============')
    rolled_seglist = np.roll(consolidated_segments[gid],
                             -roll_count, axis=0)
    print(rolled_seglist)




np.roll(np.array([1, 2, 3, 4]), -0, axis=0)

for seg in
seg1.do_i_precede


for gid in geom.gid:
    gbsegs = consolidated_segments[gid]
    for seg in gbsegs:
        pass
# ====================================================================
consolidated_segments[gid]
# ====================================================================

consolidated_segments[gid]

gid = 1
# plot_gbsegs(consolidated_segments[gid], plot_coord_order=False)
for a in consolidated_segments[gid]:
    print(a.nodes[0], a.nodes[-1])
# ====================================================================


def sort_subsets_by_original_order(CA, subsets):
    """
    Sorts subsets of coordinates based on their original order in the CA array,
    and returns the sorted subsets along with their indices.

    Args:
        CA (np.ndarray): The original 2D coordinate array (N x 2).
        subsets (list): A list of np.ndarrays, each representing a subset of coordinates.

    Returns:
        tuple: A tuple containing two elements:
            - list: A list of np.ndarrays, where each subset is sorted according to the
                    order of its points in the CA array.
            - np.ndarray: A 1D array containing the original indices of the subsets
                          in the input list, corresponding to the order of sorted subsets.
    """

    coord_to_index = {tuple(coord): idx for idx, coord in enumerate(CA)}

    sorted_subsets = []
    subset_indices = []  # To track the original indices of subsets
    for i, subset in enumerate(subsets):
        sorted_indices = np.argsort([coord_to_index[tuple(coord)] for coord in subset])
        sorted_subsets.append(subset[sorted_indices])
        subset_indices.append(i)  # Record the original index

    # Sort subset indices based on the first element of each sorted subset
    sort_order = np.argsort([coord_to_index[tuple(s[0])] for s in sorted_subsets])
    sorted_subsets = [sorted_subsets[i] for i in sort_order]
    subset_indices = np.array(subset_indices)[sort_order]  # Convert to NumPy array and reorder

    return sorted_subsets, subset_indices

sorted_segs = {gid: None for gid in geom.gid}
for gid in geom.gid:
    _, subset_indices = sort_subsets_by_original_order(gbmullines_grain_wise[gid].get_node_coords(),
                                   [seg.get_node_coords() for seg in consolidated_segments[gid]])
    sorted_segs[gid] = [consolidated_segments[gid][ssind] for ssind in subset_indices]

'''
There will still be some errors in ordering. For xample:
    [uxpo-p2d (7.5,50.5), uxpo-p2d (7.5,49.5), uxpo-p2d (2.5,49.5)]
    [uxpo-p2d (2.5,50.5), uxpo-p2d (7.5,50.5)]
    [uxpo-p2d (2.5,49.5), uxpo-p2d (2.5,50.5)]
As we see, the second segment should be the one at the end as it closes with the first.
So, we will foirst check iof the multi-segments close. If it does not, then we will re-order
the individual segments, to ensure closure.
'''
reordering_needed = []
for gid in geom.gid:
    gbsegs = sorted_segs[gid]
    if not gbsegs[0].nodes[0].eq_fast(gbsegs[-1].nodes[-1])[0]:
        reordering_needed.append(gid)

# It turns out, there are more re-orderings needed for non-end segmnets.
# So, we will use a general way to do this.
gid = 2
gbsegs = sorted_segs[gid]
create_polygon_from_segments(gbsegs)

gbsegs[0].get_node_coords()
gbsegs[1].get_node_coords()
gbsegs[2].get_node_coords()


plot_gbsegs(sorted_segs[2], plot_coord_order=True)


def plot_gbsegs(gbsegs, plot_coord_order=False):
    fig, ax = plt.subplots()
    FS = 8
    for gbsegcount, gbseg in enumerate(gbsegs, start=0):
        coords = gbseg.get_node_coords()
        color = np.random.random(3)
        ax.plot(coords[:, 0], coords[:, 1], color=color, linewidth=3)
        centroid = gbseg.centroid
        ax.plot(centroid[0], centroid[1], 'kx')
        ax.text(centroid[0], centroid[1], gbsegcount, color='red',
                fontsize=12, fontweight='bold')
        fs = FS  + (gbsegcount % 2) * 2
        offset = + (gbsegcount % 2) * 0.1
        for coord_count, coord in enumerate(coords, start=0):
            ax.text(coord[0]+offset, coord[1]+offset,
                    coord_count, color=color, fontsize=fs)
    if plot_coord_order:
        C = create_coords_from_segments(gbsegs)
        ax.plot(C[:,0], C[:,1], '-.k', linewidth=0.75)
    return ax




def sort_gbsegs(gbsegs):
    gbsegs_copy = gbsegs[:]
    sorted_gbsegs = [gbsegs_copy.pop(0)]  # Start with the first segment

    while gbsegs:
        last_segment = sorted_gbsegs[-1]
        last_node = last_segment.nodes[-1]  # End node of the last segment

        for i, segment in enumerate(gbsegs):
            start_node = segment.nodes[0]
            end_node = segment.nodes[-1]

            if last_node.eq_fast(start_node)[0]:
                # If the last node of the sorted segment matches the start node of the current segment
                sorted_gbsegs.append(segment)
                gbsegs.pop(i)
                break
            elif last_node.eq_fast(end_node)[0]:
                # If the last node of the sorted segment matches the end node of the current segment
                segment.flip()
                sorted_gbsegs.append(segment)
                gbsegs.pop(i)
                break
    return sorted_gbsegs

def sort_gbsegs_new(gbsegs):
    sorted_gbsegs = [gbsegs[0]]  # Start with the first segment
    used_indices = {0}  # Set to keep track of used indices

    while len(used_indices) < len(gbsegs):
        last_segment = sorted_gbsegs[-1]
        last_node = last_segment.nodes[-1]  # End node of the last segment

        for i, segment in enumerate(gbsegs):
            if i in used_indices:
                continue  # Skip already used segments

            start_node = segment.nodes[0]
            end_node = segment.nodes[-1]

            if last_node.eq_fast(start_node)[0]:
                # If the last node of the sorted segment matches the start node of the current segment
                sorted_gbsegs.append(segment)
                used_indices.add(i)
                break
            elif last_node.eq_fast(end_node)[0]:
                # If the last node of the sorted segment matches the end node of the current segment
                segment.flip()
                sorted_gbsegs.append(segment)
                used_indices.add(i)
                break

    return sorted_gbsegs

def sort_gbsegs_new1(gbsegs):
    if not gbsegs:
        return []

    sorted_gbsegs = [gbsegs[0]]  # Start with the first segment
    used_indices = {0}  # Set to keep track of used indices

    while len(used_indices) < len(gbsegs):
        last_segment = sorted_gbsegs[-1]
        last_node = last_segment.nodes[-1]  # End node of the last segment
        found_next_segment = False  # Flag to indicate if the next segment was found

        for i, segment in enumerate(gbsegs):
            if i in used_indices:
                continue  # Skip already used segments

            start_node = segment.nodes[0]
            end_node = segment.nodes[-1]

            if last_node.eq_fast(start_node)[0]:
                # If the last node of the sorted segment matches the start node of the current segment
                sorted_gbsegs.append(segment)
                used_indices.add(i)
                found_next_segment = True
                break
            elif last_node.eq_fast(end_node)[0]:
                # If the last node of the sorted segment matches the end node of the current segment
                segment.flip()
                sorted_gbsegs.append(segment)
                used_indices.add(i)
                found_next_segment = True
                break

        if not found_next_segment:
            raise ValueError("Cannot find a connecting segment, the segments may not form a continuous path.")

    return sorted_gbsegs


def sort_gbsegs_new2(gbsegs):
    if not gbsegs:
        return []

    sorted_gbsegs = [gbsegs[0]]  # Start with the first segment
    used_indices = {0}  # Set to keep track of used indices

    while len(used_indices) < len(gbsegs):
        found_next_segment = False  # Flag to indicate if the next segment was found

        last_segment = sorted_gbsegs[-1]
        last_node = last_segment.nodes[-1]  # End node of the last segment

        for i, segment in enumerate(gbsegs):
            if i in used_indices:
                continue  # Skip already used segments

            start_node = segment.nodes[0]
            end_node = segment.nodes[-1]

            if last_node.eq_fast(start_node)[0]:
                # If the last node of the sorted segment matches the start node of the current segment
                sorted_gbsegs.append(segment)
                used_indices.add(i)
                found_next_segment = True
                break
            elif last_node.eq_fast(end_node)[0]:
                # If the last node of the sorted segment matches the end node of the current segment
                segment.flip()
                sorted_gbsegs.append(segment)
                used_indices.add(i)
                found_next_segment = True
                break

        # Check for a segment connecting to the start of the first segment
        if not found_next_segment:
            first_segment = sorted_gbsegs[0]
            first_node = first_segment.nodes[0]  # Start node of the first segment

            for i, segment in enumerate(gbsegs):
                if i in used_indices:
                    continue  # Skip already used segments

                start_node = segment.nodes[0]
                end_node = segment.nodes[-1]

                if first_node.eq_fast(end_node)[0]:
                    # If the start node of the first segment matches the end node of the current segment
                    segment.flip()
                    sorted_gbsegs.insert(0, segment)
                    used_indices.add(i)
                    found_next_segment = True
                    break
                elif first_node.eq_fast(start_node)[0]:
                    # If the start node of the first segment matches the start node of the current segment
                    sorted_gbsegs.insert(0, segment)
                    used_indices.add(i)
                    found_next_segment = True
                    break

        if not found_next_segment:
            raise ValueError("Cannot find a connecting segment, the segments may not form a continuous path.")

    return sorted_gbsegs

def sort_gbsegs_new3(gbsegs):
    if not gbsegs:
        return []

    sorted_gbsegs = [gbsegs[0]]  # Start with the first segment
    used_indices = {0}  # Set to keep track of used indices

    while len(used_indices) < len(gbsegs):
        found_next_segment = False  # Flag to indicate if the next segment was found

        last_segment = sorted_gbsegs[-1]
        last_node_coords = last_segment.get_node_coords()[-1]  # End node coordinates of the last segment

        for i, segment in enumerate(gbsegs):
            if i in used_indices:
                continue  # Skip already used segments

            start_node_coords = segment.get_node_coords()[0]
            end_node_coords = segment.get_node_coords()[-1]

            if all(a == b for a, b in zip(last_node_coords, start_node_coords)):
                # If the last node of the sorted segment matches the start node of the current segment
                sorted_gbsegs.append(segment)
                used_indices.add(i)
                found_next_segment = True
                break
            elif all(a == b for a, b in zip(last_node_coords, end_node_coords)):
                # If the last node of the sorted segment matches the end node of the current segment
                segment.flip()
                sorted_gbsegs.append(segment)
                used_indices.add(i)
                found_next_segment = True
                break

        # Check for a segment connecting to the start of the first segment
        if not found_next_segment:
            first_segment = sorted_gbsegs[0]
            first_node_coords = first_segment.get_node_coords()[0]  # Start node coordinates of the first segment

            for i, segment in enumerate(gbsegs):
                if i in used_indices:
                    continue  # Skip already used segments

                start_node_coords = segment.get_node_coords()[0]
                end_node_coords = segment.get_node_coords()[-1]

                if all(a == b for a, b in zip(first_node_coords, end_node_coords)):
                    # If the start node of the first segment matches the end node of the current segment
                    segment.flip()
                    sorted_gbsegs.insert(0, segment)
                    used_indices.add(i)
                    found_next_segment = True
                    break
                elif all(a == b for a, b in zip(first_node_coords, start_node_coords)):
                    # If the start node of the first segment matches the start node of the current segment
                    sorted_gbsegs.insert(0, segment)
                    used_indices.add(i)
                    found_next_segment = True
                    break

        if not found_next_segment:
            raise ValueError("Cannot find a connecting segment, the segments may not form a continuous path.")

    return sorted_gbsegs

plot_gbsegs(sorted_segs[gid], plot_coord_order=True)
plot_gbsegs(sort_gbsegs_new3(sorted_segs[gid]), plot_coord_order=True)


reordering_needed

count = 0
gid = reordering_needed[count]
sorted_segs[gid]
plot_gbsegs(sorted_segs[gid], plot_coord_order=True)
plot_gbsegs(sort_gbsegs(sorted_segs[gid]), plot_coord_order=True)
count += 1



A = sorted_segs[gid]
A[0].nodes[0], A[0].nodes[-1]
A[1].nodes[0], A[1].nodes[-1]
A[2].nodes[0], A[2].nodes[-1]
A[3].nodes[0], A[3].nodes[-1]
A[4].nodes[0], A[4].nodes[-1]
A[5].nodes[0], A[5].nodes[-1]
A[6].nodes[0], A[6].nodes[-1]

A = consolidated_segments[gid]
A[0].nodes[0], A[0].nodes[-1]
A[1].nodes[0], A[1].nodes[-1]
A[2].nodes[0], A[2].nodes[-1]
A[3].nodes[0], A[3].nodes[-1]
A[4].nodes[0], A[4].nodes[-1]
A[5].nodes[0], A[5].nodes[-1]
A[6].nodes[0], A[6].nodes[-1]


resort_flags = []
for gid in geom.gid:
    resort_flags.append(1 if gid in reordering_needed else 0)

resorted_segs = {gid: None for gid in geom.gid}
for gid in geom.gid:
    if resort_flags[gid-1]:
        resorted_segs[gid] = sort_gbsegs(sorted_segs[gid])
    else:
        resorted_segs[gid] = sorted_segs[gid]







plot_gbsegs(sorted_segs[2])
print_coords_of_segments(sorted_segs[2])
create_coords_from_segments(sorted_segs[2])

def print_coords_of_segments(gbsegs):
    for gbseg in gbsegs:
        print(gbseg.get_node_coords())


gid = 19
gbsegs = sorted_segs[gid]
create_polygon_from_segments(gbsegs)
for i, gbs in enumerate(gbsegs, start=1):
    print(f'----- seg: {i}')
    print(gbs.get_node_coords())


gbsegs[0].nodes
gbsegs[1].nodes

condition_a = gbsegs[0].nodes[-1].eq_fast(gbsegs[1].nodes[0])[0]
condition_b = gbsegs[0].nodes[-1].eq_fast(gbsegs[1].nodes[-1])[0]

if condition_a:
    i_precede, flip_needed = True, False
if condition_a:
    i_precede, flip_needed = True, True
if not condition_a and not condition_b:
    i_precede, flip_needed = False, False


gbsegs[0].do_i_precede(gbsegs[1])
gbsegs[0].do_i_precede(gbsegs[2])


for i, gbs in enumerate(gbsegs[1:], start=1):
    print(gbsegs[0].do_i_precede(gbs))




indices = [0]
for i, gbseg in enumerate(gbsegs[1:], start=1):





sorted_gbsegs = sort_gbsegs(gbsegs)
create_polygon_from_segments(sorted_gbsegs)



gbsegs[0].get_node_coords()
gbsegs[1].get_node_coords()
gbsegs[2].get_node_coords()

sorted_gbsegs = sort_gbsegs(gbsegs)

sorted_gbsegs[0].lines
sorted_gbsegs[0].get_node_coords()

sorted_gbsegs[1].lines
sorted_gbsegs[1].get_node_coords()

sorted_gbsegs[2].lines
sorted_gbsegs[2].get_node_coords()

for .

create_coords_from_segments(sorted_gbsegs)

create_polygon_from_segments(sorted_gbsegs)
##############################################################
# #######################################################
newlist = [gbsegs[0]]
Start, End = gbsegs[0].nodes[0], gbsegs[0].nodes[-1]
for i, seg in enumerate(gbsegs[1:], start=1):
    i = 1
    seg = gbsegs[i]
    # ----------------------
    segstart, segend = seg.nodes[0], seg.nodes[-1]
    condition_a = End.eq_fast(segstart)[0]
    condition_b = End.eq_fast(segend)[0]
    if condition_a:
        print('Condition 1 has been satisfied')
        pass
    elif condition_b:
        print('Condition 2 has been satisfied. gbseg will be flipped.')
        seg.flip()
    # --------------------
    if condition_a or condition_b:
        newlist.append(seg)
        Start, End = seg.nodes
    else:
        if i == len(gbsegs)-1:
            newlist.append(seg)



newlist[0].get_node_coords()
newlist[1].get_node_coords()
newlist[2].get_node_coords()

'''
After sorting, some of the segments (last one!!) will need to be flipped
upside down to ensure same ordering as in gbpoints.
'''
for gid in geom.gid:
    gid = 1
    segs = sorted_segs[gid]
    segs[0].nodes
    segs[-1].nodes

if AUTO_PLOT_EVERYTHING:
    for gid in geom.gid:
        gid = 1
        fig, ax = plt.subplots()
        _, subset_indices = sort_subsets_by_original_order(gbmullines_grain_wise[gid].get_node_coords(),
                                       [seg.get_node_coords() for seg in consolidated_segments[gid]])
        sorted_segs = [consolidated_segments[gid][ssind] for ssind in subset_indices]
        for i, ss in enumerate(sorted_segs, start=0):
            ss.plot(ax)
            cen = ss.centroid
            ax.text(cen[0], cen[1], i, color='brown', fontsize=12, fontweight='bold')

def create_coords_from_segments(segments):
    # segments = gbsegs
    coords = segments[0].get_node_coords()
    for seg in segments[1:]:
        coords = np.vstack((coords, seg.get_node_coords()[1:]))
    return coords

def create_polygon_from_segments(segments):
    coords = create_coords_from_segments(segments)
    return Polygon(coords)

def create_polygon_from_coords(coords):
    return Polygon(coords)

gid = 6
create_polygon_from_coords(gbmullines_grain_wise[gid].get_node_coords())


create_polygon_from_coords(gbsegments[gid])

# --------------------------------------------
simplified_grains = []
for gid in geom.gid:
    jnp_all_sorted_coords[gid]
    simplified_grains.append(create_polygon_from_coords(jnp_all_sorted_coords[gid]))

sgs = MultiPolygon(simplified_grains)
# plot_multipolygon(sgs, invert_y=True)

detailed_grains = []
for gid in geom.gid:
    jnp_all_sorted_coords[gid]
    detailed_grains.append(create_polygon_from_coords(gbmullines_grain_wise[gid].get_node_coords()))

dgs = MultiPolygon(detailed_grains)
# plot_multipolygon(dgs, invert_y=True)

def plot_multipolygon(multipolygon, ax=None, color="blue", alpha=0.5, invert_y=False, **kwargs):
    """Plots a Shapely MultiPolygon.

    Args:
        multipolygon (MultiPolygon): The MultiPolygon object to plot.
        ax (matplotlib.axes.Axes, optional): The axes to plot on. If None, the current axes will be used.
        color (str, optional): The color of the polygons. Defaults to "blue".
        alpha (float, optional): The transparency of the polygons. Defaults to 0.5.
        **kwargs: Additional keyword arguments to pass to PolygonPatch.
    """
    from descartes import PolygonPatch  # For easier plotting of Shapely polygons
    if ax is None:
        ax = plt.gca()  # Get the current axes if not provided

    for polygon in multipolygon.geoms:
        # Use PolygonPatch for easy plotting
        patch = PolygonPatch(polygon, fc=color, ec="black", alpha=alpha, **kwargs)
        ax.add_patch(patch)

    # Set plot limits based on the multipolygon's extent
    minx, miny, maxx, maxy = multipolygon.bounds
    ax.set_xlim(minx - 0.1, maxx + 0.1)  # Add a bit of padding
    ax.set_ylim(miny - 0.1, maxy + 0.1)
    ax.set_aspect('equal')  # Ensure equal aspect ratio
    if invert_y:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymax, ymin)



plot_multipolygon(MultiPolygon(simplified_grains), invert_y=True)
fig, ax = plt.subplots()
ax.imshow(geom.lgi)



gid = 1
consolidated_segments[gid][7].plot()
segment = consolidated_segments[gid][7].get_node_coords()



import numpy as np

def moving_average(data, window_size):
    """Compute the moving average of the given data with the specified window size."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def mean_coordinates(coords, window_size):
    """
    Smooths the given 2D numpy array of coordinates using a moving average.

    Parameters:
    coords (numpy.ndarray): A 2D numpy array of shape (n, 2) where n is the number of points.
    window_size (int): The window size for the moving average.

    Returns:
    numpy.ndarray: A 2D numpy array of the smoothed coordinates.
    """
    # Check if there are enough points for the moving average
    if len(coords) < window_size:
        return coords  # Return the original coordinates if not enough points

    # Separate the coordinates into x and y components
    x = coords[:, 0]
    y = coords[:, 1]

    # Apply moving average to the x and y components separately
    x_smooth = moving_average(x, window_size)
    y_smooth = moving_average(y, window_size)

    # Add the original end points to the smoothed coordinates if there are enough points
    if len(x_smooth) > 0 and len(y_smooth) > 0:
        smoothed_coords = np.vstack([
            [x[0], y[0]],  # Start point
            np.column_stack([x_smooth, y_smooth]),
            [x[-1], y[-1]]  # End point
        ])
    else:
        smoothed_coords = coords  # If not enough points, use original coordinates

    return smoothed_coords


import matplotlib.pyplot as plt

def plot_coordinates(original, smoothed):
    """
    Plots the original and smoothed coordinates.

    Parameters:
    original (numpy.ndarray): A 2D numpy array of the original coordinates.
    smoothed (numpy.ndarray): A 2D numpy array of the smoothed coordinates.
    """
    fig, ax = plt.subplots()

    # Plot original coordinates
    ax.plot(original[:, 0], original[:, 1], 'b-', label='Original')

    # Plot smoothed coordinates
    ax.plot(smoothed[:, 0], smoothed[:, 1], 'r-', label='Smoothed')

    ax.legend()
    plt.show()

from upxo.geoEntities.mulsline2d import MSline2d

smoothing_level = 4

plt.imshow(geom.lgi)
for gid in geom.gid:
    for gbs in consolidated_segments[gid]:
        gbs_coords = gbs.get_node_coords()
        if gbs.nnodes > 4:
            smoothed_coords = mean_coordinates(gbs_coords, smoothing_level)
            plt.plot(smoothed_coords[:, 0], smoothed_coords[:, 1], '-k')
        else:
            plt.plot(gbs_coords[:, 0], gbs_coords[:, 1], '-k')
ymin, ymax = ax.get_ylim()
ax.set_ylim(ymax, ymin)

smoothed_segments_coords = {gid: [] for gid in geom.gid}
for gid in geom.gid:
    for gbs in consolidated_segments[gid]:
        gbs_coords = gbs.get_node_coords()
        if gbs.nnodes > 4:
            smoothed_coords = mean_coordinates(gbs_coords, smoothing_level)
            smoothed_segments_coords[gid].append(smoothed_coords)
        else:
            smoothed_segments_coords[gid].append(gbs_coords)

smoothed_segments = {gid: [] for gid in geom.gid}
for gid in geom.gid:
    for gbs in consolidated_segments[gid]:
        gbs_coords = gbs.get_node_coords()
        if gbs.nnodes > 4:
            smoothed_coords = mean_coordinates(gbs_coords, smoothing_level)
            segment = MSline2d.by_nodes([Point2d(sc[0], sc[1]) for sc in smoothed_coords],
                                         close=False)
            smoothed_segments[gid].append(segment)
        else:
            smoothed_segments[gid].append(gbs)

fig, ax = plt.subplots()
for gid in geom.gid:
    for seg in smoothed_segments[gid]:
        seg.plot(ax)

#
from upxo.pxtal.geogs import geogs2d
gs = geogs2d((xmin, xmax, ymin, ymax), smoothed_segments)
gs.bounds
gs.mslines2d
gs.gid
