# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 06:54:36 2024

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
from shapely.geometry import LineString, MultiLineString
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


#gid = 1
#A = np.vstack(geom.polygons[gid-1].exterior.coords.xy).T
#B = np.vstack(geom.neigh_pols[gid][0].exterior.coords.xy).T

#from upxo._sup.data_ops import find_common_coordinates

#find_common_coordinates(A, B)


#np.array(geom.raster_img_polygonisation_results[gid-1][0][0]['coordinates'][0])


#relloc_flag = [is_boundary_polygon(geom.gsmp['raw'], pol) for pol in geom.polygons]



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

border_grain_gids = np.argwhere(border_grain_flags).squeeze()+1
internal_grain_gids = np.argwhere(internal_grain_flags).squeeze()+1
corner_grain_gids = np.argwhere(corner_grain_flags).squeeze()+1
bl_grain_gids = np.argwhere(bl_grain_flags).squeeze()+1
tl_grain_gids = np.argwhere(tl_grain_flags).squeeze()+1
br_grain_gids = np.argwhere(br_grain_flags).squeeze()+1
tr_grain_gids = np.argwhere(tr_grain_flags).squeeze()+1

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
# WORKING ------------------------------ >>>>>>
gid = 1
# Build the jnp coordinates which are not there in gbp array.
jnp_indices_toinsert = []
for i, jnp in enumerate(jnp_all_coords[jnp_grain_wise_indices[gid]]):
    if not is_a_in_b(jnp, gbp_grain_wise_coords[gid]):
        '''This means jnp should be inserted into gbp_grain_wise_coords[gid]
        at the right place.'''
        jnp_indices_toinsert.append(i)
jnps_to_insert = jnp_all_coords[jnp_grain_wise_indices[gid]][jnp_indices_toinsert]

if jnps_to_insert.ndim == 1 and jnps_to_insert.size > 0:
    jnps_to_insert = jnps_to_insert[np.newaxis]

from scipy.spatial.distance import cdist
sinkarray = gbp_grain_wise_coords[gid]
# ---------------
for LOCATION in jnp_indices_toinsert:
    # LOCATION = jnp_indices_toinsert[2]
    toinsert = jnp_all_coords[jnp_grain_wise_indices[gid]][LOCATION][np.newaxis]

    distances = cdist(toinsert, gbp_grain_wise_coords[gid]).squeeze()
    neigh_points_indices = np.argsort(distances)[:2]
    neigh_points = sinkarray[neigh_points_indices]
    '''np.argsort(distances) returns neigh_points_indices with no relation
    to actual locations of points in the gbp_grain_wise_coords[gid]. neigh_points
    therefore need not be in the same order as contained in sinkarray. Hence, the
    following operations are carried out is done to address this.'''
    np0loc = find_coorda_loc_in_coords_arrayb(neigh_points[0], sinkarray)
    np1loc = find_coorda_loc_in_coords_arrayb(neigh_points[1], sinkarray)
    if np0loc > np1loc:
        # This is symptom when incoprrect order is presemt.
        # This is caught and the order is flipped.
        neigh_points_indices = np.flip(neigh_points_indices)
    neigh_points = sinkarray[neigh_points_indices]  # Corectly ordered neigh_points
    ''' Build sandwhich layers 1, 2 and 3.
    Layer 1: All points in sinkarray from start, upto and contaning the left
        neighbour of the point to be inserted: sinkarray_left
    Layer 2: Acrual point to be inserted. No need to build this sepoerately as this
        is contained already in the name of toinsert.
    Layer 3: All points in sinkarray from and containing right neighbour of the
        point to be inserted to the end: sinkarray_right
    '''
    sinkarray_left = sinkarray[:neigh_points_indices[0]+1]
    sinkarray_right = sinkarray[neigh_points_indices[1]:]
    '''Using these layers, we re-build the sinkarray by stacking the 3 layers.'''
    sinkarray = np.vstack((sinkarray_left, toinsert, sinkarray_right))


jnp_indices_toinsert_ = []
for i, jnp in enumerate(jnp_all_coords[jnp_grain_wise_indices[gid]]):
    if not is_a_in_b(jnp, sinkarray):
        jnp_indices_toinsert_.append(i)
print(jnp_indices_toinsert_)

plt.imshow(geom.lgi, cmap='viridis')
jnp = jnp_all_coords[jnp_grain_wise_indices[gid]]
plt.plot(jnp[:, 0], jnp[:, 1], 'ro', ms=10, label='Junction points', alpha=0.8)
gbp = gbp_grain_wise_coords[gid]
plt.plot(gbp[:, 0], gbp[:, 1], '-wx', mec='brown', mew=1.0, ms=10, label='GB points b/f correction')
plt.plot(sinkarray[:, 0], sinkarray[:, 1], '-.cs', lw=1, ms=6, mfc='none', mec='cyan', mew=1.0, label='GB points a/f correction')
centroid = sinkarray.mean(axis=0)
plt.plot(centroid[0], centroid[1], '^', ms=12, color='none', mec='orange', mew=1.5, label='centroid')
plt.text(centroid[0], centroid[1], gid, fontsize=12, fontweight='bold', color='white')
plt.legend(loc='upper right', facecolor='gray', framealpha=1)

# <<<<------------------------------ WORKING

# ============ ALTERNATIVE AS ABOVE HAS A BUG ==============
# WORKING ------------------------------ >>>>>>
gbp_all_coords
gbp_all_upxo
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

gid = 3
gbp_grain_wise_points[gid].tolist()
gbmullines_grain_wise[gid].nodes

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
    if len(jnps_to_insert) > 0:
        gbmullines_grain_wise[gid].add_nodes(jnps_to_insert)

for gid in geom.gid:
    gbmullines_grain_wise[gid].close(reclose=False)


plt.figure()
gid = 1
coord = gbmullines_grain_wise[gid].get_node_coords()
plt.plot(coord[:, 0], coord[:, 1])
jnp = jnp_all_coords[jnp_grain_wise_indices[gid]]
plt.plot(jnp[:, 0], jnp[:, 1], 'ro', mfc='c', alpha=0.5)
gbp = gbp_grain_wise_coords[gid]
plt.plot(gbp[:, 0], gbp[:, 1], 'kx')


plt.figure()
plt.imshow(geom.lgi)
for gid in geom.gid:
    coord = gbmullines_grain_wise[gid].get_node_coords()
    plt.plot(coord[:, 0], coord[:, 1], '-k.')


    jnp = jnp_all_coords[jnp_grain_wise_indices[gid]]
    plt.plot(jnp[:, 0], jnp[:, 1], 'ro', mfc='c', alpha=0.5)



gbp_grain_wise_coords[gid]
gbp_grain_wise_coords
gbp_grain_wise_indices

from upxo.geoEntities.mulsline2d import MSline2d
from upxo.geoEntities.point2d import Point2d
nodes = [Point2d(0,0), Point2d(1,1), Point2d(2,2), Point2d(3,3), Point2d(5,5)]
MSline2d.by_nodes(nodes, close=False)

nodes = gbp_grain_wise_points[gid].tolist()
MSline2d.by_nodes(nodes, close=True)


from upxo.geoEntities.mulsline2d import MSline2d
'''We will start by first creating UPXO line objects for grain boundaries.'''
# Build lines from gbp_grain_wise_coordinates.
gbmullines_grain_wise = {gid: [] for gid in geom.gid}
for gid in geom.gid:
    nodes = list(gbp_grain_wise_points[gid])
    gbmullines_grain_wise[gid] = MSline2d.by_nodes(nodes, close=True)
# -------------------------------
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
    mulline = gbmullines_grain_wise[gid]
    mulline.add_nodes(jnps_to_insert)


gbmullines_grain_wise[gid].plot()
gbmullines_grain_wise[gid].nodes
c = gbp_grain_wise_coords[gid]
plt.plot(c[:, 0], c[:, 1])
gbp_all_coords[gbp_grain_wise_indices[gid]]
gbp_grain_wise_indices[gid]

MSline2d.by_nodes(gbp_grain_wise_coords[gid], close=True)
# <<<<------------------------------ WORKING
for gid in geom.gid:
    plt.figure()
    plt.plot(gbp_grain_wise_coords[gid][:, 0], gbp_grain_wise_coords[gid][:, 1], '-k.')
    jnpoints = jnp_all_coords[jnp_grain_wise_indices[gid]]
    plt.plot(jnpoints[:, 0], jnpoints[:, 1], 'ro')
# #############################################################################
from upxo.geoEntities.sline2d import Sline2d as sl2d
e = sl2d.by_coord([-1, 0], [1, 0])

e.fully_contains_point(Point2d(0.0, 0.0))
e.fully_contains_point(Point2d(-1.0, 0.0))
e.fully_contains_point(Point2d(0.0, 1.0))

e.fully_contains_point(Point2d(-0.5, 0))

line = sl2d(0,0, 1,0)
line.split(method='p2d', divider=Point2d(0.05, 0), saa=True, throw=True, update='pntb')


# #############################################################################
# #############################################################################
# #############################################################################

"""Build indices of gbp for every grain. Indices will be from gbp_all_coords."""
gbp_grain_wise_coords = {}
for gid in geom.gid:
    gbp_grain_wise_coords[gid] = []
    pol_shapely = affinity.translate(geom.polygons[gid-1], xoff=-0.5, yoff=-0.5)
    for i, jnp_shapely in enumerate(gbp_all_shapely, start=0):
        if pol_shapely.touches(jnp_shapely):
            gbp_grain_wise_coords[gid].append(i)
            if




gbp_grain_wise_coords = {gid: [] for gid in geom.gid}
jnp_grain_wise_coords = {gid: [] for gid in geom.gid}
for gid in geom.gid:
   # gid = 1
    pol_shapely = affinity.translate(geom.polygons[gid-1], xoff=-0.5, yoff=-0.5)
    # _xy_ = np.vstack(pol_shapely.exterior.coords.xy).T[0]
    for i, (gbp_upxo, gbp_shapely) in enumerate(zip(gbp_all_upxo, gbp_all_shapely), start=0):
       # i = 0
        gbp_upxo, gbp_shapely = gbp_all_upxo[i], gbp_all_shapely[i]

        if pol_shapely.touches(gbp_shapely):
            gbp_grain_wise_coords[gid].append(i)
            if gbp_upxo in jnp_all_upxo_mp:
                jnp_grain_wise_coords[gid].append(i)

gbp_all_coords



plt.imshow(geom.lgi)
for gid in geom.gid:
    gbp_coord = gbp_all_coords[gbp_grain_wise_coords[gid]]
    plt.plot(gbp_coord[:, 0], gbp_coord[:, 1], 'ko', alpha=0.25)

    jnp_coord = gbp_all_coords[jnp_grain_wise_coords[gid]]
    plt.plot(jnp_coord[:, 0], jnp_coord[:, 1], 'rx')





gid = 1
from shapely import affinity
pol = affinity.translate(geom.polygons[gid-1], xoff=-0.5, yoff=-0.5)
for gbp_grain_wise_coords[gid] in

geom.polygons[gid-1].exterior.coords.xy
gbp_grain_wise_coords[gid]

_ = np.vstack(geom.polygons[gid-1].exterior.coords.xy).T
plt.plot(_[:, 0], _[:, 1], 'ko')
plt.plot(gbp_grain_wise_coords[gid][:, 0], gbp_grain_wise_coords[gid][:, 1], 'rx')



_ = np.vstack(pol.exterior.coords.xy).T
plt.plot(_[:, 0], _[:, 1], 'ko')
plt.plot(gbp_grain_wise_coords[gid][:, 0], gbp_grain_wise_coords[gid][:, 1], 'rx')




np.vstack(pol_shapely.exterior.coords.xy).T

plt.imshow(geom.lgi)
for gid in geom.gid:
    plt.plot(gbp_grain_wise_coords[gid][:, 0], gbp_grain_wise_coords[gid][:, 1], 'ko', alpha=0.25)
plt.plot(jnp_all_coords[:, 0], jnp_all_coords[:, 1], 'rx')

plt.imshow(geom.lgi)
# plt.imshow(geom.lgi == tl_grain_gids)
gid = tl_grain_gids
plt.plot(gbp_grain_wise_coords[gid][:, 0], gbp_grain_wise_coords[gid][:, 1], 'ko')
plt.plot(gbp_grain_wise_coords[gid][:, 0], gbp_grain_wise_coords[gid][:, 1], 'ko')

gbp_grain_wise_indices = {gid: [] for gid in geom.gid}
for gid in geom.gid:
    gid = tl_grain_gids
    for jnp_coord in jnp_all_coords:
        jnp_coord = jnp_all_coords[0]
        condx = gbp_grain_wise_coords[gid][:, 0] == jnp_coord[0]
        condy = gbp_grain_wise_coords[gid][:, 1] == jnp_coord[1]
        print(np.argwhere(condx & condy))
        index = np.argwhere(condx & condy)
        print(index)
        if index.size > 0:
            gbp_grain_wise_indices[gid].extend([ind for ind in index])
    gbp_grain_wise_indices.append(index)

gbp_all_coords

# #########################################################################
"""Calculate the jnp ratio"""
jnpratio = len(jnp_all_coords) / len(gbp_all_upxo)
# #########################################################################
# #########################################################################
# #########################################################################
# #########################################################################
ALLGBPCOORDS = []
for gid in geom.gid:
    _ = np.array(postpolcoords[gid-1][0][0]['coordinates'][0][:-1])-0.5
    ALLGBPCOORDS.extend(_.tolist())
ALLGBPCOORDS = np.unique(ALLGBPCOORDS, axis=0)
ALLGBPCOORDS_p2d = MPoint2d.from_xy(ALLGBPCOORDS.T)

gbp_grains_all = {}  # All grain boundary points arranged in mpoints, grain wise.
for gid in geom.gid:
    gbp_grains_all[gid] = MPoint2d.from_xy(np.array(postpolcoords[gid-1][0][0]['coordinates'][0][:-1]).T)

GRAINGBPCOORDS_p2d = {}
for gid in geom.gid:
    GRAINGBPCOORDS_p2d[gid] = []
    for allgbpcoords in ALLGBPCOORDS_p2d:
        allgbpcoords = ALLGBPCOORDS_p2d[0]
        GRAINGBPCOORDS_p2d[]

gbp_all_grains = {}
for gid in geom.gid:
    gbpoints = np.array(postpolcoords[gid-1][0][0]['coordinates'][0][:-1])-0.5
    gbpoints = gbpoints.tolist()
    # ------------------------
    chosen_points = []
    for allgbpcoords in ALLGBPCOORDS_p2d:
        for gbpoint in gbpoints:
            if gbpoint in ALLGBPCOORDS_p2d.coords:
                chosen_points
        chosen_points = [Point2d(__[0], __[1]) for __ in _]
    # ------------------------
    gbp_all_grains[gid] = [MPoint2d.from_upxo_points2d(chosen_points, zloc=0.0)]
    GRAIN = geom.polygons[gid-1]
    for jnp_, jnp_sh in zip(jnp_all_upxo, jnp_all_shapely):
        if GRAIN.touches(jnp_sh):
            if not jnp_ in gbp_all_grains[gid][0].points:
                gbp_all_grains[gid].append(jnp_)

gbp_all_grains_pass = all([len(gbp_all_grains[gid]) ==1 ])

if gbp_all_grains_pass:
    # Convert back from list to just multi-point as test has passed.
    gbp_all_grains = {gid: gbp_all_grains[gid][0] for gid in geom.gid}
    # -----------------------------
    plt.imshow(geom.lgi)
    plt.plot(JNP[:, 0], JNP[:, 1], 'ro', markersize=10, alpha=0.25)
    for gid in geom.gid:
        plt.plot(gbp_all_grains[gid].coords[:, 0], gbp_all_grains[gid].coords[:, 1], 'k.')
    # -----------------------------
    # We can now continue



JNP
gbp_grain_wise = {}
gid = 1
plt.imshow(geom.lgi)
for gid in geom.gid:
    gbp_ = np.array(postpolcoords[gid-1][0][0]['coordinates'][0][:-1])-0.5
    plt.plot(gbp_[:, 0], gbp_[:, 1], '-ko')
    gbp_grain_wise[gid] = []
    for _gbp_ in gbp_:
        _gbp_ = gbp_[0]
        _loc_ = np.argwhere((JNP[:, 0]-_gbp_[0])**2 + (JNP[:, 1]-_gbp_[1])**2 <= 1E-8)
        plt.plot(JNP[:, 0], JNP[:, 1], 'rs')

geom.polygons[0].touches


for gid in geom.gid:
    pol = geom.polygons[gid-1]
    gbp_this_grain = []
    for jnp in JNP:
        if pol.touches(jnp): GBP
            gbp_this_grain.append(jnp)

# GSJNPs = MPoint2d.from_upxo_points2d(JNP_upxo, zloc=0.0)

from_upxo_points2d
GS_GBPs_upxo = {}
for gid, gbjnp in GBJNPs.items():
    for gbp in gbp_all:
        if gsjnp in gbjnp:



print(GBJNPs[gid].points[0] in GBJNPs[gid])

JNPs.points

# #########################################################################
'''plt.imshow(geom.lgi)
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
'''
# #########################################################################
