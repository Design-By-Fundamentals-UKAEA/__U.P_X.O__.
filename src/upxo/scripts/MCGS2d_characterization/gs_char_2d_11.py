# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 16:58:32 2024

@author: rg5749
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 23:25:49 2024

@author: rg5749
"""
import cv2
import numpy as np
import gmsh
import pyvista as pv
import upxo._sup.data_ops as DO
from copy import deepcopy
from shapely import affinity

from upxo.geoEntities.mulsline2d import MSline2d
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter
from meshpy.triangle import MeshInfo, build
from upxo.ggrowth.mcgs import mcgs
from upxo.geoEntities.mulsline2d import MSline2d
from shapely.geometry import Point
from shapely.geometry import MultiPolygon, Point
from scipy.spatial import cKDTree
from shapely.geometry import Point
from upxo.geoEntities.mulsline2d import ring2d
from upxo._sup.data_ops import find_common_coordinates
from shapely.geometry import LineString, MultiLineString
from upxo.geoEntities.point2d import Point2d
from upxo.geoEntities.mulpoint2d import MPoint2d
from shapely.geometry import Point as ShPoint2d
from upxo._sup.data_ops import remove_2d_child_array_from_2d_parent_array
# ---------------------------
AUTO_PLOT_EVERYTHING = False
# ---------------------------
pxt = mcgs()
pxt.simulate()
pxt.detect_grains()
tslice = 2
pxt.char_morph_2d(tslice)
gstslice = pxt.gs[tslice]
gstslice.find_neigh()
# --------------------------------------------------------------------
plt.imshow(gstslice.lgi)
# Definition to find and mewrge single pixel grains
# STEP 1: REMOVE ANY ISLAND GRAINS
# Step 1.a. Identify the island grains and their gids
islands = np.where(np.array([len(n) for n in gstslice.neigh_gid.values()]) == 1)[0]
# ---> iteration start. @ island_grain_number
island_grain_number = 0  # <---
island = islands[island_grain_number]
gstslice.neigh_gid[island + 1]
# Step 1.b. View these island grains
gstslice.plot_grains_gids([island + 1] + list(gstslice.neigh_gid[island + 1]), cmap_name='viridis')
# Step 1.c. Note the number of grains before merging.
print(f'Number of grains before merge: {gstslice.n}')
# Step 1.d. Merge the two grains.
parent_gid = gstslice.neigh_gid[island + 1]
other_gid = island + 1
# See the size of parent grain and other grain beore merger
np.where(gstslice.lgi == parent_gid)[0].size
merge_success = gstslice.merge_two_neigh_grains(parent_gid, other_gid, check_for_neigh=False, simple_merge=True)
# See the size of parent grain and other grain after merger
np.where(gstslice.lgi == parent_gid)[0].size
np.where(gstslice.lgi == other_gid)
# Perform post merger operations
'''gstslice.perform_post_grain_merge_ops(merge_success)'''
gstslice.renumber_gid_post_merge(other_gid)

# --------------------------------------------------------------------
# STEP 2: REMOVE ANY SINGLE PIXEL GRAINS
merge_two_neigh_grains(parent_gid, other_gid, check_for_neigh=False, simple_merge=True)
gstslice.find_neigh(update_gid=True, reset_lgi=False)
# --------------------------------------------------------------------
# gstslice.neigh_gid
gstslice.find_grain_boundary_junction_points()
folder, fileName = r'D:\export_folder', 'sunil'
gstslice.export_ctf(folder, fileName, factor=1, method='nearest')
fname = r'D:\export_folder\sunil'
gstslice.set_pxtal(instance_no=1, path_filename_noext=fname)
gstslice.pxtal[1].find_gbseg1()
gstslice.pxtal[1].gbseg1
gstslice.pxtal[1].extract_gb_discrete(retrieval_method='external',
                                      chain_approximation='simple')
gstslice.pxtal[1].set_geom()
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
from upxo.pxtal.geometrification import polygonised_grain_structure as pgs
geom = pgs(gstslice.pxtal[1].lgi, gstslice.pxtal[1].gid, gstslice.pxtal[1].neigh_gid)
geom.set_up_quality_measures()
# geom.n
# gstslice.pxtal[1].n
geom.polygonize()
# geom.raster_img_polygonisation_results
geom.set_polygonization_xyoffset(0.5)
geom.make_polygonal_grains_raw()
geom.set_polygons()
geom.make_gsmp()
geom.find_neighbors()
geom.set_grain_centroids_raw()
geom.set_grain_loc_ids()
# geom.plot_gsmp(raw=True, overlay_on_lgi=True, xoffset=0.5, yoffset=0.5)
geom.gbops()
all(geom.are_grains_closed_usenodes())
all(geom.are_grains_closed_usecoords())
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
'''geom.consolidated_segments[1]
geom.sorted_segs[1]
gbsegs_can_form_rings, _ = geom.check_if_all_gbsegs_can_form_closed_rings(geom.consolidated_segments,
                                                                          _print_individual_excoord_order_=False,
                                                                          _print_statement_=True)
gbsegs_can_form_rings, _ = geom.check_if_all_gbsegs_can_form_closed_rings(geom.sorted_segs,
                                                                          _print_individual_excoord_order_=False,
                                                                          _print_statement_=True)
'''
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
geom.smooth_gbsegs(geom.GB, npasses=2,
                   max_smooth_levels=[3, 3],
                   plot=False, seriesname='kali')
geom.smooth_gbsegs(geom.smoothed['kali.1']['GB'], npasses=2,
                   max_smooth_levels=[3, 3],
                   plot=False)
geom.smooth_gbsegs(geom.smoothed['kali.2']['GB'], npasses=2,
                   max_smooth_levels=[3, 3],
                   plot=False)
geom.smoothed.keys()
geom.smoothed['kali.1'].keys()
geom.smoothed['kali.2']['GB']
geom.smoothed['kali.2']['GBCoords']

geom.plot_multipolygon(geom.POLYXTAL, lw=0.5, alpha=0.1)
geom.plot_multipolygon(geom.smoothed['kali.1']['POLYXTAL'], lw=1, alpha=0.2)
geom.plot_multipolygon(geom.smoothed['kali.2']['POLYXTAL'], lw=1, ls = ':', alpha=0.2)
geom.plot_multipolygon(geom.smoothed['kali.3']['POLYXTAL'], lw=1, ls = '--', alpha=0.2)
plt.plot(geom.JNP[:, 0], geom.JNP[:, 1], 'ko', ms=3)
plt.plot(geom.GBP_pure[:, 0], geom.GBP_pure[:, 1], '.', color='k', ms=2)
# plt.imshow(geom.lgi)
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
gid = 24
GB_smooth[gid].segids
GB_smooth[gid].segflips
GB_smooth[gid].check_closed()
GB_smooth[gid].get_coords()

GB_smooth[gid].segments

if GB_smooth[gid].segflips[GB_smooth[gid].segids.index(max(GB_smooth[gid].segids))]:
    START = GB_smooth[gid].segments[0].nodes[0]
    END = GB_smooth[gid].segments[max(GB_smooth[gid].segids)].nodes[-1]
else:
    START = GB_smooth[gid].segments[0].nodes[0]
    END = GB_smooth[gid].segments[max(GB_smooth[gid].segids)].nodes[0]
START.eq_fast(END)

print(GB_smooth[gid].segids)
print(GB_smooth[gid].check_closed())
gid += 1


coords = GBCoords_smoothed
flags = []
for gid in geom.gid:
    flag = np.abs(coords[gid][0] - coords[gid][-1]).sum() <= geom.EPS_coord_coincide
    flags.append(flag)
all(flags)
# -------------------------------

# -------------------------------
plt.imshow(geom.lgi)
geom.plot_multipolygon(geom.POLYXTAL, lw=1, alpha=0.2)
geom.plot_multipolygon(POLYXTAL_SMOOTHED, lw=2.5, alpha=0.2)
plt.plot(geom.JNP[:, 0], geom.JNP[:, 1], 'ks', ms=4)
plt.plot(geom.GBP_pure[:, 0], geom.GBP_pure[:, 1], '.', color='k', ms=3)
# -------------------------------
geom.plot_user_gbcoords(GBCoords_smoothed, lw=1.5)
geom.plot_user_gbcoords1(GBCoords_smoothed, lw=1)

plt.imshow(geom.lgi)

geom.plot_multipolygon(geom.POLYXTAL, alpha=0.2, lw=1)
geom.plot_multipolygon(POLYXTAL_SMOOTHED, alpha=0.2, lw=1.5)

GB_smooth, GBCoords_smoothed = geom.smooth_gbsegs(GB_smooth, npasses=2,
                                                  max_smooth_levels=[3, 3],
                                                  plot=False)
GRAINS_SMOOTHED, POLYXTAL_SMOOTHED = geom.construct_geometric_polyxtal_from_gbcoords(GBCoords_smoothed,
                                                                                     dtype='shapely',
                                                                                     saa=False,
                                                                                     throw=True,
                                                                                     plot_polyxtal=False)
geom.plot_multipolygon(POLYXTAL_SMOOTHED, alpha=0.2, lw=2.5)














GB_smooth[2].plot_segs()

for gbseg in GB_smooth.values():
    ax = gbseg.plot_segs(ax)

for gid in geom.gid:
    print(GBCoords_smoothed[gid][0] - GBCoords_smoothed[gid][-1])

for gid in geom.gid:
    print(geom.GBCoords[gid][0] - geom.GBCoords[gid][-1])
# ==================================================
import upxo._sup.data_ops as DO

closed = []
for gid in geom.gid:
    closed.append(DO.is_a_in_b(geom.GB[gid].get_coords()[0], geom.GB[gid].get_coords()[1:]))
closed = np.array(closed)

for open_gid in np.where(~closed)[0]:
    print(geom.GB[open_gid+1].get_coords())
    print('---------------------------------------')

np.where(~closed)[0]
gn = 38
geom.GB[gn+1].plot_segs()
plt.plot(geom.GB[gn+1].get_coords()[:, 0], geom.GB[gn+1].get_coords()[:, 1])


geom.consolidated_segments
for gid in geom.gid:
    segments = geom.consolidated_segments[gid]
    for seg in segments:
        print(DO.is_a_in_b(seg.get_node_coords()[0],
                           geom.consolidated_segments[gid].get_node_coords()[1:]))
# ==================================================

GRAINS_SMOOTHED, POLYXTAL_SMOOTHED = geom.construct_geometric_polyxtal_from_gbcoords(GBCoords_smoothed,
                                                                                     dtype='shapely',
                                                                                     saa=False,
                                                                                     throw=True,
                                                                                     plot_polyxtal=True)
geom.plot_multipolygon(geom.POLYXTAL)
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
# Start


# End
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
# Start
# End
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
# Start
# End
# JNP = deepcopy(xy)
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
# Check if all coordinates are closed and forms a ring.
postpolcoords = geom.raster_img_polygonisation_results
A = []
for gid in geom.gid:
    A.append(np.array(postpolcoords[gid-1][0][0]['coordinates'][0][0])-np.array(postpolcoords[gid-1][0][0]['coordinates'][0][-1]))
A = np.array(A)
all(A[:, 0] == A[:, 1])  # True means all coords form a ring.
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
# Start
# Assemble all grain boundary points
# End
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
# Start

# End
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
# Start

# End
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
# Start
# End
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
# Start

# End
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
if AUTO_PLOT_EVERYTHING:
    geom.plot_gsmp(raw=True, overlay_on_lgi=False, xoffset=0.5, yoffset=0.5)
    plt.imshow(geom.lgi)
    plt.plot(JNP[:, 0], JNP[:, 1], 'ro', markersize=5, alpha=1.0)
    for i, jnp in enumerate(JNP):
        plt.text(jnp[0], jnp[1], i, color = 'black')
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
# Start

# -----------------------------------------

# End
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
# Start

# End
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
# Start
# End
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
# Start

# End
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
# Start
def is_a_in_b(self, a, b):
    return any((b[:, 0] == a[0]) & (b[:, 1] == a[1]))

def find_coorda_loc_in_coords_arrayb(self, a, b):
    # DO.find_coorda_loc_in_coords_arrayb(neigh_points[1], sinkarray)
    return np.argwhere((b[:, 0] == a[0]) & (b[:, 1] == a[1]))[0][0]
# End
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
# Start

# End
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
# Start

# End
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
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

    '''gid = 3
    coord = gbmullines_grain_wise[gid].get_node_coords()
    plt.plot(coord[:, 0], coord[:, 1], '-k.')
    for i, c in enumerate(coord[:-1,:], 0):
        plt.text(c[0]+0.15, c[1]+0.15, i)
    jnp = jnp_all_coords[jnp_grain_wise_indices[gid]]
    plt.plot(jnp[:, 0], jnp[:, 1], 'ro', mfc='c', alpha=0.5)
    for i, j in enumerate(jnp):
        plt.text(j[0]+0.15, j[1], i, color='red')'''
# #############################################################################
'''def arrange_junction_point_coords(gbcoords_thisgrain, junction_points_coord):
    # Create a dictionary to map coordinates to their indices in gbcoords_thisgrain
    coord_index_map = {tuple(coord): idx for idx, coord in enumerate(gbcoords_thisgrain)}
    # Sort junction_points_coord based on their indices in gbcoords_thisgrain
    sorted_junction_points = sorted(junction_points_coord, key=lambda x: coord_index_map[tuple(x)])
    return np.array(sorted_junction_points)'''


'''def arrange_junction_points_upxo(gbpoints_thisgrain, junction_points_points):
    # Create a dictionary to map coordinates to their indices in gbpoints_thisgrain
    coord_index_map = {(point.x, point.y): idx for idx, point in enumerate(gbpoints_thisgrain)}
    # Generate a list of indices for sorting
    sorted_indices = sorted(range(len(junction_points_points)), key=lambda i: coord_index_map[(junction_points_points[i].x, junction_points_points[i].y)])
    # Sort junction_points_points based on the sorted indices
    sorted_junction_points = [junction_points_points[i] for i in sorted_indices]
    return sorted_junction_points, sorted_indices.'''
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
# Start
"""Check if nodes allways start with a junction point"""
# End
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
# Start
# End
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
# Start
# End
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
#for seg in gbsegments[11]:
#    plt.plot(seg.coords[:, 0], seg.coords[:, 1])
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
# Start
# End
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
# Start
    # return gid_pair_ids, self.gid_pair_ids_unique_lr, self.gid_pair_ids_unique_rl
# End
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
geom.neigh_gid
gbs_mid_dict = {gid: [id(seg) for seg in gbsegments[gid]] for gid in geom.gid}
gbsegments[gid]
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
# Start

## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
#def extract_gbseg_ends(gid):
    seg_ends = {}
    for count in range(len(jnp_all_sorted_coords[gid])):
        seg_ends[count] = jnp_all_sorted_coords[gid][count:count+2]
    seg_ends[count] = np.vstack((seg_ends[count],
                                 gbmullines_grain_wise[gid].get_node_coords()[-1]))
#    return seg_ends
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
# Start
# End
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
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
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
# Start
# End
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
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
# End
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
gbseg_unique = {gid: None for gid in self.gid}

pair = self.gid_pair_ids_unique_lr[0]
pair

gbsegments[pair[0]]
gbsegments[pair[1]]

gbsegments[pair[0]][0].centroid_p2dl
pair_flags = [False for _ in self.gid_pair_ids_unique_lr]
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
# Start
# End
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
# Start
# End
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
# Start
# End
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## =
# Start
# End
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## =
# Start
'''
consolidated_segments[49]
gbsegments[49]
for seg in gbsegments[49]:
    seg.plot(ax)
for cseg in consolidated_segments[49]:
    cseg.plot(ax)
'''
# End
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## =
# Start
# Assess the performance of unique grain boundary segment calculation.
# END
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## =
if AUTO_PLOT_EVERYTHING:
    fig, ax = plt.subplots()
    ax = geom.plot_gsmp(raw=True, overlay_on_lgi=True, xoffset=0.5, yoffset=0.5, ax=ax)
    ax.plot(JNP[:, 0], JNP[:, 1], 'ro', markersize=5, alpha=0.25)

if AUTO_PLOT_EVERYTHING:
    fig, ax = plt.subplots()
    ax.imshow(geom.lgi)
    for pair in self.gid_pair_ids_unique_lr:
        if GBSEGMENTS[tuple(pair)] is not None:
            for gbseg in GBSEGMENTS[tuple(pair)]:
                ax = gbseg.plot(ax=ax)
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
# Start
# End
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
"""
We see that there are segments missing from consolidated_segments[self.grain_loc_ids['bottom_left_corner']]
HOwever, gbsegments is complete. So the task is to identify wbhich segments
to transfer from gbsegments data structure into consolidated segments data
struct8ure.

This needs to be done for all the boundary grains !!!
"""
if AUTO_PLOT_EVERYTHING:
    fig, ax = plt.subplots()
    #ax.imshow(geom.lgi)
    for seg in gbsegments[self.grain_loc_ids['bottom_left_corner']]:
        seg.plot(ax)

if AUTO_PLOT_EVERYTHING:
    fig, ax = plt.subplots()
    #ax.imshow(geom.lgi)
    for seg in consolidated_segments[self.grain_loc_ids['bottom_left_corner']]:
        seg.plot(ax)

'''
gnum = self.grain_loc_ids['bottom_right_corner']
consolidated_segments[gnum]
gbsegments[gnum]
for seg in gbsegments[gnum]:
    seg.plot(ax)
for cseg in consolidated_segments[gnum]:
    cseg.plot(ax)
'''
self.grain_loc_ids.keys()
#'pure_left'
# 'pure_bottom'
# 'pure_right'
# 'pure_top'
# 'corner'
# 'bottom_left_corner'
# 'bottom_right_corner'
# 'top_right_corner'
# 'top_left_corner'
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
# Start
# End
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
# Start
# End
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
# Start
# End
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
reordering_needed = []
for gid in geom.gid:
    gbsegs = consolidated_segments[gid]
    if not gbsegs[0].nodes[0].eq_fast(gbsegs[-1].nodes[-1])[0]:
        reordering_needed.append(gid)
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
# Start
""" Useful Definitions of MulSline2D
do_i_precede(multisline2d)
do_i_proceed(multisline2d)
is_adjacent(multisline2d)
find_spatially_next_multisline2d(self, multislines2d)
"""
# End
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
# Start
# Re-assessment - segflips of the last segment.
for gid in geom.gid:
    start = GB[gid].segments[0].nodes[0]
    end0 = GB[gid].segments[-1].nodes[0]
    end1 = GB[gid].segments[-1].nodes[-1]
    condition1 = start.eq_fast(end0)[0]
    condition2 = start.eq_fast(end1)[0]
    if condition1:
        GB[gid].segflips[-1] = True
# End
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
# Start
# GB[gid].create_polygon_from_coords()
# End
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
# Start
# End
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
# Start
# End
# Call not necessary, but include the function.
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
# Start
# End
# Call not necessary, but include the function.
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
# Start
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
grain = deepcopy(GRAINS[1])
#grain_core = grain.buffer(-2.0, resolution=0, cap_style=1, join_style=1,
#                          mitre_limit=1)
# grain_core = grain_core.convex_hull
# grain_core = grain_core.simplify(tolerance=1)
# gbz = grain - grain_core
# gbz

# plot_multipolygon(POLYXTAL, invert_y=True)
# fig, ax = plt.subplots()
# ax.imshow(geom.lgi)
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
# Start
# End
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
# Start
# End
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
# Start
# all_mids, sgseg_list = AssembleGBSEGS(geom, GB)

# End
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
# Start

# End
gid = 2
midlocs = {gid: self.get_gbmid_indices_at_gid(gid, all_mids)
           for gid in geom.gid}
# sgseg_list[midlocs[gid]].tolist()
# GB[gid].segments
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
# Start
"""
Now that the individual gbsegs have been compiled into a list and the
mapping has been established between this list and the individual gbsegs in
GB data structure, we can now proceed with smoothing.
"""

# End
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
# Start

# End
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
# Start
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
# End
## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## == ## ==
mesh_size = 1
import pygmsh
import sys
geometry = pygmsh.geo.Geometry()
model = geometry.__enter__()
xtals = [model.add_polygon(GBCoords_smoothed[gid][:-1], mesh_size=mesh_size) for gid in geom.gid]
model.synchronize()
geometry.set_recombined_surfaces([xtal.surface for xtal in xtals])
gmsh.option.setNumber("Mesh.Algorithm", 8)  # Frontal-Delaunay
gmsh.option.setNumber("Mesh.RecombineAll", 1)  # Recombine triangles
gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
mesh = geometry.generate_mesh(dim=2, order=1, algorithm=6)
gmsh.write('yourfilename_i.msh')
if 'close' not in sys.argv:
    gmsh.fltk.run()

import meshio
gmsh_mesh = meshio.read("yourfilename_i.msh")
supported_cell_types = ["line", "triangle", "quad", "tetra", "hexahedron"]
filtered_cells = {
    cell_type: cells
    for cell_type, cells in gmsh_mesh.cells_dict.items()
    if cell_type in supported_cell_types
}
filtered_mesh = meshio.Mesh(
    points=gmsh_mesh.points,
    cells=filtered_cells,
    point_data=gmsh_mesh.point_data,
    cell_data=gmsh_mesh.cell_data,
    field_data=gmsh_mesh.field_data,
)
meshio.write('abaqus_input_file_upxo.inp', gmsh_mesh)
gmsh.finalize()
###############################################################################
# ===============================================================
import pygmsh
geometry = pygmsh.geo.Geometry()
model = geometry.__enter__()
for i, g in enumerate(POLYXTAL_SMOOTHED, start=0):
    coords = np.vstack((g.exterior.coords.xy[0][:-1],
                        g.exterior.coords.xy[1][:-1])).T
    model.add_polygon(coords, mesh_size=1)
model.synchronize()


import gmsh
import numpy as np
import meshio
gmsh.initialize()
model = gmsh.model
factory = model.geo
# Create Triangle Geometry
p1 = factory.addPoint(0, 0, 0)
p2 = factory.addPoint(1, 0, 0)
p3 = factory.addPoint(0.5, 1, 0)
l1 = factory.addLine(p1, p2)
l2 = factory.addLine(p2, p3)
l3 = factory.addLine(p3, p1)
cl1 = factory.addCurveLoop([l1, l2, l3])
s1 = factory.addPlaneSurface([cl1])
# Synchronize to ensure changes are applied
factory.synchronize()
# Mesh Options (Frontal-Delaunay for quads, Recombine triangles)
gmsh.option.setNumber("Mesh.Algorithm", 8)  # Frontal-Delaunay
gmsh.option.setNumber("Mesh.RecombineAll", 1)  # Recombine triangles
gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
# Generate the mesh
gmsh.model.mesh.generate(2)  # 2D mesh
# Get mesh data (assuming 2D mesh)
nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
elementTypes, elementTags, elementNodes  = gmsh.model.mesh.getElements()  # Corrected line
# Process the mesh data as needed (e.g., convert to NumPy arrays)
# Filter out quad elements (type 3 in Gmsh)
quadElementNodes = [nodes for (elemType, nodes) in zip(elementTypes, elementNodes) if elemType == 3]
# Convert filtered element nodes to NumPy array and adjust for 0-based indexing
elementNodes = np.array(quadElementNodes) - 1
# Finalize Gmsh
gmsh.finalize()
# Optional: Print or visualize the mesh data
print("Node Coordinates:")
print(nodeCoords)
print("\nElement Node Connectivity:")
print(elementNodes)
meshio.write(
    "triangle_quad_mesh.vtk",  # Filename
    meshio.Mesh(
        points=nodeCoords,
        cells={"quad": elementNodes},  # Specify cell type as "quad"
    ),
)

gmsh.finalize()



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

_, subset_indices = sort_subsets_by_original_order(gbmullines_grain_wise[gid].get_node_coords(),
                               [seg.get_node_coords() for seg in consolidated_segments[gid]])
sorted_segs = [consolidated_segments[gid][ssind] for ssind in subset_indices]

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
