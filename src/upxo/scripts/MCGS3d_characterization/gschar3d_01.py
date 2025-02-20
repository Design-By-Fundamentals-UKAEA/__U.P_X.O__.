# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 05:50:09 2024
@author: Dr. Sunil Anandatheertha
"""
from upxo.ggrowth.mcgs import mcgs
from scipy.spatial import cKDTree
import numpy as np
import pyvista as pv
from copy import deepcopy
import matplotlib.pyplot as plt
import upxo._sup.data_ops as DO
import upxo._sup.dataTypeHandlers as dth
## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## --
AUTO_PLOT_EVERYTHING = False
## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## --
pxt = mcgs()
pxt.simulate(verbose=False)
tslice = 50
gstslice = pxt.gs[tslice]
gstslice.char_morphology_of_grains(label_str_order=1)
gstslice.set_mprops()
gstslice.plot_gs_pvvox()
gstslice.plot_grains(gstslice.get_largest_gids(), opacity=1, lw=3)
gstslice.clean_gs_GMD_by_source_erosion_v1(prop='volnv', threshold=40,
                                           parameter_metric='mean',
                                           reset_pvgrid_every_iter=False,
                                           find_neigh_every_iter=False,
                                           find_grvox_every_iter=False,
                                           find_grspabnds_every_iter=False)
gstslice.n
gstslice.viz_mesh_slice_ortho(scalar='lgi', cmap='viridis', style='surface', throw=False, pvp=None)
'''
# Effect of connectivity type parameter on grain structure
connectivity = 1
gstslice.char_morphology_of_grains(label_str_order=connectivity)
# gstslice.pvgrid.plot()
gstslice.n
gstslice.set_mprops()

gids = []
vols = []
for conn in [1, 2, 3]:
    gstslice.char_morphology_of_grains(label_str_order=conn)
    gstslice.set_mprops()
    gids.append(gstslice.gid)
    vols.append(np.array(list(gstslice.mprop['volnv'].values())))

plt.figure()
sns.histplot(vols[0], label='Conn. 1', kde=True)
sns.histplot(vols[1], label='Conn. 2', kde=True)
sns.histplot(vols[2], label='Conn. 3', kde=True)
plt.legend()
plt.xlabel('Grain volume')
plt.ylabel('Count')
plt.show()
'''

'''
tslice = 19
gstslice = pxt.gs[tslice]
# gstslice.pvgrid.plot()
# ---- A ---- #
connectivity = 1
gstslice.char_morphology_of_grains(label_str_order=connectivity)
gstslice.set_mprops()
vols1_bf = np.array(list(gstslice.mprop['volnv'].values()))
gstslice.n
# CLEANING
vols1_af = np.array(list(gstslice.mprop['volnv'].values()))
gstslice.n
# ---- B ---- #
connectivity = 2
gstslice.char_morphology_of_grains(label_str_order=connectivity)
gstslice.set_mprops()
vols2_bf = np.array(list(gstslice.mprop['volnv'].values()))
gstslice.n
# CLEANING
vols2_af = np.array(list(gstslice.mprop['volnv'].values()))
gstslice.n
# ---- C ---- #
connectivity = 3
gstslice.char_morphology_of_grains(label_str_order=connectivity)
gstslice.set_mprops()
vols3_bf = np.array(list(gstslice.mprop['volnv'].values()))
gstslice.n
# CLEANING
vols3_af = np.array(list(gstslice.mprop['volnv'].values()))
gstslice.n

plt.figure()
sns.kdeplot(vols1_bf, color='red', label='Conn. 1. Before cleaning', clip=[0, 200], cumulative=False,
            linestyle="--", linewidth=1, marker='s', markevery=10, markersize=5, mfc='w', mec='r')
sns.kdeplot(vols2_bf, color='blue', label='Conn. 2. Before cleaning', clip=[0, 200], cumulative=False,
            linestyle="--", linewidth=1, marker='o', markevery=10, markersize=5, mfc='w', mec='b')
sns.kdeplot(vols3_bf, color='green', label='Conn. 3. Before cleaning', clip=[0, 200], cumulative=False,
            linestyle="--", linewidth=1, marker='<', markevery=10, markersize=5, mfc='w', mec='g')

sns.kdeplot(vols1_af, color='red', label='Conn. 1. After cleaning. LVTh=5', clip=[0, 200], cumulative=False,
            linestyle="-", linewidth=1, marker='s', markevery=10, markersize=5, mfc='r', mec='r')
sns.kdeplot(vols2_af, color='blue', label='Conn. 2. After cleaning. LVTh=5', clip=[0, 200], cumulative=False,
            linestyle="-", linewidth=1, marker='o', markevery=10, markersize=5, mfc='b', mec='b')
sns.kdeplot(vols3_af, color='green', label='Conn. 3. After cleaning. LVTh=5', clip=[0, 200], cumulative=False,
            linestyle="-", linewidth=1, marker='<', markevery=10, markersize=5, mfc='g', mec='g')
plt.legend()
plt.xlabel('Grain volume')
plt.ylabel('Density')
plt.show()

'''


gstslice.fit_ellipsoids(routine=1, regularize_data=True)
gstslice.set_mprop_arellfit(metric='max', calculate_efits=False,
                     efit_routine=1, efit_regularize_data=True)

gstslice.set_mprop_eqdia(base_size_spec='volnv')
gstslice.mprop['eqdia']
gstslice.mprop['arellfit']
# ######################################################################
# ######################################################################
loci = [0, 0, 0]
locj = [9, 9, 9]
gstslice.get_values_along_line(loci, locj, scalars='lgi')
gstslice.get_igs_properties_along_line(loci, locj, scalars='lgi')
gstslice.get_igs_along_line(loci, locj, metric='mean',
                            minimum=True, maximum=True,
                            std=True, variance=True)

gstslice.get_opposing_points_on_gs_bound_planes(plane='z',
                                           start_skip1=0, start_skip2=0,
                                           incr1=3, incr2=3,
                                           inclination='constant',
                                           inclination_extent=2,
                                           shift_seperately=False,
                                           shift_starts=True,
                                           shift_ends=True,
                                           start_shift=2, end_shift=3)

gstslice.get_igs_along_lines(metric='mean', minimum=True, maximum=True,
                             std=True, variance=True, lines_gen_method=1,
                             lines_kwargs1={'plane': 'z',
                                            'start_skip1': 0, 'start_skip2': 0,
                                            'incr1': 2, 'incr2': 2,
                                            'inclination': 'constant',
                                            'inclination_extent': 2,
                                            'shift_seperately': False,
                                            'shift_starts': False,
                                            'shift_ends': False,
                                            'start_shift': 0, 'end_shift': 0})
# ######################################################################
# ######################################################################
gstslice.pvgrid.plot_over_line()
gstslice.measure_igs_between_locations([1, 9, 0], [1, 9, 9], scalars='lgi')
# ######################################################################
# ######################################################################
gstslice.find_grains_by_mprop_range('volnv', 10, 15)
# ######################################################################
# ######################################################################
# gstslice.plot_grains(gstslice.find_grains_by_nvoxels(nvoxels=8), opacity=1.0)
# ######################################################################
# ######################################################################
gstslice.plot_grains(gstslice.get_largest_gids(), opacity=1.0, lw=3)
gstslice.plot_grains(gstslice.gid, opacity=1.0, lw=3)
gstslice.plot_grains(gstslice.gpos['internal'], opacity=1.0, lw=3)
gstslice.plot_grains(gstslice.gid, opacity=1.0, lw=3)
# ######################################################################
# ######################################################################
from upxo._sup import data_ops as DO
threshold = 5
merge_by_property = 'neigh_gid_volume'
parameter_metric = 'mean'
print(40*'-', f'\nMerging by recursive erosion::: property: {merge_by_property}, metric: {parameter_metric}')
_mvg_flag_ = False
_iteration_ = 1
while not _mvg_flag_:
    print(50*'=', f'\nIteration number: {_iteration_}')
    for tnp in np.arange(1, threshold+1, 1):
        print('\n', 40*'+', f'\n           Threshold value: {tnp}\n', 40*'+')
        mvg = gstslice.find_grains_by_nvoxels(nvoxels=tnp)
        if mvg.size == 0:
            continue
        """Break up mvg (multi-voxel grain) into multiple single voxel
        grains."""
        for gid in mvg:
            locations = np.argwhere(gstslice.lgi == gid)
            vx_neigh_gids = [list(gstslice.get_neigh_grains_next_to_location(loc))
                             for loc in locations]
            vx_neigh_gids_nneighs = [len(_) for _ in vx_neigh_gids]
            if merge_by_property == 'neigh_gid_volume':
                vx_neigh_vols = [np.array([gstslice.nvoxels[_gid_]
                                           for _gid_ in vx_neigh_gid_set])
                                 for vx_neigh_gid_set in vx_neigh_gids]
                gid_locs_in_array = [DO.find_closest_locations(vx_neigh_vol,
                                                               parameter_metric)
                                     for vx_neigh_vol in vx_neigh_vols]
                sink_gids = [vx_neigh_gid[_gla_[0]]
                             for vx_neigh_gid, _gla_ in zip(vx_neigh_gids,
                                                            gid_locs_in_array)]
                """ Now that we have the sink gids, for each pixel of the mvg,
                we will now merge the respective pixels of mvg with the
                corresponding sink gids. """
                for location, sink_gid in zip(locations, sink_gids):
                    gstslice.lgi[location[0], location[1], location[2]] = sink_gid
        # Re-number the lgi
        old_gids = np.unique(gstslice.lgi)
        new_gids = np.arange(start=1, stop=np.unique(gstslice.lgi).size+1, step=1)
        for og, ng in zip(old_gids, new_gids):
            gstslice.lgi[gstslice.lgi == og] = ng
        gstslice.set_gid()
        gstslice.calc_num_grains()
        gstslice.set_mprops()
        # gstslice.set_nvoxels()
        gstslice.make_pvgrid()
        gstslice.add_scalar_field_to_pvgrid(sf_name="lgi", sf_value=gstslice.lgi)
        gstslice.find_neigh_gid(verbose=False)
        gstslice.find_grain_voxel_locs()  # gstslice.grain_locs
        gstslice.find_spatial_bounds_of_grains()  # gstslice.spbound, gstslice.spboundex

    _iteration_ += 1
    _mvg_flag_ = all([gstslice.find_grains_by_nvoxels(nvoxels=tnp).size == 0
                      for tnp in range(threshold+1)])
gstslice.set_grain_positions(verbose=False)
# ######################################################################
# ######################################################################
# ######################################################################
gstslice.plot_grains(gstslice.get_largest_gids(), opacity=1.0, lw=3)
# ######################################################################
# ######################################################################
# ######################################################################
gstslice.neigh_gid[gid]
gstslice.spbound
# gstslice.vox_size
# gstslice.gid_s
# gstslice.s_gid
# gstslice.s_n
# gstslice.n
gstslice.nvoxels
gstslice.nvoxels_values
gstslice.get_largest_gids()
gstslice.get_smallest_gids()

gstslice.find_grain_voxel_locs()  # gstslice.grain_locs
gstslice.find_spatial_bounds_of_grains()  # gstslice.spbound, gstslice.spboundex
gstslice.set_grain_positions()
'''
# DEtails of the grain position values

1. We can look at pixel coordinates of grain sharing one of its pixel
with the origin (x=0, y=0, z=0)
gstslice.grain_locs[list(gstslice.gpos['corner']['left_back_bottom'])[0]]
'''

'''
We will now inspect gstslice.gid_imap_keys
gstslice.gid_imap_keys.keys()

This contains a list of foward and reverse maps of grian position names to
their respective position IDs. The IDs are itegrers. Do inspect values of each
of the above keys to know the ID-key pair maps. This is mainly to aid
programming.

When we know the position name of a grain, which is
When the position ID of a grain is known (say 16), we can know its loca
'''
gstslice.assign_gid_imap_keys()
gstslice.create_neigh_gid_pair_ids()
gstslice.build_gbp(verbose=True)
gstslice.build_gbp_id_mappings()
gstslice.find_gbsp()
'''
# Plot gids having a maximum presence.
gstslice.plot_grains(gstslice.get_max_presence_gids())

gstslice.get_max_presence_gids()

gstslice.plot_grains(gstslice.gpos['corner']['all'])
gstslice.plot_grains(gstslice.get_largest_gids())

areas = gstslice.nvoxels_values()
gids = []
for a in areas[np.argsort(areas)][-5:]:
    gids.append(np.argwhere(areas == a)[0][0]+1)
gstslice.plot_grains(gids)

gstslice.gid_pair_ids  # interface no or pair id no as key
gstslice.gid_pair_ids_rev  # neigh gids pairs (tuple) as key (lr by default)
gstslice.gbsurf_pids_vox  # interface no or pair id no as key
gstslice.gbp_ids  # gid as keys
gstslice.Ggbp_all  # gid as keys
gstslice.gbp_id_maps  # gbp coord as keys
gstslice.gbpstack  # Array of gbp coords
gstslice.gbpids  # Array of gbp coord ids. Starts from 0.
gstslice.gid_pair_ids_unique_lr  # Array of neigh-gid pairs (left-right)
gstslice.gid_pair_ids_unique_rl  # Array of neigh-gid pairs (right-left)
'''
# ######################################################################
"""
Task: plota grain and one of its neighbours.
Figure 1: plot all grain boundary points of core grain.
Figure 2: plot only those grain boundary points between core grain and the
    the neighbour being considered.
"""
gstslice.setup_gid_pair_gbp_IDs_DS()
gstslice.set_gid_pair_gbp_IDs(verbose=True)
gstslice.build_gid__gid_pair_IDs()

gstslice.plot_grains([1, 65])

from skimage.measure import regionprops
pr = regionprops(gstslice.lgi)
dir(pr[0])
pr[0].coords

dir(gstslice.pvgrid.threshold([10, 10], scalars='lgi'))
gstslice.pvgrid.threshold([10, 10], scalars='lgi').plot_curvature()

gstslice.pvgrid.threshold([10, 10], scalars='lgi').surface_indices()

gstslice.pvgrid.threshold([10, 10], scalars='lgi').extract_feature_edges().plot()
gstslice.pvgrid.threshold([10, 10], scalars='lgi').outline_corners().plot()
gstslice.pvgrid.threshold([10, 10], scalars='lgi').outline().plot()
gstslice.pvgrid.threshold([10, 10], scalars='lgi').points

gstslice.neigh_gid[1]
surf1 = gstslice.pvgrid.threshold([1, 1], scalars='lgi').extract_geometry().smooth(n_iter=0).triangulate()
surf2 = gstslice.pvgrid.threshold([65, 65], scalars='lgi').extract_geometry().smooth(n_iter=0).triangulate()

surf1.boolean_intersection(surf2).plot()

# gstslice.gid_pair_gbp_IDs
# gstslice.gid_gpid
gid = gstslice.get_largest_gids()[0]
# gstslice.gid_pair_ids
gid_pair_ids = list(gstslice.gid_gpid[gid])
data = {'cores': [gid], 'others': [gstslice.neigh_gid[gid]]}
coord_sets = {id_pair: gstslice.gbpstack[gstslice.gid_pair_gbp_IDs[id_pair]]
              for id_pair in gid_pair_ids}
gstslice.plot_grain_sets(data=data, scalar='lgi',
                         plot_coords=True,
                         coords=coord_sets,
                         opacities=[1.00, 0.50, 0.25, 0.50],
                         pvp=None, cmap='viridis',
                         style='wireframe', show_edges=True, lw=1,
                         opacity=1, view=None, scalar_bar_args=None,
                         axis_labels = ['001', '010', '100'], throw=False)
# ######################################################################
# ######################################################################
# ######################################################################
# ######################################################################
# ######################################################################
# ######################################################################
# ######################################################################
# ######################################################################
"""
Grain boundary points of every grain is:   gstslice.Ggbp_all
All grain boundary pints are:   gstslice.gbpstack
All grain boundary point IDs are:   gstslice.gbpids

# Grain boundaryt point coordinaes to gbp ID value mapping
gstslice.gbp_id_maps

# Grain boundary point IDs for every gid.
gstslice.gbp_ids

# Coordinates of grain boundary points of each gid
gstslice.Ggbp_all

# Coordinates of all grain boundary points
gstslice.gbpstack[0]

gstslice.gbp_id_maps[tuple(gstslice.gbpstack[0])]

"""
core_gid = 1
core_neigh_gbp_ids = {(core_gid, neigh_gid): None for neigh_gid in gstslice.neigh_gid[core_gid]}


gstslice.gid_pair_ids
gstslice.Ggbp_all
gstslice.gbp_ids
gstslice.gbpstack

gstslice.gid_pair_ids[1]
gidl = gstslice.gid_pair_ids[1][0]
gidr = gstslice.gid_pair_ids[1][1]
gstslice.gbp_ids[gidl].intersection(gstslice.gbp_ids[gidr])
# ---------------------------------------
# ---------------------------------------
gstslice.gbsurf_pids_vox
