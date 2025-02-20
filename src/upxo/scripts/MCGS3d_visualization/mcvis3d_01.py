# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 11:56:54 2024

@author: Dr. Sunil Anandatheertha
"""
from upxo.ggrowth.mcgs import mcgs
import numpy as np

pxt = mcgs()
pxt.simulate(verbose=False)
tslice = 8
gstslice = pxt.gs[tslice]
gstslice.find_grains(label_str_order=3)
gstslice.set_npixels()
gstslice.make_pvgrid()
gstslice.add_scalar_field_to_pvgrid(sf_name="lgi", sf_value=gstslice.lgi)
gstslice.find_neigh_gid()
gstslice.find_grain_voxel_locs()  # gstslice.grain_locs
gstslice.find_spatial_bounds_of_grains()  # gstslice.spbound, gstslice.spboundex
gstslice.set_grain_positions()
# -------------------------------------------------------------------
gstslice.plot_grains(gstslice.gpos['boundary'])
gstslice.plot_grains(gstslice.single_voxel_grains)
gstslice.plot_grains(gstslice.small_grains(vth=10))
gstslice.plot_grains(gstslice.find_grains_by_npixels(npixels=5))
# -------------------------------------------------------------------
gstslice.plot_grains(gstslice.single_voxel_grains)
gid = gstslice.single_voxel_grains[0]
data = {'cores': [gid], 'others': [gstslice.neigh_gid[gid]]}
gstslice.plot_grain_sets(data=data, scalar='lgi', plot_coords=False,
                         opacities=[1.00, 0.90, 0.75, 0.50],
                         pvp=None, cmap='viridis',
                         style='surface', show_edges=True, lw=0.5,
                         opacity=0.2, view=None, scalar_bar_args=None,
                         axis_labels=['001', '010', '100'], throw=False)
# -------------------------------------------------------------------
gstslice.plot_grains(gstslice.gpos['corner']['all'], opacity=1)
gstslice.plot_grains(gstslice.gpos['corner']['left_back_bottom'], opacity=1)
gstslice.plot_grains(gstslice.gpos['face']['bottom'], opacity=1)
gstslice.plot_grains(gstslice.gpos['internal'], opacity=1)
gstslice.plot_grains(gstslice.gpos['edges']['left'], opacity=1)

gstslice.plot_grains(gstslice.get_smallest_gids(), opacity=1)
gstslice.plot_grains(gstslice.get_largest_gids(), opacity=1)
# -------------------------------------------------------------------
gstslice.plot_gs_pvvox(cs_labels='user', _xname_='Z', _yname_='Y', _zname_='X')

gstslice.plot_scalar_field_slice(sf_name='lgi', slice_normal='x',
                                 slice_location=50, interpolation='nearest',
                                 vmin=1, vmax=None)

gstslice.plot_scalar_field_slices(sf_name='lgi', slice_normal='z',
                                  slice_location=0, interpolation='nearest',
                                  vmin=1, vmax=None,
                                  slice_start=0, slice_end=99, slice_incr=10,
                                  nrows=2, ncols=5, ax=None)

gstslice.plot_gs_pvvox_subset(gstslice.find_exbounding_cube_gid(5), alpha=0.5)

gstslice.plot_gs_pvvox_subset(gstslice.find_bounding_cube_gid(5), alpha=0.5,
                              isolate_gid=True, gid=5)
# -------------------------------------------------------------------
gid = list(gstslice.get_max_presence_gids())[0]
gid = list(gstslice.get_largest_gids())[0]
gid = list(gstslice.gpos['internal'])[0]
gstslice.neigh_gid[gid]
gstslice.plot_grains(gstslice.neigh_gid[gid]+[gid])
data = {'cores': [gid], 'others': [gstslice.neigh_gid[gid]]}
data = {'cores': [gid],
        'others': [gstslice.get_upto_nth_order_neighbors(gid, 1,
                                                         include_parent=False,
                                                         output_type='list')]}

gstslice.plot_grain_sets(data=data, scalar='lgi', plot_coords=True,
                         coords=gstslice.Ggbp_all[gid],
                         opacities=[1.00, 0.90, 0.75, 0.50],
                         pvp=None, cmap='viridis',
                         style='wireframe', show_edges=True, lw=0.5,
                         opacity=1, view=None, scalar_bar_args=None,
                         axis_labels = ['001', '010', '100'], throw=False)

gstslice.plot_grains(gstslice.gid, scalar='lgi', cmap='viridis', style='surface',
                     show_edges=True, lw=3.0, opacity=0.8, view=None,
                     scalar_bar_args=None, plot_coords=True,
                     coords=gstslice.gbpstack,
                     axis_labels = ['z', 'y', 'x'], pvp=None, throw=False)
# -------------------------------------------------------------------
gstslice.viz_browse_grains(scalar='lgi', cmap='viridis',
                           style='surface', show_edges=True, lw=1.0,
                           opacity=0.8, view=None, scalar_bar_args=None,
                           plot_coords=False, name='UPXO.MCGS.3D',
                           coords=None, axis_labels = ['z', 'y', 'x'],
                           add_outline=False, pvp=None)
gstslice.viz_clip_plane(normal='x', scalar='lgi', cmap='viridis',
                        invert=True, crinkle=True, normal_rotation=True,
                        throw=False, pvp=None)
gstslice.viz_mesh_slice(normal='x', scalar='lgi', cmap='viridis',
                        normal_rotation=True, throw=False, pvp=None)
gstslice.viz_mesh_slice_ortho(scalar='lgi', cmap='viridis',
                         style='surface', throw=False, pvp=None)
