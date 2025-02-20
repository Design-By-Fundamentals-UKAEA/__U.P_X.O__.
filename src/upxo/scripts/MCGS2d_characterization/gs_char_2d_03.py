# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 07:58:45 2024

@author: Dr. Sunil Anandatheertha

This converts the non-geometric MCGS2d to geometric grain strcture.
"""

import cv2
import numpy as np
import gmsh
import pyvista as pv
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter
from meshpy.triangle import MeshInfo, build
from upxo.ggrowth.mcgs import mcgs
# ---------------------------
pxt = mcgs()
pxt.simulate()
pxt.detect_grains()
tslice = 20
pxt.char_morph_2d(tslice)
gstslice = pxt.gs[tslice]
gstslice.neigh_gid
gstslice.find_grain_boundary_junction_points()
folder, fileName = r'D:\export_folder', 'sunil'
gstslice.export_ctf(folder, fileName, factor=2, method='nearest')
# gstslice.export_ctf(folder, fileName)
# ---------------------------
fname = r'D:\export_folder\sunil'
gstslice.set_pxtal(instance_no=1, path_filename_noext=fname)
# ---------------------------
gstslice.pxtal[1].g[1]['grain'].loc
gstslice.pxtal[1].mprop.keys()
# ---------------------------
gstslice.pxtal[1].mprop.keys()

gstslice.pxtal[1].mprop['aspect_ratio']

gstslice.pxtal[1].bbox
gstslice.pxtal[1].bbox_ex
gstslice.pxtal[1].bbox_bounds
gstslice.pxtal[1].bbox_ex_bounds
gstslice.pxtal[1].g
gstslice.pxtal[1].n
gstslice.pxtal[1].gbseg1[30]
gstslice.pxtal[1].neigh_gid[30]

gstslice.pxtal[1].find_gbseg1()
gstslice.pxtal[1].extract_gb_discrete(retrieval_method='external',
                                      chain_approximation='simple')

gstslice.pxtal[1].plot_gb_discrete(bjp_kwargs={'marker': 'o', 'mfc': 'yellow',
                                               'mec': 'black', 'ms': 2.5},
                                   simple_all_preference='simple')

gstslice.pxtal[1].find_lgi_subset_neigh(13, plot_gbseg=True)
gstslice.pxtal[1].mprop['npixels']



gstslice.pxtal[1].get_gids_in_params_bounds(search_gid_source='all',
                                            search_gids=None,
                                            mpnames=['area', 'solidity'],
                                            fx_stats=[np.mean,np.mean],
                                            pdslh=[[20, 20], [5, 5]],
                                            param_priority=[1, 2, 3, 2],
                                            plot_mprop=True,
                                            )

gstslice.pxtal[1].get_upto_nth_order_neighbors(26, 1,
                                               recalculate=False,
                                               include_parent=True,
                                               output_type='list', plot=True)

gstslice.pxtal[1].get_nth_order_neighbors(26, 3, recalculate=False,
                                          include_parent=True)

gstslice.pxtal[1].get_upto_nth_order_neighbors_all_grains(1, recalculate=False,
                                                          include_parent=True,
                                                          output_type='list')

gstslice.pxtal[1].get_nth_order_neighbors_all_grains(1, recalculate=False,
                                                     include_parent=True)

gstslice.pxtal[1].get_upto_nth_order_neighbors_all_grains_prob(1,
                                                 recalculate=False,
                                                 include_parent=False,
                                                 print_msg=False)

# ------------------------------------------------
from upxo._sup.data_ops import increase_grid_resolution, decrease_grid_resolution

print(gstslice.xgr.max(), gstslice.ygr.max())
print(gstslice.xgr.shape, gstslice.ygr.shape, gstslice.s.shape)
X, Y, S = decrease_grid_resolution(gstslice.xgr, gstslice.ygr, gstslice.s, 0.2)
print(X.max(), Y.max())
print(X.shape, Y.shape, S.shape)
plt.imshow(gstslice.s)
plt.imshow(S)
print('----------')
print(gstslice.xgr.max(), gstslice.ygr.max())
X, Y, S = increase_grid_resolution(gstslice.xgr, gstslice.ygr, gstslice.s, 2)
print(X.max(), Y.max())
print(X.shape, Y.shape, S.shape)
