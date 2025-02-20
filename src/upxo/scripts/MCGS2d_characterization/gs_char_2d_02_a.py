# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 07:48:46 2024

@author: rg5749

Grain boundary juncxtion point identification
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
# =====================================================================
pxt = mcgs()
pxt.simulate()
pxt.detect_grains()
tslice = 20  # Temporal slice number
pxt.char_morph_2d(tslice)
# pxt.gs[tslice].plotgs()
gstslice = pxt.gs[tslice]
gstslice.neigh_gid

#pxt.gs[tslice].plot()

hgrid = gstslice.xgr
vgrid = gstslice.ygr
mcstates = gstslice.s
nstates = pxt.uisim.S

# pxt.gs[tslice].scale(sf=2)
gstslice.export_ctf(r'D:\export_folder', 'sunil')
gstslice.find_grain_boundary_junction_points()
# -----------------------------
'''
plt.figure()
plt.imshow(pxt.gs[tslice].lgi)
for r, c in zip(np.where(pxt.gs[tslice].gbjp)[0],
                np.where(pxt.gs[tslice].gbjp)[1]):
    plt.plot(c, r, 'k.')
'''
# =====================================================================
fileName = r'D:\export_folder\sunil'
gstslice.xomap_set(path_filename_noext=fileName)

gstslice.xomap.map
gstslice.xomap_prepare()
gstslice.xomap_extract_features()

gstslice.xomap_conversion_loss()

gstslice.find_grain_boundary_junction_points(xorimap=True)
gstslice.xomap.gbjp
# =====================================================================
# plt.imshow(gstslice.lgi)
# plt.imshow(gstslice.xomap.map.grains)
# gstslice.xomap_plot_ea()
# gstslice.xomap_plotIPFMap([1, 0, 0])
gstslice.xomap.map.eulerAngleArray
gstslice.xomap.map.quatArray
gstslice.xomap.map.grainList[0].coordList
# =====================================================================
# Above is equivalent to:
# COde here
'''
plt.figure()
plt.imshow(gstslice.xomap.map.grains)
for r, c in zip(np.where(gstslice.xomap.gbjp)[0],
                np.where(gstslice.xomap.gbjp)[1]):
	plt.plot(c, r, 'k.')
'''
gids = gstslice.xomap_gid

BJP = gstslice.xomap_BJP  # Find boundary junction points
BJP
gstslice.n
gstslice.xomap.n


gstslice.xomap_find_neigh(update_gid=True)
