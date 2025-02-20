# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 12:43:57 2024

@author: rg5749
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
tslice = 5
gstslice = pxt.gs[tslice]
gstslice.char_morphology_grains(label_str_order=3)
gstslice.set_mprops()
# ######################################################################
# ######################################################################
pvp = pv.Plotter()
for ts in pxt.gs.keys():
    for gid in gstslice.get_largest_gids():
        pvp.add_mesh(gstslice.pvgrid.threshold([gid, gid], scalars=scalar),
                     show_edges=show_edges, line_width=lw,
                     style=style, opacity=opacity, cmap=cmap)
