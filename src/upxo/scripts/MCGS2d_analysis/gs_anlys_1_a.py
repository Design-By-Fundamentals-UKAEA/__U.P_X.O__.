# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 07:26:30 2024

@author: rg5749
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from upxo._sup import dataTypeHandlers as dth
from upxo.ggrowth.mcgs import mcgs
from upxo.pxtalops.Characterizer import mcgs_mchar_2d
NUMBERS, ITERABLES, RNG = dth.dt.NUMBERS, dth.dt.ITERABLES, np.random.default_rng()
# =========================================================================
mcgs = mcgs(study='independent', input_dashboard='input_dashboard.xls')
mcgs.simulate()
mcgs.detect_grains()
mcgs.gs[35].char_morph_2d(bbox=True, bbox_ex=True, area=True,
                          aspect_ratio=True, perimeter=True, solidity=True,
                          make_skim_prop=True,)
mcgs.gs[35].prop.columns
mcgs.gs[35].find_neigh()

# =========================================================================
GIDs, VALIND = mcgs.gs[35].get_gids_in_params_bounds(mpnames=['area', 'aspect_ratio', 'perimeter', 'solidity'],
                                      fx_stats=[np.mean, np.mean, np.mean, np.mean],
                                      pdslh=[[50, 30], [75, 75], [75, 75], [75, 75]],
                                      param_priority=[1, 2, 3, 2],
                                      )
# =========================================================================
'''1. GIDs value for intersection key provides the gids which satisfy all bounds
for all mpnames under consideration.'''
GIDs['intersection']
'''We can look at these grains.'''
mcgs.gs[35].plot_grains_gids(GIDs['intersection'],
                             gclr='color',
                             title="user grains",
                             cmap_name='coolwarm', )
'''The grains themselves can be easily accessed here.'''
gids_intersection = {gid: mcgs.gs[35].g[gid]['grain']
                     for gid in GIDs['intersection']}
# --------------------------
'''2. GIDs value for union key provides the gids which satisfy any of the bounds
for mpnames under consideration.'''
GIDs['union']
# --------------------------
'''3. We will now look at only those grains which satisfy bounds for
a property of choice.'''
GIDs['mpmapped'].keys()  #  Look at all property names available.
GIDs['mpmapped']['area']  #  Gids satisfying bounds on mpprop: area.
GIDs['mpmapped']['perimeter']  #  Gids satisfying bounds on mpprop: perimeter.
#  And so on.
# --------------------------
'''4. We will now look at the presence index of each gid.
DEFINITION: Presence index of a gid is the number of mprop names it has
simultaneously satisfied.

For example, consider mpnames=['area', 'aspect_ratio', 'perimeter', 'solidity']
gid=32 satisfies 'area' and 'aspect_ratio' but not other two.
gid=38 satisfies bounds of all mpnames.
Therefore, presence index of gid 32 is 2 and that of gid 38 is 4.
'''
GIDs['presence']
'''This is a dictionary with keys as union gids and values are the number
of mprops a gid has been inside the.
'''
# =========================================================================
mcgs.gs[35].plot_grains_gids(GIDs['intersection'],
                     gclr='color',
                     title="user grains",
                     cmap_name='coolwarm', )

mcgs.gs[35].plot_grains_gids(GIDs['union'],
                     gclr='color',
                     title="user grains",
                     cmap_name='coolwarm', )
# =========================================================================
GIDs, VALIND = mcgs.gs[35].get_gids_in_params_bounds(mpnames=['aspect_ratio', 'area'],
                                      fx_stats=[np.mean, np.mean],
                                      pdslh=[[50, 30], [50, 30]], plot_mprop=False
                                      )
mcgs.gs[35].map_scalar_to_lgi(GIDs['presence'], default_scalar=-1,
                      plot=True, throw_axis=True)
# =========================================================================
gid_mprop_map = mcgs.gs[35].get_gid_mprop_map('aspect_ratio',
                                              GIDs['mpmapped']['aspect_ratio'])
MPLGIAX = mcgs.gs[35].map_scalar_to_lgi(gid_mprop_map, default_scalar=-1,
                      plot=True, throw_axis=True)
