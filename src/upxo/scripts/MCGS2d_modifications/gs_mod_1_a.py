# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 22:51:46 2024

@author: Dr. Sunil Anandatheertha
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
mcgs.gs[35].char_morph_2d(bbox=True, bbox_ex=True, area=True, aspect_ratio=True,
                          make_skim_prop=True,)
mcgs.gs[35].find_neigh()

mpnames=['area', 'aspect_ratio', 'perimeter', 'solidity']
mcgs.gs[35].prop
mprop_values = mcgs.gs[35].get_mprops(mpnames, set_missing_mprop=True)
mprop_values

mcgs.gs[35].validata_gids(mcgs.gs[35].gid)

mcgs.gs[35].plotgs(plot_centroid=True, plot_gid_number=True, plot_cbar=False)
# =========================================================================
mcgs.gs[35].n
mcgs.gs[35].s_gid
mcgs.gs[35].gid_s
mcgs.gs[35].neigh_gid
mcgs.gs[35].lgi
mcgs.gs[35].prop.columns
mcgs.gs[35].g
mcgs.gs[35].neigh
# =========================================================================
parent_gid, other_gid = mcgs.gs[35].plot_two_rand_neighs()
mcgs.gs[35].neigh_gid
parent_gid, other_gid = 25, 36
mcgs.gs[35].check_for_neigh(parent_gid, other_gid)
print(f"areas of parent and other gid: {mcgs.gs[35].prop['area'][parent_gid-1]} and {mcgs.gs[35].prop['area'][other_gid-1]}")
print(f'State of parent_gid: {mcgs.gs[35].gid_s[parent_gid-1]}')
print(f'State of other_gid: {mcgs.gs[35].gid_s[other_gid-1]}')
# =========================================================================
mcgs.gs[35].merge_two_neigh_grains(parent_gid, other_gid,
                                   check_for_neigh=True,
                                   simple_merge=True)
mcgs.gs[35].plotgs(plot_centroid=True, plot_gid_number=True)
# =========================================================================
mcgs.gs[35].merge_two_neigh_grains_rand_simple(plot_bf=True, plot_af=True,
                                               return_gids=True,
                                               plot_find_kde_diff=True
                                               )
# =========================================================================
mcgs.gs[35].set_mprops(['area', 'aspect_ratio', 'npixels', 'eq_diameter'],
                       char_grain_positions=True, char_gb=False,
                       set_grain_coords=True)
mcgs.gs[35].prop.columns
mcgs.gs[35].prop.area
mcgs.gs[35].prop['area'].to_numpy()
mcgs.gs[35].prop.aspect_ratio
mcgs.gs[35].prop.npixels
mcgs.gs[35].prop.eq_diameter
# =========================================================================
mcgs.gs[35].merge_two_neigh_grains_rand_simple(return_gids=True,
                                               plot_gs_bf=True, plot_gs_af=True,
                                               plot_area_kde_diff=True,
                                               bandwidth=1.0)

mcgs.gs[35].validate_propnames(['area', 'perimeter', 'solidity'], return_type='dict')
