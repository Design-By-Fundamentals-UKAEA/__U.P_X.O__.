# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 11:57:15 2024

@author: rg5749
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from upxo.ggrowth.mcgs import mcgs
from upxo.pxtalops.Characterizer import mcgs_mchar_2d


mcgs = mcgs(study='independent',
            input_dashboard='input_dashboard.xls',
            consider_NLM_b=False,
            consider_NLM_d=False,
            AR_factor=0,
            AR_GrainAxis="-45",
            display_messages=True)
mcgs.simulate()

mcgs.detect_grains()

tslice = 10
mcgs.gs[tslice].plotgs()

# mcgs.gs[tslice].n
# mcgs.gs[tslice].s_gid
# mcgs.gs[tslice].gid_s
# len(mcgs.gs[tslice].gid_s) - mcgs.gs[tslice].n
# len(mcgs.gs[tslice].gid_s) - len([i for i in range(len(mcgs.gs[tslice].gid_s)) if mcgs.gs[tslice].gid_s[i]==0])

mcgs.gs[tslice].char_morph_2d(bbox=True, bbox_ex=True, npixels=False,
                              npixels_gb=False, area=True, eq_diameter=False,
                              perimeter=False, perimeter_crofton=False,
                              compactness=False, gb_length_px=False,
                              aspect_ratio=True,
                              solidity=False, morph_ori=False,
                              circularity=False,
                              eccentricity=False, feret_diameter=False,
                              major_axis_length=False, minor_axis_length=False,
                              euler_number=False, append=False, saa=True,
                              throw=False, char_grain_positions=False,
                              find_neigh=False, char_gb=False,
                              make_skim_prop=True, get_grain_coords=False)
mcgs.gs[tslice].plot_largest_grain()

tslice = 2
mcgs.gs[tslice].plotgs()
mcgs.gs[tslice].plot_largest_grain()

mcgs.gs[tslice].prop.columns
mcgs.gs[tslice].find_neigh()
mcgs.gs[tslice].neigh_gid
parent_gid = 1
other_gid = 14
mcgs.gs[tslice].check_for_neigh(parent_gid, other_gid)
print(f'State of parent_gid: {mcgs.gs[35].gid_s[parent_gid-1]}')
print(f'State of other_gid: {mcgs.gs[35].gid_s[other_gid-1]}')
print(f'State of other_gid: {mcgs.gs[35].gid_s[9-1]}')
print(f'Number of grains: {mcgs.gs[35].n}')

mcgs.gs[35].plot_grains_gids([1, 2, 14])

len(mcgs.gs[35].gid_s)
mcgs.gs[35].s_gid

mcgs.gs[35].single_pixel_grains
mcgs.gs[35].straight_line_grains
mcgs.gs[35].g

dir(mcgs.gs[35].g[14]['grain'])

mcgs.gs[35].g[14]['grain'].plot()
mcgs.gs[35].g[14]['grain'].neigh
mcgs.gs[35].g[14]['grain'].coords
mcgs.gs[35].g[14]['grain'].skprop

from skimage.measure import regionprops
mcgs.gs[35].g[14]['grain'].make_prop(regionprops)
dir(mcgs.gs[35].g[14]['grain'].skprop)

mcgs.gs[35].g[14]['grain'].skprop.coords
mcgs.gs[35].g[14]['grain'].skprop.centroid
mcgs.gs[35].g[14]['grain'].skprop.centroid_local
mcgs.gs[35].g[14]['grain'].skprop.area
mcgs.gs[35].g[14]['grain'].skprop.num_pixels
mcgs.gs[35].g[14]['grain'].skprop.perimeter
mcgs.gs[35].g[14]['grain'].skprop.perimeter_crofton
mcgs.gs[35].g[14]['grain'].skprop.solidity
mcgs.gs[35].g[14]['grain'].skprop.orientation
mcgs.gs[35].g[14]['grain'].skprop.eccentricity
mcgs.gs[35].g[14]['grain'].skprop.equivalent_diameter_area
mcgs.gs[35].g[14]['grain'].skprop.feret_diameter_max
mcgs.gs[35].g[14]['grain'].skprop.inertia_tensor
mcgs.gs[35].g[14]['grain'].skprop.inertia_tensor_eigvals
mcgs.gs[35].g[14]['grain'].skprop.moments
mcgs.gs[35].g[14]['grain'].skprop.moments_central
mcgs.gs[35].g[14]['grain'].skprop.moments_hu
mcgs.gs[35].g[14]['grain'].skprop.moments_normalized



mcgs.gs[tslice].plot_largest_grain()
mcgs.gs[35].plot_longest_grain()
mcgs.gs[35].prop.area.min()

gid, value, df_loc = mcgs.gs[tslice].get_gid_prop_range(PROP_NAME='aspect_ratio',
                                                    reminf=True, remnan=True,
                                                    range_type='percentage',
                                                    percentage_range=[100, 100])


mcgs.gs[tslice].plot_longest_grain()
