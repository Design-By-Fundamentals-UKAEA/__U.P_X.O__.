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
tslice = 19
gstslice = pxt.gs[tslice]
gstslice.char_morphology_of_grains(label_str_order=3)
gstslice.set_mprops()
# ---------------------------------------
igs_rshuffle = []
nshuffles = 1000
for i in range(nshuffles):
    igs = gstslice.get_igs_along_lines(metric='mean', minimum=True, maximum=True,
                                 std=True, variance=True, lines_gen_method=1,
                                 lines_kwargs1={'plane': 'z',
                                                'start_skip1': 0, 'start_skip2': 0,
                                                'incr1': 5, 'incr2': 5,
                                                'inclination': 'random',
                                                'inclination_extent': 0,
                                                'shift_seperately': True,
                                                'shift_starts': True,
                                                'shift_ends': True,
                                                'start_shift': 5, 'end_shift': 1})
    igs_rshuffle.append(igs['igs'])
    if i % 100 == 0:
        print(f'Shuffling count: {i}')

plt.plot(range(nshuffles), igs_rshuffle)
