# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 12:30:39 2024

@author: Dr. Sunil Anandatheertha

This script is to explain the details of dealing with individual UPXO.grain2d
objects.
"""
import numpy as np
from upxo.ggrowth.mcgs import mcgs
pxt = mcgs()
pxt.simulate()
pxt.detect_grains()
tslice = 20  # Temporal slice number
pxt.char_morph_2d(tslice)
# pxt.gs[tslice].plotgs()
pxt.gs[tslice].neigh_gid

#pxt.gs[tslice].plot()
# =====================================================================
# TEST SCRIPTS TO TEST EQAUALITY OF GRAIN ARTEAS
# TEST - 1: test against upxo grains
samples = [pxt.gs[tslice].g[i]['grain'] for i in pxt.gs[tslice].g.keys()]
[_.npixels for _ in samples]
upxo_sample = samples[0]
upxo_sample == samples
upxo_sample != samples
upxo_sample < samples
upxo_sample <= samples
upxo_sample > samples
upxo_sample >= samples
# TEST - 2: test against numbers
upxo_sample == [upxo_sample.npixels, 16, 17, 8, 16, 2]
upxo_sample != [upxo_sample.npixels, 16, 17, 8, 16, 2]
upxo_sample > [upxo_sample.npixels, 16, 17, 8, 16, 2]
upxo_sample <= [upxo_sample.npixels, 16, 17, 8, 16, 2]
# =====================================================================
pxt.gs[tslice].export_ctf(r'D:\export_folder', 'sunil')
fileName = r'D:\export_folder\sunil'
pxt.gs[tslice].xomap_set(path_filename_noext=fileName)

pxt.gs[tslice].xomap.map
pxt.gs[tslice].xomap_prepare()
pxt.gs[tslice].xomap_extract_features()

pxt.gs[tslice].find_grain_boundary_junction_points(xorimap=True)
pxt.gs[tslice].xomap.gbjp
# ---------------------------------
# TEST - 3: test against defDap grains
samples = pxt.gs[tslice].xomap.map.grainList
[len(_.coordList) for _ in samples]
any(upxo_sample == samples)
np.where(upxo_sample == samples)
upxo_sample != samples
