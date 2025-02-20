# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 07:05:19 2024

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
mcgs.gs[35].char_morph_2d(bbox=True, bbox_ex=True, area=True, aspect_ratio=True,
                          make_skim_prop=True,)
mcgs.gs[35].find_neigh()
