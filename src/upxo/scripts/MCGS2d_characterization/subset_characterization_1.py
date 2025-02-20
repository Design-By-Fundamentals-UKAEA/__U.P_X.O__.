"""
Created on Tue May 14 09:10:54 2024

@author: Dr. Sunil Anandatheertha
"""
'''IMPORT STATEMENTS'''
import numpy as np
import matplotlib.pyplot as plt
from upxo.ggrowth.mcgs import mcgs
from upxo.pxtalops.Characterizer import mcgs_mchar_2d
'''LETS FIRST GENERATE THE BASIC GRAIN STRUCTURE'''
pxt = mcgs()
pxt.simulate()
'''
DETECT GRAINS AND CHARACTERIZE THEW GRAIN STRUCTUER AS USUAL.
For this, I have chosen tslice = 2, you can choose any available tslice.
'''
pxt.detect_grains()
tslice = 8
pxt.char_morph_2d(tslice)
'''
* Number of grains can be obtainbed as:
    pxt.gs[tslice].n

* Grain areas can be obtained as:
    pxt.gs[tslice].areas

* Neighbouring grain IDs can be obtained as:
    pxt.gs[tslice].neigh_gid

* Properties can be obtained as as a pandas dataframe:
    pxt.gs[tslice].prop
* List of available property data colums can be accessed as:
    pxt.gs[tslice].prop.columns
* Equivalent dismater of grains can be accessed as:
    pxt.gs[tslice].prop['eq_diameter'].tolist()
'''
cell_neighbors = pxt.gs[tslice].neigh_gid
# THIS FINISHES THE USUAL GRAIN STRUCTURE GENERATION AND ANALYSIS
# -------------------------------------------------------
"""Get the field matrices. Here, we use monte-carlo states as the fmats.
"""
fmat = pxt.gs[tslice].s
fmin, fmax = 1, pxt.uisim.S
# ------------------------
"""Choose the sub-domain sizes along horizontal and vertical axes."""
hsize, vsize = int(fmat.shape[0]/2), int(fmat.shape[1]/2)
# ------------------------
"""Instantialize the mcgs 2d charectization."""
pxtchr = mcgs_mchar_2d()
"""Set field matrix you are about to subsetize and characterize."""
pxtchr.set_fmat(fmat, fmin, fmax)
"""Subsetize the field matrix now."""
fmats = pxtchr.make_fmat_subsets(hsize, vsize)
# fmats.shape
"""Characterize all field matrix sub-sets."""
pxtchr.characterize_all_subsets(fmats)
"""Characterize only the required field matrix sub-sets."""
alongh=[0, 1]
alongv = 0

characterized_subsets_indices, characterized_subsets = pxtchr.characterize_subsets(fmats, alongh, alongv)

characterized_subsets
