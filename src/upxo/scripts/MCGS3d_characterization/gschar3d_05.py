"""
Created on Mon Sep  2 14:50:17 2024

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
tslice = 19
gstslice = pxt.gs[tslice]
## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## --
gids = []
vols = []
for conn in [1, 2, 3]:
    gstslice.char_morphology_of_grains(label_str_order=conn)
    gstslice.set_mprops()
    gids.append(gstslice.gid)
    vols.append(np.array(list(gstslice.mprop['volnv'].values())))

plt.figure()
sns.histplot(vols[0], label='Conn. 1', kde=True)
sns.histplot(vols[1], label='Conn. 2', kde=True)
sns.histplot(vols[2], label='Conn. 3', kde=True)
plt.legend()
plt.xlabel('Grain volume')
plt.ylabel('Count')
plt.show()
## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## --
