"""
This script generates a MCGS2d ad explores the database
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
tslice = 2  # Temporal slice number
pxt.char_morph_2d(tslice)
pxt.gs[2].neigh_gid
#pxt.gs[tslice].plot()
hgrid = pxt.gs[tslice].xgr
vgrid = pxt.gs[tslice].ygr
mcstates = pxt.gs[tslice].s
nstates = pxt.uisim.S

def compartmentalize(matrix, hsize, vsize):
    """
    matrix = np.random.randint(0, 10, (6, 6))
    matrix
    compartmentalize(matrix, 2, 3)
    """
    subset = np.lib.stride_tricks.sliding_window_view
    return subset(matrix, (hsize, vsize))[::hsize, ::vsize]


mcstates.shape



subsets = compartmentalize(mcstates, 10, 10)


# pxt.gs[tslice].scale(sf=2)
pxt.gs[tslice].export_ctf(r'D:\export_folder', 'sunil')
