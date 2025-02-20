# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 13:18:52 2025

@author: rg5749
"""
from upxo.ggrowth.mcgs import mcgs
import numpy as np
import pyvista as pv
import pandas as pd
from openpyxl import load_workbook
# # -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## --
#       3D ALGORITHMS ONLY
# # -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## --
# Mean grain size values
GS_mean = {'alg300a': None, 'alg300b': None, 'alg301': None, 'alg302': None, }
# Maximum grain size values
GS_mean = {'alg300a': None, 'alg300b': None, 'alg301': None, 'alg302': None, }
# Number of single voxel grains
GS_nsvg = {'alg300a': None, 'alg300b': None, 'alg301': None, 'alg302': None, }
# ----------------------------------------------------------------------
file_path = r"C:\\Development\\M2MatMod\\upxo_packaged\\upxo_private\\documentation\\Alg_results_01.xlsx"
sheet_name = "3D_alg_morph_01"
book = load_workbook(file_path)
# ----------------------------------------------------------------------
# Simulation set 1: 100^3, Alg300a, M100, Q32, TempFac0.0. nsims 5.
nsims = 5
mean_gsize_sims = {sim_n: None for sim_n in range(nsims)}
max_gsize_sims = {sim_n: None for sim_n in range(nsims)}
std_gsize_sims = {sim_n: None for sim_n in range(nsims)}
num_svox_grains_sims = {sim_n: None for sim_n in range(nsims)}
WRITE_DATA = True
cell_start_rownum = 7
'''
cells_mean_gsize_sims = ["C", "D", "E", "F", "G"]
cells_max_gsize_sims = ["H", "I", "J", "K", "L"]
cells_std_gsize_sims = ["M", "N", "O", "P", "Q"]
cells_num_svox_grains_sims = ["R", "S", "T", "U", "V"]
cells_mean_gsize_sims = [_+str(cell_start_rownum)
                         for _ in cells_mean_gsize_sims]
cells_max_gsize_sims = [_+str(cell_start_rownum)
                        for _ in cells_max_gsize_sims]
cells_std_gsize_sims = [_+str(cell_start_rownum)
                        for _ in cells_std_gsize_sims]
cells_num_svox_grains_sims = [_+str(cell_start_rownum)
                              for _ in cells_num_svox_grains_sims]
'''
cells_mean_gsize_sims = list(range(2, 2+5))
cells_max_gsize_sims = list(range(7, 7+5))
cells_std_gsize_sims = list(range(12, 12+5))
cells_num_svox_grains_sims = list(range(17, 17+5))
# ---------------
for sim_n in range(nsims):
    # Individual simulations
    pxt = mcgs()
    pxt.simulate(verbose=False)
    # Mean, maximum and standard deviation of grain size values
    meangsize = np.array([None for i in range(pxt.uisim.mcsteps)])
    maxgsize = np.array([None for i in range(pxt.uisim.mcsteps)])
    stdgsize = np.array([None for i in range(pxt.uisim.mcsteps)])
    # Number of single voxel grains
    num_svox_grains = np.array([None for i in range(pxt.uisim.mcsteps)])
    for tslice, gs in pxt.gs.items():
        gs.char_morphology_of_grains(label_str_order=1)
        gs.set_mprop_volnv()
        grainsize = np.array(list(gs.mprop['volnv'].values()))
        meangsize[tslice] = grainsize.mean()
        maxgsize[tslice] = grainsize.max()
        stdgsize[tslice] = grainsize.std()
        num_svox_grains[tslice] = np.size(np.argwhere(grainsize == 1))
    mean_gsize_sims[sim_n] = meangsize
    max_gsize_sims[sim_n] = maxgsize
    std_gsize_sims[sim_n] = stdgsize
    num_svox_grains_sims[sim_n] = num_svox_grains
    if WRITE_DATA:
        df = pd.DataFrame(meangsize, columns=["Values"])
        with pd.ExcelWriter(file_path,
                            engine="openpyxl",
                            mode='a',
                            if_sheet_exists="overlay") as writer:
            writer.book = book
            df.to_excel(writer, index=False, header=False,
                        startrow=cell_start_rownum,
                        startcol=cells_mean_gsize_sims[sim_n])

# Conclusion
GS_mean['alg300a'] = mean_gsize_sims
GS_mean['alg300a'] = max_gsize_sims
GS_nsvg['alg300a'] = num_svox_grains_sims

# ----------------------------------------------------------------------
