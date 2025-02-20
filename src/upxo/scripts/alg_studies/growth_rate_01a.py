# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 13:18:52 2025

@author: rg5749
"""

from upxo.ggrowth.mcgs import mcgs
import numpy as np
import pandas as pd
from openpyxl import load_workbook

# Initialize dictionaries to store results
GS_Ng = {'alg300a': None, 'alg300b': None, 'alg301': None, 'alg302': None}
GS_mean = {'alg300a': None, 'alg300b': None, 'alg301': None, 'alg302': None}
GS_max = {'alg300a': None, 'alg300b': None, 'alg301': None, 'alg302': None}
GS_std = {'alg300a': None, 'alg300b': None, 'alg301': None, 'alg302': None}
GS_nsvg = {'alg300a': None, 'alg300b': None, 'alg301': None, 'alg302': None}

# File and sheet information
file_path = r"C:\\Development\\M2MatMod\\upxo_packaged\\upxo_private\\documentation\\Alg_results_02.xlsx"
book = load_workbook(file_path)
# ==========================================================================
sheet_name = "Alg300a"
# --------------------------------------------------------
nsims = 5  # Number of simulations. Fixed. Dont make any change.
# --------------------------------------------------------
row_starts = {'T0.0': 5, 'T0.1': 110, 'T0.2': 215}
row_start = row_starts['T0.0']
# --------------------------------------------------------
col_start = 2  # values below
'''col_start values:
    2: 50x50x50, any temperature -- Status:
        T 0.0: Done
        T 0.1: Done
        T 0.2: Done
    29: 100x100x100, any temperature -- Status:
        T 0.0: Done
        T 0.1: Done
        T 0.2:
    56: 250x250x250, any temperature -- Status:
        T 0.0: Done
        T 0.1: Done
        T 0.2:
'''
# --------------------------------------------------------
cells_mean_gsize_sims = list(range(col_start+0*nsims, col_start+1*nsims))
cells_max_gsize_sims = list(range(col_start+1*nsims, col_start+2*nsims))
cells_std_gsize_sims = list(range(col_start+2*nsims, col_start+3*nsims))
cells_num_svox_grains_sims = list(range(col_start+3*nsims, col_start+4*nsims))
cells_num_grains_sims = list(range(col_start+4*nsims, col_start+5*nsims))
# --------------------------------------------------------
# Data storage for simulation results
ngrains_sims = {}
mean_gsize_sims = {}
max_gsize_sims = {}
std_gsize_sims = {}
num_svox_grains_sims = {}

# Simulation loop
for sim_n in range(nsims):
    print(60*'#')
    print(f'Simulation number: {sim_n+1} of {nsims}')
    print(40*'#')
    # Initialize simulation
    pxt = mcgs()
    pxt.simulate(verbose=False)

    # Allocate arrays for grain size metrics
    ng = np.full(pxt.uisim.mcsteps, np.nan)
    meangsize = np.full(pxt.uisim.mcsteps, np.nan)
    maxgsize = np.full(pxt.uisim.mcsteps, np.nan)
    stdgsize = np.full(pxt.uisim.mcsteps, np.nan)
    num_svox_grains = np.full(pxt.uisim.mcsteps, np.nan)

    # Process each time slice
    for tslice, gs in pxt.gs.items():
        print(40*'=')
        print(f'Characterising grain strucure @ tslice = {tslice}')
        gs.char_morphology_of_grains(label_str_order=1)
        gs.set_mprop_volnv()
        grainsize = np.array(list(gs.mprop['volnv'].values()))

        ng[tslice] = len(grainsize)
        meangsize[tslice] = grainsize.mean()
        maxgsize[tslice] = grainsize.max()
        stdgsize[tslice] = grainsize.std()
        num_svox_grains[tslice] = np.sum(grainsize == 1)

    # Store simulation results
    ngrains_sims[sim_n] = ng
    mean_gsize_sims[sim_n] = meangsize
    max_gsize_sims[sim_n] = maxgsize
    std_gsize_sims[sim_n] = stdgsize
    num_svox_grains_sims[sim_n] = num_svox_grains

    # Write mean grain size to Excel
    df = pd.DataFrame(meangsize, columns=["Values"])
    with pd.ExcelWriter(file_path, engine="openpyxl",
                        mode='a', if_sheet_exists="overlay") as writer:
        writer.book = book
        df.to_excel(writer, sheet_name=sheet_name, index=False, header=False,
                    startrow=row_start, startcol=cells_mean_gsize_sims[sim_n])

    # Write max grain size to Excel
    df = pd.DataFrame(maxgsize, columns=["Values"])
    with pd.ExcelWriter(file_path, engine="openpyxl",
                        mode='a', if_sheet_exists="overlay") as writer:
        writer.book = book
        df.to_excel(writer, sheet_name=sheet_name, index=False, header=False,
                    startrow=row_start, startcol=cells_max_gsize_sims[sim_n])

    # Write std grain size to Excel
    df = pd.DataFrame(stdgsize, columns=["Values"])
    with pd.ExcelWriter(file_path, engine="openpyxl",
                        mode='a', if_sheet_exists="overlay") as writer:
        writer.book = book
        df.to_excel(writer, sheet_name=sheet_name, index=False, header=False,
                    startrow=row_start, startcol=cells_std_gsize_sims[sim_n])

    # Write number of single voxel grains
    df = pd.DataFrame(num_svox_grains, columns=["Values"])
    with pd.ExcelWriter(file_path, engine="openpyxl",
                        mode='a', if_sheet_exists="overlay") as writer:
        writer.book = book
        df.to_excel(writer, sheet_name=sheet_name, index=False, header=False,
                    startrow=row_start, startcol=cells_num_svox_grains_sims[sim_n])

    # Write total number of grains
    df = pd.DataFrame(ng, columns=["Values"])
    with pd.ExcelWriter(file_path, engine="openpyxl",
                        mode='a', if_sheet_exists="overlay") as writer:
        writer.book = book
        df.to_excel(writer, sheet_name=sheet_name, index=False, header=False,
                    startrow=row_start, startcol=cells_num_grains_sims[sim_n])
# --------------------------------------------------------
# Update global storage
GS_Ng['alg300a'] = ngrains_sims
GS_mean['alg300a'] = mean_gsize_sims
GS_max['alg300a'] = max_gsize_sims
GS_std['alg300a'] = std_gsize_sims
GS_nsvg['alg300a'] = num_svox_grains_sims



# --------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
file_path = r"C:\\Development\\M2MatMod\\upxo_packaged\\upxo_private\\documentation\\Alg_results_02.xlsx"
sheet_name = "Alg300a"
# -------------------------------
row_starts = {'T0.0': 5, 'T0.1': 110, 'T0.2': 215}
# -------------------------------
nsims = 5
# -------------------------------
col_start = 2
# -------------------------------
NITER = 100
# -------------------------------
domain_cols = {'50cubed': "C:AA", '100cubed': "AD:BB", '250cubed': "BE:CC"}
# -------------------------------
TMP_grain_metrics = {'T0.0': None, 'T0.1': None, 'T0.2': None}
TMP_grain_metrics_mean = {'T0.0': None, 'T0.1': None, 'T0.2': None}
TMP_grain_metrics_std = {'T0.0': None, 'T0.1': None, 'T0.2': None}
for tmp in TMP_grain_metrics.keys():
    dom_grain_metrics = {'50cubed': None, '100cubed': None, '250cubed': None}
    dom_grain_metrics_mean = {'50cubed': None, '100cubed': None, '250cubed': None}
    dom_grain_metrics_std = {'50cubed': None, '100cubed': None, '250cubed': None}
    for dom in domain_cols.keys():
        df = pd.read_excel(file_path, sheet_name=sheet_name,
                           usecols=domain_cols[dom],
                           skiprows=row_starts[tmp]-1,
                           nrows=NITER)

        dom_grain_metrics[dom] = {'meangs': df.iloc[:, col_start+0*nsims-2:col_start+1*nsims-2],
                                  'maxgs': df.iloc[:, col_start+1*nsims-2:col_start+2*nsims-2],
                                  'stdgs': df.iloc[:, col_start+2*nsims-2:col_start+3*nsims-2],
                                  'Nsvoxg': df.iloc[:, col_start+3*nsims-2:col_start+4*nsims-2],
                                  'Ng': df.iloc[:, col_start+4*nsims-2:col_start+5*nsims-2]}

        dom_grain_metrics_mean[dom] = {'meangs': dom_grain_metrics[dom]['meangs'].mean(axis=1),
                                       'maxgs': dom_grain_metrics[dom]['maxgs'].mean(axis=1),
                                       'stdgs': dom_grain_metrics[dom]['stdgs'].mean(axis=1),
                                       'Nsvoxg': dom_grain_metrics[dom]['Nsvoxg'].mean(axis=1),
                                       'Ng': dom_grain_metrics[dom]['Ng'].mean(axis=1)}

        dom_grain_metrics_std[dom] = {'meangs': dom_grain_metrics[dom]['meangs'].std(axis=1),
                                      'maxgs': dom_grain_metrics[dom]['maxgs'].std(axis=1),
                                      'stdgs': dom_grain_metrics[dom]['stdgs'].std(axis=1),
                                      'Nsvoxg': dom_grain_metrics[dom]['Nsvoxg'].std(axis=1),
                                      'Ng': dom_grain_metrics[dom]['Ng'].std(axis=1)}
    TMP_grain_metrics[tmp] = dom_grain_metrics
    TMP_grain_metrics_mean[tmp] = dom_grain_metrics_mean
    TMP_grain_metrics_std[tmp] = dom_grain_metrics_std


mciters = list(range(NITER))
# ===========================================================
mprop, ylb_str, ymax, legloc = 'meangs', 'Mean grain volume', 1000, 'upper left'
mprop, ylb_str, ymax, legloc = 'maxgs', 'Maximum grain volume', 1000000, 'upper left'
mprop, ylb_str, ymax, legloc = 'stdgs', 'Standard dev. of grain size', 1000, 'upper left'
mprop, ylb_str, ymax, legloc = 'Nsvoxg', 'Number of single voxel grains', 1.25E6, 'upper right'
mprop, ylb_str, ymax, legloc = 'Ng', 'Number of grains', 0.2E6, 'upper right'

plt.figure(figsize=(4, 4), dpi=150)
plt.errorbar(mciters,
             TMP_grain_metrics_mean['T0.0']['50cubed'][mprop],
              TMP_grain_metrics_std['T0.0']['50cubed'][mprop],
              label='$50^3, Alg300a, T0.0$',
              color='black', linestyle='-', linewidth=1.,
              ecolor='gray', elinewidth=0.5, capsize=2,
              marker='none', markerfacecolor='gray',
              markeredgecolor='k', markersize=2.5, markeredgewidth=0.5,
              )
plt.errorbar(mciters,
             TMP_grain_metrics_mean['T0.0']['100cubed'][mprop],
              TMP_grain_metrics_std['T0.0']['100cubed'][mprop],
              label='$100^3, Alg300a, T0.0$',
              color='blue', linestyle='-', linewidth=1.,
              ecolor='skyblue', elinewidth=0.5, capsize=2,
              marker='none', markerfacecolor='skyblue',
              markeredgecolor='k', markersize=2.5, markeredgewidth=0.5,
              )
plt.errorbar(mciters,
             TMP_grain_metrics_mean['T0.0']['250cubed'][mprop],
              TMP_grain_metrics_std['T0.0']['250cubed'][mprop],
              label='$250^3, Alg300a, T0.0$',
              color='red', linestyle='-', linewidth=1.,
              ecolor='lightcoral', elinewidth=0.5, capsize=2,
              marker='none', markerfacecolor='lightcoral',
              markeredgecolor='k', markersize=2.5, markeredgewidth=0.5,
              )
plt.legend(fontsize=8, loc=legloc)
plt.xlabel('Simulation time', fontsize=12)
plt.ylabel(ylb_str, fontsize=12)
plt.title('Alg300a, T0.0. Num. of sims = 5', fontsize=10)
plt.axis([0, 100, 0, ymax])
plt.tight_layout()
plt.grid(alpha=0.3)

plt.figure(figsize=(4, 4), dpi=150)
plt.errorbar(mciters,
             TMP_grain_metrics_mean['T0.1']['50cubed'][mprop],
              TMP_grain_metrics_std['T0.1']['50cubed'][mprop],
              label='$50^3, Alg300a, T0.1$',
              color='black', linestyle='--', linewidth=1.,
              ecolor='gray', elinewidth=1., capsize=2,
              marker='none', markerfacecolor='gray',
              markeredgecolor='k', markersize=2.5, markeredgewidth=0.5,
              )
plt.errorbar(mciters,
             TMP_grain_metrics_mean['T0.1']['100cubed'][mprop],
              TMP_grain_metrics_std['T0.1']['100cubed'][mprop],
              label='$100^3, Alg300a, T0.1$',
              color='blue', linestyle='--', linewidth=1.,
              ecolor='skyblue', elinewidth=1., capsize=2,
              marker='none', markerfacecolor='skyblue',
              markeredgecolor='k', markersize=2.5, markeredgewidth=0.5,
              )
plt.errorbar(mciters,
             TMP_grain_metrics_mean['T0.1']['250cubed'][mprop],
              TMP_grain_metrics_std['T0.1']['250cubed'][mprop],
              label='$250^3, Alg300a, T0.1$',
              color='red', linestyle='--', linewidth=1.,
              ecolor='lightcoral', elinewidth=1.5, capsize=2,
              marker='none', markerfacecolor='lightcoral',
              markeredgecolor='k', markersize=2.5, markeredgewidth=0.5,
              )
plt.legend(fontsize=8, loc=legloc)
plt.xlabel('Simulation time', fontsize=12)
plt.ylabel(ylb_str, fontsize=12)
plt.title('Alg300a, T0.1. Num. of sims = 5', fontsize=10)
plt.axis([0, 100, 0, ymax])
plt.tight_layout()
plt.grid(alpha=0.3)

plt.figure(figsize=(4, 4), dpi=150)
plt.errorbar(mciters,
             TMP_grain_metrics_mean['T0.2']['50cubed'][mprop],
              TMP_grain_metrics_std['T0.2']['50cubed'][mprop],
              label='$50^3, Alg300a, T0.2$',
              color='black', linestyle=':', linewidth=1.,
              ecolor='gray', elinewidth=1., capsize=2,
              marker='none', markerfacecolor='gray',
              markeredgecolor='k', markersize=2.5, markeredgewidth=0.5,
              )
plt.errorbar(mciters,
             TMP_grain_metrics_mean['T0.2']['100cubed'][mprop],
              TMP_grain_metrics_std['T0.2']['100cubed'][mprop],
              label='$100^3, Alg300a, T0.2$',
              color='blue', linestyle=':', linewidth=1.,
              ecolor='skyblue', elinewidth=1., capsize=2,
              marker='none', markerfacecolor='skyblue',
              markeredgecolor='k', markersize=2.5, markeredgewidth=0.5,
              )
plt.errorbar(mciters,
             TMP_grain_metrics_mean['T0.2']['250cubed'][mprop],
              TMP_grain_metrics_std['T0.2']['250cubed'][mprop],
              label='$250^3, Alg300a, T0.2$',
              color='red', linestyle=':', linewidth=1.,
              ecolor='lightcoral', elinewidth=1.5, capsize=2,
              marker='none', markerfacecolor='lightcoral',
              markeredgecolor='k', markersize=2.5, markeredgewidth=0.5,
              )


plt.legend(fontsize=8, loc=legloc)
plt.xlabel('Simulation time', fontsize=12)
plt.ylabel(ylb_str, fontsize=12)
plt.title('Alg300a, T0.2. Num. of sims = 5', fontsize=10)
plt.axis([0, 100, 0, ymax])
plt.tight_layout()
plt.grid(alpha=0.3)
