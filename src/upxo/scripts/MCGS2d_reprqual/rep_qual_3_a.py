# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:04:43 2024

@author: Dr. Sunil Anandatheertha

The script helps you calculate the following:
    1. Calculate the nth order neighbouring grain list for target and sample
        grain structure grains.
    2. Calculate the grsin sizes (metric: number of pixels) in target and
        sample grain structure.
    3. Construct the target and sample field dictionary.
    4. Calculate target and sample fields of: nneigh and npixels
    5. Interpolate number of neighbors field onto the underlying target and
        sample grid and visualize.
    6. Carry out O(n) sensitivity on the nneigh statistics.
    7. Calculate the Kullback-Leiber divergence R-field for target1:target2

The script helps you visualize the following:
    1. Plot the global target and sample grain structures
    2. Plot the number of O(n) neighbours for target and sample at a
        given temporal slice.
    3. Visualize the target's number of neighbours vs grain area (i.e. npixels)
        distributions in a seaborn jointplot.
    4. Visualize all grains upto nth order neighbour of a given grain id.
    5. Visualize all nth order neighbour grains of a given grain id.
    6. Visualize the R field of the grain structure for a given a
        parent-subset dataset.
"""

import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from scipy.stats import mannwhitneyu
from scipy.stats import ks_2samp
from scipy.stats import kruskal
from scipy.stats import entropy
from upxo.ggrowth.mcgs import mcgs
from upxo.pxtalops.Characterizer import mcgs_mchar_2d
from upxo._sup import dataTypeHandlers as dth
from upxo.geoEntities.mulpoint2d import MPoint2d
from upxo.interfaces.user_inputs.excel_commons import read_excel_range
from upxo.interfaces.user_inputs.excel_commons import write_array_to_excel
from upxo._sup.data_ops import find_outliers_iqr, distance_between_two_points
from scipy.interpolate import griddata
import matplotlib.ticker as ticker
from scipy.stats import entropy
from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1 import make_axes_locatable
from upxo.statops.stattests import test_rand_distr_autocorr
from upxo.statops.stattests import test_rand_distr_runs
from upxo.statops.stattests import test_rand_distr_chisquare
from upxo.statops.stattests import test_rand_distr_kolmogorovsmirnov
from upxo.statops.stattests import test_rand_distr_kullbackleibler

NUMBERS = dth.dt.NUMBERS
ITERABLES = dth.dt.ITERABLES
RNG = np.random.default_rng()
"""
NOTE: DO NOT CHANGE SETTINGS, IN TEREST OF CONFORMITY WITH TOP EXPLANATIONS.
"""
# ###################################################################
# ###################################################################
# ###################################################################
# ###################################################################
tgt = mcgs(study='independent', input_dashboard='input_dashboard.xls')
tgt.simulate()
tgt.detect_grains()
# =========================================================
smp = mcgs(study='independent', input_dashboard='input_dashboard.xls')
smp.simulate()
smp.detect_grains()
# =========================================================
"""Plot the global target and sample grain structures"""
PLOT_TARGET_GS = True
if PLOT_TARGET_GS:
    _slices_to_plot_ = [0, 4, 10, 20, 48]
    for _s_ in _slices_to_plot_:
        plt.figure(figsize=(10, 2.5), dpi=100)
        plt.imshow(tgt.gs[_s_].lgi), plt.colorbar()
        plt.title(f'Target tslice={_s_}. Ng={tgt.gs[_s_].n}')

PLOT_SAMPLE_GS = True
if PLOT_SAMPLE_GS:
    _slices_to_plot_ = [0, 4, 10, 20, 48]
    for _s_ in _slices_to_plot_:
        plt.figure(figsize=(10, 2.5), dpi=100)
        plt.imshow(smp.gs[_s_].lgi), plt.colorbar()
        plt.title(f'Sample tslice={_s_}. Ng={smp.gs[_s_].n}')
# ###################################################################
# ###################################################################
# ###################################################################
# ###################################################################
tgt_slice = 2
smp_slice = 4
# =========================================================
"""
Calculate the nth order neighbouring grain list for target and sample grain
structure grains.
"""
neigh_order = 1
tgt_nneighgids = tgt.gs[tgt_slice].get_upto_nth_order_neighbors_all_grains(neigh_order,
                                                                           include_parent=True,
                                                                           output_type='nparray')
smp_nneighgids = smp.gs[smp_slice].get_upto_nth_order_neighbors_all_grains(neigh_order,
                                                                           include_parent=True,
                                                                           output_type='nparray')
# ###################################################################
# ###################################################################
# ###################################################################
# ###################################################################
"""
Calculate the grsin sizes (metric: number of pixels) in target and sample
grain structure.
"""
tgt_npixels = tgt.gs[tgt_slice].find_grain_size_fast(metric='npixels')
smp_npixels = smp.gs[smp_slice].find_grain_size_fast(metric='npixels')
tgt_npixels_mean = tgt_npixels.mean()
smp_npixels_mean = smp_npixels.mean()
# =========================================================
"""
construct the target and sample field dictionary with the following keys.
    neigh: Neighbours of the grain number gid
    nneigh: Number of neighbours of the grain number gid
    npixels: Number of pixels in each grain
    gblength_npixels: Currently None
    aspect_ratio: Currently None
"""
tgt_fields = {gid: {'neigh': tgt_nneighgids[gid],
                    'nneigh': len(tgt_nneighgids[gid]),
                    'npixels': tgt_npixels[gid-1],
                    'gblength_npixels': None,
                    'aspect_ratio': None} for gid in tgt.gs[tgt_slice].gid}
smp_fields = {gid: {'neigh': smp_nneighgids[gid],
                    'nneigh': len(smp_nneighgids[gid]),
                    'npixels': smp_npixels[gid-1],
                    'gblength_npixels': None,
                    'aspect_ratio': None} for gid in smp.gs[smp_slice].gid}
# =========================================================
"""Access centroids of target and sample grains"""
tgt_grain_centroids = tgt.gs[tgt_slice].centroids
# tgt_grain_centroids_mp = MPoint2d.from_coords(tgt_grain_centroids)
smp_grain_centroids = smp.gs[smp_slice].centroids
# smp_grain_centroids_mp = MPoint2d.from_coords(smp_grain_centroids)
# =========================================================
"""
Calculate target and sample fields of: nneigh and npixels
"""
tgt_nneigh_field = np.array([tgt_fields[gid]['nneigh'] for gid in tgt.gs[tgt_slice].gid])
tgt_npixels_field = np.array([tgt_fields[gid]['npixels'] for gid in tgt.gs[tgt_slice].gid])

smp_nneigh_field = np.array([smp_fields[gid]['nneigh'] for gid in smp.gs[smp_slice].gid])
smp_npixels_field = np.array([smp_fields[gid]['npixels'] for gid in smp.gs[smp_slice].gid])
# =========================================================
"""
Get underlyting target and sample grid.
"""
tgt_xgrid, tgt_ygrid = tgt.gs[tgt_slice].xgr, tgt.gs[tgt_slice].ygr
smp_xgrid, smp_ygrid = smp.gs[smp_slice].xgr, smp.gs[smp_slice].ygr
# ###################################################################
# ###################################################################
# ###################################################################
# ###################################################################
"""
Interpolate number of neighbors field onto the underlying target and sample
grid and visualize.
"""
tgt_nneigh_field_grid = griddata(tgt_grain_centroids, tgt_nneigh_field, (tgt_xgrid, tgt_ygrid), method='nearest')
smp_nneigh_field_grid = griddata(smp_grain_centroids, smp_nneigh_field, (smp_xgrid, smp_ygrid), method='nearest')
min_z = min(np.nanmin(tgt_nneigh_field_grid), np.nanmin(smp_nneigh_field_grid))
max_z = np.round(max(np.nanmax(tgt_nneigh_field_grid), np.nanmax(smp_nneigh_field_grid))/10)*10
# =========================================================
"""
Plot the number of O(n) neighbours for target and sample at a given temporal
slice.
"""
PROCEED = True
# vmax_custom = tgt_nneigh_field_grid.max()
if PROCEED:
    GS_name = 'Target'  # Options: Target, Sample
    # --------------------------
    vmax_custom = 500
    cbar_tick_incr = 50
    # --------------------------
    if GS_name in ('target', 'Target'):
        temporal_slice = tgt_slice
        num_of_grains = tgt.gs[tgt_slice].n
        nneigh_field_grid = tgt_nneigh_field_grid
        mgs = np.round(tgt_npixels_mean, 1)
    elif GS_name in ('sample', 'Sample'):
        temporal_slice = smp_slice
        num_of_grains = smp.gs[smp_slice].n
        nneigh_field_grid = smp_nneigh_field_grid
        mgs = np.round(smp_npixels_mean, 1)
    # --------------------------
    title_mgs = mgs
    # --------------------------
    plt.figure(figsize=(6, 5), dpi=150)
    levels = np.arange(0, vmax_custom+2, 2)
    contour = plt.contourf(nneigh_field_grid,
                           levels=levels, vmin=0, vmax=vmax_custom,
                           cmap='nipy_spectral')
    # plt.scatter(tgt_grain_centroids[:,0], tgt_grain_centroids[:,1], s=1, color='black')
    plt.xlabel('x-axis'), plt.ylabel('y-axis')

    plt.title(f'{GS_name} tslice={temporal_slice}. Ng={num_of_grains}. AGS.<Npx>={title_mgs}. Number of O({neigh_order}) neighbours', fontsize=12)
    axs = plt.gca()
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    # REF: https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    divider = make_axes_locatable(axs)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    # plt.colorbar(contour, format=ticker.FormatStrFormatter('%.0f'), location='right')
    cbar = plt.colorbar(contour, format=ticker.FormatStrFormatter('%.0f'),
                        cax=cax, ticks=range(0, vmax_custom+cbar_tick_incr, cbar_tick_incr),
                        label=f'No. of  O({neigh_order}) neighbours')
    cbar.ax.tick_params(labelsize=10)
    axs.set_aspect('equal', 'box')
# ###################################################################
# ###################################################################
# ###################################################################
# ###################################################################
"""
Carry out O(n) sensitivity on the nneigh statistics.
"""
DO_THIS_STUDY = True

if DO_THIS_STUDY:
    GS_name = 'Sample'
    neighbour_orders = np.arange(1, 10, 1)

    if GS_name in ('target', 'Target'):
        time_slice = tgt_slice
        GSTR = tgt.gs
    elif GS_name in ('sample', 'Sample'):
        time_slice = smp_slice
        GSTR = smp.gs

    GSTR_tslice = GSTR[time_slice]

    nneigh_no = {}

    for no in neighbour_orders:
        print(f'{GS_name}. Working on neighbour order: {no}')
        gstr_nneighgids = GSTR_tslice.get_upto_nth_order_neighbors_all_grains(no, include_parent=True,
                                                                      output_type='list')
        gstr_nneigh_field = np.array([len(gstr_nneighgids[gid]) for gid in GSTR_tslice.gid])

        nneigh_no[no] = list(gstr_nneigh_field)
    nneigh_no = pd.DataFrame(nneigh_no)

    tslices = np.array(list(GSTR.keys()))
    tslices = np.hstack((tslices[1:10:2], tslices[10::4]))
    neighbour_orders = np.arange(1, 10, 1)
    mean_nneigh_no_all_tslices = {}
    for tslice in tslices:
        GSTR_tslice = GSTR[tslice]
        nneigh_no = {}
        print(40*'-')
        for no in neighbour_orders:
            print(f'{GS_name}. tslice: {tslice}. Working on neighbour order: {no}')
            gstr_nneighgids = GSTR_tslice.get_upto_nth_order_neighbors_all_grains(no, include_parent=True,
                                                                          output_type='list')
            gstr_nneigh_field = np.array([len(gstr_nneighgids[gid]) for gid in GSTR_tslice.gid])
            nneigh_no[no] = list(gstr_nneigh_field)
        nneigh_no = pd.DataFrame(nneigh_no)
        mean_nneigh_no_all_tslices[tslice] = nneigh_no.mean()

    grain_areas_all_tslices = {}
    for tslice in list(GSTR.keys()):
        print(f'{GS_name}. Extracting grain pixel areas @tslice={tslice}')
        grain_areas_all_tslices[tslice] = GSTR[tslice].find_grain_size_fast(metric='npixels')

    fig, ax = plt.subplots(1, 3, figsize=(18, 5), dpi=100)
    snsbph = sns.boxplot(nneigh_no, notch=True, ax=ax[0])
    snsbph.set_xlabel('Neighbour order, O(n)', fontsize=14)
    snsbph.set_ylabel(f'Number of neighbours, {GS_name} @tslice={time_slice}', fontsize=13)
    snsbph.set_xticklabels(snsbph.get_xticklabels(), fontsize=12)
    snsbph.set_yticklabels(snsbph.get_yticklabels(), fontsize=12)

    for tslice, nneigh_no_tslice in zip(tslices, mean_nneigh_no_all_tslices.values()):
        ax[1].plot(neighbour_orders, nneigh_no_tslice, label=f'Ng={GSTR[tslice].n} @tslice={tslice}')
    ax[1].legend()
    ax[1].set_xlabel('Neighbour order, O(n)', fontsize=14)
    ax[1].set_ylabel(f'Mean number of neighbours, {GS_name}', fontsize=14)
    ax[1].set_xlim([0, neighbour_orders.max()+1])

    data = list(grain_areas_all_tslices.values())
    keys = grain_areas_all_tslices.keys()
    colormap = cm.get_cmap('nipy_spectral')
    COLORS = [colormap(i / len(keys)) for i in range(len(keys))]

    mplboxh = ax[2].boxplot(data, patch_artist=True)
    for patch, color in zip(mplboxh['boxes'], COLORS):
        patch.set_facecolor(color)
    NOutliers = []
    for key, _data_, color in zip(keys, data, COLORS):
        ax[2].plot([], [], color=color, label=f'#outliers: {len(find_outliers_iqr(_data_))}')
    ax[2].set_xticks(ticks=np.arange(1, len(keys) + 1), labels=keys)
    for flier in mplboxh['fliers']:
        flier.set(marker='.', color='red', alpha=0.5, markersize=0.5)
    for median in mplboxh['medians']:
        median.set(color='black', linewidth=2)
    ax[2].set_xlabel('Temporal slices', fontsize=14)
    ax[2].set_ylabel(f'Grain size. Metric: npixels, {GS_name}', fontsize=14)
    ax[2].legend(fontsize=8, ncol=2)
# =========================================================
# tgt.gs[tgt_slice].plot_grains(tgt_nneighgids[150])
# tgt.gs[tgt_slice].n
# =========================================================
# ###################################################################
# ###################################################################
# ###################################################################
# ###################################################################
"""
Visualize the target's number of neighbours vs grain area (i.e. npixels)
distributions in a seaborn jointplot.
"""
# g = sns.jointplot(x=tgt_npixels, y=tgt_nneigh_field, kind='reg')
# g = sns.jointplot(x=tgt_npixels, y=tgt_nneigh_field, kind='hist')
PROCEED = True
case = 'target-target'

if case == 'target-target':
    GS_name = 'Target'
    gstr_npixels = tgt_npixels
    gstr_nneigh_field = tgt_nneigh_field
    temporal_slice = tgt_slice
elif case == 'sample-sample':
    GS_name = 'Sample'
    gstr_npixels = smp_npixels
    gstr_nneigh_field = smp_nneigh_field
    temporal_slice = smp_slice

vmax = max(tgt_nneigh_field.max(), smp_nneigh_field.max())

plot_points = True
nrandpoints_to_plot_factor = 0.1

fit_regression_line = False
plot_regression_line = False
fit_polynomial_order = 1
fit_factor = 2
cmap_name = 'nipy_spectral'
if PROCEED:
    g = sns.jointplot(x=gstr_npixels, y=gstr_nneigh_field,
                      kind='hex', gridsize=20,
                      cmap=cmap_name,
                      marginal_kws=dict(bins=100,
                                        stat='density',
                                        cumulative=False,
                                        element='step',
                                        edgecolor='none',
                                        hist_kws=dict(color='lightgray'),
                                        kde=False,
                                        line_kws=dict(linewidth=2,
                                                      color='black'
                                                      ),
                                        fill=True,
                                        color='gray'
                                        ),
                      marginal_ticks=True,
                      height=5, ratio=5)
    fig, ax = g.fig, g.ax_joint
    hb = ax.collections[0]

    counts = hb.get_array()  # Get the array of counts
    probabilities = counts / counts.sum()  # Normalize counts to probabilities
    hb.set_array(probabilities)  # Update the hexbin plot with probabilities
    # hb.set_norm(plt.Normalize(vmin=0, vmax=2000))
    hb.set_clim(0, 1.0)
    cbar = fig.colorbar(hb,
                        ax=ax,
                        orientation='vertical',
                        format=ticker.FormatStrFormatter('%.1f'),
                        ticks=np.arange(0, 1.0+0.1, 0.1)
                        )
    # cbar.set_label('Counts')
    # g.plot_marginals(sns.histplot, bins=50, kde=True, color='black', fill=True)
    # g.fig.suptitle('Customized Jointplot of A and B', y=1.02)
    g.set_axis_labels(f'Grain area (N Pixels), {GS_name} @tslice={temporal_slice}',
                      f'Number of (O{neigh_order}) neighbours, {GS_name} @tslice={temporal_slice}')
    g.ax_joint.grid(True, linestyle='--', alpha=0.7)

    if plot_points or nrandpoints_to_plot_factor > 1E-2:
        Npoints_to_plot = int(nrandpoints_to_plot_factor * gstr_npixels.size)
        points_to_plot = RNG.choice(np.arange(0, Npoints_to_plot, 1),
                                    size=Npoints_to_plot, replace=False)
        plt.scatter(gstr_npixels[points_to_plot],
                    gstr_nneigh_field[points_to_plot],
                    s=1, color='black', alpha=0.1)

    if fit_regression_line:
        fit_polynomial_order = fit_polynomial_order
        fit_factor = fit_factor
        sort_indices = np.argsort(gstr_npixels)
        gstr_npixels_limited = gstr_npixels[sort_indices][gstr_npixels[sort_indices] <= fit_factor*gstr_npixels.mean()]
        gstr_nneigh_field_limited = gstr_nneigh_field[sort_indices][gstr_npixels[sort_indices] <= fit_factor*gstr_npixels.mean()]
        coefficients = np.polyfit(gstr_npixels_limited, gstr_nneigh_field_limited, fit_polynomial_order)
        polynomial = np.poly1d(coefficients)
        gstr_nneigh_field_fit = polynomial(gstr_npixels_limited)
        if plot_regression_line:
            plt.plot(gstr_npixels_limited,
                     gstr_nneigh_field_fit,
                     'orange')
    plt.show()

# g = sns.jointplot(x=tgt_npixels, y=tgt_nneigh_field, kind='reg')
# g = sns.jointplot(x=tgt_npixels, y=tgt_nneigh_field, kind='hist')

# ###################################################################
# ###################################################################
# ###################################################################
# ###################################################################
"""
# Visualize all grains upto nth order neighbour of a given grain id.

plt.figure()
ngids = tgt.gs[tgt_slice].get_upto_nth_order_neighbors(10,
                                                       10,
                                                       recalculate=False,
                                                       include_parent=False,
                                                       output_type='list')
if type(ngids) not in dth.dt.ITERABLES:
    ngids = [ngids]
tgt.gs[tgt_slice].plot_grains(ngids)

# ---------------------------------
# Visualize all nth order neighbour grains of a given grain id.

plt.figure()
ngids = tgt.gs[tgt_slice].get_nth_order_neighbors(10,
                                                  10,
                                                  recalculate=False,
                                                  include_parent=False)
if type(ngids) not in dth.dt.ITERABLES:
    ngids = [ngids]
tgt.gs[tgt_slice].plot_grains(ngids)
"""
# ********************************************************************
"""
Calculate the Kullback-Leiber divergence R-field for target1:target2, where,
    target1: target npixels field for the entire target grain structure
    target2: sample npixels field of all O(n) neighbours of every grain in the
        target grain structure
NOTE: This uses number of neighbours, tgt_npixels.
"""
PROCEED = True
plot_centroids = False
vmax_custom = 20
cbar_tick_incr = 2

case = 'target-target'  # 'target-target', 'sample-sample', 'target-sample'

if case == 'target-target':
    gstrT = tgt.gs[tgt_slice]
    gstrS = smp.gs[smp_slice]
    gstrT_NPIXELS_FIELD = tgt_npixels_field
    gstrS_NPIXELS_FIELD = tgt_npixels_field
    gstrT_nneighgids = tgt_nneighgids
    gstrS_nneighgids = smp_nneighgids
    gstr_grain_centroids = tgt_grain_centroids
    gstr_xgrid, gstr_ygrid = tgt_xgrid, tgt_ygrid
    infot, infos = 'tgt', 'tgt'
elif case == 'sample-sample':
    gstr_NPIXELS_FIELD = smp_npixels_field
    gstr_nneighgids = smp_nneighgids
    gstr_grain_centroids = smp_grain_centroids
    gstr_xgrid, gstr_ygrid = smp_xgrid, smp_ygrid
    infot, infos = 'tgt', 'tgt'
elif case == 'target-sample':
    gstrT_NPIXELS_FIELD = tgt_npixels_field
    gstrS_NPIXELS_FIELD = smp_npixels_field
    gstr_nneighgids = smp_nneighgids
    gstr_grain_centroids = tgt_grain_centroids
    gstr_xgrid, gstr_ygrid = tgt_xgrid, tgt_ygrid
    infot, infos = 'tgt', 'tgt'

if PROCEED:
    KLD = []
    '''Estimate the bin using Target grain striucture'''
    bins_maxT = int(np.array([len(gstrT_nneighgids[gid])
                              for gid in gstrT.gid]).mean())
    '''Estimate the bin using Sample grain striucture'''
    bins_maxS = int(np.array([len(gstrS_nneighgids[gid])
                              for gid in gstrS.gid]).mean())
    '''Estimate the mean bin'''
    bins_max = int(np.mean([bins_maxT, bins_maxS]))
    """
    Definitions
    -----------
    R field is the field of local representativeness of a grian structure
    with values calculated at grain centroids using using O(n) local
    neighbourhood of each grain, where n is the chosen neighbour order.

    Global target grain structure (GTGS) and global sample grain struycture
    (GSGS): GTGS represents the reference grain structure against which the
    representativeness of GSGS is beng globally assessed.

    GSTRT: Target grain structure in this repr qualification. Could be the
        global target grain structure, or global global sample grain structure.
    GSTRS: Sample grain structure in this repr qualification. Could be the
        global target grain structure, or global global sample grain structure.

    Concept
    -------
    We check the local neighbourhood similarity of grains in a GSTRS with
    GSTRT. The GSTRS and GSTRT could each be either the Target or the Sample
    grain structure trhemseolves.

    Trivials
    --------
    1. Different n in neighbour order O(n) gives different repr values in the
    R field.

    Hypotheses
    ----------
    1.

    Procedure
    ---------
    STEP 1. Calculate Base O(n) R-field for global target: R(X):gt|gt:O(n).
        a. take the GTGS's distribution for parameter of choice
        b. for each grain id, gid, in the GTGS, calculate R:gt|gt(gid.O(n))
        c. Repeat b. for all gids in the GTGS.
        d. Map the above R:gt|gt(gid.O(n)) to grain centroids to get
            R(Xc):gt|gt:O(n).
        e. INterpolate this onto a grid bounded by GTGS bounds to get,
            R(X):gt|gt:O(n).

    STEP 2. Calculate Base O(n) R-field for global sample: R(X):gs|gs:O(n).
        a. take the GSGS's distribution for parameter of choice
        b. for each grain id, gid, in the GSGS, calculate R:gs|gs(gid.O(n))
        c. Repeat b. for all gids in the GSGS.
        d. Map the above R:gs|gs(gid.O(n)) to grain centroids to get
            R(Xc):gs|gs:O(n).
        e. INterpolate this onto a grid bounded by GSGS bounds to get,
            R(X):gs|gs:O(n).

    STEP 3. Calculate the spatial uniformity value, SUV.O(n) for the present
        parameter of choice. SUV.O(n) has a range of [0.0, 1.0]. A high value
        suggests the presence of a high spatial uniformity in the chosen
        parameter's distribution over the entire domain of the grain structure.
        Further, as these values are calculated on the relative R value of T
        and S, a uniform spatial distribution in |ΔR(T-S)| suggests the
        similarity of S with T in terms of spatial distribution of R-value over
        the morphological parameter under consideration.
        A. If R(X) of target and sample have the same underlying grid:
            a. Calculate ΔR(T-S) = R(X):gt|gt:O(n) - R(X):gs|gs:O(n).
            b. Calculate quartiles |ΔR(T-S)|.Qi, where, i is in (0,1,2,3,4).
            c. Bin the gid over quartiles to get: [|ΔR(T-S)|].Qi
            d. Calculate the median ΔR~ = |ΔR(T-S)|~
            e. Calculate the standard deviatiuon σ = |ΔR(T-S)|
            f. Calculate ΔR~ + σ/2
            g. Distribute [0, ΔR~-σ/2], (ΔR~-σ/2, ΔR~+σ/2] and (ΔR~+σ/2, ΔRmax]
               into n1, n2 and n3 bins. These represent the three zones Z1, Z2
               and Z3. Refer to my gemini AI chat to find the optimial bin
               width here: https://g.co/gemini/share/057c950070f4
               Z1 = [0, ΔR~-σ/2]. Each bin in Z1 is a sub-zone: Z1.1, Z1.2,..
               Z2 = (ΔR~-σ/2, ΔR~+σ/2]
               Z3 = (ΔR~+σ/2, ΔRmax]
            h. Identify gids_pq in each subzone q of zone p, where,
               q = 1, 2, 3, ..., Q and p = 1, 2, 3. Q: Number of zones.
            i. For each gid set in gids_pq, get the centroidal distance array
               of seperation using the following example. Call this distance
               as D_pq.
                   from scipy.spatial.distance import pdist, squareform
                   centroids = [(1, 2), (3, 4), (5, 6), (7, 8)]
                   distances_matrix = squareform(pdist(centroids))
                   triu_indices = np.triu_indices_from(distances_matrix, k=1)
                   distances = distances_matrix[triu_indices]
            j. Calculate isrand = {'test_method_1': bool,...
                                   'test_method_n': bool}.
               Available test methods are:
                   . test_rand_distr_autocorr (test1)
                   . test_rand_distr_runs (test2)
                   . test_rand_distr_chisquare (test3)
                   . test_rand_distr_kolmogorovsmirnov (test4)
                   . test_rand_distr_kullbackleibler (test5)
            k. Access the user input (isrand_assess_method) on how to use
               isrand. Options for isrand_assess_method include the following.
                   . 'accept_if_any_true':
                           tests = (test1, test2, test3, test4, test5)
                           result_tests_pq = True if any(tests) else False
                           # Example:
                           # Ex: tests = (False, True, True, True, True)
                           # result_tests_pq = True if any(tests) else False
                   . 'accept_if_all_true'
                           tests = (test1, test2, test3, test4, test5)
                           result_tests_pq = True if all(tests) else False
                           # Example:
                           # Ex: tests = (False, True, True, True, True)
                           # result_tests_pq = True if all(tests) else False
                   . 'reject_if_any_false'
                           tests = (test1, test2, test3, test4, test5)
                           result_tests_pq = True if not any(not test for test in tests) else False
                           # Example:
                           # tests = (False, True, True, True, True)
                           # result_tests_pq = True if not any(not test for test in tests) else False
                   . 'specific'
                           tests = {'test_rand_distr_autocorr': False,
                                    'test_rand_distr_runs': True,
                                    'test_rand_distr_chisquare': True,
                                    'test_rand_distr_kolmogorovsmirnov': True,
                                    'test_rand_distr_kullbackleibler': True}
                           specific = ['test_rand_distr_autocorr',
                                       'test_rand_distr_kullbackleibler',
                                       'test_rand_distr_chisquare']
                           result_tests_pq = all([tests[mt] for mt in mandatory])
            l. If result_tests_pq is True, then the distances in D_pq is indeed
               random. This proves that the grains falling in the zone p and
               subzone q have a uniform spatial distribution.
            m. Repeat j, k and l over all gids and assimilate result_tests_pq
               for all zones and sub-zones. result_tests_pq would then be a
               fully populated q x p matrix. First column represents Z1, second
               Z2 and third Z3. The third row of second column would represent
               thew third subzone of second zone.
            o. Calculate the spatial uniformity value in |ΔR(T-S)|@O(n)
               represented by SUV_n = SUV.O(n) for nth order neighbour as:
                   SUV_n = np.where(result_tests_pq)[0].size/result_tests_pq.size
            -------------------------------------------------------------------
        B. If R(X) of target and sample have different underlying grids:
            B1. Proceed by extracting a similar sized representative subset
                from the target and proceed with step A. To get the similar
                size, optimize the Ng.
            B2. Take the statistical distribution approach.

    STEP 3. Calculate O(n) R-field for global target - global sample: R(X):gt|gs.
        a. take the GSGS's distribution for parameter of choice
        b. for each grain id, gid, in the GSGS, calculate R:gs|gs(gid.O(n))
        c. Repeat b. for all gids in the GSGS.
        d. Map the above R:gs|gs(gid.O(n)) to grain centroids to get
            R(Xc):gs|gs(gid.O(n)).
        e. INterpolate this onto a grid bounded by GSGS bounds to get,
            R(X):gs|gs(gid.O(n)).
    """
    for gid in tgt.gs[tgt_slice].gid:
        if gid % 500 == 0:
            print(f'Estimating @O({neigh_order}) Non-local representativeness field value [RE-KLD (T-{infot}|S-{infos}.grains)] for grain no. {gid}')
        """ Get the npixels property field of current grain in the target grain
        structure. This will form the sample being repr tested.
        """
        sample_npixels_field_gid = tgt_npixels_field[np.array(tgt_nneighgids[gid])-1]
        """ Use sample_npixels_field_gid alolng with npixels of entire target
        and calculkate bins. """
        bins = np.histogram_bin_edges(np.concatenate([tgt_npixels_field,
                                                      sample_npixels_field_gid]),
                                      bins=bins_max)
        '''Calculate the target histogram.'''
        target_hist, _ = np.histogram(tgt_npixels_field, bins=bins, density=True)
        target_hist = np.where(target_hist == 0, 1e-10, target_hist)
        '''Calculate the sample histogram.'''
        sample_hist, _ = np.histogram(sample_npixels_field_gid, bins=bins, density=True)
        sample_hist = np.where(sample_hist == 0, 1e-10, sample_hist)
        """Calculate repr measure for this grain. Once this loop completes,
        these repr values would represent the repr values of individual grains
        of non-local O(n) parameter reprsentativeness field. As each greain
        possess a unique repr measure, we will map the repr measure value to
        the centroid of the grain. This makes the repr measure a field in the
        physical space of the grain structurte."""
        KLD.append(entropy(target_hist, sample_hist))
    """
    Now that the repr field has been established at the grain centroids, we
    will now interpolate these values onto the grid defined by the bounds of
    the corresponding grain structure. We will only use the nearest option
    to carry out this interpolation as other are not expected to yield any
    significant benefits over nearest. Furtherm using options other than
    nearest would lead to unavailable values at places onthe grid near
    boundaries, where interpolations would not be possible to be carried out.
    """
    tgt_KLD_grid = griddata(tgt_grain_centroids, KLD, (tgt_xgrid, tgt_ygrid), method='nearest')
    min_z = np.nanmin(tgt_KLD_grid)
    max_z = np.nanmax(tgt_KLD_grid)
    # --------------------------------------
    """
    Visualize the R field of the grain structure for a given a parent-subset
    dataset.
    """
    plt.figure(figsize=(6, 5), dpi=150)
    levels = np.arange(0, vmax_custom, 0.1)
    ax = plt.gca()
    contour = plt.contourf(tgt_KLD_grid, levels=levels, cmap='nipy_spectral')
    # contour = plt.contourf(smp_KLD_grid, cmap='nipy_spectral')
    if plot_centroids:
        plt.scatter(smp_grain_centroids[:,0], smp_grain_centroids[:,1], s=0.5, color = 'black', alpha=0.5)
    plt.xlabel('x-axis'), plt.ylabel('y-axis')
    plt.title(f'Non-local R-field: KLD (T-tgt|S-tgt.grains.O({neigh_order})) neighbours')
    ax.set_aspect('equal', 'box')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    plt.colorbar(contour,
                 cax=cax,
                 orientation='vertical',
                 format=ticker.FormatStrFormatter('%.0f'),
                 ticks=range(0,
                             vmax_custom+cbar_tick_incr,
                             cbar_tick_incr)
                 )
# ********************************************************************
"""
Calculate the Kullback-Leiber divergence for target1:target2, where,
    target1: target npixels field for the entire target grain structure
    target2: sample npixels field of all O(n) neighbours of every grain in the
        target grain structure
NOTE: This uses number of neighbours, tgt_npixels.
"""
PROCEED = True
plot_centroids = False
vmax_custom = 20
cbar_tick_incr = 2
if PROCEED:
    KLD = []
    bins_max = int(np.array([len(tgt_nneighgids[gid]) for gid in tgt.gs[tgt_slice].gid]).mean())
    for gid in tgt.gs[tgt_slice].gid:
        if gid % 500 == 0:
            print(f'Estimating @O({neigh_order}) Non-local representativeness field value [RE-KLD (T-tgt|S-tgt.grains)] for grain no. {gid}')
        tgt_npixels_field_gid = tgt_npixels_field[np.array(tgt_nneighgids[gid])-1]
        bins = np.histogram_bin_edges(np.concatenate([tgt_npixels_field,
                                                      tgt_npixels_field_gid]),
                                      bins=bins_max)
        target_hist, _ = np.histogram(tgt_npixels_field, bins=bins, density=True)
        sample_hist, _ = np.histogram(tgt_npixels_field_gid, bins=bins, density=True)
        target_hist = np.where(target_hist == 0, 1e-10, target_hist)
        sample_hist = np.where(sample_hist == 0, 1e-10, sample_hist)
        KLD.append(entropy(target_hist, sample_hist))
    tgt_KLD_grid = griddata(tgt_grain_centroids, KLD, (tgt_xgrid, tgt_ygrid), method='nearest')
    min_z = np.nanmin(tgt_KLD_grid)
    max_z = np.nanmax(tgt_KLD_grid)
    # --------------------------------------
    plt.figure(figsize=(6, 5), dpi=150)
    levels = np.arange(0, vmax_custom, 0.1)
    ax = plt.gca()
    contour = plt.contourf(tgt_KLD_grid, levels=levels, cmap='nipy_spectral')
    # contour = plt.contourf(smp_KLD_grid, cmap='nipy_spectral')
    if plot_centroids:
        plt.scatter(smp_grain_centroids[:,0], smp_grain_centroids[:,1], s=0.5, color = 'black', alpha=0.5)
    plt.xlabel('x-axis'), plt.ylabel('y-axis')
    plt.title(f'Non-local R-field: KLD (T-tgt|S-tgt.grains.O({neigh_order})) neighbours')
    ax.set_aspect('equal', 'box')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    plt.colorbar(contour,
                 cax=cax,
                 orientation='vertical',
                 format=ticker.FormatStrFormatter('%.0f'),
                 ticks=range(0,
                             vmax_custom+cbar_tick_incr,
                             cbar_tick_incr)
                 )
# g = sns.jointplot(x=tgt_npixels, y=KLD, kind='reg')
# g = sns.jointplot(x=tgt_nneigh_field, y=KLD, kind='reg')

"""
Calculate the Kullback-Leiber divergence for target1:target2, where,
    target1: target npixels field for the entire sample grain structure
    target2: sample npixels field of all O(n) neighbours of every grain in the
        sample grain structure
NOTE: This uses number of neighbours, smp_npixels.
"""
PROCEED = True
plot_centroids = True
vmax_custom = 18
cbar_tick_incr = 2

if PROCEED:
    KLD = []
    bins_max1 = int(np.array([len(tgt_nneighgids[gid]) for gid in tgt.gs[tgt_slice].gid]).mean())
    bins_max2 = int(np.array([len(smp_nneighgids[gid]) for gid in smp.gs[smp_slice].gid]).mean())
    bins_max = int(np.mean([bins_max1,bins_max2]))
    for gid in smp.gs[smp_slice].gid:
        smp_npixels_field_gid = smp_npixels_field[np.array(smp_nneighgids[gid])-1]
        bins = np.histogram_bin_edges(np.concatenate([smp_npixels_field,
                                                      smp_npixels_field_gid]),
                                      bins=bins_max)
        target_hist, _ = np.histogram(smp_npixels_field, bins=bins, density=True)
        sample_hist, _ = np.histogram(smp_npixels_field_gid, bins=bins, density=True)
        target_hist = np.where(target_hist == 0, 1e-10, target_hist)
        sample_hist = np.where(sample_hist == 0, 1e-10, sample_hist)
        KLD.append(entropy(target_hist, sample_hist))

    smp_KLD_grid = griddata(smp_grain_centroids, KLD, (smp_xgrid, smp_ygrid), method='nearest')
    min_z, max_z = np.nanmin(smp_KLD_grid), np.nanmax(smp_KLD_grid)
    # --------------------------------------
    plt.figure(figsize=(6, 5), dpi=150)
    levels = np.arange(0, vmax_custom, 0.5)
    ax = plt.gca()
    contour = plt.contourf(tgt_KLD_grid, levels=levels, cmap='nipy_spectral')
    # contour = plt.contourf(smp_KLD_grid, cmap='nipy_spectral')
    if plot_centroids:
        plt.scatter(smp_grain_centroids[:,0], smp_grain_centroids[:,1], s=3, color = 'black')
    plt.xlabel('x-axis'), plt.ylabel('y-axis')
    plt.title(f'R-field: RE-KLD (T-smp|S-smp.grains.O({neigh_order})) neighbours')
    ax.set_aspect('equal', 'box')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    plt.colorbar(contour,
                 cax=cax,
                 orientation='vertical',
                 format=ticker.FormatStrFormatter('%.0f'),
                 ticks=range(0,
                             vmax_custom+cbar_tick_incr,
                             cbar_tick_incr)
                 )

# g = sns.jointplot(x=smp_nneigh_field, y=KLD, kind='reg')
"""
Calculate the Kullback-Leiber divergence for target1:target2, where,
    target1: target npixels field for the entire target grain structure
    target2: sample npixels field of all O(n) neighbours of every grain in the
        sample grain structure
NOTE: This uses number of neighbours, tgt_nneigh, smp_npixels.
"""
PROCEED = True
plot_centroids = True
if PROCEED:
    KLD = []
    bins_max_tgt = int(np.array([len(tgt_nneighgids[gid]) for gid in tgt.gs[tgt_slice].gid]).mean())
    bins_max_smp = int(np.array([len(smp_nneighgids[gid]) for gid in smp.gs[smp_slice].gid]).mean())
    bins_max = int((bins_max_tgt + bins_max_smp)/2)
    for gid in smp.gs[smp_slice].gid:
        smp_npixels_field_gid = smp_npixels_field[np.array(smp_nneighgids[gid])-1]
        bins = np.histogram_bin_edges(np.concatenate([tgt_npixels_field,
                                                      smp_npixels_field_gid]),
                                      bins=bins_max)
        target_hist, _ = np.histogram(tgt_npixels_field, bins=bins, density=True)
        sample_hist, _ = np.histogram(smp_npixels_field_gid, bins=bins, density=True)
        target_hist = np.where(target_hist == 0, 1e-10, target_hist)
        sample_hist = np.where(sample_hist == 0, 1e-10, sample_hist)
        KLD.append(entropy(target_hist, sample_hist))

    trgsmp_KLD_grid = griddata(smp_grain_centroids, KLD, (smp_xgrid, smp_ygrid), method='nearest')
    min_z = np.nanmin(trgsmp_KLD_grid)
    max_z = np.nanmax(trgsmp_KLD_grid)

    plt.figure(figsize=(6, 5), dpi=150)
    levels = np.arange(0, 18, 2)
    contour = plt.contourf(tgt_KLD_grid, levels=levels, cmap='nipy_spectral')
    #contour = plt.contourf(trgsmp_KLD_grid, cmap='nipy_spectral')
    plt.colorbar(contour, format=ticker.FormatStrFormatter('%.1f'), location='bottom')
    if plot_centroids:
        plt.scatter(smp_grain_centroids[:,0], smp_grain_centroids[:,1], s=3, color = 'black')
    plt.xlabel('x-axis'), plt.ylabel('y-axis')
    plt.title(f'R-field: RE-KLD (T-tgt|S-smp.grains.O({neigh_order})) neighbours')
    axs = plt.gca()
    axs.set_aspect('equal', 'box')

PROCEED = True
if PROCEED:
    g = sns.jointplot(x=smp_nneigh_field, y=KLD, kind='reg')

"""
Calculate the Kullback-Leiber divergence for target1:target2, where,
    target1: target npixels field for the entire target grain structure
    target2: sample npixels field of all O(n) neighbours of every grain in the
        sample grain structure
NOTE: This uses number of neighbours, tgt_nneigh, smp_nneigh.
"""
PROCEED = True
if PROCEED:
    KLD = []
    bins_max_tgt = int(np.array([len(tgt_nneighgids[gid]) for gid in tgt.gs[tgt_slice].gid]).mean())
    bins_max_smp = int(np.array([len(smp_nneighgids[gid]) for gid in smp.gs[smp_slice].gid]).mean())
    bins_max = int((bins_max_tgt + bins_max_smp)/2)
    for gid in smp.gs[smp_slice].gid:
        smp_nneigh_field_gid = smp_nneigh_field[np.array(smp_nneighgids[gid])-1]
        bins = np.histogram_bin_edges(np.concatenate([tgt_nneigh_field,
                                                      smp_nneigh_field_gid]),
                                      bins=bins_max)
        target_hist, _ = np.histogram(tgt_nneigh_field, bins=bins, density=True)
        sample_hist, _ = np.histogram(smp_nneigh_field_gid, bins=bins, density=True)
        target_hist = np.where(target_hist == 0, 1e-10, target_hist)
        sample_hist = np.where(sample_hist == 0, 1e-10, sample_hist)
        KLD.append(entropy(target_hist, sample_hist))

    trgsmp_KLD_grid = griddata(smp_grain_centroids, KLD, (smp_xgrid, smp_ygrid), method='nearest')
    min_z = np.nanmin(trgsmp_KLD_grid)
    max_z = np.nanmax(trgsmp_KLD_grid)

    plt.figure(figsize=(6, 5), dpi=150)
    levels = np.arange(0, 18, 1)
    contour = plt.contourf(tgt_KLD_grid, levels=levels, cmap='nipy_spectral')
    # contour = plt.contourf(trgsmp_KLD_grid, cmap='nipy_spectral')
    plt.colorbar(contour, format=ticker.FormatStrFormatter('%.1f'), location='bottom')
    plt.scatter(smp_grain_centroids[:,0], smp_grain_centroids[:,1], s=3, color = 'black')
    plt.xlabel('x-axis'), plt.ylabel('y-axis')
    plt.title(f'R-field: RE-KLD (T-tgt|S-smp.grains.O({neigh_order})) neighbours')
    axs = plt.gca()
    axs.set_aspect('equal', 'box')
