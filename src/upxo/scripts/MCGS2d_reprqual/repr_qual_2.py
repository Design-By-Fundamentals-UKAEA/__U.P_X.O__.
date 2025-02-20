"""
Created on Tue May 14 12:44:36 2024

@author: Dr. Sunil Anandatheertha

Explanations
------------
This example provides codes using UPXO to caculate KL representativeness
measure between a target parent grain structure and sample grain structurwsa.
All samples are subset grain structures derived from parent by window sliding
by its lengtyh and height with zero overlap.

Following must be noted:
    * Target is synthetic
    * Samples are synthetic
    * Target's 20th temporal slice is ised as target in each above set
        slicenumber = 20.
    * In each above set, samples of slices [2, 6, 10, 14, 20, 24, 28, 32]
        are used. tslice = [2, 6, 10, 14, 20, 24, 28, 32].
"""
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors
from scipy.stats import entropy
from scipy.stats import gaussian_kde
from scipy.spatial import cKDTree
from upxo.ggrowth.mcgs import mcgs
from upxo.pxtalops.Characterizer import mcgs_mchar_2d
from upxo._sup import dataTypeHandlers as dth
from scipy.spatial.distance import pdist
from upxo.geoEntities.sline2d import Sline2d
from upxo.geoEntities.mulpoint2d import MPoint2d
from upxo.interfaces.user_inputs.excel_commons import read_excel_range
from upxo.interfaces.user_inputs.excel_commons import write_array_to_excel
from upxo._sup.data_ops import find_outliers_iqr

plt.rcParams['text.usetex'] = False
NUMBERS = dth.dt.NUMBERS
ITERABLES = dth.dt.ITERABLES
"""
NOTE: DO NOT CHANGE SETTINGS, IN THE INTEREST OF TOP EXPLANATIONS.
"""

"""Generate grain structure and detect the grains."""
tgt = mcgs(study='independent', input_dashboard='input_dashboard.xls')
tgt.simulate()
tgt.detect_grains()
"""Choose a temporal slice and characterize it."""
tslices = [2, 4, 10, 16, 20, 30, 40, 48]
"""We will choose to get only the grain areas and ignore the others.
We can adopt two approaches:
    Approach 1: Use tgt.char_morph_2d(...)
    Approach 2: Use tgt.find_grain_areas_fast(...)

The first one does a whole lot than just calculate grain areas. It has options
to control what you characterize. However, we will use the second approacj
as it is significantly faster and moreover, we are not intersted in anythig
else other than the grain areas as of this moment.

The Approach 1 is:
    tgt.char_morph_2d(tslices, bbox=False, bbox_ex=True, npixels=False,
                      npixels_gb=False, area=True, eq_diameter=False,
                      perimeter=False, perimeter_crofton=False,
                      compactness=False, gb_length_px=False,
                      aspect_ratio=False, solidity=False, morph_ori=False,
                      circularity=False, eccentricity=False,
                      feret_diameter=False, major_axis_length=False,
                      minor_axis_length=False, euler_number=False,
                      append=False, saa=True, throw=False,
                      char_grain_positions=True, find_neigh=True, char_gb=True)
    tgt.gs[tslice].areas

The Approach 2 is:
    grain_areas_all_tslices = find_grain_areas_fast(tslices)

Both approaches provide number of pixe4ls in the grain.
"""
grain_areas_all_tslices = tgt.find_grain_areas_fast(tslices)

# border_gids, internal_gids, lgi_border, lgi_internal = tgt.find_border_internal_grains_fast(tslices)
# plt.imshow(lgi_border[48])

# border_grain_areas = tgt.find_npixels_border_grains_fast(tslices)
# internal_grain_areas = tgt.find_npixels_internal_grains_fast(tslices)

"""We will now store the excel cell IDs we need to read-write data."""
cell_locs = ['B11', 'B28', 'B45', 'B62', 'B79', 'B96', 'B113', 'B130']
cell_locs = {tslice: cell_loc for tslice, cell_loc in zip(tslices, cell_locs)}
"""Specify the excel file."""
file_name = r"C:\Development\M2MatMod\upxo_packaged\upxo_private\data\repr_dummy.xlsx"
"""Provide sheet names."""
sheet_names = ["Sheet1a", "Sheet1b", "Sheet1c", "Sheet1d"]
# ==============================================
"""Select"""
case = 0
sheet_name = sheet_names[case]
case_flags = [(True, True),
              (True, False),
              (False, True),
              (False, False)]

REMOVE_SMALLGRAINS, REMOVE_OUTLIERS = case_flags[case]
SmallGrainArea = 1.0
remove_small_grain_ineq_rule = 'le'

REMOVE_OUTLIERS_SUBSETS = False
REMOVE_SMALLGRAINS_SUBSETS = REMOVE_SMALLGRAINS
SmallGrainArea_subsets = SmallGrainArea
remove_small_grain_ineq_rule_subsets = remove_small_grain_ineq_rule

PERFORM_SAMPLE_TARGET_ASYMMETRY_TEST = False

ngrains = []
TARGET_GRAIN_AREAS = []
# ==============================================
"""LETS WORK WITH ONE TEMPORAL SLICE AT ONCE."""
for tslice in tslices:
    start_cell = cell_locs[tslice]
    """Lets plot the grain structure."""
    PLOT_PARENT_GS = False
    if PLOT_PARENT_GS:
        plt.figure(figsize=(6,5), dpi=150)
        tgt.gs[tslice].plot(cmap='viridis')
        plt.title(f'tslice={tslice}. Ng={len(tgt.gs[tslice].areas)}')
    """Get the areas of all grains in the tslice target grain structure."""
    target_grain_areas = deepcopy(grain_areas_all_tslices[tslice])
    # len(tgt.gs[tslice].areas)
    # tgt.gs[tslice].n
    """Extract the field matrix.
    Prescibe its minimum and maximum. We should not acvquire thwse directly from
    fmat as fmat may not allways have the minimum and maximum contained in it due
    to grain struvctuer evolution.
    """
    fmat = tgt.gs[tslice].s
    fmin, fmax = 1, tgt.uisim.S
    """Lets store all samples area here. We can use theser later to plot
    hostograms."""
    sample_areas_all = []
    """Set up flags for outlier and small grain removal.
    Also set up the small grain threshold and its inequality rule.

    NOTE: These only affect the area distribution array and not the actyual grain
    structure.
    """

    if REMOVE_OUTLIERS:
        tgt_indices = np.array(list(set(range(target_grain_areas.size)) - set(find_outliers_iqr(target_grain_areas))))
        target_grain_areas = target_grain_areas[tgt_indices]

    if REMOVE_SMALLGRAINS:
        if remove_small_grain_ineq_rule == 'le':
            target_grain_areas = target_grain_areas[np.argwhere(target_grain_areas > SmallGrainArea).T.squeeze()]
        elif remove_small_grain_ineq_rule == 'lt':
            target_grain_areas = target_grain_areas[np.argwhere(target_grain_areas >= SmallGrainArea).T.squeeze()]
    # len(tgt.gs[tslice].areas), len(target_grain_areas)
    TARGET_GRAIN_AREAS.append(target_grain_areas)


    ngrains_tgt = target_grain_areas.size
    """Instantialize the mcgs 2d charectization."""
    pxtchr = mcgs_mchar_2d()
    """Set field matrix you are about to subsetize and characterize."""
    pxtchr.set_fmat(fmat, fmin, fmax)
    """
    Subsetize the field matrix now.

    hfac: Specifies how many subsets we wish to have along y-axis
    vfac: Specifies how many subsets we wish to have along x-axis

    Example @(hfac,vfac):(1, 2)
    If fmat is:
    ########
    ########
    ########
    ########
    then subsets would be:
    ####     ####
    #### and ####
    ####     ####
    ####     ####

    Example @(hfac,vfac):(2, 1)
    If fmat is:
    ########
    ########
    ########
    ########
    then, subsets would be:
    ########
    ########
    and
    ########
    ########

    Example @(hfac,vfac):(2, 2)
    If fmat is:
    ########
    ########
    ########
    ########
    then, subsets would be:
    ####  ####
    ####  ####

    ####  ####
    ####  ####
    """
    PLOT_GS = False
    # ============================================
    """ Wwe will now create subsets. Lets choose a factor which divies the parent
    set into subsets. We will choose factors for horizontally dividing the parent
    ans also to v ertic ally divde the parent. This factor must obviously be > 1
    and less than size(parent) along h and v directions. The subsets are in
    fmats. These are subsets of the parent field matrix fmat.

    We will choose many such factors. They will be in hfacvfac
    """
    hfacvfac = [[2.0, 2.0],
                [2.25, 2.25],
                [2.5, 2.5],
                [3.0, 3.0],
                [3.5, 3.5],
                [4.0, 4.0],
                [5.0, 5.0],
                [6.0, 6.0],
                [8.0, 8.0],
                [10.0, 10.0],
                [12.5, 12.5],
                [15.0, 15.0]
                ]

    avg_number_of_grains = []
    # Subset feature
    ssfeat_nxm = {f'data_{hvfac[0]}x{hvfac[1]}': None for hvfac in hfacvfac}
    # Subset data
    data_nxm = {f'data_{hvfac[0]}x{hvfac[1]}': None for hvfac in hfacvfac}
    data_nxm_kldfit = {f'data_{hvfac[0]}x{hvfac[1]}': None for hvfac in hfacvfac}
    data_nxm_kldfit_Sline2d = {f'data_{hvfac[0]}x{hvfac[1]}': None for hvfac in hfacvfac}
    data_nxm_kldfit_mpoints = {f'data_{hvfac[0]}x{hvfac[1]}': None for hvfac in hfacvfac}
    data_nxm_kldfit_centroids = {f'data_{hvfac[0]}x{hvfac[1]}': None for hvfac in hfacvfac}
    data_nxm_kldfit_std = {f'data_{hvfac[0]}x{hvfac[1]}': None for hvfac in hfacvfac}
    data_keys = []
    hsizevsize = []
    for hvfac in hfacvfac:
        data_key = f'data_{hvfac[0]}x{hvfac[1]}'
        data_keys.append(data_key)
        hfac, vfac = hvfac
        hsize, vsize = int(fmat.shape[0]/hfac), int(fmat.shape[1]/vfac)
        hsizevsize.append([hsize, vsize])
        fmats = pxtchr.make_fmat_subsets(hsize, vsize)
        """Lets visualize the subset grain struvctures."""
        if PLOT_GS:
            fig, ax = plt.subplots(nrows=int(hfac), ncols=int(vfac),
                                   sharex=True, sharey=True, squeeze=True)
            images = [[None for v in range(fmats.shape[1])]
                      for h in range(fmats.shape[0])]
            for h in range(int(hfac)):
                for v in range(int(vfac)):
                    images[h][v] = ax[h, v].imshow(fmats[h][v])
            norm = colors.Normalize(vmin=fmat.min(), vmax=fmat.max())
            for h in range(int(hfac)):
                for v in range(int(vfac)):
                    images[h][v].set_norm(norm)
            fig.colorbar(images[h][v], ax=ax.ravel().tolist(), orientation='vertical',
                         fraction=.1)
        # fmats.shape
        """Now, characterize all field matrix sub-sets."""
        characterized_subsets_all = pxtchr.characterize_all_subsets(fmats)
        # characterized_subsets_all[0][0].keys()
        """We will now build the subset grain area database."""
        subset_grain_areas = [[None for v in range(fmats.shape[1])]
                              for h in range(fmats.shape[0])]
        """We will now build the number of grains database fir subsets."""
        subset_ng = [[None for v in range(fmats.shape[1])]
                     for h in range(fmats.shape[0])]
        for h in range(fmats.shape[0]):
            for v in range(fmats.shape[1]):
                subset_grain_areas[h][v] = np.array(characterized_subsets_all[h][v]['gid_npxl'])
                # --------------------------------
                if REMOVE_OUTLIERS_SUBSETS:
                    tgt_indices = np.array(list(set(range(subset_grain_areas[h][v].size)) - set(find_outliers_iqr(subset_grain_areas[h][v]))))
                    subset_grain_areas[h][v] = subset_grain_areas[h][v][tgt_indices]
            # --------------------------------
                if REMOVE_SMALLGRAINS_SUBSETS:
                    if remove_small_grain_ineq_rule_subsets == 'le':
                        subset_grain_areas[h][v] = subset_grain_areas[h][v][np.argwhere(subset_grain_areas[h][v] > SmallGrainArea_subsets).T.squeeze()]
                    elif remove_small_grain_ineq_rule_subsets == 'lt':
                        subset_grain_areas[h][v] = subset_grain_areas[h][v][np.argwhere(subset_grain_areas[h][v] >= SmallGrainArea_subsets).T.squeeze()]
                # len(tgt.gs[tslice].areas), len(subset_grain_areas[h][v])
                # --------------------------------
                if type(subset_grain_areas[h][v]) in (int, np.int32):
                    subset_ng[h][v] = 1
                    subset_grain_areas[h][v] = np.array([subset_grain_areas[h][v]])
                else:
                    subset_ng[h][v] = len(subset_grain_areas[h][v])
        subset_ng = np.array(subset_ng)
        ssfeat_nxm[data_key] = subset_ng
        """Make space for kullback leibler divergence (KLD) metric values.
        KLD_ts: KLD Target | Sample, KLD_st: KLD Sample | Target
        """
        KLD_ts = [[None for v in range(fmats.shape[1])] for h in range(fmats.shape[0])]
        KLD_st = [[None for v in range(fmats.shape[1])] for h in range(fmats.shape[0])]
        """
        We will now build the histogram-based probability distributions of the above
        grain area database. Then, we will calculate relative entropy of the
        distributions Target:Sample.
        """
        for h in range(fmats.shape[0]):
            for v in range(fmats.shape[1]):
                sample_grain_areas = subset_grain_areas[h][v]
                bins = np.histogram_bin_edges(np.concatenate([target_grain_areas,
                                                              sample_grain_areas]),
                                              bins=50)
                target_hist, _ = np.histogram(target_grain_areas,
                                              bins=bins, density=True)
                sample_hist, _ = np.histogram(sample_grain_areas,
                                              bins=bins, density=True)
                # Avoiding zero values to prevent issues with log(0)
                target_hist = np.where(target_hist == 0, 1e-10, target_hist)
                sample_hist = np.where(sample_hist == 0, 1e-10, sample_hist)
                # Calculate the relative entropy of the distributions -- Target:Sample
                kl_divergence_ts = entropy(target_hist, sample_hist)
                KLD_ts[h][v] = kl_divergence_ts
                if PERFORM_SAMPLE_TARGET_ASYMMETRY_TEST:
                    kl_divergence_st = entropy(target_hist, sample_hist)
                    KLD_st[h][v] = kl_divergence_st
        KLD_ts, KLD_st = np.array(KLD_ts), np.array(KLD_st)
        # -------------
        PLOT_KLD_HEATMAP = False
        if PLOT_KLD_HEATMAP:
            # fig = plt.figure(figsize=(5, 4), dpi=125)
            # ax = plt.gca()
            # sns.heatmap(KLD_ts, vmin=0, vmax=20, annot=True)
            # ax.set(xlabel='subset x location', ylabel='subset y location')
            # -------------
            # fig = plt.figure(figsize=(5, 4), dpi=125)
            # ax = plt.gca()
            # # sns.heatmap(KLD_ts, vmin=0, vmax=20, annot=True)
            # sns.heatmap(KLD_ts, annot=True, fmt='2.1f',
            #             annot_kws={'fontsize': 7},
            #             cbar_kws={'label': 'R metric: relative entropy (T | S)'})
            # ax.set(xlabel='subset x location', ylabel='subset y location')
            # -------------
            fig = plt.figure(figsize=(5, 4), dpi=125)
            ax = plt.gca()
            sns.heatmap(KLD_ts, vmin=0, vmax=20, annot=True, fmt='2.1f',
                        annot_kws={'fontsize': 7},
                        cbar=True,
                        cmap='nipy_spectral',
                        cbar_kws={'label': 'R metric: relative entropy (T | S)'})
            ax.set(xlabel='subset x location', ylabel='subset y location')
        # =============================================================
        """Lets plot the number of grains."""
        PLOT_NGRAINS_HEATMAP = False
        if PLOT_NGRAINS_HEATMAP:
            fig = plt.figure(figsize=(5, 4), dpi=125)
            ax = plt.gca()
            subset_ng_norm = subset_ng/ngrains_tgt
            sns.heatmap(subset_ng_norm, vmin=0, vmax=0.5, annot=True, fmt='4.3f',
                        cmap='nipy_spectral',
                        annot_kws={'fontsize': 7},
                        cbar_kws={'label': f'Ng (subset) / Ng (target)={tgt.gs[tslice].n}'})
            ax.set(xlabel='subset x location', ylabel='subset y location')
        # =============================================================
        """lets see how the metric behaves with grain area values."""
        PLOT_NG_KLD_SCATTER = False
        if PLOT_NG_KLD_SCATTER:
            plt.figure(figsize=(5, 5), dpi=150)
            plt.scatter(subset_ng.ravel(), KLD_ts.ravel(), s=10, c='black')
            plt.xlabel('Number of grains')
            plt.ylabel('R metric: relative entropy (T | S)')
        # =============================================================
        avg_number_of_grains.append(subset_ng.mean())
        # =============================================================
        data_nxm[data_key] = np.vstack((np.array(subset_ng).ravel(),
                                        np.array(KLD_ts).ravel())).T
        data_nxm_kldfit[data_key] = np.polyfit(data_nxm[data_key][:, 0],
                                                 data_nxm[data_key][:, 1],
                                                 1)
        gradient = data_nxm_kldfit[data_key][0]
        intercept = data_nxm_kldfit[data_key][1]
        length = np.median(pdist(data_nxm[data_key]))
        centre = data_nxm[data_key].mean(axis=0)
        data_nxm_kldfit_Sline2d[data_key] = Sline2d.by_MCLC(gradient,
                                                            intercept,
                                                            1.0*length,
                                                            centre)
        data_nxm_kldfit_mpoints[data_key] = MPoint2d.from_coords(data_nxm[data_key])
        data_nxm_kldfit_centroids[data_key] = data_nxm_kldfit_mpoints[data_key].centroid
        data_nxm_kldfit_std[data_key] = data_nxm[data_key][:, 1].std()
    # -----------------------------
    avg_number_of_grains = np.array(avg_number_of_grains)
    # -----------------------------
    lines = np.array([[line.x0,line.y0,line.x1,line.y1] for line in data_nxm_kldfit_Sline2d.values()])
    # -----------------------------
    r_scatter_centroids = np.array([[centroid[0], centroid[1]]
                                    for centroid in data_nxm_kldfit_centroids.values()])
    # -----------------------------
    array_to_excel = np.vstack((avg_number_of_grains,
                                avg_number_of_grains/tgt.gs[tslice].n,
                                r_scatter_centroids[:,1].T,
                                lines.T,
                                np.array(hsizevsize).T,
                                np.array([_ng_.std() for _ng_ in ssfeat_nxm.values()]),
                                np.array(list(data_nxm_kldfit_std.values()))
                                )).T
    array_to_excel.shape
    write_array_to_excel(array_to_excel, file_name, sheet_name, start_cell)
    # -----------------------------
    ngrains.append(ngrains_tgt)
    # -----------------------------
    CALCULATE_KDE_FIELD = False
    if CALCULATE_KDE_FIELD:
        _xy_ = np.vstack((data_nxm.values()))
        xlimits = (0*np.min(_xy_[:, 0]), np.max(_xy_[:, 0]))
        ylimits = (0, 20)
        tree = cKDTree(_xy_)
        distances, _ = tree.query(_xy_, k=2, workers=4)  # k=2 because the nearest neighbor is the point itself
        if np.min(distances[:, 1]) < 1E-5:
            DMIN = distances[:, 1][distances[:, 1] > 1E-5].min()
        DMINx = DMIN*(xlimits[1] - xlimits[0])
        DMINy = DMIN*(ylimits[1] - ylimits[0])
        # -----------------------------
        # DMINx = DMIN
        # DMINy = DMIN
        # Construct a rectangular grid based on limits and minimum distance
        DMINx, DMINy = np.mean([DMINx, DMINy]), np.mean([DMINx, DMINy])
        DMINx, DMINy = 0.05, 0.005
        x_grid = np.arange(xlimits[0], xlimits[1], DMINx)
        y_grid = np.arange(ylimits[0], ylimits[1], DMINy)
        x_grid.shape
        X, Y = np.meshgrid(x_grid, y_grid)
        positions = np.vstack([X.ravel(), Y.ravel()])
        # -----------------------------
        kdes = [gaussian_kde(data.T) for data in data_nxm.values()]
        # -----------------------------
        densities = []
        for i, kde in enumerate(kdes):
            print(f'Evaluating KDE {i} of {len(kdes)}')
            densities.append(kde.evaluate(positions))
        # -----------------------------


        densities = [np.reshape(density, (X.shape[0], X.shape[1]))
                     for density in densities]
        # -----------------------------
        combined_density = np.zeros_like(densities[0])
        for density in densities:
            mask = density > combined_density
            combined_density[mask] = density[mask]
        # -----------------------------
    # -----------------------------
    DO_NOT_PLOT = True
    if CALCULATE_KDE_FIELD and not DO_NOT_PLOT:
        plt.figure(figsize=(6.5, 4.5), dpi=200)
        plt.contourf(X, Y, combined_density, cmap='viridis')
        plt.colorbar(label='Kullback Leibler Relative entropy (density)')
        for data in data_nxm.values():
            plt.scatter(data[:, 0], data[:, 1], s=1, c=np.random.random(3), alpha=0.25)
        # PLot the centroids
        plt.plot(r_scatter_centroids[:,0], r_scatter_centroids[:,1],
                 '-ko', lw=0.5, ms=5, mfc='None', mec='k', mew=0.8, alpha=0.5)
        # PLot the regression lines for each dataset.
        plt.plot(lines[:, [0, 2]].T, lines[:, [1, 3]].T, '-k', lw=1)
        plt.xlim(xlimits)
        plt.ylim(ylimits)
        plt.xlabel('Number of grains, Ng')
        plt.ylabel('Representativeness, R: KL-re T|S')
        # plt.title('R: Relative entropy (target|sample) against Ng')
        plt.show()
    # ============================================================================
    tgt.gs[tslice].n
    # ============================================================================
    ANALYZE = False
    cell_starts = ['B10', 'B27', 'B44', 'B61', 'B78', 'B95', 'B112', 'B129']
    cell_ends = ['L22', 'L39', 'L56', 'L73', 'L90', 'L107', 'L124', 'L141']
    cell_ranges = [a+':'+b for a, b in zip(cell_starts, cell_ends)]
    DF = [read_excel_range(file_name, sheet_name, cell_range) for cell_range in cell_ranges]
    subplot_locs =  np.array([(1,1), (1,2), (1,3), (1,4), (2,1), (2,2), (2,3), (2,4)])
    if ANALYZE:
        cell_starts = ['B10', 'B27', 'B44', 'B61', 'B78', 'B95', 'B112', 'B129']
        cell_ends = ['L22', 'L39', 'L56', 'L73', 'L90', 'L107', 'L124', 'L141']
        cell_ranges = [a+':'+b for a, b in zip(cell_starts, cell_ends)]
        DF = [read_excel_range(file_name, sheet_name, cell_range) for cell_range in cell_ranges]
        subplot_locs =  np.array([(1,1), (1,2), (1,3), (1,4), (2,1), (2,2), (2,3), (2,4)])
        # ============================================================================
        # ============================================================================
        DO_NOT_PLOT = True
        if not DO_NOT_PLOT:
            YMAXa = np.round(np.array([df['R density'].max() for df in DF]).max())
            YMAXb = np.round(np.array([df['R density. std'].max() for df in DF]).max())
            YMAX = np.round((YMAXa + YMAXb)/10)*10
            fig, axs = plt.subplots(nrows=2, ncols=4,
                                    sharex=False, sharey=False, squeeze=True,
                                    figsize=(15, 7), dpi=100)
            for spr, spc, df, ng in zip(subplot_locs[:, 0], subplot_locs[:, 1],
                                        DF, ngrains):
                axs[spr-1, spc-1].plot(df['n grains'], df['R density'], 'o',
                                       lw=1.0, ms=4, mfc='k', mec='k', c='k',
                                       mew=0.8, alpha=1.0)
                lines = df.loc[:, ['x0', 'y0', 'x1', 'y1']].to_numpy()
                axs[spr-1, spc-1].plot(lines[:, [0, 2]].T, lines[:, [1, 3]].T, '-r', lw=1)
                axs[spr-1, spc-1].errorbar(df['n grains'], df['R density'],
                                           xerr=df['s grains. std'],
                                           yerr=df['R density. std'],
                                           lw=1.5, mfc='k', ecolor='gray', alpha=0.5)
                axs[spr-1, spc-1].text(df['n grains'].min(),
                                       YMAX*0.9,
                                       f'No. of grains in parent = {ng}',
                                       fontsize=10)
                axs[spr-1, spc-1].set_xlabel('Mean average no. of grains', fontsize=11)
                axs[spr-1, spc-1].set_ylabel('Representativeness, R: KL-re T|S', fontsize=11)
                axs[spr-1, spc-1].set_ylim([0, YMAX])
        # ============================================================================
        DO_NOT_PLOT = True
        if not DO_NOT_PLOT:
            XMAX = np.round(np.array([df['ngrains/Ng'].max() for df in DF]).max()*10)/10
            YMAXa = np.round(np.array([df['R density'].max() for df in DF]).max())
            YMAXb = np.round(np.array([df['R density. std'].max() for df in DF]).max())
            YMAX = np.round((YMAXa + YMAXb)/10)*10
            fig, axs = plt.subplots(nrows=2, ncols=4,
                                    sharex=False, sharey=False, squeeze=True,
                                    figsize=(15, 7), dpi=100)
            for spr, spc, df, ng in zip(subplot_locs[:, 0], subplot_locs[:, 1],
                                        DF, ngrains):
                axs[spr-1, spc-1].plot(df['ngrains/Ng'], df['R density'], 'o',
                                       lw=1.0, ms=4, mfc='k', mec='k', c='k',
                                       mew=0.8, alpha=1.0)
                lines = df.loc[:, ['x0', 'y0', 'x1', 'y1']].to_numpy()
                axs[spr-1, spc-1].plot(lines[:, [0, 2]].T/ng, lines[:, [1, 3]].T, '-r',
                                       lw=1, alpha=0.5)
                axs[spr-1, spc-1].errorbar(df['ngrains/Ng'], df['R density'],
                                           xerr=df['s grains. std']/ng,
                                           yerr=df['R density. std'],
                                           lw=1.5, mfc='k', ecolor='gray', alpha=0.5)
                axs[spr-1, spc-1].text(df['ngrains/Ng'].min(),
                                       YMAX*0.9,
                                       f'No. of grains in parent (Ngp) = {ng}',
                                       fontsize=10)
                axs[spr-1, spc-1].set_xlabel('<Ng of subsets> / Ng of parent', fontsize=11)
                axs[spr-1, spc-1].set_ylabel('Representativeness, R: KL-re T|S', fontsize=11)
                axs[spr-1, spc-1].set_ylim([0, XMAX])
                axs[spr-1, spc-1].set_ylim([0, YMAX])
        # ============================================================================
        DO_NOT_PLOT = True
        if not DO_NOT_PLOT:
            XMAX = np.round(np.array([df['ngrains/Ng'].max() for df in DF]).max()*10)/10
            YMAXa = np.round(np.array([df['R density'].max() for df in DF]).max())
            YMAXb = np.round(np.array([df['R density. std'].max() for df in DF]).max())
            YMAX = np.round((YMAXa + YMAXb)/10)*10

            fig = plt.figure(figsize=(7.5, 7.5), dpi=125)
            axs = plt.gca()
            markers = ['o', 's', 'x', 'd', '<', 'h', '*', '^']
            for df, ng, marker in zip(DF, ngrains, markers):
                plt.errorbar(df['ngrains/Ng'], df['R density'],
                                           xerr=df['s grains. std']/ng,
                                           yerr=df['R density. std'],
                                           marker=marker,
                                           lw=1.5, ecolor='gray', elinewidth=0.5, alpha=1,
                                           label=f'Ng of parent = {ng}'
                                           )

                axs.set_xlabel('<Ng of subsets> / Ng of parent', fontsize=12)
                axs.set_ylabel('Representativeness, R: KL-re T|S', fontsize=12)
                axs.set_ylim([0, XMAX])
                axs.set_ylim([0, YMAX])
            plt.legend()
