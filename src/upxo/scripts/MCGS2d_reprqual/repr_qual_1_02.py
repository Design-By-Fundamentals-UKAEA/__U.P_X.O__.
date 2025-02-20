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
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors
from scipy.stats import entropy
import scipy.stats as st
from scipy.stats import gaussian_kde
from scipy.spatial import cKDTree
from upxo.ggrowth.mcgs import mcgs
from upxo.pxtalops.Characterizer import mcgs_mchar_2d
from upxo._sup import dataTypeHandlers as dth

NUMBERS = dth.dt.NUMBERS
ITERABLES = dth.dt.ITERABLES
"""
NOTE: DO NOT CHANGE SETTINGS, IN THE INTEREST OF TOP EXPLANATIONS.
"""


def find_outliers_iqr(data):
    """
    Find outliers in data.

    Parameters
    ----------
    data: input data: Iterable

    Return
    ------
    outlier_indices: indices of outliers in data.
    """
    if type(data) not in ITERABLES:
        raise TypeError('Invalid data type specified.')
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_indices = np.where((data < lower_bound) | (data > upper_bound))[0]
    return outlier_indices


"""Generate grain structure and detect the grains."""
tgt = mcgs(study='independent', input_dashboard='input_dashboard.xls')
tgt.simulate()
tgt.detect_grains()
"""Choose a temporal slice and characterize it."""
tslice = 4
tgt.char_morph_2d(tslice)
"""Lets plot the grain structure."""
tgt.gs[tslice].plot(cmap='viridis')
"""Get the areas of all grains in the tslice target grain structure."""
target_grain_areas = tgt.gs[tslice].areas
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
REMOVE_OUTLIERS = True
REMOVE_SMALLGRAINS = True
SmallGrainArea = 1.0
remove_small_grain_ineq_rule = 'le'

if REMOVE_OUTLIERS:
    tgt_indices = np.array(list(set(range(target_grain_areas.size)) - set(find_outliers_iqr(target_grain_areas))))
    target_grain_areas = target_grain_areas[tgt_indices]

if REMOVE_SMALLGRAINS:
    if remove_small_grain_ineq_rule == 'le':
        target_grain_areas = target_grain_areas[np.argwhere(target_grain_areas > SmallGrainArea).T.squeeze()]
    elif remove_small_grain_ineq_rule == 'lt':
        target_grain_areas = target_grain_areas[np.argwhere(target_grain_areas >= SmallGrainArea).T.squeeze()]


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
"""
hfac, vfac = 2.5, 2.5
hsize, vsize = int(fmat.shape[0]/hfac), int(fmat.shape[1]/vfac)
fmats = pxtchr.make_fmat_subsets(hsize, vsize)
"""Lets visualize the subset grain struvctures."""
if PLOT_GS:
    fig, ax = plt.subplots(nrows=hfac, ncols=vfac,
                           sharex=True, sharey=True, squeeze=True)
    images = [[None for v in range(fmats.shape[1])]
              for h in range(fmats.shape[0])]
    for h in range(hfac):
        for v in range(vfac):
            images[h][v] = ax[h, v].imshow(fmats[h][v])
    norm = colors.Normalize(vmin=fmat.min(), vmax=fmat.max())
    for h in range(hfac):
        for v in range(vfac):
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
        subset_grain_areas[h][v] = characterized_subsets_all[h][v]['gid_npxl']
        subset_ng[h][v] = len(subset_grain_areas[h][v])
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
        kl_divergence_st = entropy(target_hist, sample_hist)
        KLD_st[h][v] = kl_divergence_st

print(25*'#')
print(KLD_ts)
print(np.array(KLD_ts).mean())
print(25*'-')
print(KLD_st)
print(25*'#')
# -------------
PLOT_KLD_HEATMAP = False
if PLOT_KLD_HEATMAP:
    fig = plt.figure(figsize=(5, 4), dpi=125)
    ax = plt.gca()
    sns.heatmap(KLD_ts, vmin=0, vmax=20, annot=True)
    ax.set(xlabel='subset x location', ylabel='subset y location')
    # -------------
    fig = plt.figure(figsize=(5, 4), dpi=125)
    ax = plt.gca()
    # sns.heatmap(KLD_ts, vmin=0, vmax=20, annot=True)
    sns.heatmap(KLD_ts, annot=True, fmt='2.1f',
                annot_kws={'fontsize': 7},
                cbar_kws={'label': 'R metric: relative entropy (T | S)'})
    ax.set(xlabel='subset x location', ylabel='subset y location')
    # -------------
    fig = plt.figure(figsize=(5, 4), dpi=125)
    ax = plt.gca()
    sns.heatmap(KLD_ts, vmin=0, vmax=20, annot=True, fmt='2.1f',
                annot_kws={'fontsize': 7},
                cbar_kws={'label': 'R metric: relative entropy (T | S)'})
    ax.set(xlabel='subset x location', ylabel='subset y location')
# =============================================================
"""Lets plot the number of grains."""
PLOT_NGRAINS_HEATMAP = False
if PLOT_NGRAINS_HEATMAP:
    fig = plt.figure(figsize=(5, 4), dpi=125)
    ax = plt.gca()
    subset_ng_norm = np.array(subset_ng)/ngrains_tgt
    sns.heatmap(subset_ng_norm, vmin=0, vmax=1.0, annot=True, fmt='4.3f',
                cmap='nipy_spectral',
                annot_kws={'fontsize': 7},
                cbar_kws={'label': 'Ng (subset) / Ng (target)'})
    ax.set(xlabel='subset x location', ylabel='subset y location')
# =============================================================
"""lets see how the metric behaves with grain area values."""
PLOT_NG_KLD_SCATTER = False
if PLOT_NG_KLD_SCATTER:
    subset_ng, KLD_ts = np.array(subset_ng), np.array(KLD_ts)
    plt.figure(figsize=(5, 5), dpi=150)
    plt.scatter(subset_ng.ravel(), KLD_ts.ravel(), s=10, c='black')
    plt.xlabel('Number of grains')
    plt.ylabel('R metric: relative entropy (T | S)')
# =============================================================
data_2x2 = np.vstack((np.array(subset_ng).ravel(),
                      np.array(KLD_ts).ravel())).T  # DONE
data_2p5x2p5 = np.vstack((np.array(subset_ng).ravel(),
                      np.array(KLD_ts).ravel())).T  # DONE
data_3x3 = np.vstack((np.array(subset_ng).ravel(),
                      np.array(KLD_ts).ravel())).T  # DONE
data_4x4 = np.vstack((np.array(subset_ng).ravel(),
                      np.array(KLD_ts).ravel())).T  # DONE
data_6x6 = np.vstack((np.array(subset_ng).ravel(),
                      np.array(KLD_ts).ravel())).T  # DONE
data_8x8 = np.vstack((np.array(subset_ng).ravel(),
                      np.array(KLD_ts).ravel())).T  # DONE
data_10x10 = np.vstack((np.array(subset_ng).ravel(),
                        np.array(KLD_ts).ravel())).T  # DONE
# -----------------------------
datas = [data_2x2, data_2p5x2p5, data_4x4, data_6x6, data_8x8,data_10x10]
_xy_ = np.vstack((data_2x2,data_2p5x2p5,  data_4x4, data_6x6, data_8x8,data_10x10))
# -----------------------------
from upxo.geoEntities.mulpoint2d import MPoint2d
mpoints = [MPoint2d.from_coords(data) for data in datas]
centroids = np.array([mp.centroid for mp in mpoints])
# -----------------------------
xlimits = (np.min(_xy_[:, 0]), np.max(_xy_[:, 0]))
ylimits = (np.min(_xy_[:, 1]), np.max(_xy_[:, 1]))
tree = cKDTree(_xy_)
distances, _ = tree.query(_xy_, k=2)  # k=2 because the nearest neighbor is the point itself
DMIN = np.min(distances[:, 1])
DMINx =  DMIN* (xlimits[1]-xlimits[0])
DMINy = DMIN * (ylimits[1]-ylimits[0])
# Construct a rectangular grid based on limits and minimum distance
x_grid = np.arange(xlimits[0], xlimits[1], DMINx)
y_grid = np.arange(ylimits[0], ylimits[1], DMINy)
X, Y = np.meshgrid(x_grid, y_grid)
positions = np.vstack([X.ravel(), Y.ravel()])

kdes = [gaussian_kde(data.T) for data in datas]

densities = [np.reshape(kde(positions).T, X.shape) for kde in kdes]

combined_density = np.zeros_like(densities[0])
for density in densities:
    mask = density > combined_density
    combined_density[mask] = density[mask]
# -----------------------------
plt.figure(figsize=(7, 5), dpi=200)
plt.contourf(X, Y, combined_density, cmap='viridis')
plt.colorbar(label='Kullback Leibler Relative entropy')
for data in datas:
    plt.scatter(data[:, 0], data[:, 1], s=1)
plt.plot(centroids[:,0], centroids[:,1], 'o', lw=1, ms=5, mfc='None', mec='k', mew=0.8)
plt.xlim(xlimits)
plt.ylim(ylimits)
plt.xlabel('Number of grains, Ng')
plt.ylabel('Representativeness, R: KL-re T|S')
# plt.title('R: Relative entropy (target|sample) against Ng')
plt.show()
