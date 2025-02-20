"""
Created on Tue May 14 12:44:36 2024

@author: Dr. Sunil Anandatheertha

Explanations
------------
This example provides codes using UPXO to caculate KL diverg3ence
representativeness measure between a synthetic target grain structure and
multiple synthetic sample grain structures.

Following must be noted:
    * Target is synthetic
    * Samples are synthetic
    * 10 (target : samples) sets are used. nsims = 10.
    * Target's 20th temporal slice is ised as target in each above set
        slicenumber = 20.
    * In each above set, samples of slices [2, 6, 10, 14, 20, 24, 28, 32]
        are used. tslice = [2, 6, 10, 14, 20, 24, 28, 32].
"""

'''IMPORT STATEMENTS'''


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from scipy.stats import ks_2samp
from scipy.stats import kruskal
from scipy.stats import entropy
from upxo.ggrowth.mcgs import mcgs
from upxo.pxtalops.Characterizer import mcgs_mchar_2d
from upxo._sup import dataTypeHandlers as dth

NUMBERS = dth.dt.NUMBERS
ITERABLES = dth.dt.ITERABLES
"""
NOTE: DO NOT CHANGE SETTINGS, IN TEREST OF CONFORMITY WITH TOP EXPLANATIONS.
"""
nsims = 10
slicenumber = 20
tslice = [2, 6, 10, 14, 20, 24, 28, 32]
KLD_all = []
sample_areas_all = []

REMOVE_OUTLIERS = True
REMOVE_SMALLGRAINS = True

SmallGrainAreas = [1.0]


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
    iqr = q3-q1
    lower_bound = q1-1.5*iqr
    upper_bound = q3+1.5*iqr
    outlier_indices = np.where((data < lower_bound) | (data > upper_bound))[0]
    return outlier_indices


for nsim in range(nsims):
    '''LETS FIRST GENERATE THE BASIC GRAIN STRUCTURE.'''
    tgt = mcgs(study='independent', input_dashboard='input_dashboard.xls')
    smp = mcgs(study='independent', input_dashboard='input_dashboard.xls')
    tgt.simulate()
    smp.simulate()
    '''DETECT GRAINS AND CHARACTERIZE THEW GRAIN STRUCTUER AS USUAL.'''
    tgt.detect_grains()
    smp.detect_grains()

    tgt.char_morph_2d(slicenumber)
    smp.char_morph_2d(tslice)
    # =========================================================
    KLD = []
    target_area = tgt.gs[slicenumber].areas
    sample_areas = []
    for tsl in tslice:
        target = tgt.gs[slicenumber].areas
        sample = smp.gs[tsl].areas
        sample_areas.append(sample)
        # ********************************************************************
        if REMOVE_OUTLIERS:
            tgt_indices = np.array(list(set(range(target.size)) - set(find_outliers_iqr(target))))
            smp_indices = np.array(list(set(range(sample.size)) - set(find_outliers_iqr(sample))))
            target = target[tgt_indices]
            sample = sample[smp_indices]

        # np.count_nonzero(target == target.min())
        # np.count_nonzero(sample == sample.min())
        if REMOVE_SMALLGRAINS:
            for sga in SmallGrainAreas:
                target = target[np.argwhere(target != sga).T.squeeze()]
                sample = sample[np.argwhere(sample != sga).T.squeeze()]
        # target.size, sample.size
        # ********************************************************************
        # Create histogram-based probability distributions
        bins = np.histogram_bin_edges(np.concatenate([target,
                                                      sample]), bins=50)
        target_hist, _ = np.histogram(target, bins=bins, density=True)
        sample_hist, _ = np.histogram(sample, bins=bins, density=True)
        # Avoid zero values to prevent issues with log(0)
        target_hist = np.where(target_hist == 0, 1e-10, target_hist)
        sample_hist = np.where(sample_hist == 0, 1e-10, sample_hist)
        # Compute KL divergence
        kl_divergence = entropy(target_hist, sample_hist)
        KLD.append(kl_divergence)
        print(f"KL Divergence: {kl_divergence}")
        # ********************************************************************
    KLD_all.append(KLD)
    sample_areas_all.append(sample_areas)


KLD_all = np.array(KLD_all)
KLD_all.mean(axis=0)
KLD_all.min(axis=0)
KLD_all.max(axis=0)
KLD_all.std(axis=0)

sample_areas
AREA = [sa.mean() for sa in sample_areas]
AREA_std = [sa.std() for sa in sample_areas]

plt.figure(figsize=(5, 5), dpi=150)
plt.errorbar(tslice, KLD_all.mean(axis=0), yerr=KLD_all.std(axis=0), fmt='-o')
plt.xlabel('Sample temporal slice numbers', fontsize=12)
plt.ylabel('R.KL (target : sample). Grain areas.', fontsize=12)
plt.title(f'Target @ tslice={slicenumber}', fontsize=12)
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=12)
ax.tick_params(axis='both', which='minor', labelsize=8)

plt.figure(figsize=(5, 5), dpi=150)
plt.errorbar(tslice, AREA, yerr=AREA_std, fmt='-o')
plt.xlabel('Sample temporal slice numbers', fontsize=12)
plt.ylabel('Sample areas.', fontsize=12)
plt.title(f'Temporal grain area.', fontsize=12)
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=12)
ax.tick_params(axis='both', which='minor', labelsize=8)
