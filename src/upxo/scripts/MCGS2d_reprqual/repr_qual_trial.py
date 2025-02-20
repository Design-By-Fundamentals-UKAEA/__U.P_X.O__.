# -*- coding: utf-8 -*-
"""
Created on Wed May 15 12:29:41 2024

@author: rg5749
"""


KLD_all = np.array(KLD_all)
KLD_all.mean(axis=0)
KLD_all.min(axis=0)
KLD_all.max(axis=0)
KLD_all.std(axis=0)

AREA = [sa.mean() for sa in sample_areas]
sample_areas = np.array(sample_areas)
sample_areas.mean(axis=0)
sample_areas.std(axis=0)

fig, ax0 = plt.subplots(nrows=1, sharex=True)
ax0.errorbar(tslice, KLD_all.mean(axis=0), yerr=KLD_all.std(axis=0), fmt='-o')

plt.plot(tslice, AREA, '-o')

plt.plot(tslice, KLD)

# **********************************
plt.imshow(tgt.gs[slicenumber].lgi)
plt.imshow(smp.gs[slicenumber].lgi)
# **********************************
sns.histplot(target, color='skyblue', label='target', kde=True, alpha=0.5)
sns.histplot(sample, color='salmon', label='sample', kde=True, alpha=0.5)
plt.xlabel('Grain area')
plt.ylabel('Number of grains')
plt.legend()
plt.show()
# **********************************
# [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[

nsamples = np.arange(1, sample.size, 2)
nsamples_pstd_mw = np.hstack((nsamples[np.newaxis].T, nsamples[np.newaxis].T)).astype(float)
nsamples_pstd_ks = np.hstack((nsamples[np.newaxis].T, nsamples[np.newaxis].T)).astype(float)
nsamples_pstd_kr = np.hstack((nsamples[np.newaxis].T, nsamples[np.newaxis].T)).astype(float)
nsamples_pmean_mw = np.hstack((nsamples[np.newaxis].T, nsamples[np.newaxis].T)).astype(float)
nsamples_pmean_ks = np.hstack((nsamples[np.newaxis].T, nsamples[np.newaxis].T)).astype(float)
nsamples_pmean_kr = np.hstack((nsamples[np.newaxis].T, nsamples[np.newaxis].T)).astype(float)
for i, ns in enumerate(nsamples):
    p_values_mw = []
    p_values_ks = []
    p_values_kr = []
    for _ in range(ntests):
        _, p_value_mw = mannwhitneyu(target, np.random.choice(sample, ns, replace=False))
        _, p_value_ks = ks_2samp(target, np.random.choice(sample, ns, replace=False))
        _, p_value_kr = kruskal(target, np.random.choice(sample, ns, replace=False))
        p_values_mw.append(p_value_mw)
        p_values_ks.append(p_value_ks)
        p_values_kr.append(p_value_kr)
    nsamples_pstd_mw[i][1] = np.array(p_values_mw).std()
    nsamples_pstd_ks[i][1] = np.array(p_values_ks).std()
    nsamples_pstd_kr[i][1] = np.array(p_values_kr).std()
    nsamples_pmean_mw[i][1] = np.array(p_values_mw).mean()
    nsamples_pmean_ks[i][1] = np.array(p_values_ks).mean()
    nsamples_pmean_kr[i][1] = np.array(p_values_kr).mean()
# ]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
plt.figure(figsize=(5,5), dpi = 150)
plt.plot(nsamples_pmean_mw[:,0], nsamples_pmean_mw[:,1], label=f'<p> tslice={slicenumber}')
plt.plot(nsamples_pstd_mw[:,0], nsamples_pstd_mw[:,1], label=f'std(p) tslice={slicenumber}')
plt.xlabel('Number of randomly drawn samples')
plt.ylabel(f'P-values from {ntests} tests.')
plt.title(f'Spread in P-values with Sample size \n tslice={slicenumber}')
ax = plt.gca()
#ax.set_xlim([0, 600])
#ax.set_ylim([0, 1.0])
plt.legend()

# -----------------------------------------------------------
plt.figure(figsize=(5,5), dpi = 150)
plt.plot(nsamples_pmean_ks[:,0], nsamples_pmean_ks[:,1], label=f'<p> tslice={slicenumber}')
plt.plot(nsamples_pstd_ks[:,0], nsamples_pstd_ks[:,1], label=f'std(p) tslice={slicenumber}')
plt.xlabel('Number of randomly drawn samples')
plt.ylabel(f'P-values from {ntests} tests.')
plt.title(f'Spread in P-values with Sample size \n tslice={slicenumber}')
ax = plt.gca()
#ax.set_xlim([0, 600])
#ax.set_ylim([0, 1.0])
plt.legend()



plt.figure(figsize=(5,5), dpi = 150)
plt.plot(nsamples_pmean_kr[:,0], nsamples_pmean_kr[:,1], label=f'<p> tslice={slicenumber}')
plt.plot(nsamples_pstd_kr[:,0], nsamples_pstd_kr[:,1], label=f'std(p) tslice={slicenumber}')
plt.xlabel('Number of randomly drawn samples')
plt.ylabel(f'P-values from {ntests} tests.')
plt.title(f'Spread in P-values with Sample size \n tslice={slicenumber}')
ax = plt.gca()
#ax.set_xlim([0, 600])
#ax.set_ylim([0, 1.0])
plt.legend()


target.size
sample.size


import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)  # for reproducibility

# Exponential distribution (right-skewed)
data1 = stats.expon.rvs(scale=2, size=100)

# Log-normal distribution (right-skewed)
data2 = stats.lognorm.rvs(s=0.5, scale=np.exp(1), size=50)
data2 = stats.expon.rvs(scale=2, size=10)

# Visualize the distributions
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(data1, bins=20, density=True, alpha=0.7, label='Exponential')
plt.title('Exponential Distribution')
plt.subplot(1, 2, 2)
plt.hist(data2, bins=20, density=True, alpha=0.7, label='Log-normal')
plt.title('Log-normal Distribution')
plt.show()



from scipy.stats import ks_2samp

# Perform the Kolmogorov-Smirnov test
statistic, p_value = ks_2samp(data1, data2)

print(f'Kolmogorov-Smirnov statistic: {statistic:.3f}')
print(f'P-value: {p_value:.3f}')





from scipy.stats import ks_2samp
from numpy.random import permutation

def permutation_test(data1, data2, n_permutations=1000):
    observed_stat, _ = ks_2samp(data1, data2)
    permuted_stats = []
    combined_data = np.concatenate([data1, data2])
    for _ in range(n_permutations):
        permuted_data = permutation(combined_data)
        perm_stat, _ = ks_2samp(permuted_data[:len(data1)], permuted_data[len(data1):])
        permuted_stats.append(perm_stat)
    permuted_stats = np.array(permuted_stats)  # Convert to NumPy array to enable the comparison.
    p_value = np.mean(permuted_stats >= observed_stat)
    return p_value

p_value_permutation = permutation_test(data1, data2)
print(f'Permutation test p-value: {p_value_permutation:.3f}')
