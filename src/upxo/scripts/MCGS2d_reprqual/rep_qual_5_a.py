# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 12:33:03 2024

@author: rg5749
"""

# Python Script to Read Grain Structure Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

# Define file path
file_path = r'D:\EBSD Datasets\Daniella\M4-AxialRadial_GrainData.xlsx'

# Load the data from specific range
sheet_name = 0
use_cols = 'J:Q'
skiprows = 3  # Skip first three rows since data starts from A4

data = pd.read_excel(file_path, sheet_name=sheet_name, usecols=use_cols, skiprows=skiprows)

# Rename columns for better understanding
data.columns = [
    'Grain_ID', 'Num_Pixels', 'Grain_Size_um2', 'Aspect_Ratio',
    'Bunge_Euler_1_rad', 'Bunge_Euler_2_rad', 'Bunge_Euler_3_rad', 'Grain_Orientation_Spread'
]

# Display the first few rows
print(data.head())

distr_data = data['Grain_Size_um2'].to_numpy()
distr_data_parent = distr_data[~np.isnan(distr_data)]

distr_area = data['Grain_Size_um2'].to_numpy()
distr_area_parent = distr_area[~np.isnan(distr_area)]
distr_ar = data['Aspect_Ratio'].to_numpy()
distr_ar_parent = distr_ar[~np.isnan(distr_ar)]


def compute_kl_divergence(parent_dist, subset_dist, bins=50):
    parent_hist, bin_edges = np.histogram(parent_dist, bins=bins, density=True)
    subset_hist, _ = np.histogram(subset_dist, bins=bin_edges, density=True)
    kl_div = entropy(subset_hist + 1e-9, parent_hist + 1e-9)  # Add small value to avoid log(0)
    return kl_div

def extract_random_subset_and_compare(distr_data_parent, subset_size):
    distr_data_subset = np.random.choice(distr_data_parent, subset_size, replace=False)
    kl_divergence = compute_kl_divergence(distr_data_parent, distr_data_subset)
    return distr_data_subset, kl_divergence

distr_data_subset, kl_value = extract_random_subset_and_compare(distr_data_parent, 25)
print(kl_value)


distr_data_subset = np.random.choice(distr_data_parent, 1000, replace=False)
from scipy.stats import ks_2samp
from scipy.stats import anderson_ksamp
from scipy.stats import mannwhitneyu
from scipy.stats import energy_distance

ks_stat, ks_p = ks_2samp(distr_data_parent, distr_data_subset)
print(f"KS Test: Stat={ks_stat}, p={ks_p}")
ad_stat, ad_crit, ad_signif = anderson_ksamp([distr_data_parent, distr_data_subset])
print(f"Anderson-Darling Test: Stat={ad_stat}, Significance={ad_signif}")

mw_stat, mw_p = mannwhitneyu(distr_data_parent, distr_data_subset, alternative='two-sided')
print(f"Mann-Whitney U Test: Stat={mw_stat}, p={mw_p}")

energy_dist = energy_distance(distr_data_parent, distr_data_subset)
print(f"Energy Distance: {energy_dist}")

from scipy.stats import wasserstein_distance

# Calculate Earth Mover's Distance (Wasserstein Distance)
emd = wasserstein_distance(distr_data_parent, distr_data_subset)
print(f"Earth Mover's Distance: {emd}")


from scipy.stats import skew, moment
from statsmodels import robust
from scipy.stats import entropy
from scipy.stats import kurtosis

def compute_stats(arr):
    stats = {}
    abs_diff = np.abs(arr - np.mean(arr))

    # Basic statistics
    stats['min'] = np.min(arr)
    stats['mean'] = np.mean(arr)
    stats['median'] = np.median(arr)
    stats['max'] = np.max(arr)
    stats['range'] = np.ptp(arr)

    # Quartiles and IQR
    stats['1st_quartile'] = np.percentile(arr, 25)
    stats['3rd_quartile'] = np.percentile(arr, 75)
    stats['iqr'] = stats['3rd_quartile'] - stats['1st_quartile']

    # Descriptive statistrics
    stats['variance'] = np.var(arr)

    # Skewness
    stats['skewness'] = skew(arr)
    stats['kurtosis'] = kurtosis(arr)

    # Moments
    stats['moment_1'] = moment(arr, moment=1)
    stats['moment_2'] = moment(arr, moment=2)
    stats['moment_3'] = moment(arr, moment=3)
    stats['moment_4'] = moment(arr, moment=4)
    stats['moment_5'] = moment(arr, moment=5)
    stats['moment_6'] = moment(arr, moment=6)

    stats['robust_mad'] = robust.mad(arr)
    stats['entropy'] = entropy(arr)

    return stats

import seaborn as sns

from scipy.stats import lognorm



distr_data_subset = np.random.choice(distr_data_parent, 3000, replace=False)

distr_data_parent_stats = compute_stats(distr_data_parent)
distr_data_subset_stats = compute_stats(distr_data_subset)


percent_differences = {
    key: 100 * abs(distr_data_subset_stats[key] - distr_data_parent_stats[key]) / distr_data_parent_stats[key]
    for key in distr_data_parent_stats if distr_data_parent_stats[key] != 0
}

# Display results
percent_diff_df = pd.DataFrame(list(percent_differences.items()), columns=['Statistic',
                                                                           'Percentage Difference'])


import numpy as np
from seaborn import kdeplot

my_kde = sns.kdeplot(distr_area_parent, bw_adjust=0.25, fill=False, color = 'gray', alpha=0.3,
                     label="KDE",cumulative=False)
line = my_kde.lines[0]
x, y = line.get_data()

fig, ax = plt.subplots()
ax.plot(x, y)

# sns.reset_orig()
# sns.set(font_scale=1, style="whitegrid", rc={"axes.edgecolor": "black", "axes.linewidth": 1})

# =========================================================================
hidden_fig, hidden_ax = plt.subplots()
kde = sns.kdeplot(distr_area_parent, bw_adjust=0.25, fill=False, color = 'gray', alpha=0.3, label="KDE", cumulative=False, ax = hidden_ax)
kde_line = kde.lines[0]
xarea, yarea = kde_line.get_data()
plt.close(hidden_fig)

hidden_fig, hidden_ax = plt.subplots()
kde = sns.kdeplot(distr_ar_parent, bw_adjust=0.25, fill=False, color = 'gray', alpha=0.3, label="KDE", cumulative=False, ax = hidden_ax)
kde_line = kde.lines[0]
xar, yar = kde_line.get_data()
plt.close(hidden_fig)

h, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].plot(xarea, yarea)
axs[0].set_xlabel(r'Grain area, $\mu m^2$', fontsize=12)
axs[0].set_ylabel('Kernel density', fontsize=12)
axs[0].set_xlim([0, max(xarea)])
axs[0].set_ylim([0, max(yarea)])
axs[1].plot(xar, yar)
axs[1].set_xlabel(r'Grain aspect ratio', fontsize=12)
axs[1].set_ylabel('Kernel density', fontsize=12)
axs[1].set_xlim([0, max(xar)])
axs[1].set_ylim([0, max(yar)])

plt.tight_layout()
plt.show()
# =========================================================================
from scipy.stats import kstest, lognorm
# =========================================================================
nsamples = 8000
# Log normal fit
shape, loc, scale = lognorm.fit(distr_area_parent, floc=0)
pdf_x_area = np.linspace(min(distr_area_parent), max(distr_area_parent), nsamples)
pdf_y_area = lognorm.pdf(pdf_x_area, shape, loc, scale)
distr_area_parent_from_fit = lognorm.rvs(shape, loc=loc, scale=scale, size=nsamples)

h, axs = plt.subplots(1, 2, figsize=(8, 4))
# plt.figure(figsize=(8, 6))
axs[0].plot(xarea, yarea, 'k-', linewidth=1.0, label='Raw data')
axs[0].plot(pdf_x_area, pdf_y_area, 'r--', linewidth=1.0, label=f"Log-Normal Fit\n $\sigma$={shape:.2f}, x'={loc:.2f}, $e^\mu$={scale:.2f}")
axs[0].set_xlabel(r"Grain area, $\mu m^2$", fontsize=12)
axs[0].set_ylabel("Kernel density", fontsize=12)
axs[0].legend()

shape, loc, scale = lognorm.fit(distr_ar_parent, floc=0)
pdf_x_ar = np.linspace(min(distr_ar_parent), max(distr_ar_parent), nsamples)
pdf_y_ar = lognorm.pdf(pdf_x_ar, shape, loc, scale)
distr_ar_parent_from_fit = lognorm.rvs(shape, loc=loc, scale=scale, size=nsamples)

axs[1].plot(xar, yar, 'k-', linewidth=1.0, label=f"Raw data")
axs[1].plot(pdf_x_ar, pdf_y_ar, 'r--', linewidth=1.0, label=f"Log-Normal Fit\n $\sigma$={shape:.2f}, x'={loc:.2f}, $e^\mu$={scale:.2f}")
axs[1].set_xlabel("Grain aspect ratio", fontsize=12)
axs[1].set_ylabel("Kernel density", fontsize=12)
axs[1].legend()
plt.show()
# =========================================================================
distr_area_parent.mean()
distr_area_parent.std()
distr_area_parent.max()

ks_stat, ks_p = ks_2samp(distr_area_parent, distr_area_parent_from_fit)
ad_stat, ad_crit, ad_signif = anderson_ksamp([distr_area_parent, distr_area_parent_from_fit])
mw_stat, mw_p = mannwhitneyu(distr_area_parent, distr_area_parent_from_fit,
                             alternative='two-sided')
energy_dist = energy_distance(distr_area_parent, distr_area_parent_from_fit)
emd = wasserstein_distance(distr_area_parent, distr_area_parent_from_fit)
print(f"KS Test: Stat={ks_stat}, p={ks_p}")
print(f"Anderson-Darling Test: Stat={ad_stat}, Significance={ad_signif}")
print(f"Mann-Whitney U Test: Stat={mw_stat}, p={mw_p}")
print(f"Energy Distance: {energy_dist}")
print(f"EMD: {emd}")

ks_stat, ks_p = ks_2samp(distr_ar_parent, distr_ar_parent_from_fit)
ad_stat, ad_crit, ad_signif = anderson_ksamp([distr_ar_parent, distr_ar_parent_from_fit])
mw_stat, mw_p = mannwhitneyu(distr_ar_parent, distr_ar_parent_from_fit,
                             alternative='two-sided')
energy_dist = energy_distance(distr_ar_parent, distr_ar_parent_from_fit)
emd = wasserstein_distance(distr_ar_parent, distr_ar_parent_from_fit)
print(f"KS Test: Stat={ks_stat}, p={ks_p}")
print(f"Anderson-Darling Test: Stat={ad_stat}, Significance={ad_signif}")
print(f"Mann-Whitney U Test: Stat={mw_stat}, p={mw_p}")
print(f"Energy Distance: {energy_dist}")
print(f"EMD: {emd}")
# =========================================================================
distribution_data = distr_area_parent
distribution_data_fit = distr_area_parent_from_fit

x_vals = np.linspace(min(min(distribution_data), min(distribution_data_fit)),
                     max(max(distribution_data), max(distribution_data_fit)), 1000)
def empirical_cdf(data, x_vals):
    cdf = np.array([np.sum(data <= x) / len(data) for x in x_vals])
    return cdf

distr_data_parent_cdf = empirical_cdf(distribution_data, x_vals)
distr_data_parent_from_fit_cdf = empirical_cdf(distribution_data_fit, x_vals)
max_diff = np.max(np.abs(distr_data_parent_cdf - distr_data_parent_from_fit_cdf))
max_diff_index = np.argmax(np.abs(distr_data_parent_cdf - distr_data_parent_from_fit_cdf))
x_max_diff = x_vals[max_diff_index]

# Plot the CDFs
plt.figure(figsize=(8, 6))
plt.plot(x_vals, distr_data_parent_cdf,
         label="CDF of original data", color='blue')
plt.plot(x_vals, distr_data_parent_from_fit_cdf,
         label="CDF of fit data", color='green')

# Highlight maximum difference
plt.vlines(x_max_diff, distr_data_parent_cdf[max_diff_index],
           distr_data_parent_from_fit_cdf[max_diff_index],
           color='red', linestyle='--', label=f'Max Diff: {max_diff:.4f}')

plt.scatter([x_max_diff], [distr_data_parent_cdf[max_diff_index]],
            color='blue', s=100, zorder=5)
plt.scatter([x_max_diff], [distr_data_parent_from_fit_cdf[max_diff_index]],
            color='green', s=100, zorder=5)

#plt.title("CDF Comparison and Maximum Difference")
# plt.xlabel("Grain aspect ratio", fontsize=12)
plt.xlabel(r"Grain area $\mu m^2$", fontsize=12)
plt.ylabel("Cumulative Probability", fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
# =========================================================================
from scipy.stats import lognorm, weibull_min, gamma, expon, genextreme
def plot_fit(distr_data, dist, dist_name, params):
    x = np.linspace(min(distr_data), max(distr_data), 1000)
    pdf = dist.pdf(x, *params)
    plt.figure(figsize=(8, 6))
    plt.hist(distr_data, bins=30, density=True, alpha=0.6, color='gray', label="Data")
    plt.plot(x, pdf, 'r-', label=f'{dist_name} Fit')
    plt.title(f"{dist_name} Distribution Fit")
    plt.xlabel("Grain Area")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

# Fit and Plot Log-Normal
shape, loc, scale = lognorm.fit(distr_ar_parent, floc=0)
plot_fit(distr_ar_parent, lognorm, "Log-Normal", (shape, loc, scale))

# Fit and Plot Weibull
c, loc, scale = weibull_min.fit(distr_ar_parent, floc=0)
plot_fit(distr_ar_parent, weibull_min, "Weibull", (c, loc, scale))

# Fit and Plot Gamma
a, loc, scale = gamma.fit(distr_ar_parent, floc=0)
plot_fit(distr_ar_parent, gamma, "Gamma", (a, loc, scale))

# Fit and Plot Exponential
loc, scale = expon.fit(distr_ar_parent, floc=0)
plot_fit(distr_ar_parent, expon, "Exponential", (loc, scale))

# Fit and Plot Generalized Extreme Value (GEV)
shape, loc, scale = genextreme.fit(distr_ar_parent)
plot_fit(distr_ar_parent, genextreme, "Generalized Extreme Value (GEV)", (shape, loc, scale))
# =========================================================================
