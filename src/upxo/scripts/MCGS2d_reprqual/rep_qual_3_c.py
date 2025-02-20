# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 23:11:21 2024

@author: rg5749
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
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from networkx.algorithms import community
from scipy.stats import wasserstein_distance, ks_2samp, energy_distance
import netlsd

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
# ###################################################################
# ###################################################################
# ###################################################################
# ###################################################################
# =========================================================
def create_grain_network_nx(neighbor_dict):
    """Creates a networkx graph from the entire neighbor_dict."""
    G = nx.Graph()
    G.add_edges_from([(gid, neighbor) for gid, neighbors in neighbor_dict.items() for neighbor in neighbors])
    return G

# Define Jaccard similarity function
def jaccard_similarity(G1, G2):
    """Calculates Jaccard similarity between the node sets of two graphs."""
    nodes1 = set(G1.nodes())
    nodes2 = set(G2.nodes())
    intersection = len(nodes1 & nodes2)
    union = len(nodes1 | nodes2)
    return intersection / union

def degree_distribution_similarity(G1, G2, method='wasserstein', use_same_bins=True):
    """
    Calculates degree distribution similarity between two networks.

    Args:
        G1 (nx.Graph): The first network.
        G2 (nx.Graph): The second network.
        method (str, optional): The similarity measure to use.
                                Options: 'wasserstein', 'kolmogorov-smirnov', 'energy'.
                                Default: 'wasserstein'.

    Returns:
        float: The similarity score (higher is more similar).
    """

    # Calculate degree distributions
    degrees1 = [G1.degree(node) for node in G1.nodes()]
    degrees2 = [G2.degree(node) for node in G2.nodes()]

    # Convert to numpy arrays for easier calculations
    degrees1 = np.array(degrees1)
    degrees2 = np.array(degrees2)

    if use_same_bins:
        '''
        The KS test is sensitive to differences in the cumulative distribution
        functions (CDFs) of the data. Directly using the raw degree sequences
        can be misleading if the sample sizes (number of nodes) differ
        significantly between the two networks. Histograms normalize the data,
        providing a better representation of the underlying probability
        distributions.
        '''
        # Calculate histograms with the same bins for fair comparison
        min_degree = min(np.min(degrees1), np.min(degrees2))
        max_degree = max(np.max(degrees1), np.max(degrees2))
        bins = np.arange(min_degree, max_degree + 2)  # +2 for inclusive range
        degrees1, _ = np.histogram(degrees1, bins=bins, density=True)
        degrees2, _ = np.histogram(degrees2, bins=bins, density=True)

    if method == 'wasserstein':
        # Wasserstein Distance (Earth Mover's Distance)
        distance = wasserstein_distance(degrees1, degrees2)
        similarity = 1 / (1 + distance)  # Convert distance to similarity
    elif method == 'kolmogorov-smirnov':
        # Kolmogorov-Smirnov Test
        _, p_value = ks_2samp(degrees1, degrees2)
        similarity = p_value  # Higher p-value indicates more similar distributions
    elif method == 'energy':
        # Energy Distance (a non-parametric measure of statistical distance)
        distance = energy_distance(degrees1, degrees2)
        similarity = 1 / (1 + distance)
    else:
        raise ValueError(f"Invalid method: {method}")

    return similarity

def calculate_netlsd_similarity(G1, G2,
                                timescales=np.logspace(-2, 2, 20),
                                ):  # Changed to 'timescales'
    """
    Calculates NetLSD similarity between two networks.

    Args:
        G1 (nx.Graph): The first network.
        G2 (nx.Graph): The second network.
        timescales (int, optional): Number of timescales to use (default: 10).

    Returns:
        float: The NetLSD distance (lower is more similar).
    """
    descriptor1 = netlsd.heat(G1, timescales=timescales)
    descriptor2 = netlsd.heat(G2, timescales=timescales)
    distance = np.linalg.norm(descriptor1 - descriptor2)

    # Convert distance to similarity (optional)
    similarity = 1 / (1 + distance)

    return similarity


def calculate_angular_distance(coord1, coord2):
    """
    Calculates the angle in radians between two position vectors
    formed from the origin to the given coordinates.

    Args:
        coord1 (tuple or list): The (x, y) or (x, y, z) coordinates of the first point.
        coord2 (tuple or list): The (x, y) or (x, y, z) coordinates of the second point.

    Returns:
        float: The angle between the position vectors in radians (0 to pi).
    """
    # Convert coordinates to NumPy arrays (if not already)
    vec1 = np.array(coord1)
    vec2 = np.array(coord2)

    # Input validation (optional)
    if vec1.shape != vec2.shape:
        raise ValueError("Input coordinates must have the same dimensions.")

    # Calculate the dot product
    dot_product = np.dot(vec1, vec2)

    # Calculate the magnitudes
    mag1 = np.linalg.norm(vec1)
    mag2 = np.linalg.norm(vec2)

    # Handle zero magnitudes (avoid division by zero)
    if mag1 == 0 or mag2 == 0:
        return 0.0  # Angle is 0 if either vector is the zero vector

    # Calculate the cosine of the angle
    cos_theta = dot_product / (mag1 * mag2)

    # Calculate the angle (arccos) and ensure it's within the valid range
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    return angle_rad

def calculate_density_bins(A, n_bins=10):
    """
    Calculates non-uniform bins for a matrix A of random numbers between 0 and 1,
    based on the density distribution of values.

    Args:
        A (np.ndarray): A 2D array of random numbers between 0 and 1.
        n_bins (int, optional): The desired number of bins (default: 10).

    Returns:
        np.ndarray: An array of bin edges.

    Example
    -------
    A = np.random.randn(100, 100)
    bin_edges = calculate_density_bins(A, n_bins=15)

    plt.hist(A.ravel(), bins=bin_edges, density=True)
    plt.title('Histogram with Density-Based Bins')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.show()
    """

    # Flatten the array for analysis
    values = A.ravel()

    # Calculate the cumulative distribution function (CDF)
    values_sorted = np.sort(values)
    cdf = np.arange(1, len(values) + 1) / len(values)

    # Determine bin edges based on equal spacing in the CDF
    bin_edges = np.interp(np.linspace(0, 1, n_bins + 1), cdf, values_sorted)

    return bin_edges

def approximate_to_bin_means(A, n_bins=50):
    """
    Approximates each element in array A to the mean of its corresponding bin edges.

    Args:
        A (np.ndarray): A 2D array of random numbers between 0 and 1.
        bin_edges (np.ndarray): An array of bin edges calculated using `calculate_density_bins`.

    Returns:
        np.ndarray: A new 2D array with elements approximated to their bin means.
    """
    bin_edges = calculate_density_bins(A, n_bins=n_bins)
    # Create a copy of A to avoid modifying the original array
    A_approx = A.copy()

    # Digitize to find the bin index for each value
    bin_indices = np.digitize(A_approx, bin_edges) - 1
    # Calculate bin means
    bin_means = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Replace values with their corresponding bin means
    for i in range(A_approx.shape[0]):
        for j in range(A_approx.shape[1]):
            bin_index = bin_indices[i, j]-1
            A_approx[i, j] = bin_means[bin_index]

    return bin_means, A_approx
# =========================================================
neigh_orders = [1, 2, 3, 5, 12]
calc_netlsd_sim = False
jaccard_sim = {n: np.zeros((len(smp.gs.keys()), len(tgt.gs.keys()))) for n in neigh_orders}
wasserstein_sim = {n: np.zeros((len(smp.gs.keys()), len(tgt.gs.keys()))) for n in neigh_orders}
ks_sim = {n: np.zeros((len(smp.gs.keys()), len(tgt.gs.keys()))) for n in neigh_orders}
energy_sim = {n: np.zeros((len(smp.gs.keys()), len(tgt.gs.keys()))) for n in neigh_orders}
netlsd_sim = {n: np.zeros((len(smp.gs.keys()), len(tgt.gs.keys()))) for n in neigh_orders}
use_same_bins = True
# =========================================================
for neigh_order in neigh_orders:
    ti = 0
    for tslice in list(tgt.gs.keys()):
        if tslice == 0:
            pass
        else:
            tgt_nneighgids = tgt.gs[tslice].get_upto_nth_order_neighbors_all_grains(neigh_order,
                                                                                    include_parent=True,
                                                                                    output_type='nparray')
            G1 = create_grain_network_nx(tgt_nneighgids)
        si = 0
        for sslice in list(smp.gs.keys()):
            if tslice == 0 or sslice == 0:
                jaccard_sim[neigh_order][si, ti] = 0
                wasserstein_sim[neigh_order][si, ti] = 0
                ks_sim[neigh_order][si, ti] = 0
                energy_sim[neigh_order][si, ti] = 0
                if calc_netlsd_sim:
                    netlsd_sim[neigh_order][si, ti] = 0
            else:
                print(f'Working on O(n)={neigh_order}, tgt_slice={tslice}, smp_slice={sslice}')
                smp_nneighgids = smp.gs[sslice].get_upto_nth_order_neighbors_all_grains(neigh_order,
                                                                                        include_parent=True,
                                                                                        output_type='nparray')
                G2 = create_grain_network_nx(smp_nneighgids)
                # Calculate similarities using different methods
                jaccard_sim[neigh_order][si, ti] = jaccard_similarity(G1, G2)
                wasserstein_sim[neigh_order][si, ti] = degree_distribution_similarity(G1, G2, method='wasserstein', use_same_bins=use_same_bins)
                ks_sim[neigh_order][si, ti] = degree_distribution_similarity(G1, G2, method='kolmogorov-smirnov', use_same_bins=use_same_bins)
                energy_sim[neigh_order][si, ti] = degree_distribution_similarity(G1, G2, method='energy', use_same_bins=use_same_bins)
                if calc_netlsd_sim:
                    netlsd_sim[neigh_order][si, ti] = calculate_netlsd_similarity(G1, G2)
            si += 1
        ti += 1


neigh_orders_to_use = [1, 2, 5, 12]
fig, ax = plt.subplots(len(neigh_orders_to_use), 5, figsize=(7, 5), dpi=120, constrained_layout=True, sharex=True, sharey=True)
xtick_incr, ytick_incr = 2, 2
xticks = np.arange(0, len(tgt.gs.keys()), xtick_incr)
yticks = np.arange(0, len(smp.gs.keys()), ytick_incr)
plot_netsld = True
lfs = 7  # Label Font Size
tfs = 8  # Title Font Size
count = 0
for neigh_order in neigh_orders_to_use:

    axis_num = 0
    imh = ax[count, axis_num].imshow(jaccard_sim[neigh_order], cmap='nipy_spectral')
    # plt.colorbar(imh, ax=ax[axis_num])
    ax[count, axis_num].set_xlabel('Target GS tempral slice number', fontsize=lfs)
    ax[count, axis_num].set_ylabel('Sample GS tempral slice number', fontsize=lfs)
    ax[count, axis_num].set_title(f'Jaccard sim. measure,\n O(n)={neigh_order}', fontsize=tfs)
    ax[count, axis_num].invert_yaxis()
    ax[count, axis_num].set_xticks(xticks)
    ax[count, axis_num].set_yticks(yticks)

    axis_num = 1
    imh = ax[count, axis_num].imshow(wasserstein_sim[neigh_order], cmap='nipy_spectral', vmin=0, vmax=1)
    # plt.colorbar(imh, ax=ax[axis_num])
    ax[count, axis_num].set_xlabel('Target GS tempral slice number', fontsize=lfs)
    ax[count, axis_num].set_ylabel('Sample GS tempral slice number', fontsize=lfs)
    ax[count, axis_num].set_title(f'Wasserstein distance based sim.\n measure, O(n)={neigh_order}', fontsize=tfs)
    ax[count, axis_num].invert_yaxis()
    ax[count, axis_num].set_xticks(xticks)
    ax[count, axis_num].set_yticks(yticks)

    axis_num = 2
    imh = ax[count, axis_num].imshow(ks_sim[neigh_order], cmap='nipy_spectral', vmin=0, vmax=1)
    # plt.colorbar(imh, ax=ax[axis_num])
    ax[count, axis_num].set_xlabel('Target GS tempral slice number', fontsize=lfs)
    ax[count, axis_num].set_ylabel('Sample GS tempral slice number', fontsize=lfs)
    ax[count, axis_num].set_title(f'Kolmogorov-Smirnov P-value based\n sim. measure, O(n)={neigh_order}. Inequal bins', fontsize=tfs)
    ax[count, axis_num].invert_yaxis()
    ax[count, axis_num].set_xticks(xticks)
    ax[count, axis_num].set_yticks(yticks)

    axis_num = 3
    imh = ax[count, axis_num].imshow(energy_sim[neigh_order], cmap='nipy_spectral', vmin=0, vmax=1)
    # plt.colorbar(imh, ax=ax[axis_num])
    ax[count, axis_num].set_xlabel('Target GS tempral slice number', fontsize=lfs)
    ax[count, axis_num].set_ylabel('Sample GS tempral slice number', fontsize=lfs)
    ax[count, axis_num].set_title(f'Energy distance based\n sim. measure, O(n)={neigh_order}', fontsize=tfs)
    ax[count, axis_num].invert_yaxis()
    ax[count, axis_num].set_xticks(xticks)
    ax[count, axis_num].set_yticks(yticks)
    if plot_netsld:
        axis_num = 4
        imh = ax[count, axis_num].imshow(netlsd_sim[neigh_order], cmap='nipy_spectral', vmin=0, vmax=1)
        ax[count, axis_num].set_xlabel('Target GS tempral slice number', fontsize=lfs)
        ax[count, axis_num].set_ylabel('Sample GS tempral slice number', fontsize=lfs)
        ax[count, axis_num].set_title(f'NetLSD sim. measure,\n O(n)={neigh_order}', fontsize=tfs)
        ax[count, axis_num].invert_yaxis()
        ax[count, axis_num].set_xticks(xticks)
        ax[count, axis_num].set_yticks(yticks)

    count += 1

cbar = plt.colorbar(imh, ax=ax[:], fraction=0.046, pad=0.04,
                    orientation='vertical', aspect=30,
                    ticks=np.arange(0, 1.1, 0.1))
cbar.set_label('Measure of representativeness R(S|T)', fontsize=10)


DATA = ks_sim
DATA_TITLE = 'R: Similarity(Kolmogorov-Smirnov test P-value)'

DATA = wasserstein_sim
DATA_TITLE = 'R: Wasserstein distance based sim.\n measure'

DATA = netlsd_sim
DATA_TITLE = 'R: NetLSD sim. measure'

DATA = energy_sim
DATA_TITLE = 'Energy distance based sim. measure'

DATA = jaccard_sim
DATA_TITLE = 'Jaccard sim. measure'

neigh_orders_to_use = neigh_orders[:-2]
# neigh_orders_to_use = [1]
n_bins = 30
ANG_DISTANCE = {i: {'bin_means': None, 'min':None, 'mean':None, 'max':None, 'std':None} for i in neigh_orders_to_use}
for neigh_order in neigh_orders_to_use:
    print(f'neighbour order: {neigh_order}')
    bin_means, DATA_approx = approximate_to_bin_means(DATA[neigh_order], n_bins=n_bins)
    angular_distance_min = np.zeros_like(bin_means)
    angular_distance_mean = np.zeros_like(bin_means)
    angular_distance_max = np.zeros_like(bin_means)
    angular_distance_std = np.zeros_like(bin_means)

    for bm_i, bm in enumerate(bin_means):
        bm_locs = np.argwhere(DATA_approx == bm)
        bin_means_sparse = np.zeros((bm_locs.shape[0], bm_locs.shape[0]))
        angular_distance_sparse = np.zeros((bm_locs.shape[0], bm_locs.shape[0]))
        for i in range(bm_locs.shape[0]):
            for j in range(bm_locs.shape[0]):
                if i>j:
                    # Only find the upper triangular matrix, thats enough.
                    angular_distance_sparse[j, i] = calculate_angular_distance(bm_locs[j], bm_locs[i])
                else:
                    # Nothing left to do here.
                    pass
        # plt.imshow(angular_distance_sparse)
        angular_distance_sparse = np.unique(angular_distance_sparse)
        angular_distance_sparse_compact = angular_distance_sparse[np.nonzero(angular_distance_sparse)[0]]
        if angular_distance_sparse_compact.size == 0:
            angular_distance_min[bm_i] = np.NaN
            angular_distance_mean[bm_i] = np.NaN
            angular_distance_max[bm_i] = np.NaN
            angular_distance_std[bm_i] = np.NaN
        else:
            angular_distance_min[bm_i] = angular_distance_sparse_compact.min()
            angular_distance_mean[bm_i] = angular_distance_sparse_compact.mean()
            angular_distance_max[bm_i] = angular_distance_sparse_compact.max()
            angular_distance_std[bm_i] = angular_distance_sparse_compact.std()
    ANG_DISTANCE[neigh_order]['bin_means'] = bin_means
    ANG_DISTANCE[neigh_order]['min'] = angular_distance_min
    ANG_DISTANCE[neigh_order]['mean'] = angular_distance_mean
    ANG_DISTANCE[neigh_order]['max'] = angular_distance_max
    ANG_DISTANCE[neigh_order]['std'] = angular_distance_std


plt.figure(figsize=(5, 5), dpi=150, constrained_layout=True)
# Choose a colormap (e.g., 'viridis', 'plasma', 'tab20')
cmap = cm.get_cmap('nipy_spectral')
num_colors = len(neigh_orders_to_use)  # Number of colors needed
legends, legend_names = [], []
color_increment = 1.0 / (len(neigh_orders_to_use) + 1)  # Add 1 to avoid using the last color in the colormap, which is often too light
for i, neigh_order in enumerate(neigh_orders_to_use):
    color = cmap(color_increment * (i + 1))  # Use color_increment to space out the colors
    line_1, = plt.plot(ANG_DISTANCE[neigh_order]['bin_means'],
                       ANG_DISTANCE[neigh_order]['mean'],
                       linestyle='-', color=color,
                       marker='s', markersize=5, markerfacecolor=color)
    fill_1 = plt.fill_between(ANG_DISTANCE[neigh_order]['bin_means'],
                              ANG_DISTANCE[neigh_order]['mean'] - ANG_DISTANCE[neigh_order]['std'],
                              ANG_DISTANCE[neigh_order]['mean'] + ANG_DISTANCE[neigh_order]['std'],
                              color=color, alpha=0.2)
    legends.append((line_1, fill_1))
    legend_names.append(f'Neigh order, O({neigh_order})')
plt.margins(x=0)
plt.legend(legends, legend_names, facecolor='none', edgecolor='none', loc=1)
ax=plt.gca()
ax.set_xlim(0, 1.0)
ax.set_ylim(0, 1.6)
ax.set_xlabel(DATA_TITLE, fontsize=12)
ax.set_ylabel('Mean angular distance, @Iso-valued R (radians)', fontsize=12)
ax.set_xticks(np.arange(0, 1.1, 0.1))
plt.grid(True, linestyle=':', color='gray', alpha=0.2)



ax.set_xlim(0.94, 1.0)
ax.set_xticks(np.arange(0.94, 1.0, 0.02))




'''
mean_mean = []
std_mean = []
min_mean = []
for i, neigh_order in enumerate(neigh_orders_to_use):
    min_mean.append(np.nanmin(ANG_DISTANCE[neigh_order]['mean']))
    mean_mean.append(np.nanmean(ANG_DISTANCE[neigh_order]['mean']))
    std_mean.append(np.nanstd(ANG_DISTANCE[neigh_order]['mean']))
clr = 'r'
min_mean, mean_mean, std_mean = np.array(min_mean), np.array(mean_mean), np.array(std_mean)
plt.figure(figsize=(5, 5), dpi=150)
line_1, = plt.plot(neigh_orders_to_use, mean_mean, linestyle='-', color=clr, marker='s',
                   label='Mean of angular distance mean')
fill_1 = plt.fill_between(neigh_orders_to_use,
                          mean_mean - std_mean, mean_mean + std_mean,
                          color=clr, alpha=0.2,
                          label='2 Std. of angular distance mean')
line_2 = plt.plot(neigh_orders_to_use, min_mean, '-ko', markersize=5,
                  mfc='black', label='Min. of angular distance mean')
plt.margins(x=0)
plt.legend()
ax=plt.gca()
ax.set_xlim(min(neigh_orders_to_use), max(neigh_orders_to_use))
ax.set_ylim(0, 1)
ax.set_xlabel('O(n)', fontsize=12)
ax.set_ylabel('Mean of mean angular distance, Iso-valued R, radians', fontsize=12)
ax.set_xticks(neigh_orders_to_use)
ax.set_title(DATA_TITLE)
'''
