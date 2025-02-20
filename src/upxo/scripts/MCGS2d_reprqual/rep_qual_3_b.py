# -*- coding: utf-8 -*-
"""
Created on Sat May 25 03:46:49 2024

@author: rg5749
"""
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
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

NUMBERS = dth.dt.NUMBERS
ITERABLES = dth.dt.ITERABLES
"""
NOTE: DO NOT CHANGE SETTINGS, IN TEREST OF CONFORMITY WITH TOP EXPLANATIONS.
"""
tgt_slice = 18
smp_slice = 18
# =========================================================
tgt = mcgs(study='independent', input_dashboard='input_dashboard.xls')
tgt.simulate()
tgt.detect_grains()
tgt.gs[tgt_slice].n
# =========================================================
smp = mcgs(study='independent', input_dashboard='input_dashboard.xls')
smp.simulate()
smp.detect_grains()
# =========================================================
neigh_order = 5
tgt_nneighgids = tgt.gs[tgt_slice].get_upto_nth_order_neighbors_all_grains(neigh_order,
                                                                           include_parent=True,
                                                                           output_type='nparray')
smp_nneighgids = smp.gs[tgt_slice].get_upto_nth_order_neighbors_all_grains(neigh_order,
                                                                           include_parent=True,
                                                                           output_type='nparray')
# =========================================================
tgt_npixels = tgt.gs[tgt_slice].find_grain_size_fast(metric='npixels')
smp_npixels = smp.gs[tgt_slice].find_grain_size_fast(metric='npixels')

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
tgt_grain_centroids = tgt.gs[tgt_slice].centroids
tgt_grain_centroids_mp = MPoint2d.from_coords(tgt_grain_centroids)
tgt_nneigh_field = np.array([tgt_fields[gid]['nneigh'] for gid in tgt.gs[tgt_slice].gid])
tgt_npixels_field = np.array([tgt_fields[gid]['npixels'] for gid in tgt.gs[tgt_slice].gid])

smp_grain_centroids = smp.gs[smp_slice].centroids
smp_grain_centroids_mp = MPoint2d.from_coords(smp_grain_centroids)
smp_nneigh_field = np.array([smp_fields[gid]['nneigh'] for gid in smp.gs[smp_slice].gid])
smp_npixels_field = np.array([smp_fields[gid]['npixels'] for gid in smp.gs[smp_slice].gid])

tgt_neigh_field = []
for gid in tgt.gs[tgt_slice].gid:
    tgt_neigh_field.append(tgt_nneighgids[gid])
# =========================================================

# =========================================================

tgtX, tgtY = tgt.gs[tgt_slice].xgr, tgt.gs[tgt_slice].ygr
smpX, smpY = smp.gs[smp_slice].xgr, smp.gs[smp_slice].ygr

tgt_z = griddata(tgt_grain_centroids, tgt_nneigh_field, (tgtX, tgtY), method='nearest')
smp_z = griddata(smp_grain_centroids, smp_nneigh_field, (smpX, smpY), method='nearest')
min_z = min(np.nanmin(tgt_z), np.nanmin(smp_z))
max_z = np.round(max(np.nanmax(tgt_z), np.nanmax(smp_z))/10)*10


plt.figure(figsize=(6, 5), dpi=150)
contour = plt.contourf(tgt_z, vmin=min_z, vmax=tgt_z.max(), cmap='nipy_spectral')
plt.colorbar(contour, format=ticker.FormatStrFormatter('%.0f'))
plt.scatter(tgt_grain_centroids[:,0], tgt_grain_centroids[:,1], s=3, color = 'black')
plt.xlabel('x-axis'), plt.ylabel('y-axis')
plt.title(f'Number of O({neigh_order}) neighbours')

tgt.gs[tgt_slice].plot_grains(tgt_nneighgids[50])
tgt.gs[tgt_slice].n

# g = sns.jointplot(x=tgt_npixels, y=tgt_nneigh_field, kind='reg')
# g = sns.jointplot(x=tgt_npixels, y=tgt_nneigh_field, kind='hist')
g = sns.jointplot(x=tgt_npixels, y=tgt_nneigh_field, kind='hex', gridsize=25, cmap='viridis',
                  marginal_kws=dict(bins=50, fill=True))
g.plot_marginals(sns.histplot, bins=50, kde=True, color='gray', fill=True)
g.fig.suptitle('Customized Jointplot of A and B', y=1.02)
g.set_axis_labels('A values', 'B values')
g.ax_joint.grid(True, linestyle='--', alpha=0.7)
plt.show()
plt.scatter(tgt_npixels, tgt_nneigh_field, s=3, color='black', alpha=0.25)

fit_polynomial_order = 2
factor = 2
sort_indices = np.argsort(tgt_npixels)
tgt_npixels_limited = tgt_npixels[sort_indices][tgt_npixels[sort_indices] <= factor*tgt_npixels.mean()]
tgt_nneigh_field_limited = tgt_nneigh_field[sort_indices][tgt_npixels[sort_indices] <= factor*tgt_npixels.mean()]
coefficients = np.polyfit(tgt_npixels_limited, tgt_nneigh_field_limited, fit_polynomial_order)
polynomial = np.poly1d(coefficients)
tgt_nneigh_field_fit = polynomial(tgt_npixels_limited)
plt.plot(tgt_npixels_limited, tgt_nneigh_field_fit, 'k')

# =========================================================
plt.figure()
plt.imshow(tgt.gs[tgt_slice].lgi)

ngids = tgt.gs[tgt_slice].get_upto_nth_order_neighbors(10,
                                                       10,
                                                       recalculate=False,
                                                       include_parent=False,
                                                       output_type='list')
ngids = tgt.gs[tgt_slice].get_nth_order_neighbors(10,
                                                  10,
                                                  recalculate=False,
                                                  include_parent=False)
if type(ngids) not in dth.dt.ITERABLES:
    ngids = [ngids]

tgt.gs[tgt_slice].plot_grains(ngids)

# ********************************************************************
KLD = []
bins_max = int(np.array([len(tgt_nneighgids[gid]) for gid in tgt.gs[tgt_slice].gid]).mean())
for gid in tgt.gs[tgt_slice].gid:
    tgt_npixels_field_gid = tgt_npixels_field[np.array(tgt_nneighgids[gid])-1]
    bins = np.histogram_bin_edges(np.concatenate([tgt_npixels_field,
                                                  tgt_npixels_field_gid]),
                                  bins=bins_max)
    target_hist, _ = np.histogram(tgt_npixels_field, bins=bins, density=True)
    sample_hist, _ = np.histogram(tgt_npixels_field_gid, bins=bins, density=True)
    target_hist = np.where(target_hist == 0, 1e-10, target_hist)
    sample_hist = np.where(sample_hist == 0, 1e-10, sample_hist)
    KLD.append(1/entropy(target_hist, sample_hist))
# ********************************************************************
tgt_z = griddata(tgt_grain_centroids, KLD, (tgtX, tgtY), method='nearest')
tgt_z = tgt_z/tgt_z.max()
min_z = np.nanmin(tgt_z)
max_z = np.nanmax(tgt_z)

plt.figure(figsize=(6, 5), dpi=150)
levels = np.arange(0, 1.1, 0.1)
#contour = plt.contourf(tgt_z, levels=levels, cmap='nipy_spectral')
contour = plt.contourf(tgt_z, levels=levels, cmap='nipy_spectral')
plt.colorbar(contour)#, format=ticker.FormatStrFormatter('%.1f'))
plt.scatter(tgt_grain_centroids[:,0], tgt_grain_centroids[:,1], s=3, color = 'black')
plt.xlabel('x-axis'), plt.ylabel('y-axis')
plt.title(f'R-field: RE-KLD (T|S). O({neigh_order}) neighbours')
