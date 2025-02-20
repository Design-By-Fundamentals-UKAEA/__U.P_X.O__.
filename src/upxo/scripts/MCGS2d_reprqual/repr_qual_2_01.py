"""
Created on Fri May 17 09:25:49 2024

@author: Dr. Sunil Anandatheertha

Explanations
------------
This example assesses the representativeness of 3D grain structure to 2D slices
taken from it. For simplicity, we will construct 3D voxellated  Voronoi tessellated grain structure.
"""
import damask
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
# -----------------------------------------------------
size = np.ones(3)
cells = [250,250,250]
N_grains = 1000
seeds = damask.seeds.from_random(size,N_grains,cells)
grid = damask.GeomGrid.from_Voronoi_tessellation(cells,size,seeds)
grid.save(f'Polycystal_{N_grains}_{cells[0]}x{cells[1]}x{cells[2]}')
grid
# -----------------------------------------------------
fmat = grid.material
# -----------------------------------------------------
fids = np.unique(fmat)
"""
'''Lets plot a few grains in the 3D grain structure'''
features = {fid: None for fid in fids}
for fid in fids:
    features[fid] = np.argwhere(fmat == fid)

# Lerts plot the first two grains
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for fid in range(2):
    xyz = np.argwhere(fmat == fid)
    plt.plot(xyz[:,0], xyz[:,1], xyz[:,2], 'o')
'''We see that due to symmetry of Voronoi tessellation, damask ends up
assigining grains on opposite faces, the same ID. We need to overcome this.
Lets use scikit ndimage to find connected components and label the image
instead.'''
"""
# -----------------------------------------------------
"""Lets now label the 3D grain structure using feature matrix."""
from scipy.ndimage import label, generate_binary_structure
struct = generate_binary_structure(3, 1)
"""glabels: grain labels"""
glabels = np.zeros_like(fmat)
for fid in fids:
    print(f'GRain set no. {fid}')
    labeled_array, num_features = label(fmat==fid, structure=struct)
    if fid == 0:
        glabels = labeled_array
    else:
        labeled_array[labeled_array != 0] += glabels.max()
        glabels += labeled_array
"""We will now construct IDs of all grains"""
gids = np.unique(glabels)
"""Lets calculate th3 total number of grains in the grain structure"""
ngrains = glabels.max()
"""If need be, consider plotting. Can be quite time consuming.
Not a good visualization anyway. But should do for most simple purposes."""
PLOT_GRAINS = False
if PLOT_GRAINS:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for fid_ in list(np.unique(glabels))[1:]:
        xyz = np.argwhere(glabels == fid_)
        plt.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], 'o')
'''Lets, now segmewnt into grains.'''
grains = {gid: np.argwhere(glabels == gid) for gid in gids}
grain_vol = [grains[gid].shape[0] for gid in gids]
'''Lets calculate the Equivalent Sphere Diameter'''
ESD = np.cbrt(6*np.array(grain_vol)/np.pi)
ESD_mean = ESD.mean()
ESD_std = ESD.std()
# sns.histplot(grain_vol)
# sns.histplot(ESD)
"""
Lets now shift our attention to slices.
We will not slice the field matrix into slices by specifying the slicing
normal. Slicing normal is either x or y or z axis.
"""
Nslicesx, Nslicesy, Nslicesz = fmat.shape
slice_axis = 'z'
slice_start = 0
slice_end = -1
slice_incr = 25
# -----------------------------------------------------
if slice_axis == 'x':
    slicelocs = np.arange(0, Nslicesx, 1)[::slice_incr]
    slices = [fmat[slc,:,:] for slc in slicelocs]
elif slice_axis == 'y':
    slicelocs = np.arange(0, Nslicesy, 1)[::slice_incr]
    slices = [fmat[:,slc,:] for slc in slicelocs]
elif slice_axis == 'z':
    slicelocs = np.arange(0, Nslicesz, 1)[::slice_incr]
    slices = [fmat[:,:,slc] for slc in slicelocs]

len(slicelocs)
# -----------------------------------------------------
slicelocs_norm = np.reshape(slicelocs/slice_incr, (1, len(slicelocs))).astype(int)

fmatslices = []
for r in range(slicelocs_norm.shape[0]):
    fmatslices_c = []
    for c in range(slicelocs_norm.shape[1]):
        fmatslices_c.append(slices[slicelocs_norm[r, c]])
    fmatslices.append(fmatslices_c)
fmatslices = np.array(fmatslices)

fmatslices.shape
# -----------------------------------------------------
PLOT_GS = True
if PLOT_GS:
    fig, ax = plt.subplots(nrows=1, ncols=len(slicelocs),
                           sharex=True, sharey=True, squeeze=True)
    images = np.zeros_like(slicelocs_norm).tolist()
    for r in range(slicelocs_norm.shape[0]):
        for c in range(slicelocs_norm.shape[1]):
            if slicelocs_norm.shape[0] == 1:
                images[r][c] = ax[c].imshow(fmatslices[r][c])
            else:
                images[r][c] = ax[r, c].imshow(fmatslices[r][c])
    norm = colors.Normalize(vmin=fmat.min(), vmax=fmat.max())
    for r in range(slicelocs_norm.shape[0]):
        for c in range(slicelocs_norm.shape[1]):
            images[r][c].set_norm(norm)
    fig.colorbar(images[r][c],
                 ax=ax.ravel().tolist(),
                 orientation='horizontal',
                 fraction=.02,
                 label='Grain number'
                 )
# -----------------------------------------------------
slices2d = [[mcgs_mchar_2d() for c in range(slicelocs_norm.shape[1])]
            for r in range(slicelocs_norm.shape[0])]
grains2d = [[None for c in range(slicelocs_norm.shape[1])]
            for r in range(slicelocs_norm.shape[0])]
Ngrains2d = [[None for c in range(slicelocs_norm.shape[1])]
            for r in range(slicelocs_norm.shape[0])]
grain_areas_2d = [[None for c in range(slicelocs_norm.shape[1])]
            for r in range(slicelocs_norm.shape[0])]
ECD = [[None for c in range(slicelocs_norm.shape[1])]
            for r in range(slicelocs_norm.shape[0])]
fmin, fmax = 1, glabels.max()

for r in range(slicelocs_norm.shape[0]):
    for c in range(slicelocs_norm.shape[1]):
        slices2d[r][c].set_fmat(fmatslices[r][c], fmin, fmax)
        grains2d[r][c] = slices2d[r][c].find_grains(library='opencv',
                                                    fmat=fmatslices[r][c],
                                                    kernel_order=2)
        Ngrains2d[r][c] = grains2d[r][c]['Ngrains']
        grain_areas_2d[r][c] = grains2d[r][c]['gid_npxl']
        ECD[r][c] = 2*np.sqrt(np.array(grain_areas_2d[r][c])/np.pi)
# -----------------------------------------------------
grain_areas_2d_mean = [[np.array(grain_areas_2d[r][c]).mean()
                        for c in range(slicelocs_norm.shape[1])]
                       for r in range(slicelocs_norm.shape[0])]
grain_areas_2d_std = [[np.array(grain_areas_2d[r][c]).std()
                        for c in range(slicelocs_norm.shape[1])]
                       for r in range(slicelocs_norm.shape[0])]

grain_areas_2d_min = [[np.array(grain_areas_2d[r][c]).min()
                        for c in range(slicelocs_norm.shape[1])]
                       for r in range(slicelocs_norm.shape[0])]

grain_areas_2d_max = [[np.array(grain_areas_2d[r][c]).max()
                        for c in range(slicelocs_norm.shape[1])]
                       for r in range(slicelocs_norm.shape[0])]
# -----------------------------------------------------
ECD_mean = [[np.array(ECD[r][c]).mean()
                        for c in range(slicelocs_norm.shape[1])]
                       for r in range(slicelocs_norm.shape[0])]
ECD_std = [[np.array(ECD[r][c]).std()
                        for c in range(slicelocs_norm.shape[1])]
                       for r in range(slicelocs_norm.shape[0])]
ECD_max = np.array([[np.array(ECD[r][c]).max()
                        for c in range(slicelocs_norm.shape[1])]
                       for r in range(slicelocs_norm.shape[0])])
# -----------------------------------------------------
"""Lets now see how the equivalent diameters measure up against each other.

NOTES:
    * Bin widths are minimum of each equivalent diameter data.
"""
fig = plt.figure(figsize=(7, 5), dpi=150)
ax = plt.gca()
sns.kdeplot(ESD, bw_adjust=ESD.min(),
            fill=True, color='maroon', cumulative=False, common_norm=True,
            label='ESD. Parent.')
_kdeevalmax_ = max(max(ESD), ECD_max.max())
_eqdia_ = np.arange(0, _kdeevalmax_, 0.2)

for r in range(slicelocs_norm.shape[0]):
    for c in range(slicelocs_norm.shape[1]):
        sns.kdeplot(ECD[r][c], bw_adjust=ECD[r][c].min(),
                    fill=False, cumulative=False, common_norm=True,
                    label=f'ECD. Slice-{slice_axis}-({r},{c})')
plt.legend(fontsize=8)
_xmax_ = 1.4*max(max(ESD), ECD_max.max())
ax.set_xlim(0, _xmax_)
plt.xlabel('Equivalent diameter')
plt.ylabel('Probability Density')


ECD_reevaluated = [None for line in ax.lines]
ESD_sorted = np.sort(ESD)
for i, line in enumerate(ax.lines):
    x_data, y_data = line.get_xdata(), line.get_ydata()
    densities = []
    for _x_ in ESD_sorted:
        densities.append(np.interp(_x_, x_data, y_data))
    ECD_reevaluated[i] = densities

ECD_reevaluated = np.array(ECD_reevaluated)
ECD_reevaluated.mean(axis=0)
ECD_reevaluated.std(axis=0)

fig = plt.figure(figsize=(7, 5), dpi=150)
ax = plt.gca()
plt.errorbar(ESD_sorted, ECD_reevaluated.mean(axis=0), yerr=ECD_reevaluated.std(axis=0))
sns.kdeplot(ESD, bw_adjust=ESD.min(),
            fill=True, color='maroon', cumulative=False, common_norm=True,
            label='ESD. Parent.')



fig = plt.figure(figsize=(7, 5), dpi=150)
ax = plt.gca()
sns.kdeplot(ESD, bw_adjust=ESD.min(), clip=[ESD.min(), ESD.max()],
            fill=True, color='maroon', cumulative=False, common_norm=True,
            label='ESD: parent 3d', alpha=0.5)
plt.plot(ESD_sorted, ECD_reevaluated.mean(axis=0), '-b',
         label='ECD: slices 2d Mean')
plt.fill_between(ESD_sorted,
                 ECD_reevaluated.mean(axis=0)-ECD_reevaluated.std(axis=0),
                 ECD_reevaluated.mean(axis=0)+ECD_reevaluated.std(axis=0),
                 alpha=0.25, label='ECD: slices 2d Std.')
plt.xlabel('Equivalent diameter')
plt.ylabel('Probability density')
ax.set_xlim(ESD.min(), ESD.max())
plt.legend(fontsize=8, loc='upper left')
plt.title(f'Equivalent diameters of 3D voxelated VTGS vs. 2D slices (normal: {slice_axis}-axis), nslices={len(slicelocs)}', fontsize=9)


sns.histplot(ESD)
