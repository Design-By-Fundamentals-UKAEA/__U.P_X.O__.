"""
Development code. Introduction of twins in grains
"""

from upxo.ggrowth.mcgs import mcgs
import numpy as np
import pyvista as pv
from upxo.geoEntities.mulpoint3d import MPoint3d as mp3d
from upxo.geoEntities.plane import Plane
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# =============================================================================
pxt = mcgs()
pxt.simulate(verbose=False)
tslice = 99
gstslice = pxt.gs[tslice]
gstslice.char_morphology_of_grains(label_str_order=1)
gstslice.set_mprops(volnv=True, eqdia=False, eqdia_base_size_spec='volnv',
                    arbbox=False, arbbox_fmt='gid_dict', arellfit=False,
                    arellfit_metric='max', arellfit_calculate_efits=False,
                    arellfit_efit_routine=1,
                    arellfit_efit_regularize_data=False, solidity=False,
                    sol_nan_treatment='replace', sol_inf_treatment='replace',
                    sol_nan_replacement=-1, sol_inf_replacement=-1)
# =============================================================================
plot_now = False
'''Plot the grain structure'''
if plot_now:
    gstslice.make_pvgrid()
    gstslice.add_scalar_field_to_pvgrid(sf_name="lgi", sf_value=None)
    gstslice.plot_gs_pvvox(show_edges=False)
# =============================================================================
plot_now = False
'''Plot the largest grain. Trial twins will be introduced in this'''
largest_grain_gid = gstslice.get_largest_gids()
if plot_now:
    gstslice.plot_grains(largest_grain_gid, opacity=0.8, lw=1)
# =============================================================================
# ---------- Grain structure cleaning operation ----------
'''Remove all single voxel grains by erosion assisted mergers'''
grainsize = np.array(list(gstslice.mprop['volnv'].values()))
print(f'No. of unit vox. grains = {np.sum(grainsize == 1)} of {gstslice.n} grains')
gstslice.clean_gs_GMD_by_source_erosion_v1(prop='volnv',
                                           threshold=1.5,
                                           parameter_metric='mean',
                                           reset_pvgrid_every_iter=True,
                                           find_neigh_every_iter=False,
                                           find_grvox_every_iter=True,
                                           find_grspabnds_every_iter=True)
# ---------- Grain structure post-cleaning operations ----------
# 1. Recalculate grain sizes
gstslice.set_mprop_volnv()
grainsize = np.array(list(gstslice.mprop['volnv'].values()))
print(f'No. of unit vox. grains = {np.sum(grainsize == 1)} of {gstslice.n} grains')
# 2. Re-identify the largwest grain and plot
largest_grain_gid = gstslice.get_largest_gids()[0]
if plot_now:
    gstslice.plot_grains([largest_grain_gid], opacity=1.0, lw=1)
# =============================================================================
# ---------- Prepare background data: Coordinates and lgi bounding box
# 1. Find bounding box coordinates
gid = largest_grain_gid-1
xsl = slice(gstslice.spbound['xmins'][gid], gstslice.spbound['xmaxs'][gid]+1)
ysl = slice(gstslice.spbound['ymins'][gid], gstslice.spbound['ymaxs'][gid]+1)
zsl = slice(gstslice.spbound['zmins'][gid], gstslice.spbound['zmaxs'][gid]+1)
BBX = pxt.xgr[zsl, ysl, xsl]
BBY = pxt.ygr[zsl, ysl, xsl]
BBZ = pxt.zgr[zsl, ysl, xsl]
# 2. Offset to start from origin
BBX = BBX - BBX.min()
BBY = BBY - BBY.min()
BBZ = BBZ - BBZ.min()
# 2. Find bounding box lgi i.e. gid values
BBLGI = gstslice.find_bounding_cube_gid(largest_grain_gid)
# 3. Create bounding lgi box masked with the specific gid
BBLGI_mask = BBLGI == largest_grain_gid
# 4. Identify locations where BBLGI == specific gid
BBLGI_locs = np.argwhere(BBLGI == largest_grain_gid)
# Cross-identify the coordinates of the grain
BBLGI_coords_X = np.array([BBX[crd[0], crd[1], crd[2]] for crd in BBLGI_locs])
BBLGI_coords_Y = np.array([BBY[crd[0], crd[1], crd[2]] for crd in BBLGI_locs])
BBLGI_coords_Z = np.array([BBZ[crd[0], crd[1], crd[2]] for crd in BBLGI_locs])
BBLGI_coords = np.vstack((BBLGI_coords_X, BBLGI_coords_Y, BBLGI_coords_Z)).T
# =============================================================================
# ---------- Construct the UPXO multi-point object
BBcoords = np.vstack((BBX.ravel(), BBY.ravel(), BBZ.ravel())).T
BBC_mpoint = mp3d.from_coords(BBcoords)
# ---------- Construct the planes which make the twins
plane = Plane(point=(3.0, 3.0, 3.0), normal=(1, 1, 1))
num_planes, translation_vector = 3, np.array([5, 5, 5])
planes = plane.create_translated_planes(translation_vector, num_planes)
# ---------- Calc. perp distances from each plane to all bounding box coords
D = [p.calc_perp_distances(BBC_mpoint.coords, signed=False) for p in planes]
# ---------- Identify BBC points which can form twins as per thickness
twin_thick = 1.2
TWIN_COORDS = [BBC_mpoint.coords[np.argwhere(d <= 2*twin_thick)].squeeze()
               for d in D]
# ---------- Identify BBLGI points which are a subset of the TWIN_COORDS
'''BBLGI_TWIN = {twin_i: None for twin_i in range(len(TWIN_COORDS))}
for twin_i, twin_coords in zip(range(len(TWIN_COORDS)), TWIN_COORDS):
    BBLGI_TWIN[twin_i] = np.array([coord for coord in twin_coords if any(np.array_equal(coord, b) for b in BBLGI_coords)])'''
BBLGI_TWIN = {}
for twin_i, twin_coords in enumerate(TWIN_COORDS):
    print('')
    matches = np.any(np.all(twin_coords[:, None, :] == BBLGI_coords[None, :, :],
                            axis=2), axis=1)
    BBLGI_TWIN[twin_i] = twin_coords[matches]

# =============================================================================
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(BBC_mpoint.coords[:, 0],
           BBC_mpoint.coords[:, 1],
           BBC_mpoint.coords[:, 2],
           c='c', marker='o', alpha=0.01,
           s=50, edgecolors='black')
ax.scatter(BBLGI_coords[:, 0],
           BBLGI_coords[:, 1],
           BBLGI_coords[:, 2],
           c='b', marker='o', alpha=0.025,
           s=80, edgecolors='black')

colors = ['red', 'black', 'green']
for twin_i, twin_coord in BBLGI_TWIN.items():
    ax.scatter(twin_coord[:, 0],
               twin_coord[:, 1],
               twin_coord[:, 2],
               c=colors[twin_i], marker='o', alpha=1.0,
               s=30, edgecolors='black')
# =============================================================================
