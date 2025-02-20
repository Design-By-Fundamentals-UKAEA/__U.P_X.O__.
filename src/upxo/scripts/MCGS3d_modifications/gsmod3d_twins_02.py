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
from scipy.spatial import cKDTree
# =============================================================================
pxt = mcgs()
pxt.simulate(verbose=False)
tslice = 199
gstslice = pxt.gs[tslice]
gstslice.char_morphology_of_grains(label_str_order=1,
                                   find_grain_voxel_locs=True,
                                   find_spatial_bounds_of_grains=True)
gstslice.set_mprops(volnv=True, eqdia=False, eqdia_base_size_spec='volnv',
                    arbbox=False, arbbox_fmt='gid_dict', arellfit=False,
                    arellfit_metric='max', arellfit_calculate_efits=False,
                    arellfit_efit_routine=1,
                    arellfit_efit_regularize_data=False, solidity=False,
                    sol_nan_treatment='replace', sol_inf_treatment='replace',
                    sol_nan_replacement=-1, sol_inf_replacement=-1,)
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
    gstslice.plot_grains([largest_grain_gid], opacity=0.5, lw=1)
# =============================================================================
'''
gstslice.build_gbp()
gbp_grain = gstslice.Ggbp_all[2]


from upxo.geoEntities.mulpoint3d import MPoint3d as mp3d
mp_gbp_grain = mp3d.from_coords(gbp_grain)

gbp_grain_tree = mp_gbp_grain.maketree(saa=True, throw=True)
gbp_grain_tree.querry

gbp_random = gbp_grain[np.random.choice(range(0, gbp_grain.shape[0]), 1, replace=False)]

coords = mp_gbp_grain.find_first_order_neigh_CUBIC(gbp_random,
                                                   0.5001,
                                                   return_indices=False,
                                                   return_coords=True,
                                                   return_input_coord=False)[0]
'''
# ------------------------------------
# Select the gid of interest and get its coordinates
GID = gstslice.get_largest_gids()[0]
# gcoords = gstslice.grain_locs[GID]
# Extract grain boundary voxels of this grain
# tree = cKDTree(gcoords)
# neighbor_counts = tree.query_ball_point(gcoords, r=np.sqrt(3)*1.00001, return_length=True)
# boundary_coords = gcoords[neighbor_counts < 26]
# Get the bounding box lgi of this grain
BBLGI = gstslice.find_bounding_cube_gid(GID)
# Mask the bounding box lgi of this grain with the grain ID
BBLGI_mask = BBLGI == GID
# Find the locations of grain voxels in the bounding box
BBLGI_locs = np.argwhere(BBLGI == GID)
# Construct tree of the grain voxel locations
BBLGI_locstree = cKDTree(BBLGI_locs)
# Find the number of nearest neighbours of every voxel in the grain
neighbor_counts = BBLGI_locstree.query_ball_point(BBLGI_locs,
                                                  r=np.sqrt(3)*1.00001,
                                                  return_length=True)
# Boundary coordinates are those which have less than 26 neighbours
boundary_coords = BBLGI_locs[neighbor_counts < 26]
"""
TODO: We can identify those locations on the grain boundary which will create
problems for conformal meshing by using cleverly the number of nearest
neighbours. For example, a value of exactly equal to 25 would mean the
presence of a pit of exactly 1 voxel depth. Ironing it out would smoothen the
grain boundary surface significatly, thus enabling a smoother conformal
meshing operation.
"""
# Update the mask to a new variable and seperate the grain boundary from core
BBLGI_mask_ = BBLGI_mask.astype(int)
for bc in boundary_coords:
    BBLGI_mask_[bc[0], bc[1], bc[2]] = -1

BBLGI_mask_gb = np.copy(BBLGI_mask_)
BBLGI_mask_gb[BBLGI_mask_gb != -1] = 0
BBLGI_mask_gb = np.abs(BBLGI_mask_gb)

BBLGI_mask_core = np.copy(BBLGI_mask_)
BBLGI_mask_core[BBLGI_mask_core == -1] = 0
CORE_coords = np.argwhere(BBLGI_mask_core == 1)

# Choose a random grain boundary point on this grain and find the first order
# nearest neigbours for this point.

mp_gbp_grain = mp3d.from_coords(boundary_coords)
gbp_rand_id = np.random.choice(range(boundary_coords.shape[0]),
                               1, replace=False)[0]
gbp_rand_coord = boundary_coords[gbp_rand_id]
coords = mp_gbp_grain.find_first_order_neigh_CUBIC(gbp_rand_coord, 1.0,
                                                   return_indices=False,
                                                   return_coords=True,
                                                   return_input_coord=False)[0]
# Find the nearest neighbours of gbp_rand_coord in CORE_coords.
CORE_tree = cKDTree(CORE_coords)
K = 10
_, nearest_ids = CORE_tree.query(gbp_rand_coord, k=K)
nearest_coords_in_core = CORE_tree.data[nearest_ids]

n = np.random.choice(range(K), 2, replace=False)
tp = Plane.from_three_points(gbp_rand_coord,
                             nearest_coords_in_core[n[0]],
                             nearest_coords_in_core[n[1]])
# ---------- Construct the planes which make the twins
num_planes, translation_vector = 10, np.array([7, 7, 7])
tps1 = tp.create_translated_planes(translation_vector, num_planes)
tps2 = tp.create_translated_planes(-translation_vector, num_planes)
tps = tuple(tps1) + tuple(tps2)
# ---------- Calc. perp distances from each plane to all bounding box coords
D_gbz = [p.calc_perp_distances(boundary_coords, signed=False) for p in tps]
D_core = [p.calc_perp_distances(CORE_coords, signed=False) for p in tps]
D = [p.calc_perp_distances(BBLGI_locs, signed=False) for p in tps]

# ---------- Identify BBC points which can form twins as per thickness
twin_thick = 0.5
TWIN_COORDS_gbz = [boundary_coords[np.argwhere(d <= 2*twin_thick)].squeeze()
                   for d in D_gbz]
TWIN_COORDS_core = [CORE_coords[np.argwhere(d <= 2*twin_thick)].squeeze()
                    for d in D_core]
TWIN_COORDS = [BBLGI_locs[np.argwhere(d <= 2*twin_thick)].squeeze()
               for d in D]
# =============================================================================





# Extend each of the lines [gbp_rand_coord - nearest_coords_in_core]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(boundary_coords[:, 0], boundary_coords[:, 1], boundary_coords[:, 2],
           c='c', marker='o', alpha=0.02, s=60, edgecolors='none')
# ax.scatter(CORE_coords[::4, 0], CORE_coords[::4, 1], CORE_coords[::4, 2],
#            c='maroon', marker='o', alpha=0.05, s=40, edgecolors='none')
ax.scatter(gbp_rand_coord[0], gbp_rand_coord[1], gbp_rand_coord[2],
           c='b', marker='o', alpha=1.0, s=40, edgecolors='black')
ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
           c='red', marker='x', alpha=1.0, s=5, edgecolors='black')
ax.scatter(nearest_coords_in_core[:, 0],
           nearest_coords_in_core[:, 1],
           nearest_coords_in_core[:, 2],
           c='k', marker='o', alpha=1.0, s=10, edgecolors='black')
# Starting points of vectors
vix, viy, viz = gbp_rand_coord
vjx, vjy, vjz = nearest_coords_in_core.T
U, V, W = vjx - vix, vjy - viy, vjz - viz
ax.quiver(vix, viy, viz, U, V, W, color='blue')

for tcgbz, tcc in zip(TWIN_COORDS_gbz, TWIN_COORDS_core):
    ax.scatter(tcgbz[:, 0],
               tcgbz[:, 1],
               tcgbz[:, 2],
               c='black', marker='o', alpha=0.25, s=20, edgecolors='black')
    ax.scatter(tcc[:, 0],
               tcc[:, 1],
               tcc[:, 2],
               c='red', marker='o', alpha=0.25, s=20, edgecolors='red')
