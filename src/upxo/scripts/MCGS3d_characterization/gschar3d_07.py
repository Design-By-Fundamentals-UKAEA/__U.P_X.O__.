# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 23:59:41 2024

@author: rg5749
"""

from upxo.ggrowth.mcgs import mcgs
import numpy as np
import pyvista as pv
# # -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## --
pxt = mcgs()
pxt.simulate(verbose=False)
tslice = 49
gstslice = pxt.gs[tslice]
gstslice.char_morphology_of_grains(label_str_order=1,
                                   make_pvgrid=False,
                                   find_neigh=[False, [1]],
                                   find_grain_voxel_locs=True,
                                   find_spatial_bounds_of_grains=True,
                                   find_grain_locations=True,
                                   force_compute=True,)
gstslice.set_mprops(volnv=True, eqdia=False,
                    eqdia_base_size_spec='volnv',
                    arbbox=True, arbbox_fmt='gid_dict',
                    arellfit=False, arellfit_metric='max',
                    arellfit_calculate_efits=False,
                    arellfit_efit_routine=1,
                    arellfit_efit_regularize_data=False,
                    solidity=False, sol_nan_treatment='replace',
                    sol_inf_treatment='replace',
                    sol_nan_replacement=-1, sol_inf_replacement=-1)

gstslice.make_pvgrid()
gstslice.add_scalar_field_to_pvgrid(sf_name="lgi", sf_value=None)

gstslice.plot_gs_pvvox(show_edges=False)
#gstslice.plot_grains(gstslice.get_largest_gids(), opacity=1, lw=3)
gstslice.clean_gs_GMD_by_source_erosion_v1(prop='volnv',
                                           threshold=4,
                                           parameter_metric='mean',
                                           reset_pvgrid_every_iter=True,
                                           find_neigh_every_iter=False,
                                           find_grvox_every_iter=True,
                                           find_grspabnds_every_iter=True)

gstslice.n

# gstslice.viz_mesh_slice_ortho(scalar='lgi', cmap='viridis', style='surface', throw=False, pvp=None)

from zmesh import Mesher
labels = gstslice.lgi
# -----------------------
gids = gstslice.gid
# Mesh in zmesh
mesher = Mesher( (10, 10, 10) )
mesher.mesh(labels, close=True)
meshes = []
mesher_ids = []
for obj_id in mesher.ids():
  meshes.append(mesher.get(obj_id, normals=False, reduction_factor=0, max_error=0.1, voxel_centered=False,))
  mesher_ids.append(obj_id)
  mesher.erase(obj_id)
mesher.clear()
mesher_ids = np.array(mesher_ids)
sortids = np.argsort(mesher_ids)
meshes = [meshes[si] for si in sortids]
## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## --
## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## --
## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## --
subset = [meshes[gid-1] for gid in gids]
## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## --
## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## --
filename_base = r'C:\Development\M2MatMod\upxo_packaged\upxo_private\src\upxo\scripts\MCGS3d_conformal_meshing\objs'
# Write the surface mesh to file for import to tetgen for tet meshing
for count, mesh in enumerate(subset, start=1):
    mesh = mesher.simplify(mesh, reduction_factor=0, max_error=0.5, compute_normals=False)
    with open(filename_base+f'\{gids[count-1]}.obj', 'wb') as f:
      f.write(mesh.to_obj())
## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## --
## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## --
## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## --
## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## --
## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## --
import tetgen
# Create tetrahedral meshes
tetmeshes = []
for count, mesh in enumerate(subset, start=1):
    mesh = pv.get_reader(filename_base + f'\{gids[count-1]}.obj').read()
    tet = tetgen.TetGen(mesh)
    tet.tetrahedralize(order=1, quality=True, mindihedral=20, minratio=1.0,
                       verbose=False, refine=0.5,
                       nobisect=False, steinerleft=1E6, smooth_alpha=0.0)
    tetmeshes.append(tet)
## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## --
## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## --
## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## --
## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## --
## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## --
combined_mesh = pv.UnstructuredGrid()
for mesh in tetmeshes:
    combined_mesh = combined_mesh.merge(mesh.grid)

combined_mesh.plot(show_edges=True)

crinkled = combined_mesh.clip(normal=[0, 0.6, 1], crinkle=True)
crinkled.plot(show_edges=True)
