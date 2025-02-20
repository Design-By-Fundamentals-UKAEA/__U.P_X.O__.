# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 12:22:51 2024

@author: rg5749
"""
from upxo.ggrowth.mcgs import mcgs
from scipy.spatial import cKDTree
import numpy as np
import pyvista as pv
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter
import upxo._sup.data_ops as DO
import upxo._sup.dataTypeHandlers as dth
from skimage.segmentation import find_boundaries
# ---------------------------
AUTO_PLOT_EVERYTHING = False
# ---------------------------
## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## --
## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## --
## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## --
## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## --
## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## --
pxt = mcgs()
pxt.simulate()
tslice = 19
gstslice = pxt.gs[tslice]
# gstslice.find_grains_scilab_ndimage_3d()
gstslice.find_grains(label_str_order=3)
gstslice.calc_num_grains()
# gstslice.set_npixels()
# gstslice.px_size
# gstslice.gid_s
# gstslice.s_gid
# gstslice.s_n
# gstslice.n
gstslice.make_pvgrid()
gstslice.add_scalar_field_to_pvgrid(sf_name="lgi", sf_value=gstslice.lgi)
# gstslice.npixels
# gstslice.npixels_values
# gstslice.get_largest_gids()
# gstslice.get_smallest_gids()
gstslice.find_grain_voxel_locs()
gstslice.find_spatial_bounds_of_grains()
gstslice.set_grain_positions()
# gstslice.plot_gs_pvvox(cs_labels='user', _xname_='Z', _yname_='Y', _zname_='X')
gids = gstslice.gpos['face']['bottom']
gstslice.plot_grains(gids, scalar='lgi',
                     cmap='viridis',
                     style='surface', show_edges=True, lw=1.0,
                     opacity=1, view=None,
                     scalar_bar_args=None,
                     axis_labels = ['001', '010', '100'],
                     throw=False)

gids = np.array(list(gids))
## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## --
## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## --
## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## --
## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## --
## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## --
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
## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## --
## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## --
## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## --
## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## --
## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## --
cell_qual = subgrid.compute_cell_quality()['CellQuality']
subgrid.plot(scalars=cell_qual, stitle='Quality', cmap='bwr', clim=[0, 1],
             flip_scalars=True, show_edges=True)






tm = tetmeshes[10]
# Create object from vertex and face arrays
meshfix = pymeshfix.MeshFix(tm.v, tm.f)
# Plot input
meshfix.plot()
# Repair input mesh
meshfix.repair()
# Access the repaired mesh with vtk
mesh = meshfix.mesh
# Or, access the resulting arrays directly from the object
meshfix.v # numpy np.float64 array
meshfix.f # numpy np.int32 array
# View the repaired mesh (requires vtkInterface)
meshfix.plot()
# Save the mesh
meshfix.write('out.ply')


###################################################################
filename_base = r'C:\Development\M2MatMod\upxo_packaged\upxo_private\src\upxo\scripts\MCGS3d_conformal_meshing\objs'
import pygalmesh
import numpy as np

x_ = np.linspace(-1.0, 1.0, 50)
y_ = np.linspace(-1.0, 1.0, 50)
z_ = np.linspace(-1.0, 1.0, 50)
x, y, z = np.meshgrid(x_, y_, z_)

vol = np.empty((50, 50, 50), dtype=np.uint8)
idx = x**2 + y**2 + z**2 < 0.5**2
vol[idx] = 1
vol[~idx] = 0

voxel_size = (0.1, 0.1, 0.1)

mesh = pygalmesh.generate_from_array(
    vol, voxel_size, max_facet_distance=0.2, max_cell_circumradius=0.1
)
mesh.write(filename_base+r"\ball.vtk")

# ################################################################
# ################################################################
# ################################################################
# ################################################################

gstslice.gpos['internal']

gid = gstslice.get_max_presence_gids()[0]
p = pv.Plotter()
lgisubset = pv.UniformGrid()
lgi = gstslice.find_exbounding_cube_gid(gid)
bounded = gstslice.find_exbounding_cube_gid(gid)
bounded_ug = pv.UniformGrid()

locs = np.argwhere(bounded == gid)
mean_gid_loc = locs.mean(axis=0)

# offset = gstslice.grain_locs[gid].mean(axis=0)
lgisubset.dimensions = np.array(lgi.shape) + 1
lgisubset.origin, lgisubset.spacing = (0, 0, 0), (1, 1, 1)
lgisubset.cell_data['lgi'] = lgi.flatten(order="F")

bounded_ug.dimensions = np.array(bounded.shape) + 1
bounded_ug.origin, bounded_ug.spacing = (0, 0, 0), (1, 1, 1)
bounded_ug.cell_data['lgi'] = bounded.flatten(order="F")

p.add_mesh(bounded_ug.threshold([gid, gid]), show_edges=True, opacity=0.5, cmap='nipy_spectral', clim=[gid-1, gid+1])
# p.add_mesh(bounded_ug, style="points", point_size=10, render_points_as_spheres=True)

b = np.array(find_boundaries(gstslice.make_zero_non_gids_in_lgisubset(lgi, [gid]),
                             connectivity=1, mode='subpixel',
                             background=0), dtype=int)
b[b > 0] = 1
gblocs = np.argwhere(b>0)/2
offset = np.array([gstslice.spboundex['xmins'][gid-1],
                   gstslice.spboundex['ymins'][gid-1],
                   gstslice.spboundex['zmins'][gid-1]])

# gbpoints = pv.PolyData(gblocs-0*mean_gid_loc+np.array([0.5, 0.5, 0.5])-np.ones(3))
gbpoints = pv.PolyData(gblocs + 0.5)
p.add_mesh(gbpoints, point_size=12)
_ = p.add_axes(
    line_width=5,
    cone_radius=0.6,
    shaft_length=0.7,
    tip_length=0.3,
    ambient=0.5,
    label_size=(0.4, 0.16),
)
p.show()

# ######################################################################
# ######################################################################
# ######################################################################
# ######################################################################
# ######################################################################
# ######################################################################
# ######################################################################
# ######################################################################
gid = list(gstslice.gpos['internal'])[3]
lgiss = gstslice.find_exbounding_cube_gid(gid)
locs = np.argwhere(lgiss == gid)
mean_gid_loc = locs.mean(axis=0)
gbp = np.array(find_boundaries(gstslice.make_zero_non_gids_in_lgisubset(lgiss, [gid]),
                               connectivity=1, mode='subpixel',
                               background=0), dtype=int)
gbp[gbp > 0] = 1
gblocs = np.argwhere(gbp > 0)/2



pvp = pv.Plotter()
gbpoints = pv.PolyData(gblocs + 0.5)
pvp.add_mesh(gbpoints, point_size=12)
_ = pvp.add_axes(line_width=5, cone_radius=0.6, shaft_length=0.7,
                 tip_length=0.3, ambient=0.5, label_size=(0.4, 0.16),)

bounded_ug = pv.UniformGrid()
bounded_ug.dimensions = np.array(lgiss.shape) + 1
bounded_ug.origin, bounded_ug.spacing = (0, 0, 0), (1, 1, 1)
bounded_ug.cell_data['lgi'] = lgiss.flatten(order="F")
pvp.add_mesh(bounded_ug.threshold([gid, gid]), show_edges=True, opacity=0.5, cmap='nipy_spectral', clim=[gid-1, gid+1])

pvp.show()
# ######################################################################
# ######################################################################
# ######################################################################
# ######################################################################
# ######################################################################
# ######################################################################
# ######################################################################
# ######################################################################




    def set_gbpoints_global_point_cloud(self, points=None):
        self.pointclouds_pv['gbp_global'] = pv.PolyData(points)

    def plot_gbpoint_cloud_global(self):
        """
        plot all the grain boundary points clouds.
        """
        self.pointclouds_pv['gbp_global'].plot(eye_dome_lighting=True)
# -------------------------------------------------
## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## --
## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## --
## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## --
## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## --
## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## --
gstslice.find_bounding_cube_gid(1)
gstslice.find_exbounding_cube_gid(1)
gstslice.set_bounding_cube_all()


for gid, _ in enumerate(range(gstslice.n), start=1):
    print(f'Finding boundaries for grain: {gid}')
    if gid == 1:
        b = np.array(find_boundaries(gstslice.make_zero_non_gids_in_lgi([gid]),
                                     connectivity=1, mode='subpixel',
                                     background=0), dtype=int)
    else:
        b += np.array(find_boundaries(gstslice.make_zero_non_gids_in_lgi([gid]),
                                     connectivity=1, mode='subpixel',
                                     background=0), dtype=int)

b[b>0] = 1
plt.imshow(b[2])
gblocs = np.argwhere(b>0)/2

gstslice.plot_gs_pvvox(alpha=1, points=gblocs, render_points_as_spheres=True, point_size=200)

gstslice.set_gbpoints_global_point_cloud(points=gblocs)
gstslice.plot_gbpoint_cloud_global()


import pyvista as pv
p = pv.Plotter()
p.add_mesh(gstslice.pvgrid, show_edges=True, opacity=0.8)
p.add_mesh(pv.PolyData(gblocs+0.5))
p.show()
###############################################################################
xloc = 4
gblocs_slice = gblocs[np.argwhere(gblocs[:, 2] == xloc).T.squeeze()]
gstslice.plot_scalar_field_slice(sf_name='lgi', slice_normal='x',
                                            slice_location=xloc, interpolation='nearest',
                                            vmin=1, vmax=gstslice.n)
plt.plot(gblocs_slice[:, 1], gblocs_slice[:, 0], 'ko')





xloc = 4
gstslice.plot_scalar_field_slice(sf_name='lgi', slice_normal='x',
                                            slice_location=xloc, interpolation='nearest',
                                            vmin=1, vmax=gstslice.n)
gblocs_slice = gblocs[np.argwhere(gblocs[:, 2] == xloc+0.5).T.squeeze()]
plt.plot(gblocs_slice[:, 1], gblocs_slice[:, 0], 'ko')

gstslice.plot_scalar_field_slice(sf_name='lgi', slice_normal='x',
                                            slice_location=xloc+1, interpolation='nearest',
                                            vmin=1, vmax=gstslice.n)
plt.plot(gblocs_slice[:, 1], gblocs_slice[:, 0], 'ko')
###############################################################################

plot_tslices_field_data(b[0::2], slice_incr=1, nrows=2, ncols=5)
plot_gs_vox(b[0::2], [0, lgi.shape[2]], [0, lgi.shape[0]], [0, lgi.shape[1]], vmin=0, vmax=1, alpha=1.0)

gblocs = np.argwhere(b>0)/2
vdpoints = vd.Points(gblocs)
p = vdpoints.points()
vdpoints.cmap('viridis', vdpoints.points(), name="mydata")
vd.show(vdpoints)
vd.close

zloc = 2
gblocs_slice = gblocs[np.argwhere(gblocs[:, 2] == zloc).T.squeeze()]
plt.imshow(gstslice.lgi[:, :, zloc])
plt.plot(gblocs_slice[:, 1], gblocs_slice[:, 0], 'ko')
## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## --
## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## --
## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## --
## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## --
## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## --
