# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 03:47:50 2022

@author: rg5749
"""

import numpy as np
import pygmsh
import pyvista as pv
import meshio

with pygmsh.geo.Geometry() as geom:
# =============================================================================
#     poly = geom.add_polygon([ [0.0, 0.0], [4.0, 0.0], [2.0, 2.0], ], mesh_size = 0.2, )
# =============================================================================
    poly = geom.add_polygon([ [0.0, 0.0],
                              [0.5, -0.5],
                              [1.0, -0.5],
                              [1.5, 0.0],
                              [1.0, 0.5],
                              [0.5, 0.5],
                              ],
                            mesh_size = 0.5, )
    # Define field for edges
    field0 = geom.add_boundary_layer(edges_list = [poly.curves[0], poly.curves[1]],
                                     lcmin = 0.05, lcmax = 0.50, distmin = 0.20, distmax = 0.20, num_points_per_curve = 20)
    # Define field for points
    field3 = geom.add_boundary_layer(nodes_list = [poly.points[0]], lcmin = 0.02, lcmax = 0.40, distmin = 0.10, distmax = 0.20, )
    geom.set_background_mesh([field0], operator="Min")
    #geom.set_recombined_surfaces([poly.surface])
    mesh = geom.generate_mesh()
    #pygmsh.optimize(mesh, method="")
    mesh.write("sunil.vtk")
# =============================================================================
# mesh = meshio.read("sunil.vtk")
# optimized_mesh = pygmsh.optimize(mesh, method="")
# =============================================================================
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
grid = pv.read("sunil.vtk")
pv.global_theme.background='maroon'
plotter = pv.Plotter(window_size=(1400,800))
_ = plotter.add_axes_at_origin(x_color = 'red', y_color = 'green', z_color = 'blue',
                               line_width = 1,
                               xlabel = 'x', ylabel = 'y', zlabel = 'z',
                               labels_off = True)
_ = plotter.add_points(np.array([0,0,0]),
                       render_points_as_spheres = True,
                       point_size = 25)
_ = plotter.add_points(grid.points,
                       render_points_as_spheres = True,
                       point_size = 2)
#_ = plotter.add_bounding_box(line_width=2, color='black')
_ = plotter.add_mesh(grid,
                     show_edges = True,
                     edge_color = 'black',
                     line_width = 1,
                     render_points_as_spheres = True,
                     point_size = 10,
                     style = 'wireframe')
plotter.view_xy()
plotter.camera.zoom(2.0)
plotter.show()
# =============================================================================
# grid.compute_cell_quality()
# =============================================================================
# =============================================================================
# grid.plot(show_scalar_bar=False,
#           show_axes=True,
#           show_edges=True,
#           style='wireframe',
#           opacity=1.0)
# # =============================================================================
# # pv.Plotter.show_bounds(grid='front',
# #                        location='outer',
# #                        all_edges=True,
# #                        )
# # =============================================================================
#=============================================================================
quality_min_angle = grid.compute_cell_quality(quality_measure = 'min_angle')
quality_max_angle = grid.compute_cell_quality(quality_measure = 'max_angle')
quality_min_angle.plot(cpos = 'xy',
                       scalars = 'CellQuality',
                       show_edges = True,
                       cmap = 'nipy_spectral',
                       scalar_bar_args={'title': "Min. Angle (deg.)"},
                       )
quality_max_angle.plot(cpos = 'xy',
                       scalars = 'CellQuality',
                       show_edges = True,
                       cmap = 'nipy_spectral',
                       scalar_bar_args={'title': "Max. Angle (deg.)"},
                       )
#=============================================================================
MinAngle = np.delete(quality_min_angle.cell_data['CellQuality'], quality_min_angle.cell_data['CellQuality'] == -1)
MaxAngle = np.delete(quality_max_angle.cell_data['CellQuality'], quality_min_angle.cell_data['CellQuality'] == -1)

import matplotlib.pyplot as plt
import seaborn as sb

sb.set(style="darkgrid")
fig, axs = plt.subplots(2,2, figsize = (10,10), dpi = 600)

sb.histplot(data = MinAngle,
            color = 'skyblue',
            binwidth = 1.0,
            alpha = 0.75,
            element = 'step',
            cumulative = False,
            kde = False,
            ax = axs[0,0],
            )
axs[0, 0].set_xlim(0,60)
axs[0, 0].set_xlabel("Min. Angle (deg.)", size = 15, ha = 'center')
axs[0, 0].set_ylabel("Count", size = 15, ha = 'center')

sb.histplot(data = MaxAngle,
            color = 'teal',
            binwidth = 1.0,
            alpha = 0.75,
            element = 'step',
            cumulative = False,
            kde = False,
            ax = axs[0,1]
            )
axs[0, 1].set_xlim(60,120)
axs[0, 1].set_xlabel("Max. Angle (deg.)", size = 15, ha = 'center')
axs[0, 1].set_ylabel("Count", size = 15, ha = 'center')
fig.tight_layout()
plt.show()