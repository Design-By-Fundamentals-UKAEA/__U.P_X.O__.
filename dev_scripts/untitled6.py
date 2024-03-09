import pygmsh
import numpy as np
import pyvista as pv
with pygmsh.geo.Geometry() as geom:
    poly = geom.add_polygon(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [3.0, 1.0],
            [1.0, 2.0],
            [0.0, 1.0],
        ],
        mesh_size=0.3,
    )

    field0 = geom.add_boundary_layer(
        edges_list=[poly.curves[0]],
        lcmin=0.05,
        lcmax=0.2,
        distmin=0.0,
        distmax=0.2,
    )
    field1 = geom.add_boundary_layer(
        nodes_list=[poly.points[2]],
        lcmin=0.05,
        lcmax=0.2,
        distmin=0.1,
        distmax=0.4,
    )
    geom.set_background_mesh([field0, field1], operator="Min")

    mesh = geom.generate_mesh()
    
    mesh.write("sunil.vtk")

    grid = pv.read("sunil.vtk")
    pv.global_theme.background='maroon'
    plotter = pv.Plotter(window_size = (1400,800))
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
    plotter.camera.zoom(1.5)
    plotter.show()
    
    
    
    
