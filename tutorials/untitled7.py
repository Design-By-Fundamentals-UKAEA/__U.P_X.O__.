from math import sqrt
import pygmsh


with pygmsh.occ.Geometry() as geom:
    geom.add_ball([0.0, 0.0, 0.0], 1.0)

    geom.set_mesh_size_callback(
        lambda dim, tag, x, y, z: abs(sqrt(x**2 + y**2 + z**2) - 0.5) + 0.1
    )
    mesh = geom.generate_mesh()
    import meshio
    mesh = meshio.read("sunil.vtk")
    optimized_mesh = pygmsh.optimize(mesh, method="")
