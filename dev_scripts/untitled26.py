import numpy as np
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
import pyvista as pv  # For advanced 3D visualization

# Step 1: Generate 10 random points within a cuboid
x_min, x_max = 0, 1
y_min, y_max = 0, 1
z_min, z_max = 0, 1
points = np.random.rand(20, 3)  # Generate random points
points[:, 0] = points[:, 0] * (x_max - x_min) + x_min  # Scale x coordinates
points[:, 1] = points[:, 1] * (y_max - y_min) + y_min  # Scale y coordinates
points[:, 2] = points[:, 2] * (z_max - z_min) + z_min  # Scale z coordinates


# Assume `points` is your array of points for Voronoi tessellation
vor = Voronoi(points)

# Step 1: Identify the region for a specific point of interest
point_index = 2  # For example, the first point in your array
region_index = vor.point_region[point_index]
# Step 2: Get vertices of the region
region_vertices_indices = vor.regions[region_index]

if -1 in region_vertices_indices:  # Check for unbounded region
    print("The region is unbounded and may not have a well-defined face.")
else:
    # Step 3: Identify faces (ridges) belonging to the region
    faces = []
    for ridge_vertices, (p1, p2) in zip(vor.ridge_vertices, vor.ridge_points):
        if point_index in (p1, p2) and all(v in region_vertices_indices or v == -1 for v in ridge_vertices):
            # Step 4: Construct the face from ridge vertices
            face = [vor.vertices[v] for v in ridge_vertices if v != -1]  # Exclude point at infinity
            faces.append(face)

unique_vertices, indices = np.unique(np.vstack(faces), axis=0, return_inverse=True)

# Step 2: Create the PyVista faces (number of points followed by the indices of the vertices)
pyvista_faces = []
index_offset = 0
for face in faces:
    face_size = len(face)
    face_indices = indices[index_offset:index_offset + face_size]
    pyvista_faces.extend([face_size, *face_indices])
    index_offset += face_size

# Step 3: Create a PyVista mesh (PolyData) using the vertices and faces
surface_mesh = pv.PolyData(unique_vertices, pyvista_faces)

solid_volume = surface_mesh.delaunay_3d(alpha=1)
# Optionally, extract the outer surface of the solid to visualize or further process
outer_surface = solid_volume.extract_geometry()
outer_surface = outer_surface.extract_surface()

# Visualize the solid volume
plotter = pv.Plotter()
plotter.add_mesh(outer_surface, show_edges=True, color='lightblue', opacity=1)
plotter.show()

tri_mesh = surface_mesh.triangulate()


all_triangles = tri_mesh.is_all_triangles()
feature_edges = tri_mesh.extract_feature_edges(boundary_edges=True,
                                               non_manifold_edges=True,
                                               feature_edges=False,
                                               manifold_edges=False)
is_closed = all_triangles and feature_edges.n_points == 0
print(f"The triangulated surface mesh is {'closed' if is_closed else 'not closed'}.")


tri_mesh.save("tri_mesh.stl")
surface_mesh.save("surface_mesh.stl")
solid_volume.save("solid_volume.vtk")


import gmsh
gmsh = gmsh
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)  # Enable message printing
# Load the STL file
path_to_stl = "part1.stl"
gmsh.merge(path_to_stl)
# Get the entities in the mesh
surfaces = gmsh.model.getEntities(dim=2)
# Create a surface loop from all surfaces
loop = gmsh.model.geo.addSurfaceLoop([s[1] for s in surfaces])
# Define a volume based on the surface loop
vol = gmsh.model.geo.addVolume([loop])
# Define a physical group for the volume
#gmsh.model.addPhysicalGroup(3, [vol], tag=1)
#gmsh.model.setPhysicalName(3, 1, "VolumeMesh")
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.01)

gmsh.model.geo.synchronize()
gmsh.option.setNumber("Mesh.Algorithm3D", 5)
gmsh.model.mesh.generate(3)
gmsh.model.mesh.optimize("Laplace2D")
gmsh.model.mesh.optimize("Netgen")
gmsh.write("optimized_tet_mesh.vtk")
gmsh.finalize()
mesh = pv.read("optimized_tet_mesh.vtk")

# Visualize the tetrahedral mesh
plotter = pv.Plotter()
plotter.add_mesh(mesh, show_edges=True, opacity=0.5)
plotter.show()


















import gmsh
import numpy as np

# Initialize Gmsh
gmsh.initialize()
gmsh.model.add("polyhedra")

# Your list of faces with points
faces = [
    # Your faces and points go here
]

# Step 1: Flatten the list and find unique points
all_points = np.vstack([np.array(face) for face in faces])
unique_points, indices = np.unique(all_points, axis=0, return_inverse=True)

# Step 2: Create Gmsh points
point_tags = []
for point in unique_points:
    tag = gmsh.model.geo.addPoint(*point, meshSize=1.0)
    point_tags.append(tag)

# Step 3 & 4: Define lines and create plane surfaces for each face
surface_tags = []
for face in faces:
    line_tags = []
    for i in range(len(face)):
        start_point = indices[np.where((unique_points == face[i]).all(axis=1))[0][0]]
        end_point = indices[np.where((unique_points == face[(i + 1) % len(face)]).all(axis=1))[0][0]]
        line_tag = gmsh.model.geo.addLine(point_tags[start_point], point_tags[end_point])
        line_tags.append(line_tag)
    loop_tag = gmsh.model.geo.addCurveLoop(line_tags)
    surface_tag = gmsh.model.geo.addPlaneSurface([loop_tag])
    surface_tags.append(surface_tag)

# Step 5: Create a surface loop from all the plane surfaces
surf_loop = gmsh.model.geo.addSurfaceLoop(surface_tags)

# Step 6: Create the volume
vol = gmsh.model.geo.addVolume([surf_loop])

gmsh.model.geo.synchronize()

# Optional: Mesh the volume and save the mesh
gmsh.model.mesh.generate(3)
gmsh.write("polyhedra.msh")

gmsh.finalize()
