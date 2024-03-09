import pyvista as pv
import numpy as np

# Create a sample polyhedron (e.g., a cube)
cube = pv.Cube()

# Define the plane for clipping
# Point on the plane
point = np.array([0.5, 0, 0])
# Normal vector to the plane
normal = np.array([1, 0, 0])

# Perform the intersection (clipping)
clipped_mesh = cube.clip(normal=normal, origin=point)

# Visualize the original and clipped mesh
p = pv.Plotter()
p.add_mesh(cube, style='wireframe', color='blue', label='Original Polyhedron')
#p.add_mesh(clipped_mesh, color='red', label='Clipped/Intersected Polyhedron')
p.add_legend()
p.show()




# Perform the clipping and retain both parts
clipped_mesh, discarded_mesh = cube.clip(normal=normal, origin=point, return_clipped=True)

# Visualize both parts
p = pv.Plotter()
p.add_mesh(clipped_mesh, color='green', label='Clipped Part (Inside)')
p.add_mesh(discarded_mesh, color='red', label='Discarded Part (Outside)')
p.add_legend()
p.show()





import pyvista as pv
import numpy as np

# Create a sample polyhedron (e.g., a cube)
cube = pv.Cube()

# Define the slicing plane
# Point on the plane
point = np.array([0.5, 0, 0])
# Normal vector to the plane
normal = np.array([1, 0, 0])

# Use clipping to separate the polyhedron into two parts
part_1 = cube.clip(normal=normal, origin=point)
part_2 = cube.clip(normal=-normal, origin=point)  # Note the inverted normal

# Visualize the original polyhedron and the slicing plane
p = pv.Plotter()
p.add_mesh(cube, style='wireframe', color='blue', label='Original Polyhedron')
p.add_mesh(part_1, color='green', label='Part 1')
p.add_mesh(part_2, color='red', label='Part 2')
p.add_plane_widget(normal=normal, origin=point, assign_to_axis=None, color='yellow')
p.add_legend()
p.show()


import pyvista as pv
import numpy as np

# Define your polyhedron, for example, a cube
cube = pv.Cube()

# Define the cutting plane
# Point on the plane
point = np.array([0.5, 0, 0])
# Normal vector to the plane
normal = np.array([1, 1, 1])

# Perform clipping to simulate cutting, keeping both sides of the cut
part_1 = cube.clip(normal=normal, origin=point)
part_2 = cube.clip(normal=-normal, origin=point)

# Visualize the parts
p = pv.Plotter()
p.add_mesh(part_1, color='blue', label='Part 1')
#p.add_mesh(part_2, color='red', label='Part 2')
p.add_legend()
p.show()






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

# Step 2: Compute Voronoi Tessellation
x = Voronoi(points)

# Basic 3D Visualization using matplotlib (Optional)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b')  # Plot the points
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('3D Voronoi Tessellation - Points')
plt.show()

# Advanced 3D Visualization using PyVista (Optional)
# Note: This requires understanding of PyVista's mesh and plotter objects
# and is best suited for advanced 3D visualization needs.
import numpy as np
from scipy.spatial import Voronoi

# Assume `points` is your array of points for Voronoi tessellation
vor = Voronoi(points)

# Step 1: Identify the region for a specific point of interest
point_index = 4  # For example, the first point in your array
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

# `faces` now contains the vertices that define each face of the Voronoi region for the chosen point

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Initialize 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot each face's vertices as points
for face in faces:
    for vertex in face:
        ax.scatter(vertex[0], vertex[1], vertex[2], c='r', marker='o')

# Set labels
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title('Scatter Plot of Voronoi Face Points')

# Show plot
plt.show()




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
mesh = pv.PolyData(unique_vertices, pyvista_faces)

# Step 4: Plot the mesh
plotter = pv.Plotter()
plotter.add_mesh(mesh, show_edges=True, color='lightblue')
plotter.show()





# Create the surface mesh using PyVista PolyData
surface_mesh = pv.PolyData(unique_vertices, pyvista_faces)

# Check if the surface mesh is watertight
if surface_mesh.is_all_triangles() and surface_mesh.is_closed():
    # Create a solid volume from the surface mesh
    solid_volume = surface_mesh.delaunay_3d(alpha=1)

    # Optionally, extract the outer surface of the solid to visualize or further process
    outer_surface = solid_volume.extract_geometry()

    # Visualize the solid volume
    plotter = pv.Plotter()
    plotter.add_mesh(outer_surface, show_edges=True, color='lightblue', opacity=0.5)
    plotter.show()
else:
    print("The surface mesh is not closed or not all faces are triangles. Cannot create a solid.")







# Define the cutting plane
# Point on the plane
point = solid_volume.points.T.mean(axis = 1)
# Normal vector to the plane
normal = np.array([1, 1, 1])

# Perform clipping to simulate cutting, keeping both sides of the cut
part_1 = solid_volume.clip(normal=normal, origin=point)
part_2 = solid_volume.clip(normal=-normal, origin=point)

# Visualize the parts
p = pv.Plotter()
p.add_mesh(part_1, color='blue', label='Part 1')
p.add_mesh(part_2, color='red', label='Part 2')
p.add_legend()
p.show()

part_1_surface = part_1.extract_surface()
part_1_surface.save("part1.stl")




import gmsh
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
gmsh.model.geo.synchronize()
# Define a physical group for the volume
gmsh.model.addPhysicalGroup(3, [vol], tag=1)
gmsh.model.setPhysicalName(3, 1, "VolumeMesh")
gmsh.model.geo.synchronize()
gmsh.option.setNumber("Mesh.Algorithm3D", 4)
gmsh.model.mesh.generate(3)
gmsh.model.mesh.optimize("Laplace2D")
gmsh.model.mesh.optimize("Netgen")
gmsh.write("optimized_tet_mesh.vtk")
mesh = pv.read("optimized_tet_mesh.vtk")

# Visualize the tetrahedral mesh
plotter = pv.Plotter()
plotter.add_mesh(mesh, show_edges=True, opacity=0.5)
plotter.show()
