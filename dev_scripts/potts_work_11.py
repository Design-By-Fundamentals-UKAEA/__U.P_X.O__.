import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import random

def find_clusters(array):
    visited = np.zeros_like(array)
    clusters = []

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if visited[i, j] == 0:
                cluster = set()
                dfs(array, i, j, visited, cluster)
                if len(cluster) > 0:
                    clusters.append(cluster)

    return clusters

def dfs(array, i, j, visited, cluster):
    stack = [(i, j)]
    target = array[i, j]

    while stack:
        x, y = stack.pop()
        if visited[x, y] == 1:
            continue
        visited[x, y] = 1
        cluster.add((x, y))

        if x - 1 >= 0 and array[x - 1, y] == target and visited[x - 1, y] == 0:
            stack.append((x - 1, y))
        if x + 1 < array.shape[0] and array[x + 1, y] == target and visited[x + 1, y] == 0:
            stack.append((x + 1, y))
        if y - 1 >= 0 and array[x, y - 1] == target and visited[x, y - 1] == 0:
            stack.append((x, y - 1))
        if y + 1 < array.shape[1] and array[x, y + 1] == target and visited[x, y + 1] == 0:
            stack.append((x, y + 1))


def find_cluster_boundaries(clusters):
    boundaries = []

    for cluster in clusters:
        boundary = set()

        for point in cluster:
            x, y = point
            neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]

            for nx, ny in neighbors:
                if (nx, ny) not in cluster:
                    boundary.add((nx, ny))

        sorted_boundary = sort_boundary_points(boundary)
        boundaries.append(sorted_boundary)

    return boundaries

def sort_boundary_points(boundary):
    boundary_array = np.array(list(boundary))
    center = np.mean(boundary_array, axis=0)
    sorted_boundary = sorted(boundary_array, key=lambda point: np.arctan2(point[1] - center[1], point[0] - center[0]))

    return sorted_boundary



def plot_clusters1(clusters):
    plotter = pv.Plotter()
    plotter.background_color = "white"

    for cluster in clusters:
        points = np.array(list(cluster))
        points_3d = np.c_[points, np.zeros(len(points))]

        colors = np.random.rand(len(points))

        cloud = pv.PolyData(points_3d)
        cloud.point_arrays["Colors"] = colors

        plotter.add_mesh(cloud, render_points_as_spheres=True, point_size=20)

    plotter.show()

def plot_clusters2(clusters):
    plotter = pv.Plotter()
    plotter.background_color = "white"

    for cluster in clusters:
        points = np.array(list(cluster))
        points_3d = np.c_[points, np.zeros(len(points))]

        colors = np.random.rand(len(points))
        scalars = colors * (len(points) - 1)

        cloud = pv.PolyData(points_3d)
        cloud.point_arrays["Colors"] = scalars

        plotter.add_mesh(cloud, render_points_as_spheres=True, point_size=10, cmap="hsv")

    plotter.show()


def plot_clusters3(clusters, boundaries):
    # Convert cluster points to PyVista mesh
    meshes = []
    for cluster in clusters:
        points = np.array(list(cluster))
        points_3d = np.hstack((points, np.zeros((points.shape[0], 1))))
        mesh = pv.PolyData(points_3d)
        meshes.append(mesh)

    # Create a PyVista plotter
    plotter = pv.Plotter()

    # Add cluster meshes to the plotter
    for mesh in meshes:
        plotter.add_mesh(mesh, color='blue', show_edges=True, opacity=0.5)

    boundary_points = np.array(list(boundaries[0]))
    # Create a PyVista PolyData object for the boundary points
    boundary_points_3d = np.hstack((boundary_points, np.zeros((boundary_points.shape[0], 1))))
    boundary_mesh = pv.PolyData(boundary_points_3d)

    # Add the boundary mesh to the plotter
    plotter.add_mesh(boundary_mesh, color='red', line_width=3, render_lines_as_tubes=True)

    # Set the background color to white
    plotter.background_color = 'white'

    # Set a larger point size for better visibility
    # plotter.set_point_size(10)

    # Show the plot
    plotter.show()

def plot_clusters4(clusters, boundaries):
    # Convert cluster points to PyVista mesh
    meshes = []
    for cluster in clusters:
        points = np.array(list(cluster))
        points_3d = np.hstack((points, np.zeros((points.shape[0], 1))))
        mesh = pv.PolyData(points_3d)
        meshes.append(mesh)

    # Create a PyVista plotter
    plotter = pv.Plotter()

    # Add cluster meshes to the plotter
    for mesh in meshes:
        plotter.add_mesh(mesh, color='blue', show_edges=True, opacity=0.5)

    boundary_points = np.array(list(boundaries[0]))
    # Create a PyVista PolyData object for the boundary points
    boundary_points_3d = np.hstack((boundary_points, np.zeros((boundary_points.shape[0], 1))))
    boundary_mesh = pv.PolyData(boundary_points_3d)

    # Add the boundary mesh to the plotter
    plotter.add_mesh(boundary_mesh, color='red', line_width=3, render_lines_as_tubes=True)

    # Set the background color to white
    plotter.background_color = 'white'

    # Set a larger point size for better visibility
    # plotter.set_point_size(10)
    for cluster in clusters:
        points = np.array(list(cluster))
        points_3d = np.c_[points, np.zeros(len(points))]
        colors = np.random.rand(len(points))
        scalars = colors * (len(points) - 1)
        cloud = pv.PolyData(points_3d)
        cloud.point_arrays["Colors"] = scalars
        plotter.add_mesh(cloud, render_points_as_spheres=True, point_size=10, cmap="hsv")

    # Show the plot
    plotter.show()




def plot_clusters5(clusters, boundaries):
    import vtk
    # Create a VTK polydata for each cluster
    polydatas = []
    for cluster in clusters:
        points = np.array(list(cluster))

        # Create VTK points
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(vtk.vtkFloatArray())
        for point in points:
            vtk_points.InsertNextPoint(point[0], point[1], 0)

        # Create VTK polygon
        polygon = vtk.vtkPolygon()
        polygon.GetPointIds().SetNumberOfIds(len(points))
        for i, _ in enumerate(points):
            polygon.GetPointIds().SetId(i, i)

        # Create VTK cell array
        cell_array = vtk.vtkCellArray()
        cell_array.InsertNextCell(polygon)

        # Create VTK polydata
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_points)
        polydata.SetPolys(cell_array)

        polydatas.append(polydata)

    # Create a VTK polydata for the boundaries
    boundary_polydatas = []
    for polydata in polydatas:
        boundary_filter = vtk.vtkFeatureEdges()
        boundary_filter.SetInputData(polydata)
        boundary_filter.BoundaryEdgesOn()
        boundary_filter.ManifoldEdgesOff()
        boundary_filter.NonManifoldEdgesOff()
        boundary_filter.FeatureEdgesOff()
        boundary_filter.Update()

        boundary_polydatas.append(boundary_filter.GetOutput())

    # Create VTK renderers and render windows
    renderers = []
    render_windows = []
    for _ in range(len(polydatas) + len(boundary_polydatas)):
        renderer = vtk.vtkRenderer()
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)

        renderers.append(renderer)
        render_windows.append(render_window)

    # Set up renderers for cluster visualization
    for i, polydata in enumerate(polydatas):
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0, 0, 1)  # Set color to blue

        renderers[i].AddActor(actor)

    # Set up renderers for boundary visualization
    for i, boundary_polydata in enumerate(boundary_polydatas):
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(boundary_polydata)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1, 0, 0)  # Set color to red
        actor.GetProperty().SetLineWidth(2)  # Set line width

        renderers[i + len(polydatas)].AddActor(actor)

    # Create VTK render window interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_windows[0])
    interactor.Initialize()

    # Set the background color to white for all render windows
    for render_window in render_windows:
        renderer = render_window.GetRenderers().GetFirstRenderer()
        renderer.SetBackground(1, 1, 1)

    # Render all the windows
    for render_window in render_windows:
        render_window.Render()

    # Start the interaction
    interactor.Start()



def plot_clusters6(clusters, boundaries):
    import vtk
    # Create a VTK polydata for each cluster
    polydatas = []
    for cluster in clusters:
        points = np.array(list(cluster))

        # Create VTK points
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(vtk.vtkFloatArray())
        for point in points:
            vtk_points.InsertNextPoint(point[0], point[1], 0)

        # Create VTK polygon
        polygon = vtk.vtkPolygon()
        polygon.GetPointIds().SetNumberOfIds(len(points))
        for i, _ in enumerate(points):
            polygon.GetPointIds().SetId(i, i)

        # Create VTK cell array
        cell_array = vtk.vtkCellArray()
        cell_array.InsertNextCell(polygon)

        # Create VTK polydata
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_points)
        polydata.SetPolys(cell_array)

        polydatas.append(polydata)

    # Create a VTK polydata for the boundaries
    boundary_polydatas = []
    for polydata in polydatas:
        boundary_filter = vtk.vtkFeatureEdges()
        boundary_filter.SetInputData(polydata)
        boundary_filter.BoundaryEdgesOn()
        boundary_filter.ManifoldEdgesOff()
        boundary_filter.NonManifoldEdgesOff()
        boundary_filter.FeatureEdgesOff()
        boundary_filter.Update()

        boundary_polydatas.append(boundary_filter.GetOutput())

    # Create a VTK renderer and render window
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)

    # Set the background color to white
    renderer.SetBackground(1, 1, 1)

    # Set up renderers for cluster visualization
    for polydata in polydatas:
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0, 0, 1)  # Set color to blue

        renderer.AddActor(actor)

    # Set up renderers for boundary visualization
    for boundary_polydata in boundary_polydatas:
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(boundary_polydata)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1, 0, 0)  # Set color to red
        actor.GetProperty().SetLineWidth(2)  # Set line width

        renderer.AddActor(actor)

    # Create a VTK render window interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.Initialize()

    # Render the window
    render_window.Render()

    # Start the interaction
    interactor.Start()


def plot_clusters7(clusters, boundaries):
    import vtk
    # Create a VTK renderer and render window
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)

    # Set the background color to white
    renderer.SetBackground(1, 1, 1)

    # Set up renderers for cluster visualization
    for cluster in clusters:
        points = np.array(list(cluster))

        # Create VTK points
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(vtk.vtkFloatArray())
        for point in points:
            vtk_points.InsertNextPoint(point[0], point[1], 0)

        # Create VTK polygon
        polygon = vtk.vtkPolygon()
        polygon.GetPointIds().SetNumberOfIds(len(points))
        for i, _ in enumerate(points):
            polygon.GetPointIds().SetId(i, i)

        # Create VTK cell array
        cell_array = vtk.vtkCellArray()
        cell_array.InsertNextCell(polygon)

        # Create VTK polydata
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_points)
        polydata.SetPolys(cell_array)

        # Generate random RGB color
        color = [random.random(), random.random(), random.random()]

        # Create VTK mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)

        renderer.AddActor(actor)

    # Create a VTK render window interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.Initialize()

    # Render the window
    render_window.Render()

    # Start the interaction
    interactor.Start()




def plot_clusters8(clusters, boundaries):
    from vedo import *
    # Create a plotter
    plotter = Plotter(bg='white')

    # Plot each cluster with a random color and set the point size
    for cluster in clusters:
        points = np.array(list(cluster))

        # Create a Points object from the cluster points
        points_obj = Points(points)

        # Set the point size of the Points object
        #points_obj.pointSize(10)

        # Generate a random color
        color = np.random.rand(3)

        # Add the Points object to the plotter with the specified color
        #plotter.add(points_obj, color=color)
        plotter.add(points_obj)

    # Show the plotter window
    plotter.show()















# Given array
array = np.array([[2, 2, 1, 1, 1, 2, 2, 2, 2, 2],
                  [3, 2, 2, 1, 1, 2, 2, 2, 2, 2],
                  [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                  [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                  [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                  [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                  [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                  [2, 2, 3, 2, 2, 2, 2, 2, 2, 2],
                  [2, 2, 2, 2, 1, 2, 3, 3, 2, 2],
                  [2, 2, 3, 3, 1, 2, 3, 3, 2, 2]])

# Find individual clusters
clusters = find_clusters(array)

# Find cluster boundaries and sort them clockwise
boundaries = find_cluster_boundaries(clusters)

# Plot the clusters
plt.figure(figsize=(6, 6))
plt.imshow(array, cmap='jet', origin='lower')

colors = plt.cm.get_cmap('tab10').colors  # Generate colors for clusters
for i, cluster in enumerate(clusters):
    for point in cluster:
        plt.text(point[1], point[0], str(array[point]), color=colors[i], ha='center', va='center')

plt.xticks([])
plt.yticks([])
plt.title('Individual Clusters')
plt.show()




# Plot the boundaries
plt.figure(figsize=(6, 6))
plt.imshow(array, cmap='jet', origin='lower')

for boundary in boundaries:
    for point in boundary:
        x, y = point
        plt.plot(y, x, 'ro', markersize=5)

plt.xticks([])
plt.yticks([])
plt.title('Cluster Boundaries')
plt.show()




# Plot each cluster using PyVista
plot_clusters1(clusters)
plot_clusters2(clusters)

plot_clusters3(clusters, boundaries)
plot_clusters4(clusters, boundaries)
plot_clusters5(clusters, boundaries) # AVOID
plot_clusters6(clusters, boundaries)
plot_clusters7(clusters, boundaries)
