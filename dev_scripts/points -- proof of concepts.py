"""
Creating VTK points
"""
import vtk

# Create a vtkPoints object
points = vtk.vtkPoints()

# Insert points into the vtkPoints object
points.InsertNextPoint(1.0, 2.0, 3.0)
points.InsertNextPoint(4.0, 5.0, 6.0)
points.InsertNextPoint(7.0, 8.0, 9.0)

# Create a vtkPolyData object
polyData = vtk.vtkPolyData()
polyData.SetPoints(points)

# Setup actor and mapper
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputData(polyData)

actor = vtk.vtkActor()
actor.SetMapper(mapper)
actor.GetProperty().SetPointSize(5)  # Set point size

# Create a renderer and render window
renderer = vtk.vtkRenderer()
renderWindow = vtk.vtkRenderWindow()
renderWindow.AddRenderer(renderer)

# Create a render window interactor
renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)

# Add the actor to the scene
renderer.AddActor(actor)
renderer.SetBackground(0.1, 0.2, 0.3)  # Background color dark blue

# Render and interact
renderWindow.Render()
renderWindowInteractor.Start()
# --------------------------------------------------------------------------
