import pyvista as pv

xgr, ygr, zgr = np.meshgrid(np.arange(10), np.arange(8), np.arange(6), indexing='ij')

# Create a structured grid from the xgr, ygr, zgr coordinates
grid = pv.StructuredGrid(xgr, ygr, zgr)

random_integer_array = np.random.randint(1, 11, size=(10, 8, 6))

# Add the random_integer_array as a scalar data attribute to the grid
grid['values'] = random_integer_array.flatten(order='F')  # Flatten in Fortran order to match the VTK's indexing

# Save the grid to a VTK file
grid.save('random_integer_grid.vtk')

print("VTK file saved successfully.")
