"""
Verion V1.1
-----------
Change:
    As against V1.0, V1.1 incorporates Michael Atkinson's suggestion
    for exectuion speed up using jit.

Script to generate 3D/2D grain structure and extract subset. 24-10-2024
    NOTE 1: Setting any one dimension length (i.e. nx, ny, nz) to 1
            provides you with a 2D grain structure.
    NOTE 2: The current 3D grain structure generation in core UPXO is not
            optimal. The code V1 provided herein, is a highly optimized beta
            version which will be integrated into the core UPXO.

You would need to provide the following inputs:
    * Domain size:    nx, ny, nz = 500, 100, 25
    * Number of unique states: S=5. Increasing this will provide smaller
      grains, which requires more number of iterations to converge to the same
      grain size.
    * iterations = 10.   Increasing "iterations" will take longer runs but will
      result in longer computation times and larger grain sizes.

Computational time for the above input values: ~ 10 Minutes with these specs:
    AMD Ryzen 7 5700U with Radeon Graphics
    1.80 GHz
    64-bit operating system, x64-based processor
    Python 3.9.13

Purpose:
    To test whether we are able to generate the grain structure on a Linux
    system with its native python environment.

Author: vaasu
"""
from itertools import product
import numpy as np
import pyvista as pv
from numba import njit
from skimage.measure import label

STRUCTURE = list(product([-1, 0, 1], repeat=3))
@njit
def monte_carlo_iterations(STRUCTURE, state_matrix, nx, ny, nz, S, iterations):
    for iteration in range(iterations):
        print(f'Iteration: {iteration}')
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    E1, E2 = [], []
                    current_state = state_matrix[x, y, z]
                    for dx, dy, dz in STRUCTURE:
                        x2, y2, z2 = (x + dx) % nx, (y + dy) % ny, (z + dz) % nz
                        E1.append(current_state == state_matrix[x2, y2, z2])
                    new_state = np.random.randint(1, S + 1)
                    while new_state == current_state:
                        new_state = np.random.randint(1, S + 1)
                    for dx, dy, dz in STRUCTURE:
                        x2, y2, z2 = (x + dx) % nx, (y + dy) % ny, (z + dz) % nz
                        E2.append(new_state == state_matrix[x2, y2, z2])
                    E1_total, E2_total = 27 - sum(E1), 27 - sum(E2)
                    if E2_total - E1_total < 0:
                        state_matrix[x, y, z] = new_state
    return state_matrix

# Parameters
nx, ny, nz = 50, 50, 50
S = 8
iterations = 100

# Initialize the state matrix
state_matrix = np.random.randint(1, S + 1, size=(nx, ny, nz))
state_matrix = monte_carlo_iterations(STRUCTURE, state_matrix, nx, ny, nz, S, iterations)
# ============================================================================
grid = pv.StructuredGrid(*np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), np.arange(nz + 1), indexing='ij'))
grid.cell_data["scalars"] = state_matrix.flatten(order="F")[:]
plotter = pv.Plotter()
plotter.add_mesh(grid, cmap="cividis", show_edges=False)
plotter.show()

labeled_grains = label(state_matrix)
# grain_ids = np.unique(labeled_grains)
grid = pv.StructuredGrid(*np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), np.arange(nz + 1), indexing='ij'))
grid.cell_data["scalars"] = labeled_grains.flatten(order="F")[:]
plotter = pv.Plotter()
plotter.add_mesh(grid, cmap="nipy_spectral", show_edges=False)
plotter.show()

# ============================================================================

x_min, x_max = 1, 50
y_min, y_max = 1, 20
z_min, z_max = 1, 5

subset_grid = grid.extract_subset([x_min, x_max, y_min, y_max, z_min, z_max])
subset_grid_image = subset_grid.cell_data["scalars"]
subset_grid_image = np.array(subset_grid_image.reshape((x_max-x_min,
                                                        y_max-y_min,
                                                        z_max-z_min),
                                                       order="F"))

plotter = pv.Plotter()
plotter.add_mesh(subset_grid, cmap="cividis", show_edges=False)
plotter.show()

# ============================================================================
cube = [0, 0, 0]
rotcube = [45, 0, 0]
goss = [0, 45, 90]
brass = [32.26, 45, 90]
s1 = [59, 29, 63]
s2 = [47, 37, 63]
cu = [90, 35, 45]
t1 = [42, 71, 20]
t2 = [90, 27, 45]

cube_deltj = 5
rotcube_deltj = 5
goss_deltj = 5
brass_deltj = 5
s1_deltj = 5
s2_deltj = 5
cu_deltj = 5
t1_deltj = 5
t2_deltj = 5

cube_flag = 0
rotcube_flag = 0
goss_flag = 0
brass_flag = 0
s1_flag = 0
s2_flag = 0
cu_flag = 0
t1_flag = 0
t2_flag = 0
