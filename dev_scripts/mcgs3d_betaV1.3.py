# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 20:24:02 2024

@author: rg5749
"""

from itertools import product
import numpy as np
from numba import njit
import pyvista as pv
from skimage.measure import label
from numba.typed import List as NBList

STRUCTURE = NBList(product([-1, 0, 1], repeat=3))
@njit
def monte_carlo_iterations(STRUCTURE, state_matrix, state_matrices, nx, ny, nz, S, iterations, dataStep):
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
        state_matrices[iteration] = state_matrix.copy()
    return state_matrices

# Parameters
nx, ny, nz = 50, 50, 1
S = 2
iterations = 500
dataStep = 500
# Initialize the state matrix
state_matrix = np.random.randint(1, S + 1, size=(nx, ny, nz))
state_matrices = np.empty((iterations, nx, ny, nz), dtype=np.int32)
state_matrices = monte_carlo_iterations(STRUCTURE, state_matrix, state_matrices, nx, ny, nz, S, iterations, dataStep)

for iteration in range(iterations):
    if iteration % 50 == 0:
        grid = pv.StructuredGrid(*np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), np.arange(nz + 1), indexing='ij'))
        grid.cell_data["scalars"] = state_matrices[iteration].flatten(order="F")[:]
        # -------------------------------------
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(grid, cmap="cividis", show_edges=False, show_scalar_bar=False)
        plotter.screenshot(fr"D:\HSDI\your_visualization{iteration}.png",
                           transparent_background=True, window_size=(1920, 1080))
        plotter.close()
# -------------------------------------


"""
labeled_grains = label(state_matrix)
# grain_ids = np.unique(labeled_grains)
# grain_ids.size
grid = pv.StructuredGrid(*np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), np.arange(nz + 1), indexing='ij'))
grid.cell_data["scalars"] = labeled_grains.flatten(order="F")[:]
plotter = pv.Plotter()
plotter.add_mesh(grid, cmap="nipy_spectral", show_edges=False)
plotter.show()
"""
