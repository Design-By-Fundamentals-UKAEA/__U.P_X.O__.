import matplotlib.pyplot as plt
from itertools import product
import numpy as np
import pyvista as pv
from skimage.measure import label
# ====================================================================
nx, ny, nz = 500, 100, 25
S = 5
iterations = 10
# ====================================================================
state_matrix = np.random.randint(1, S + 1, size=(nx, ny, nz))
STRUCTURE = list(product([-1, 0, 1], repeat=3))
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
                E1_total, E2_total = 27-sum(E1), 27-sum(E2)
                if E2_total - E1_total < 0:
                    state_matrix[x, y, z] = new_state
# ====================================================================
labeled_grains = label(state_matrix)
grain_ids = np.unique(labeled_grains)
vol = np.zeros(grain_ids.size)
for gid in grain_ids:
    gloc = np.argwhere(labeled_grains==gid)
    vol[gid-1] = gloc.shape[0]
# ====================================================================
grid = pv.StructuredGrid(*np.meshgrid(np.arange(nx + 1),
                                      np.arange(ny + 1),
                                      np.arange(nz + 1),
                                      indexing='ij'))
grid.cell_data["scalars"] = state_matrix.flatten(order="F")[:]
# ====================================================================
plotter = pv.Plotter()
plotter.add_mesh(grid, cmap="cividis", show_edges=False)
plotter.show()
# ====================================================================
x_min, x_max = 1, 50
y_min, y_max = 1, 20
z_min, z_max = 1, 5
subset_grid = grid.extract_subset([x_min, x_max, y_min, y_max, z_min, z_max])

plotter = pv.Plotter()
plotter.add_mesh(subset_grid, cmap="cividis", show_edges=False)
plotter.show()
# ====================================================================
