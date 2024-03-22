import matplotlib.pyplot as plt
from itertools import product
import numpy as np
import pyvista as pv
from skimage.measure import label
from math import floor
# ====================================================================

nx, ny, nz = 25, 25, 30  # Small grid dimensions for demonstration
S = 8 # Number of states
iterations = 20  # Number of iterations

# ====================================================================
TFIELD = np.zeros((nx, ny, nz))

tspecs_x = {0: [[0, 2], [0.0001, 0.0001] ],
		    1: [[3, 20], [0.001, 1.5] ],
            2: [[21, nx], [1.5, 0.00001] ], }
for __ in tspecs_x.keys():
    loc_bounds = tspecs_x[__][0]
    temp_bounds = tspecs_x[__][1]
    locations = np.arange(loc_bounds[0], loc_bounds[1])
    temp_1d = np.interp(locations, loc_bounds, temp_bounds)
    for loc, temp in zip(locations, temp_1d):
        TFIELD[:, loc, :] += temp

tspecs_y = {0: [[0, 5], [0.0001, 0.0001] ],
		    1: [[6, 12], [0.001, 5] ],
            2: [[13, ny], [0.5, 0.00001] ], }
for __ in tspecs_y.keys():
    loc_bounds = tspecs_y[__][0]
    temp_bounds = tspecs_y[__][1]
    locations = np.arange(loc_bounds[0], loc_bounds[1])
    temp_1d = np.interp(locations, loc_bounds, temp_bounds)
    for loc, temp in zip(locations, temp_1d):
        TFIELD[:, loc, :] += temp

tspecs_z = {0: [[0, 10], [0.0001, 0.0001] ],
		    1: [[11, 18], [0.001, 1] ],
            2: [[19, nz], [1, 0.00001] ], }
for __ in tspecs_z.keys():
    loc_bounds = tspecs_z[__][0]
    temp_bounds = tspecs_z[__][1]
    locations = np.arange(loc_bounds[0], loc_bounds[1])
    temp_1d = np.interp(locations, loc_bounds, temp_bounds)
    for loc, temp in zip(locations, temp_1d):
        TFIELD[:, :, loc] += temp


x = np.arange(nx + 1)
y = np.arange(ny + 1)
z = np.arange(nz + 1)
grid = pv.StructuredGrid(*np.meshgrid(x, y, z, indexing='ij'))
scalar_data = TFIELD.flatten(order="F")[:]  # Flatten and adjust size if necessary
grid.cell_data["TFIELD"] = scalar_data  # Use 'cell_data' instead of 'cell_arrays'

plotter = pv.Plotter()
plotter.add_mesh(grid, cmap="cividis", show_edges=True)
plotter.show_axes()
plotter.show()
# ====================================================================

# ====================================================================

# Initialize the state matrix with random states
state_matrix = np.random.randint(1, S + 1, size=(nx, ny, nz))
hamiltonians = np.zeros(iterations)  # Array to store Hamiltonian values for all iterations
STRUCTURE = list(product([-1, 0, 1], repeat=3))

wm_map = np.array([[[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]],
               [[10, 11, 12],
                [13, 14, 15],
                [16, 17, 18]],
               [[19, 20, 21],
                [22, 23, 24],
                [25, 26, 27]]])

wm = np.array([[[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]],
               [[10, 11, 12],
                [13, 14, 15],
                [16, 17, 18]],
               [[19, 20, 21],
                [22, 23, 24],
                [25, 26, 27]]])

# Plate like grains normal to x
wm = np.ones((3,3,3))
wm[0,:,:] = -1
wm[1,:,:] = 5
wm[2,:,:] = -1

# Plate like grains normal to y
wm = np.ones((3,3,3))
wm[:,0,:] = -1
wm[:,1,:] = 5
wm[:,2,:] = -1

# Plate like grains normal to z
wm = np.ones((3,3,3))
wm[:,:,0] = -1
wm[:,:,1] = 5
wm[:,1,:] = 0
wm[:,:,2] = -1

# wm = np.random.random((3,3,3))
wm = np.ones((3,3,3))



# Mapping the numerical structure to string descriptors
def map_structure_to_descriptors(structure, wm_map):
    w = wm_map
    # Mapping for individual components
    # map_dict = {-1: 'b', 0: '', 1: 'f'}  # Back and Front for x-axis
    # map_dict_y = {-1: 'l', 0: '', 1: 'r'}  # Left and Right for y-axis (inverting to match conventional usage)
    # map_dict_z = {-1: 'd', 0: '', 1: 't'}  # Down and Top for z-axis

    map_dict_x = {-1: 'xleft_',
                  +0: 'xcent_',
                  +1: 'xright_'}  # x-axis

    map_dict_y = {-1: 'yback_',
                  +0: 'ycent_',
                  +1: 'yfront_'}  # y-axis

    map_dict_z = {-1: 'zbot_',
                  +0: 'zcent_',
                  +1: 'ztop_'}  # z-axis

    # Mapping each tuple to its descriptor
    descriptors = [
        map_dict_x[dx] + map_dict_y[dy] + map_dict_z[dz]
        for dx, dy, dz in structure]
    return {d: [s, tuple(_+1 for _ in s) ] for d, s in zip(descriptors, structure)}
# Original STRUCTURE array from itertools.product
structure = list(product([-1, 0, 1], repeat=3))
# Get the corresponding array of string descriptors
descriptors = map_structure_to_descriptors(structure, wm_map)
descriptors

for k, v in descriptors.items():
    ind = descriptors[k][1]
    descriptors[k].append(wm_map[ind[0], ind[1], ind[2]])

# ====================================================================
'''
plt.figure()
plt.imshow(state_matrix[1, :, :])
'''
# ====================================================================


# Main simulation loop
for iteration in range(iterations):
    HAM = np.zeros((nx, ny, nz))
    print(f'Iteration: {iteration}')
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                temperature = TFIELD[x, y, z]
                # --------------------------------
                # Calculate energy at site
                E1, E2 = [], []
                current_state = state_matrix[x, y, z]
                for dx, dy, dz in STRUCTURE:
                    x2, y2, z2 = (x + dx) % nx, (y + dy) % ny, (z + dz) % nz
                    E1.append(wm[dx+1, dy+1, dz+1]*(current_state == state_matrix[x2, y2, z2]))
                # --------------------------------
                # Flip the state
                new_state = np.random.randint(1, S + 1)
                while new_state == current_state:
                    new_state = np.random.randint(1, S + 1)
                # --------------------------------
                # Calculate new energy at site
                for dx, dy, dz in STRUCTURE:
                    x2, y2, z2 = (x + dx) % nx, (y + dy) % ny, (z + dz) % nz
                    E2.append(wm[dx+1, dy+1, dz+1]*(new_state == state_matrix[x2, y2, z2]))
                # --------------------------------
                # Compare the energies
                E1_total, E2_total = 27-sum(E1), 27-sum(E2)
                delE = E2_total - E1_total
                # --------------------------------
                # Accept or reject the new state
                if delE < 0 or np.random.rand() < np.exp(-delE / temperature):
                    state_matrix[x, y, z] = new_state
                    HAM[x, y, z] = E2_total
                else:
                    HAM[x, y, z] = E1_total
    hamiltonians[iteration] = HAM.sum()


# Assuming `state_matrix` is your 3D numpy array representing the grain structure
# Label connected components (grains)
labeled_grains = label(state_matrix)

grain_ids = np.unique(labeled_grains)
vol = np.zeros(grain_ids.size)
for gid in grain_ids:
    gloc = np.argwhere(labeled_grains==gid)
    vol[gid-1] = gloc.shape[0]

gid_largest = np.argwhere(vol==vol.max())[0][0]



'''
for i in range(nx):
    plt.figure()
    plt.imshow(state_matrix[i, :, :])


for i in range(ny):
    plt.figure()
    plt.imshow(state_matrix[:, i, :])

for i in range(nz):
    plt.figure()
    plt.imshow(state_matrix[:, :, i])


for i in range(nz):
    plt.figure()
    plt.imshow(labeled_grains[:, :, i]==gid_largest+1)
'''



# Create the spatial reference
x = np.arange(nx + 1)
y = np.arange(ny + 1)
z = np.arange(nz + 1)
grid = pv.StructuredGrid(*np.meshgrid(x, y, z, indexing='ij'))

# Adjust the scalar data to match the number of cells
scalar_data = state_matrix.flatten(order="F")[:]  # Flatten and adjust size if necessary
# Assign the scalar values to the grid
grid.cell_data["mcstates"] = scalar_data  # Use 'cell_data' instead of 'cell_arrays'

'''
# Adjust the scalar data to match the number of cells
scalar_data = labeled_grains.flatten(order="F")[:]  # Flatten and adjust size if necessary
# Assign the scalar values to the grid
grid.cell_data["gid"] = scalar_data  # Use 'cell_data' instead of 'cell_arrays'
'''

# Make sure to tell PyVista what the active scalars are for volume rendering
grid.set_active_scalars("mcstates")




'''
# Visualization
plotter = pv.Plotter()
plotter.add_volume(grid, cmap="hot", scalar_bar_args={'title': "Grain Structure"})
plotter.show_axes()
plotter.show_bounds(grid="front", location="outer", all_edges=True)
plotter.show()

# Plot the isosurface
plotter = pv.Plotter()
point_data = grid.cell_data_to_point_data()
isosurfaces = point_data.contour(isosurfaces=[3]).smooth(n_iter=10000)
plotter.add_mesh(isosurfaces, color="cyan", opacity=1.0)
plotter.show()
'''




# Plot the isosurface
plotter = pv.Plotter()
plotter.add_mesh(grid, cmap="cividis", show_edges=True)
plotter.show()

