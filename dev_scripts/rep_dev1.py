import numpy as np
import matplotlib.pyplot as plt
from mcgs import monte_carlo_grain_structure as mcgs
# ===================================================
make_new_gs = 1
if make_new_gs:
    PXGS = mcgs()
    PXGS.simulate()
    tslice = 18
    PXGS.detect_grains(tslice)
    PXGS.gs[tslice].char_morph_2d()
    PXGS.gs[tslice].neigh()

PXGS.gs[tslice].vtgs2d(visualize=True)

plt.figure()
plt.imshow(PXGS.gs[tslice].s)
print(PXGS.gs[tslice].n, PXGS.gs[tslice].vtgs.L0.xtals_n)
# ==============================================================
elshape = 'tri'
elorder = 1
algorithm = 6

from pxtalmesh_01 import pxtalmesh
pxtal_mesh = pxtalmesh(meshing_tool = 'pygmsh',
                       pxtal = PXGS.gs[tslice].vtgs,
                       level = 0,
                       elshape = elshape,
                       elorder = elorder,
                       algorithm = algorithm,
                       elsize_global = [2., 2.5, 2.5],
                       optimize = True,
                       sta = True,
                       wtfs = True,
                       ff = ['vtk', 'inp'],
                       throw = False
                       )

PXGS.gs[tslice].vtgs.write_abapy_input_coords(identification_point_type = 'L0_xtals_centroids')


filename = 'femesh'
fileformat = 'vtk'

mesh_quality_measures = ['aspect_ratio',
                         'skew',
                         'min_angle',
                         'area',
                         ]

INTERACTIVE_MESH_QUALITY_ASSESSMENT = True

import pyvista as pv
# make filename for storing meshdatain vtk format
mesh_filename = f'{filename}.{fileformat}'
# access gmsh mesh data variable mesh
mesh = pxtal_mesh.mesh[3]
# Write the vtk mesh file
mesh.write(mesh_filename)


# Load the vtk mesh file
grid = pv.read(mesh_filename)
#--------------------------
# Visualize using pyvista
pxtal_mesh.vis_pyvista(data_to_vis = 'mesh > mesh > all',
                       rfia = True,
                       grid = grid
                       )
# Extract the t and q elements
tel, qel = pxtal_mesh.get_pygmsh_qt_elements(mesh = mesh)
# Get the number of t and q elements
tel_n, qel_n, allel_n = pxtal_mesh.get_pygmsh_tq_n(method = 'from_el_list',
                                                   tel = tel,
                                                   qel = qel
                                                   )
print(f'Average number of elements per xtal: {int(round(allel_n/PXGS.gs[tslice].vtgs.L0.xtals_n, -1))}')
#--------------------------
# Calculate mesh quality
mqm_data, mqm_dataframe = pxtal_mesh.assess_pygmsh(grid = grid,
                                                   mesh_quality_measures = mesh_quality_measures,
                                                   elshape = elshape,
                                                   elorder = elorder,
                                                   algorithm = algorithm
                                                   )
#--------------------------
# Visualize the mesh quality field parameter
mesh_quality_measures = ['aspect_ratio',
                         'skew',
                         'min_angle',
                         'area']
clims = [[1.0, 2.5],
         [-1.0, 1.0],
         [0.0, 90.0],
         [0.0*mqm_dataframe['area'].max(), mqm_dataframe['area'].max()]
         ]
pxtal_mesh.vis_pyvista(data_to_vis = 'mesh > quality > field',
                       mesh_qual_fields = mqm_data,
                       mesh_qual_field_vis_par = {'mesh_quality_measures': mesh_quality_measures,
                                                  'cpos': 'xy',
                                                  'scalars': 'CellQuality',
                                                  'show_edges': False,
                                                  'cmap': 'viridis',
                                                  'clims': clims,
                                                  'below_color': 'white',
                                                  'above_color': 'black',
                                                  }
                       )


# KDE- plot of data
band_widths = 4*[0.25]
colors = ['red', 'green', 'blue', 'gray']
pxtal_mesh.vis_kde(data = mqm_dataframe,
                   datatype = 'pandas_df',
                   df_names = mesh_quality_measures,
                   clips = clims,
                   cumulative = False,
                   band_widths = band_widths,
                   colors = colors
                   )
# ==============================================================
PXGS.gs[tslice].areas
PXGS.gs[tslice].perimeters
PXGS.gs[tslice].ratio_p_a
# ==============================================================
for grain in PXGS.gs[tslice]:
    print(grain.centroid)
# ==============================================================
PXGS.gs[tslice].single_pixel_grains
PXGS.gs[tslice].plot_grains_gids(PXGS.gs[tslice].single_pixel_grains)
# ==============================================================
PXGS.gs[tslice].straight_line_grains
PXGS.gs[tslice].plot_grains_gids(PXGS.gs[tslice].straight_line_grains[0])
# ==============================================================
mp = PXGS.gs[tslice].make_mulpoint2d_grain_centroids(visualize=True, overlay_on_mcgs=True)
# ==============================================================
PXGS.gs[tslice].AF_bgrains_igrains
# ==============================================================
# ==============================================================
import seaborn as sns

dir(PXGS.gs[tslice])


fig, ax = plt.subplots()
sns.histplot(PXGS.gs[tslice].vtgs.L0.xtal_ape_dstr.data,
             binwidth=5,
             kde=True,
             stat='probability', cumulative=False,legend=True)
sns.histplot(PXGS.gs[tslice].areas, binwidth = 5, kde=True,
             stat='probability', cumulative=False, legend=True)
ax.set_xlim(0, 200)
ax.set_ylim(0, 0.3)
ax.set_xlabel('Area, $\mu m^2$')
ax.set_xticks(range(0,201, 25))
ax.legend(labels=['MCGS2d', 'MCGS2d$_{centroids}$ --> VTGS2d'])
# ==============================================================


PXGS.gs[tslice].vtgs.L0.n_vert
PXGS.gs[tslice].vtgs.L0.xtals_ids

PXGS.gs[tslice].vtgs.L0.xtal_ape_val
PXGS.gs[tslice].vtgs.L0.xtal_ble_val
