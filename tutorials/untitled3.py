# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 03:48:40 2022

@author: rg5749
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 17:10:32 2022

@author: rg5749
"""










# GBZ: Grain boundary zone
def form_GBZ_hosters_dict(METHOD, Quantity, NUM_grains, NAMES_GRAINS_dict):
    ID_Available_Grains  = list(NAMES_GRAINS_dict['idNames'].keys())
    num_Available_Grains = len(ID_Available_Grains)
    """
    METHOD: (1) 'byNumVolumeFraction'
            (2) 'byAreaVolumeFraction'
            (2) 'byNumPercentage'
            (3) 'byNumber'
    """
        # If the GBZ_hosters are to be identified from volume fraction
        # here, volume fraction is applied to the number of grains
        # TO-DO: Accurately, this must be done through grain areas and not number of grains
                # See 'byAreaVolumeFraction' branch
    if METHOD=='byNumVolumeFraction':
        num_GBZ_hosters = round(abs(float(Quantity)*NUM_grains))
        GBZ_hosters     = random.sample(ID_Available_Grains, k = num_GBZ_hosters)
    elif METHOD=='byAreaVolumeFraction':
        #TODO: byAreaVolumeFraction
        pass
    elif METHOD=='byNumPercentage':
        num_GBZ_hosters = round(abs(float(Quantity/100)*NUM_grains))
        GBZ_hosters     = random.sample(ID_Available_Grains, k = num_GBZ_hosters)
    elif METHOD=='byNumber':
        if Quantity <= NUM_grains:
            GBZ_hosters = random.sample(ID_Available_Grains, k = Quantity)
    else:
        pass
    
    # Sort the id numbers in the ascending order
    GBZ_hosters.sort()
    
    # Make the dictionary
    GBZ_hosterData = {'GBZ_hoster_ids' : GBZ_hosters,
                      'num_GBZ_hosters': len(GBZ_hosters),
                     }
    return GBZ_hosterData

# Identify the grains which host Grain Boundary Zones (GBZ)
GBZ_hosterData = form_GBZ_hosters_dict(METHOD            = 'byNumPercentage',
                                       Quantity          = 100,
                                       NUM_grains        = NUM_grains,
                                       NAMES_GRAINS_dict = NAMES_GRAINS_dict)
###############################################################################
# Base grain structure - LEVEL 0 ---- COMPLETED
# Grain boundary zone  - LEVEL 1 ---- START
###############################################################################
def build_dict_GS(GBZ_hosterData):
    # Unpack the total number of grains hosting grain boundary zones
    num_GBZ_hosters       = GBZ_hosterData['num_GBZ_hosters']
    num_GBZ_hosters_Range = range(num_GBZ_hosters)
    # Unpack the ID's of grains hosting grain boundary zones
    IDs_GBZ_hosters = GBZ_hosterData['GBZ_hoster_ids']

    GRAIN_STRUCTURE_L1 = {IDs_GBZ_hosters[GBZ_hoster_count]: {'core_id'        : GBZ_hoster_count,
                                                              'SuperHosterID'  : IDs_GBZ_hosters[GBZ_hoster_count],
                                                              'core_V_id'      : [],
                                                              'core_V_coord'   : [],
                                                              'core_C_coord'   : [],
                                                              'core_param'     : {'area': [],
                                                                                  'perimeter':[],
                                                                                  'aspectRatio':[]},
                                                              'gbz_id'         : GBZ_hoster_count,
                                                              'SuperHosterID'  : IDs_GBZ_hosters[GBZ_hoster_count],
                                                              'gbz_deflation'  : [],
                                                              'gbz_V_int_id'   : [],
                                                              'gbz_V_int_coord': [],
                                                              'gbz_V_ext_id'   : [],
                                                              'gbz_V_ext_coord': [],
                                                              'gbz_C_coord'    : [],
                                                              'gbz_param'      : {'area': [],
                                                                                  'perimeter':[],
                                                                                  'aspectRatio':[]},
                                                              } for GBZ_hoster_count in num_GBZ_hosters_Range}
    return GRAIN_STRUCTURE_L1

GRAIN_STRUCTURE_L1 = build_dict_GS(GBZ_hosterData)
###############################################################################
# def populate_dict_GS_CORE():
###############################################################################

###############################################################################
'''
    Below shows extracting info from the above dictionary
    
    # 1. Accessing core_id
        GRAIN_STRUCTURE_L1[GBZ_hoster_count]['core_id']
    # 2. Accessing SuperHposterID
        GRAIN_STRUCTURE_L1[GBZ_hoster_count]['SuperHosterID']
    # 3. Accessing core_V_id
        GRAIN_STRUCTURE_L1[GBZ_hoster_count]['core_V_id']
    # 4. Accessing core_V_coord
        GRAIN_STRUCTURE_L1[GBZ_hoster_count]['core_V_coord']
    # 5. Accessing core_C_coord
        GRAIN_STRUCTURE_L1[GBZ_hoster_count]['core_C_coord']
    # 6. Accessing core_param
        GRAIN_STRUCTURE_L1[GBZ_hoster_count]['core_param']
        # 6a. Accessing core_param: area
            GRAIN_STRUCTURE_L1[GBZ_hoster_count]['core_param']['area']
        # 6b. Accessing core_param: perimeter
            GRAIN_STRUCTURE_L1[GBZ_hoster_count]['core_param']['perimeter']
        # 6c. Accessing core_param: aspectRatio
            GRAIN_STRUCTURE_L1[GBZ_hoster_count]['core_param']['aspectRatio']
    # 7. Accessing gbz_id
        GRAIN_STRUCTURE_L1[GBZ_hoster_count]['gbz_id']
    # 8. Accessing SuperHosterID
        GRAIN_STRUCTURE_L1[GBZ_hoster_count]['SuperHosterID']
    # 9. Accessing gbz_deflation
        GRAIN_STRUCTURE_L1[GBZ_hoster_count]['gbz_deflation']
    # 10. Accessing gbz_V_int_id
        GRAIN_STRUCTURE_L1[GBZ_hoster_count]['gbz_V_int_id']
    # 11. Accessing gbz_V_int_coord
        GRAIN_STRUCTURE_L1[GBZ_hoster_count]['gbz_V_int_coord']
    # 12. Accessing gbz_V_ext_id
        GRAIN_STRUCTURE_L1[GBZ_hoster_count]['gbz_V_ext_id']
    # 13. Accessing gbz_V_ext_coord
        GRAIN_STRUCTURE_L1[GBZ_hoster_count]['gbz_V_ext_coord']
    # 14. Accessing gbz_C_coord
        GRAIN_STRUCTURE_L1[GBZ_hoster_count]['gbz_C_coord']
    # 15. Accessing gbz_param
        GRAIN_STRUCTURE_L1[GBZ_hoster_count]['gbz_param']
        # 15a. Accessing gbz_param: area
            GRAIN_STRUCTURE_L1[GBZ_hoster_count]['gbz_param']['area']
        # 15b. Accessing gbz_param: perimeter
            GRAIN_STRUCTURE_L1[GBZ_hoster_count]['gbz_param']['perimeter']
        # 15c. Accessing gbz_param: aspectRatio
            GRAIN_STRUCTURE_L1[GBZ_hoster_count]['gbz_param']['aspectRatio']
'''
#Saturday night:
#    Selected grains grain boundary zone.
#    Naming and tagging
#    Parameter extractoin and saving in dictionaries
#    Meshing of the grain boundary zones

###############################################################################
"""Grain core coordinates"""
GRAINS_GRC_coord_dict = {count: [] for count in range(NUM_grains)}

"""GBZ POLYGON OBJECTS"""
GRAINS_GBZ_POL_dict = {count: [] for count in range(NUM_grains)}

"""GRCore POLYGON OBJECTS"""
GRAINS_GRC_POL_dict = {count: [] for count in range(NUM_grains)} 

"""Grain centroidal coordinates of the full (parent) grain"""
GRAINS_GRF_CEN_coord_dict = {count: [] for count in range(NUM_grains)}

"""Grain centroidal coordinates of grain core"""
GRAINS_GRC_CEN_coord_dict = {count: [] for count in range(NUM_grains)}

"""Grain boundary zone area"""
GRAINS_GBZ_areas_dict = {count: [] for count in range(NUM_grains)}

"""Grain core"""
GRAINS_GRC_areas_dict = {count: [] for count in range(NUM_grains)}

"""Grain boundary zone area"""
GRAINS_GBZ_perim_dict = {count: [] for count in range(NUM_grains)}

"""Grain core"""
GRAINS_GRC_perim_dict = {count: [] for count in range(NUM_grains)}

for count in range(NUM_grains):
    grain_coord = GRAINS_V_coord_dict[count]
    GRP_F_0 = Polygon(grain_coord) # Grain polygon level 0
    #GRP_F_0.area
    
    #[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
    #]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
    # 2. Set the grain boundary zone creation parameters
    # 2a. Set the grain deflation value
    # CALL FUNCTION TO CALCUALTE DEFLATION
    GRP0_deflation = np.abs(0.01)
    
    #[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
    #]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
    # 2b. Set the deflation direction
    GRP0_deflation_dir = 'right' # DO NO CHANGE
    
    #[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
    #]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
    # 2b. Set the GB_curve-chain-links joining style
    GRP0_defl_GB_join_style = 1 # 1:Round 2:Mitre 3:Bevel
    
    #[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
    #]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
    # 2c. Set the mitre limit for joining style
    GRP0_defl_GB_mitre_limit = 0.1
    
    #[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
    #]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
    # Use the above parameters and make the grain boundary zone
    # 3a. Deflate the parent grain
    defgr = GRP_F_0.exterior.parallel_offset(-GRP0_deflation, GRP0_deflation_dir, join_style = GRP0_defl_GB_join_style, mitre_limit = GRP0_defl_GB_mitre_limit)
    
    #[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
    #]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
    # 3b. Make shapely polygon object from deflated deflated upper level's boundary curve_chain
    GRP_F_1 = Polygon(defgr) # Grain polygon level 1
    GRAINS_GRC_POL_dict[count] = GRP_F_1
    
    #[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
    #]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
    # GRP_F_1.area
    # 3c. Construct the Grain boundary zone (GBZ) as the diff. pol. from grains at lev 0 & 1
    GBZ_G0_G1 = GRP_F_0.difference(GRP_F_1) # OR GBZ_G0_G1 = Polygon(list(GRP_0_coord), [list(innPOL_coord)]) ?
    GRAINS_GBZ_POL_dict[count] = GBZ_G0_G1
    
    #[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
    #]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
    # CONSTRUCT VERTICES COORDINATE ARRAYS
    # a. Construct vertices coordinate data of Hosting grain (0) and Core grain (1)
    GRP_0_coord = np.hstack((np.array(GRP_F_0.exterior.xy[0])[np.newaxis].T, np.array(GRP_F_0.exterior.xy[1])[np.newaxis].T)) # Hosting grain
    GRP_1_coord = np.hstack((np.array(GRP_F_1.exterior.xy[0])[np.newaxis].T, np.array(GRP_F_1.exterior.xy[1])[np.newaxis].T)) # Core grain
    #[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
    #]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
    # b. Construct vertices coordinate data of outer and inner boundaries of GBZ
    GBZ_G0_G1_outer_coord = np.hstack((np.array(GBZ_G0_G1.boundary.geoms[0].xy[0])[np.newaxis].T, np.array(GBZ_G0_G1.boundary.geoms[0].xy[1])[np.newaxis].T))
    GBZ_G0_G1_inner_coord = np.hstack((np.array(GBZ_G0_G1.boundary.geoms[1].xy[0])[np.newaxis].T, np.array(GBZ_G0_G1.boundary.geoms[1].xy[1])[np.newaxis].T))
    # NOTE 1: GBZ_G0_G1_outer_coord = GRP_0_coord. They must be the same. Could use them to validate polygon subtraction output!
    # NOTE 2: GBZ_G0_G1_inner_coord = GRP_1_coord. They must be the same. Could use them to validate polygon subtraction output!

    GRAINS_GRC_coord_dict[count] = GBZ_G0_G1_inner_coord

    # CALCULATE GEOMETRY PROPERTIES
    GRAINS_GBZ_areas_dict[count] = GBZ_G0_G1.area
    GRAINS_GBZ_perim_dict[count] = GBZ_G0_G1.length
    
    GRAINS_GRC_areas_dict[count] = GRP_F_1.area
    GRAINS_GRC_perim_dict[count] = GRP_F_1.length
    
    GRAINS_GRF_CEN_coord_dict[count] = GRP_F_0.centroid.xy
    GRAINS_GRC_CEN_coord_dict[count] = GRP_F_0.centroid.xy

    # GRZ_G0_G1_deflationERROR = GRP_F_0.area - (GRP_F_1.area + GBZ_G0_G1.area)

#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
#]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
fig, ax1 = plt.subplots(dpi = 400)
for count in range(NUM_grains):
    ax1.fill(np.array(GRAINS_GBZ_POL_dict[count].exterior.xy[0]),
             np.array(GRAINS_GBZ_POL_dict[count].exterior.xy[1]),
             facecolor = 'blue', 
             edgecolor = 'black', linestyle = '-', linewidth = 1,
             rasterized = True, alpha = 1.0)
    ax1.fill(np.array(GRAINS_GRC_POL_dict[count].exterior.xy[0]),
             np.array(GRAINS_GRC_POL_dict[count].exterior.xy[1]),
             facecolor = 'yellow', 
             edgecolor = 'black', linestyle = '-', linewidth = 1,
             rasterized = True, alpha = 1.0)
plt.axis('equal')  # square, equal
fig.tight_layout()
plt.show()
# =============================================================================
# fig, ax1 = plt.subplots(dpi = 100)
# for count in range(2000):
#     x = mesh.points[mesh.cells[1].data[count]][:,0]
#     y = mesh.points[mesh.cells[1].data[count]][:,1]
#     ax1.fill(x, y, facecolor = 'cyan', edgecolor = 'black')
# plt.show()
# =============================================================================
# =============================================================================
#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
#]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
# Set mesh resolution factors for global, core and grain boundary zones
MSH_RES_global     = 0.06
MSH_RES_GCore_FAC  = 0.05
MSH_RES_GBZone_FAC = 0.05

#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
#]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
# Set flags for element type in the grain core and grain boundary zone
FLAG_quads_GCore  = 1
FLAG_quads_GBZone = 0

#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
#]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
# Set dimensionality, mesh order and meshing algorithm
dimensionality    = 2
Mesh_Order        = 1
Meshing_Algorithm = 8

# Make empty geometry object from GMSH built in geometry kernel
geometry = pygmsh.geo.Geometry()

# Make object to add grain features to the "geometry" object
AGAFM = geometry.__enter__() # AGAFM: Add Grain Area Feature to Model

# Add relavant geometric features to the geometry object
for count in range(NUM_grains):
    GCore  = AGAFM.add_polygon(GRAINS_GRC_coord_dict[count][:-1,:], mesh_size = MSH_RES_GCore_FAC * MSH_RES_global)
    GBZone = AGAFM.add_polygon(GRAINS_V_coord_dict[count][:-1,:], holes = [GCore], mesh_size = MSH_RES_GBZone_FAC * MSH_RES_global)
    # Enable triangle recombination if quad elements are needed
    if FLAG_quads_GCore == 1:
        AGAFM.set_recombined_surfaces([GCore.surface])
    if FLAG_quads_GBZone == 1:
        AGAFM.set_recombined_surfaces([GBZone.surface])

# Synchronize everything before setting up to mesh
AGAFM.synchronize()

# Generate the mesh
mesh = geometry.generate_mesh(dim = dimensionality, order = Mesh_Order, algorithm = Meshing_Algorithm)

#points, cells, point_data, cell_data, field_data = pygmsh.generate_mesh(geom)
print('-'*40)
print('Mesh generation complete')
###############################################################################
print('-'*40)

# =============================================================================
# import optimesh
# optimesh.optimize(mesh, "laplace", 1.0e-2, 100, verbose = False)
# optimesh.odt.fixed_point(mesh.points, mesh.cells, 1.0e-2, 100, verbose = False)
# =============================================================================
#pygmsh.optimize(mesh, method = "Laplace2D", verbose = False)

#mesh = OMesh

#import optimesh
#optimesh.laplace.fixed_point(mesh.points, mesh.cells[1], 0.0, 50, step_filename_format="haha.png")
#haha = optimesh.optimize(mesh, "laplace", 1E-5, False)

# =============================================================================
# if FLAG_quads == 1:
#     if mesh.cells[1].type[0:3] == 'tri':
#         print('Primary element category:', mesh.cells[2].type.upper())
#         print('Mesh also contains ' + mesh.cells[1].type.upper() + ' elements')
#         TYPE_elem = {'NUM_'+mesh.cells[1].type: len(mesh.cells_dict[mesh.cells[1].type]),
#                       'NUM_'+mesh.cells[2].type: len(mesh.cells_dict[mesh.cells[2].type]),
#                       'NUM_ALL_ELEM': len(mesh.cells_dict[mesh.cells[1].type])+len(mesh.cells_dict[mesh.cells[2].type])}
#         SHARING_nodes = {'NUM nodes shared by '+mesh.cells[1].type+' elements': len(np.unique(np.concatenate(mesh.cells_dict[mesh.cells[1].type]))),
#                           'NUM nodes shared by '+mesh.cells[2].type+' elements': len(np.unique(np.concatenate(mesh.cells_dict[mesh.cells[2].type]))),
#                           'NUM Total nodes: ': len(np.union1d(np.concatenate(mesh.cells_dict[mesh.cells[1].type]),np.concatenate(mesh.cells_dict[mesh.cells[2].type])))}
#     else:
#         print('Primary element category:', mesh.cells[1].type.upper())
#         TYPE_elem = {'NUM_'+mesh.cells[1].type: len(mesh.cells_dict[mesh.cells[1].type]),
#                       'NUM_ALL_ELEM': len(mesh.cells_dict[mesh.cells[1].type])}
#         SHARING_nodes = {'NUM nodes shared by '+mesh.cells[1].type+' elements': len(np.unique(np.concatenate(mesh.cells_dict[mesh.cells[1].type]))),
#                           'NUM Total nodes: ': len(np.union1d(np.concatenate(mesh.cells_dict[mesh.cells[1].type]),np.concatenate(mesh.cells_dict[mesh.cells[2].type])))}
# else:
#     print('Primary element category:', mesh.cells[1].type.upper())
#     TYPE_elem = {'NUM_'+mesh.cells[1].type: len(mesh.cells_dict[mesh.cells[1].type]),
#                   'NUM_ALL_ELEM': len(mesh.cells_dict[mesh.cells[1].type])}
#     SHARING_nodes = {'NUM nodes shared by '+mesh.cells[1].type+' elements': len(np.unique(np.concatenate(mesh.cells_dict[mesh.cells[1].type]))),
#                       'NUM Total nodes: ': len(np.union1d(np.concatenate(mesh.cells_dict[mesh.cells[1].type]),np.concatenate(mesh.cells_dict[mesh.cells[2].type])))}
# 
# print(TYPE_elem)
# print(SHARING_nodes)
# =============================================================================
###############################################################################
print('Writing mesh file - FILENAME.vtk')
# =============================================================================
# optmesh = gmsh.model.mesh.optimize("Laplace2D", force = False, niter = 1, dimTags = [])
# mesh = optmesh
# =============================================================================
#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
#]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
mesh.write("sunil.vtk")

###############################################################################
# MESH VISUALIZATION MODULE - PYVISTA
print('#'*40)
print('STARTING MESH VISUALIZATION')
print('-'*10)
print('Reading mesh file - FILENAME.vtk')
print('-'*10)

#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
#]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
grid = pv.read("sunil.vtk")
#mesh1 = meshio.read("sunil.vtk")

#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
#]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
print('Mesh viz.')
# Set plotter window properties
pv.global_theme.background='maroon'
plotter = pv.Plotter(window_size = (900, 600))
# =============================================================================
# _ = plotter.add_axes_at_origin(x_color = 'red', y_color = 'green', z_color = 'blue',
#                                line_width = 1,
#                                xlabel = 'x', ylabel = 'y', zlabel = 'z',
#                                labels_off = True)
# =============================================================================
_ = plotter.add_points(np.array([0,0,0]),
                        render_points_as_spheres = True,
                        point_size = 25)
_ = plotter.add_points(grid.points,
                        render_points_as_spheres = True,
                        point_size = 2)
#_ = plotter.add_bounding_box(line_width=2, color='black')
_ = plotter.add_mesh(grid,
                      show_edges = True,
                      edge_color = 'black',
                      line_width = 0.5,
                      render_points_as_spheres = False,
                      point_size = 1,
                      style = 'wireframe')
plotter.view_xy()
plotter.camera.zoom(1.0)
plotter.show()

#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
#]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
# extract only triangular elements
ELEMENT_ID_TRI = 3
tri_cells = [i for i in range(grid.n_cells) if grid.cell_n_points(i) == ELEMENT_ID_TRI]
grid_subset_TRI = grid.extract_cells(tri_cells)

#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
#]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
# extract only quadrilateral elements
ELEMENT_ID_QUAD = 3
tri_cells = [i for i in range(grid.n_cells) if grid.cell_n_points(i) == ELEMENT_ID_QUAD]
grid_subset_QUAD = grid.extract_cells(tri_cells)

# =============================================================================
# # extract only triangular elements
# CELLDATA = grid.cells
# # vtk.VTK_TRIANGLE
# triggerID = 3
# TRIANGLES = np.array([[.0,.0,.0]])
# 
# actualcount = 0
# increment = int(CELLDATA[0])
# for count in range(len(CELLDATA)):
#     if CELLDATA[actualcount] == triggerID:
#         TRIANGLES = np.append(TRIANGLES,
#                               np.array(CELLDATA[actualcount+1 : actualcount+1+increment])[np.newaxis],
#                               axis = 0)
#     actualcount += increment+1
#     if actualcount < len(CELLDATA)-1:
#         increment = CELLDATA[actualcount]
#     else: 
#         break
# TRIANGLES = np.delete(TRIANGLES, 0, 0)
# 
# fig, ax1 = plt.subplots(dpi = 400)
# for count in range(np.shape(TRIANGLES)[0]):
#     x = [grid.points[int(TRIANGLES[count][0])][0], grid.points[int(TRIANGLES[count][1])][0], grid.points[int(TRIANGLES[count][2])][0]]
#     y = [grid.points[int(TRIANGLES[count][1])][1], grid.points[int(TRIANGLES[count][2])][1], grid.points[int(TRIANGLES[count][2])][1]]
#     ax1.fill(x, y, facecolor = 'none', edgecolor = 'black')
# plt.show()
# =============================================================================
# =============================================================================
# dir(grid)
# index=grid.find_cells_along_line([0.0, 0.0, 0.0], [2.0, 0.0, 0.0])
# =============================================================================

# =============================================================================
# index = grid.find_cells_along_line([0.0, 0.0, 0.0], [0.04, 0.04, 0.0])
# subset = grid.extract_cells(index)
# 
# pv.global_theme.background='maroon'
# plotter = pv.Plotter(window_size = (900, 600))
# # =============================================================================
# # _ = plotter.add_axes_at_origin(x_color = 'red', y_color = 'green', z_color = 'blue',
# #                                line_width = 1,
# #                                xlabel = 'x', ylabel = 'y', zlabel = 'z',
# #                                labels_off = True)
# # =============================================================================
# _ = plotter.add_points(np.array([0,0,0]),
#                         render_points_as_spheres = True,
#                         point_size = 25)
# _ = plotter.add_points(grid.points,
#                         render_points_as_spheres = True,
#                         point_size = 2)
# #_ = plotter.add_bounding_box(line_width=2, color='black')
# _ = plotter.add_mesh(subset,
#                       show_edges = True,
#                       edge_color = 'black',
#                       line_width = 0.5,
#                       render_points_as_spheres = True,
#                       point_size = 10,
#                       style = 'wireframe')
# plotter.view_xy()
# plotter.camera.zoom(1.0)
# plotter.show()
# =============================================================================
# =============================================================================

pv.global_theme.font.label_size = 16
pv.global_theme.font.title_size = 20

#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
#]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
# Calculate 
quality_min_angle = grid.compute_cell_quality(quality_measure = 'min_angle')
quality_max_angle = grid.compute_cell_quality(quality_measure = 'max_angle')
quality_Jacobian  = grid.compute_cell_quality(quality_measure = 'jacobian')
quality_aspect_ratio  = grid.compute_cell_quality(quality_measure = 'aspect_ratio')
quality_aspect_frobenius = grid.compute_cell_quality(quality_measure = 'aspect_frobenius')
quality_distortion = grid.compute_cell_quality(quality_measure = 'distortion')
#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
#]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
quality_min_angle.plot(cpos = 'xy',
                       scalars = 'CellQuality',
                       show_edges = False,
                       cmap = 'nipy_spectral',
                       clim = [45, 90],
                       scalar_bar_args={'title': "Min. Angle (deg.)"},
                       )

#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
#]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
quality_max_angle.plot(cpos = 'xy',
                       scalars = 'CellQuality',
                       show_edges = False,
                       cmap = 'nipy_spectral',
                       clim = [90, 180],
                       scalar_bar_args={'title': "Max. Angle (deg.)"},
                       )

#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
#]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
quality_Jacobian.plot(cpos = 'xy',
                      scalars = 'CellQuality',
                      show_edges = False,
                      cmap = 'nipy_spectral',
                      scalar_bar_args={'title': "Max. Angle (deg.)"},
                      )

#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
#]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
quality_aspect_ratio.plot(cpos = 'xy',
                  scalars = 'CellQuality',
                  show_edges = False,
                  cmap = 'nipy_spectral',
                  clim = [1.2, 1.3],
                  below_color = 'white',
                  above_color = 'black',
                  scalar_bar_args={'title': "Element aspect ratio"},
                 )

#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
#]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
quality_aspect_frobenius.plot(cpos = 'xy',
                  scalars = 'CellQuality',
                  show_edges = False,
                  cmap = 'nipy_spectral',
                  clim = [1.1, 1.3],
                  below_color = 'white',
                  above_color = 'black',
                  scalar_bar_args={'title': "Element Aspect Frobenius"},
                 )

#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
#]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
quality_distortion.plot(cpos = 'xy',
                  scalars = 'CellQuality',
                  show_edges = False,
                  cmap = 'nipy_spectral',
                  clim = [0.5, 1.0],
                  below_color = 'white',
                  above_color = 'black',
                  scalar_bar_args={'title': "distortion"},
                 )

#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
#]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
MinAngle   = np.delete(quality_min_angle.cell_data['CellQuality'], quality_min_angle.cell_data['CellQuality'] == -1)
MaxAngle   = np.delete(quality_max_angle.cell_data['CellQuality'], quality_min_angle.cell_data['CellQuality'] == -1)
Quality_AR = np.delete(quality_aspect_ratio.cell_data['CellQuality'], quality_aspect_ratio.cell_data['CellQuality'] == -1)
Quality_DISTORTION = np.delete(quality_distortion.cell_data['CellQuality'], quality_distortion.cell_data['CellQuality'] == -1)
Quality_DISTORTION = np.delete(Quality_DISTORTION, Quality_DISTORTION == 1)

#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
#]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
# Plot histograms of mesh quality parameters
num_bins = 100
fig, axs = plt.subplots(2, 2, figsize = (10, 10), dpi = 100, sharey = False, facecolor = 'white')
plt.box(on = True)

axs[0,0].set_facecolor("white")
hist_areas = axs[0,0].hist(MinAngle, bins = num_bins, facecolor = 'blue', edgecolor = 'k', linewidth = 2)
axs[0,0].set_xlabel('Minimum angles', fontsize = 18)
axs[0,0].set_ylabel('Count', fontsize = 18)
axs[0,0].tick_params(axis = 'x', labelsize = 14)
axs[0,0].tick_params(axis = 'y', labelsize = 14)

axs[0,1].set_facecolor("white")
hist_perimeters = axs[0,1].hist(MaxAngle, bins = num_bins, facecolor = 'cyan', edgecolor = 'k', linewidth = 2)
axs[0,1].set_xlabel('Maximum angles', fontsize = 18)
axs[0,1].set_ylabel('Count', fontsize = 18)
axs[0,1].tick_params(axis = 'x', labelsize = 14)
axs[0,1].tick_params(axis = 'y', labelsize = 14)

axs[1,0].set_facecolor("white")
hist_aspect_ratio = axs[1,0].hist(Quality_AR, bins = num_bins, facecolor = 'cyan', edgecolor = 'k', linewidth = 2)
axs[1,0].set_xlabel('Aspect ratio', fontsize = 18)
axs[1,0].set_ylabel('Count', fontsize = 18)
axs[1,0].tick_params(axis = 'x', labelsize = 14)
axs[1,0].tick_params(axis = 'y', labelsize = 14)

axs[1,1].set_facecolor("white")
hist_Quality_DISTORTION = axs[1,1].hist(Quality_DISTORTION, bins = num_bins, facecolor = 'cyan', edgecolor = 'k', linewidth = 2)
axs[1,1].set_xlabel('Distortion', fontsize = 18)
axs[1,1].set_ylabel('Count', fontsize = 18)
axs[1,1].tick_params(axis = 'x', labelsize = 14)
axs[1,1].tick_params(axis = 'y', labelsize = 14)

fig.tight_layout()
plt.show()




###############################################################################
# Twin zones           - Level 2
###############################################################################
