# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 03:49:07 2022

@author: rg5749
"""

import random
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import pygmsh
import pyvista as pv
import meshio
import gmsh as gmsh
import vtk
###############################################################################
from shapely.geometry import Polygon
###############################################################################
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.stats import gaussian_kde as KDE_scipy_gaussian
from scipy.signal import find_peaks
from scipy.stats import skew
from scipy.stats import kurtosis
###############################################################################
# Move to setters
a       = 0.1 # float
startx  = 0   # float
endx    = 1.0 # float
nyunits = 2   # int
angle   = 60  # float
###############################################################################
UnitCellDIM = {'LengthAngle': {'UnitLength': a,
                                   'UnitAngle' : angle}}
xdomain     = {'StartEnd'  : (startx, endx)}
ydomain     = {'NumCopies' : (nyunits)}
###############################################################################
def gen_RAND_Lattice_Coord(UnitCellDIM, xdomain, ydomain):
    """
    Generate random lattice coordinate data.
    
    Input
    -----
        UnitCellDIM: Dictionary: data needed to generate unit cell
        xdomain    : Information needed to make the x data
        ydomain    : Information needed to make the y data

    Return
    ------
        x: x-coordinate data array
        y: y-coordinate data array
    """
    a = 1
    return a
###############################################################################
def gen_REC_Lattice_Coord(UnitCellDIM, xdomain, ydomain):
    """
    Generate rectangular lattice coordinate data.
    
    Input
    -----
        UnitCellDIM: Dictionary containing data needed to generate unit cell
        xdomain: Information needed to make the x data
        ydomain: Information needed to make the y data
    
    Return
    ------
        x: x-coordinate data array
        y: y-coordinate data array
    """
    npx = 5
    npy = 5
    
    x0 = 1
    x1 = 10
    y0 = 1
    y1 = 10
    
    x = np.linspace(x0,x1,npx)
    y = np.linspace(y0,y1,npy)
    
    x, y = np.meshgrid(x, y) + 0.00*np.random.random((npy, npx))
    
    return x, y
###############################################################################
# Pointer to gradientType is stored in the key 'gradientType'
gradientData = {'none'  : [],
                'log10' : {'cx': 0.0,
                           'cy': 0.0,
                           'kx': 1.0,
                           'ky': 1.0,
                           'powerx': 1.0,
                           'powery': 1.0},
                'powerLaw' : {'cx': 0.0,
                              'cy': 0.0,
                              'kx': 1.0,
                              'ky': 1.0,
                              'powerx': 0.5,
                              'powery': -0.5}
                }
###############################################################################
gradientType = 'none'
###############################################################################
def operate_REC_LatCoord_apply_gradient(gradientType, gradientData, x, y):
    """
    Operate on the rectangular lattice coordinates to make gradient
    
    Inputs
    ------
        gradientType: specifies what sort of gradient need be introduced
        gradientData: Data giving paramter value of the gradient making rule or function
    Return
    ------
        x: gradient-applied x-coordinate of the lattice
        y: gradient-applied y-coordinate of the lattice
        
    Definition call
    ---------------
        x, y = operate_REC_LatCoord_apply_gradient(gradientType, gradientData, x, y)
    """
    if gradientType == 'none':
        pass
    elif gradientType == 'log10':
        cx = gradientData[gradientType]['cx']
        cy = gradientData[gradientType]['cy']
        kx = gradientData[gradientType]['kx']
        ky = gradientData[gradientType]['ky']
        powerx = gradientData[gradientType]['powerx']
        powery = gradientData[gradientType]['powerx']
        x = np.log10(x)
        y = np.log10(y)
    elif gradientType == 'powerlaw':
        power_x = gradientData['gradientParam']
        x = np.power(x, +0.5)
        y = np.power(y, +0.5)
    return x, y
###############################################################################
def gen_HEX_Lattice_Coord(UnitCellDIM, xdomain, ydomain):
    a = 1
    return a
###############################################################################
def gen_TRI_REC_mixed_Lattice_Coord(UnitCellDIM, xdomain, ydomain):
    a = 1
    return a
###############################################################################
def gen_TRI_HEX_mixed_Lattice_Coord(UnitCellDIM, xdomain, ydomain):
    a = 1
    return a
###############################################################################
def gen_mixed_Lattice_Coord(UnitCellDIM, xdomain, ydomain):
    a = 1
    return a
###############################################################################
def gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain):
    """
    Generate triangular lattice coordinate data.
    
    Input
    ------
        UnitCellDIM: Dictionary to get inputs on lattice
        xdomain: data specifying x-bounds of lattice
        ydomain: data specifying y-bounds of lattice
    Return
    ------
        x: x-coordinate of the lattice
        y: y-coordinate of the lattice
        
    Definition call
    ---------------
        x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

    Note
    ----
        none now.
    """
    if list(UnitCellDIM.keys())[0]=='LengthAngle':
        data  = UnitCellDIM[list(UnitCellDIM.keys())[0]]
        a     = data['UnitLength']
        angle = data['UnitAngle']
        
    if list(xdomain.keys())[0]=='StartEnd':
        startx, endx = list(xdomain.values())[0]
        
    if list(ydomain.keys())[0]=='NumCopies':
        nyunits = list(ydomain.values())[0]
        
    # Row 1 - x
    x  = np.arange(startx, endx, a)
    x1 = np.copy(x)
    # Row 2 - x
    x2 = x1 + a*np.cos(np.deg2rad(angle))
    # Row 1 and Row 2 - x
    xi = np.vstack((x1, x2))
    # Row 1 - y
    y1 = np.zeros(np.shape(x1))
    # Row 2 - y
    y2 = y1 + a*np.sin(np.deg2rad(angle))
    # Row 1 and Row 2 - y
    yi = np.vstack((y1,y2))
    # Make pattern by Translation
    x = np.copy(xi)
    y = np.copy(yi)
    for count in range(nyunits):
        x = np.vstack((x, xi))
        y = np.vstack((y, yi+(count+1)*2*a*np.sin(np.deg2rad(angle))))
    
    return x, y
###############################################################################
def add_Perturb_coordinates(x, y, method, factors):
    """
    Add perturbations to the NpArray.
    
    Input
    ------
        x      : numpy.ndarray: (:, 1): x-coordinates of the lattice
        y      : numpy.ndarray: (:, 1): y-coordinates of the lattice
        method : str          : Which method to use to introduce lattice vertex distributions
        factors: Factors specifying the amount of lattice vertex perturbation
        
    Return
    ------
        x: perturbed x-coordinates of the lattice
        y: perturbed y-coordinates of the lattice
        
    Definition call
    ---------------
        x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

    Note
    ----
        none now.

    """
    if method == 'RandomUniform':
        facx, facy = factors
        x = x + facx*np.random.random(np.shape(x))
        y = y + facy*np.random.random(np.shape(y))
    return x, y
###############################################################################
def vis_coords(x, y):
    """
    Plot the point distribution.
    
    Input
    ------
        x: x-coordinate of the lattice
        y: y-coordinate of the lattice
        
    Return
    ------
        nothing to return
        
    Definition call
    ---------------
        x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

    Note
    ----
        note here
    """
    plt.scatter(x, y, s = 10, c = 'b', marker = 's')
    plt.show()
###############################################################################
def form_2D_Coord_Array(x, y):
    """
    Prepare co-ordinate data format.
    
    Input
    -----
        x: x-coordinate of the data
        y: y-coordinate of the data
        
    Return
    ------
        x: x-coordinate of the data
        y: y-coordinate of the data
        
    Definition call
    ---------------
        x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

    Note
    ----
        note here
    """
    x = np.reshape(x, np.size(x))[np.newaxis].T
    y = np.reshape(y, np.size(y))[np.newaxis].T
    return x, y
###############################################################################
def form_2D_VorTess_Seeds(x, y):
    """
    Prepare the voronoi seed values.
    
    Input
    ------
        x: x-coordinaste data of the lattice coordinates
        y: y-coordinate data of the lattice coordinates
        
    Return
    ------
        vseeds: seed coordinate array needed to make the Voronoi tesseallation
        
    Definition call
    ---------------
        x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

    Note
    ----
        note here
    """
    vseeds = np.hstack((x, y))
    return vseeds
# x = [np.hstack((x, (count+1)*(a+xi))) for count in range(nxunits)]
# x = np.tile(x, nxunits)
# x = np.append(x, x+a)
# x = [ for count in range(nxunits)]
# x = np.linspace(0.0, 1.0, nxunits)[np.newaxis]
# y = np.zeros((1, nxunits), dtype = 'float')
###############################################################################
def form_2D_VorTess_Object(vseeds):
    """
    Calculate the voronoi object (vo).
    
    Input
    ------
        vseeds: seed coordinate array needed to Voronoi tessellate 
        
    Return
    ------
        vo: Voronoi object
        
    Definition call
    ---------------
        x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

    Note
    ----
        note here
    """
    vo = Voronoi(vseeds)
    return vo
###############################################################################
def vis_Vor(VorTess2D_object):
    """
    Plot the voronoi tessellation.
    
    Input
    ------
        VorTess2D_object: Voronoi tessellation object (Returned from the sci py Voronoi)
       
    Return
    ------
        Just a plot of the Voronoi tessellation
        
    Definition call
    ---------------
        x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

    Note
    ----
        note here
    """
    voronoi_plot_2d(VorTess2D_object)
###############################################################################
def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite regions.
    
    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.
        
    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)
###############################################################################
###############################################################################
def ini_Grain_param(paramSize):
    """
    Write the summary line here.
    
    #TODO: Describe this definition
    #TODO: Investigate what is happening to centroid in thie definition codes
    
    Input
    ------
        paramSize: Size of hwe parameter dictionary
        
    Return
    ------
        GrainPar: Dictionary for storing the Grain structure parameters
        
    Definition call
    ---------------
        x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

    Note
    ----
        note here
    """
    areas      = np.zeros(paramSize)
    centroid   = areas.copy()#<<<<<<<<<<<<<<<<<<
    perimeters = areas.copy()
    # Collect to dict
    GrainPar   = {'areas'     : areas,
                  'perimeters': perimeters}
    return GrainPar
###############################################################################
def make_Super_Bounding_Polygon(VoronoiObject, METHOD):
    """
    Generate bounding box for the VTGS.
    
    VoronoiObject: The voronoi tessellation object data
    METHOD:        (1) useVObounds_and_offsets: str
    BoundingData:  (1) For METHOD=useVObounds_and_offsets, it is the offset values
                       for left edge, right edge, bottom edge and top edge
                       arranged in this order, in a list
    
    Input
    ------
        VoronoiObject: Shapely object: Voronoi object of the grain structure. Unbounded Voronoi grain structure
        METHOD       : str           : Specifies how bounding box is constructed
                                       useVObounds_and_offsets

    # TODO: Rename definition name from "make_Super_Bounding_Polygon" to "make_L0GS_Super_Bounding_Polygon"

    Return
    ------
        PolygBBox_VT: Shapely polygon object. Represents the bounding box.
        
    Definition call
    ---------------
        PolygBBox_VT = make_Super_Bounding_Polygon(VoronoiObject, METHOD)

    Note
    ----
        note here
    """
    if METHOD=='useVObounds_and_offsets':
        min_x = vo.min_bound[0] - 0.00
        max_x = vo.max_bound[0] + 0.00
        min_y = vo.min_bound[1] - 0.00
        max_y = vo.max_bound[1] + 0.00

        # PolygBBox_VT: Polygonal bounding box for Voronoi tessellation
        PolygBBox_VT = Polygon([[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]])
    return PolygBBox_VT
###############################################################################
def L0GS_calc_num_VorGrains(VGrains):
    """
    Calculate the number of grains in the Voronoi tessellation.
    
    Input
    -----
        VGrains: list: Voronoi grains: A list of external vertices indices of all grains

    # TODO: rename "VGrains" to "L0GS_extVGrains"
    # TODO: rename definition name "L0GS_calc_num_VorGrains" to "L0GS_calc_num_intVorGrains"
    # TODO: Create definition "L0GS_calc_num_extVorGrains"

    Return
    ------
        L0GS_NGrains
    """
    L0GS_NGrains = len(VGrains)
    return L0GS_NGrains
###############################################################################
# TODO : Variable name change - L0GS_NGrains to L0GS_NGrains
def build_GSL0_idname(L0GS_NGrains):
    """
    ID and string name of the grain in the GSL0.
    
    GRn: grainCount
    L0GS_gridnames: L0 Grain Structure grain id names
    
    Input
    ------
        L0GS_NGrains: int: Level 0 Grain Structure - Number of grains
        
    Return
    ------
        L0GS_gridnames: Dictionary: Level 0 Grain Structure  -- Grain ID Names
        
    Definition call
    ---------------
        x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

    Note
    ----
        It is sanely assumed that the maximum number of grains is < 10000000
    """
    L0GS_gridnames = {GRn: ['L0GRAIN-%07d'%GRn, []] for GRn in range(L0GS_NGrains)}
    return L0GS_gridnames
# - - - - - - - - - - - - - -
def build_GSL0_vid(L0GS_NGrains):
    """
    Vertices numbers of the grain in GSL0. As returned by shapely.
    
    GRn: grainCount
    L0GS_vid: L0 Grain Structure vertices id
    
    Input
    ------
        L0GS_NGrains: int: Number of the grains in the L0 Grain Structure
        
    Return
    ------
        L0GS_vid: Dictionary: LO0 Grain Structure - Vertex ID
        
    Definition call
    ---------------
        x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

    Note
    ----
        note here
    """
    L0GS_vid = {GRn: ['L0GRAIN-%07d'%GRn, []] for GRn in range(L0GS_NGrains)}
    return L0GS_vid
# - - - - - - - - - - - - - -
def build_GSL0_edgeid(L0GS_NGrains):
    """
    ID of edges of the grain in the GSL0 & correspodning Edge Vertices ID.
    
    GRn: grainCount
    L0GS_gridnames: L0 Grain Structure grain id names
    
    Input
    ------
        L0GS_NGrains: int: Level 0 Grain Struture Number of Grains
        
    Return
    ------
        L0GS_edgeid: Level 0 Grain strucvture - Grain external-boundary edges - IDs
        
    # TODO: Rename "L0GS_edgeid" to "L0GS_extedgeid"
    # TODO: Rename the definition name "build_GSL0_edgeid" to "build_GSL0_extedgeid"
    
    # TODO: Create definition names "build_GSL0_intedgeid"
    
    Definition call
    ---------------
        x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

    Note
    ----
        note here
    """
    L0GS_edgeid = {GRn: ['L0GRAIN-%07d'%GRn, []] for GRn in range(L0GS_NGrains)}
    return L0GS_edgeid
# - - - - - - - - - - - - - -
def build_GSL0_vcoord(L0GS_NGrains):
    """
    X-Y Coordinates of the vertices of the grain in GSL0.
    
    GRn: grainCount
    L0GS_vcoord: L0 Grain Structure vertices coordinates
    
    Input
    ------
        L0GS_NGrains: int: Level 0 Grain Structure Number of grains
        
    Return
    ------
        L0GS_vcoord: Level 0 Grain Structure Vertices coordinates - External boundary vertices

    # TODO: Rename "L0GS_vcoord" to "L0GS_extvcoord"
    # TODO: Rename "build_GSL0_vcoord" to "build_GSL0_extvcoord"
    # TODO: Create definition named "build_GSL0_intvcoord"

    Definition call
    ---------------
        x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

    Note
    ----
        note here
    """
    L0GS_vcoord = {GRn: ['L0GRAIN-%07d'%GRn, []] for GRn in range(L0GS_NGrains)}
    return L0GS_vcoord
# - - - - - - - - - - - - - -
def build_GSL0_ccoord(L0GS_NGrains):
    """
    X-Y Coordinates of the centroid of the grain in GSL0.
    
    GRn: grainCount
    L0GS_ccoord: L0 Grain Structure centroid coordinates
    
    Input
    ------
        L0GS_NGrains: int: Level 0 Grain Structure Number of Grains
        
    Return
    ------
        L0GS_ccoord: Level 0 Grain Structure Centroid coordinates
        
    Definition call
    ---------------
        x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

    Note
    ----
        note here
    """
    L0GS_ccoord = {GRn: ['L0GRAIN-%07d'%GRn, []] for GRn in range(L0GS_NGrains)}
    return L0GS_ccoord
# - - - - - - - - - - - - - -
def build_GSL0_area(L0GS_NGrains):
    """
    Geometric area of the grain in GSL0.
    
    GRn: grainCount
    L0GS_area: L0 Grain Structure area
    
    Input
    ------
        L0GS_NGrains: int: Level 0 Grain Structure Number of Grains
        
    Return
    ------
        L0GS_area: Level 0 Grain Structure parameter Area
        
    Definition call
    ---------------
        x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

    Note
    ----
        note here
    """
    L0GS_area = {GRn: ['L0GRAIN-%07d'%GRn, []] for GRn in range(L0GS_NGrains)}
    return L0GS_area
# - - - - - - - - - - - - - -
def build_GSL0_extperim(L0GS_NGrains):
    """
    Geometric external-perimeter of the grain in GSL0.
    
        If the grain has no fully enveloped holes or fully submerged precipitates,
        then the grain will only have external perimeter
    GRn: grainCount
    L0GS_extperim: L0 Grain Structure external perimeter
    
    Input
    ------
        L0GS_NGrains: int: Level 0 Grain Structure Number of Grains
        
    Return
    ------
        L0GS_extperim: Level 0 Grain Structure parameter external boundary perimeter
        
    Definition call
    ---------------
        x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

    Note
    ----
        note here
    """
    L0GS_extperim = {GRn: ['L0GRAIN-%07d'%GRn, []] for GRn in range(L0GS_NGrains)}
    return L0GS_extperim
# - - - - - - - - - - - - - -
def build_GSL0_intperim(L0GS_NGrains):
    """
    Geometric internal-perimeter(s) of the grain in GSL0.
    
        If the grain has fully enveloped holes or fully submerged precipitates,
        then the grain will have internal perimeter(s) also.
        
        For each hole/precipitate, there will be a seperate internal perimeter
        in the list data structure. Just append inside the list, as needed, with
        correct feature id (either hole or precipitate) appropriately
        
    GRn: grainCount
    L0GS_intperim: L0 Grain Structure external perimeter
    
    Input
    ------
        L0GS_NGrains: int: Level 0 Grain Structure Number of Grains
        
    Return
    ------
        L0GS_intperim: Dictionary: Level 0 Grain Structure parameter internal boundary perimeter
        
    Definition call
    ---------------
        L0GS_intperim = build_GSL0_intperim(L0GS_NGrains)

    Note
    ----
        note here
    """
    L0GS_intperim = {GRn: ['L0GRAIN-%07d'%GRn, []] for GRn in range(L0GS_NGrains)}
    return L0GS_intperim
# - - - - - - - - - - - - - -
def build_GSL0_nextedges(L0GS_NGrains):
    """
    To get the number of edges on the external boundary in the grain in GSL0.
    
    GRn: grainCount
    L0GS_nextedges: L0 Grain Structure number external edges
    
    Input
    ------
        L0GS_NGrains: int: Level 0 Grain Structure Number of Grains
        
    Return
    ------
        L0GS_nextedges: Dictionary: Level 0 Grain Structure parameter Number of external edges
        
    Definition call
    ---------------
        x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

    Note
    ----
        note here
    """
    L0GS_nextedges = {GRn: ['L0GRAIN-%07d'%GRn, []] for GRn in range(L0GS_NGrains)}
    return L0GS_nextedges
# - - - - - - - - - - - - - -
def build_GSL0_nedgesextv(L0GS_NGrains):
    """
    To get the number of edges for every vertex.
    
    GRn: grainCount
    L0GS_nedgesv: L0 Grain Structure
    
    Input
    ------
        L0GS_NGrains: int: Level 0 Grain Structure Number of Grains
        
    Return
    ------
        L0GS_nedgesextv: Dictionary: Level 0 Grain Structure parameter Number of edges for every external vertex of the grain
        
    Definition call
    ---------------
        x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

    Note
    ----
        note here
    """
    L0GS_nedgesextv = {GRn: ['L0GRAIN-%07d'%GRn, []] for GRn in range(L0GS_NGrains)}
    return L0GS_nedgesextv
# - - - - - - - - - - - - - -
def build_GSL0_mindiag(L0GS_NGrains):
    """
    Minimum of the diagonal lengths of the grain.
    
    GRn: grainCount
    L0GS_mindiag: L0 Grain Structure 
    
    Input
    ------
        L0GS_NGrains: int: Level 0 Grain Structure Number of Grains
        
    Return
    ------
        L0GS_mindiag: Dictionary: Level 0 Grain Structure parameter Minimum diagonal length
        
    Definition call
    ---------------
        x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

    Note
    ----
        note here
    """
    L0GS_mindiag = {GRn: ['L0GRAIN-%07d'%GRn, []] for GRn in range(L0GS_NGrains)}
    return L0GS_mindiag
# - - - - - - - - - - - - - -
def build_GSL0_maxdiag(L0GS_NGrains):
    """
    Minimum of the diagonal lengths of the grain.
    
    GRn: grainCount
    L0GS_maxdiag: L0 Grain Structure 
    
    Input
    ------
        L0GS_NGrains: int: Level 0 Grain Structure Number of Grains
        
    Return
    ------
        L0GS_maxdiag: Dictionary: Level 0 Grain Structure parameter Maximum diagonal length
        
    Definition call
    ---------------
        x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

    Note
    ----
        note here
    """
    L0GS_maxdiag = {GRn: ['L0GRAIN-%07d'%GRn, []] for GRn in range(L0GS_NGrains)}
    return L0GS_maxdiag
# - - - - - - - - - - - - - -
def build_GSL0_meandiag(L0GS_NGrains):
    """
    Mean of the diagonal lengths of the grain.
    
    GRn: grainCount
    L0GS_mindiag: L0 Grain Structure 
    
    DEFINITION CALL
    
    Input
    ------
        L0GS_NGrains: int: Level 0 Grain Structure Number of Grains
        
    Return
    ------
        L0GS_meandiag: Dictionary: Level 0 Grain Structure parameter Mean diagonal length
        
    Definition call
    ---------------
        x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

    Note
    ----
        note here
    """
    L0GS_meandiag = {GRn: ['L0GRAIN-%07d'%GRn, []] for GRn in range(L0GS_NGrains)}
    return L0GS_meandiag
# - - - - - - - - - - - - - -
def build_GSL0_stddiag(L0GS_NGrains):
    """
    To build Standard deviation data of the diagonal lengths of the grain.
    
    GRn: grainCount
    L0GS_stddiag: L0 Grain Structure 
    
    DEFINITION CALL
    
    Input
    ------
        L0GS_NGrains: int: Level 0 Grain Structure Number of Grains
        
    Return
    ------
        L0GS_stddiag: Dictionary: Level 0 Grain Structure parameter standard deviation of the diagonal lengths
        
    Definition call
    ---------------
        x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

    Note
    ----
        note here
    """
    L0GS_stddiag = {GRn: ['L0GRAIN-%07d'%GRn, []] for GRn in range(L0GS_NGrains)}
    return L0GS_stddiag
# - - - - - - - - - - - - - -
def build_GSL0_extvintangle(L0GS_NGrains):
    """
    To build internal angles data at all external vertices of the grain in GSL0.
    
    GRn: grainCount
    L0GS_extvintangle: L0 Grain Structure external vertice internal angle
    
    Input
    ------
        L0GS_NGrains: int: Level 0 Grain Structure Number of Grains
        
    Return
    ------
        L0GS_extvintangle: Dictionary: Level 0 Grain Structure parameter External Vertices Internal Angle
        
    Definition call
    ---------------
        x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

    Note
    ----
        note here
    """
    L0GS_extvintangle = {GRn: ['L0GRAIN-%07d'%GRn, []] for GRn in range(L0GS_NGrains)}
    return L0GS_extvintangle
# - - - - - - - - - - - - - -
def build_GSL0_ctovdist(L0GS_NGrains):
    """
    DEFINITION: ctovdist:: .
    
        C: Centroid of the grain
        V: Vertices of the grain
            Calculation:
            1. Find Eucleadian distance between centroid and all vertices of the grain
            2. Populate GRSTR_L0_parameters['C_to_V_dist'][grainCount]
        One value set for every grain
    Input
    -----
        L0GS_NGrains: int: Level 0 Grain Structure Number of Grains
        
    Return
    ------
        L0GS_ctovdist: Dictionary: Level 0 Grain Structure parameter Centroid to internal vertices distances
    
    # TODO: Rename "L0GS_ctovdist" to "L0GS_c_to_extv_dist"
    # TODO: Rename the definition "build_GSL0_ctovdist" to "build_L0GS_c_to_extv_dist"
    # TODO: Make the definition "build_L0GS_c_to_intv_dist"

    Definition call
    ---------------
        x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

    Note
    ----
        note here
    """
    L0GS_ctovdist = {GRn: ['L0GRAIN-%07d'%GRn, []] for GRn in range(L0GS_NGrains)}
    return L0GS_ctovdist
# - - - - - - - - - - - - - -
def build_hist_data():
    """
    Build dictionary to store distrribution data.
    
    Input
    ------
        1
        
    Return
    ------
        1
        
    Definition call
    ---------------
        x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

    Note
    ----
        note here
    """
    hist_data = {'numbins' : [],
                 'bins'    : [],
                 'counts'  : [],
                 'prob'    : [],
                 'cumprob' : [],
                 'pdf_data': (),
                 'cdf_data': (),
                 'modality': [],
                 'skewness': [],
                 'width'   : (),
                 'peaks@'  : (),
                 }
    return hist_data
# - - - - - - - - - - - - - -
def build_GSL0_pdstr_area(hist_data = build_hist_data(), BIN_N = [10, 50, 100, 500]):
    """
    Probability distribution data of grain area of GSL0.
    
    DEFINITION CALL:
        L0GS_pdist_area = build_GSL0_pdstr_area()
    Input
    ------
        1
        
    Return
    ------
        1
        
    Definition call
    ---------------
        x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

    Note
    ----
        note here
    """
    L0GS_pdist_area = {'nbin=%d'%binCount: hist_data for binCount in BIN_N}
    L0GS_pdist_area['nbin=nbin'] = hist_data # Will allways be added to the end
    return L0GS_pdist_area
# - - - - - - - - - - - - - -
def build_GSL0_pdstr_extperim(hist_data = build_hist_data(), BIN_N = [10, 50, 100, 500]):
    """
    Probability distribution data of external perimeters of GSL0.
    
    DEFINITION CALL:
        L0GS_pdist_extperim = build_GSL0_pdstr_extperim()
    Input
    ------
        1
        
    Return
    ------
        1
        
    Definition call
    ---------------
        x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

    Note
    ----
        note here
    """
    L0GS_pdist_extperim = {'nbin=%d'%binCount: hist_data for binCount in BIN_N}
    L0GS_pdist_extperim['nbin=nbin'] = hist_data # Will allways be added to the end
    return L0GS_pdist_extperim
# - - - - - - - - - - - - - -
def build_GSL0_pdstr_intperim(hist_data = build_hist_data(), BIN_N = [10, 50, 100, 500]):
    """
    Probability distribution data of internal perimeters of GSL0.
    
    DEFINITION CALL:
        L0GS_pdist_intperim = build_GSL0_pdstr_intperim()
    Input
    ------
        1
        
    Return
    ------
        1
        
    Definition call
    ---------------
        x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

    Note
    ----
        note here
    """
    L0GS_pdist_intperim = {'nbin=%d'%binCount: hist_data for binCount in BIN_N}
    L0GS_pdist_intperim['nbin=nbin'] = hist_data # Will allways be added to the end
    return L0GS_pdist_intperim
# - - - - - - - - - - - - - -
def build_GSL0_pdstr_nextedges(hist_data = build_hist_data(), BIN_N = [10, 50, 100, 500]):
    """
    # SUMMARY: Probability distribution data of number of external edges of GSL0.
    
    # DEFINITION CALL: L0GS_pdist_nextedges = build_GSL0_pdstr_nextedges()
    Input
    ------
        1
        
    Return
    ------
        1
        
    Definition call
    ---------------
        x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

    Note
    ----
        note here
    """
    L0GS_pdist_nextedges = {'nbin=%d'%binCount: hist_data for binCount in BIN_N}
    L0GS_pdist_nextedges['nbin=nbin'] = hist_data # Will allways be added to the end
    return L0GS_pdist_nextedges
# - - - - - - - - - - - - - -
def build_GSL0_pdstr_nedgesextv(hist_data = build_hist_data(), BIN_N = [10, 50, 100, 500]):
    """
    # SUMMARY: Probability distribution data of number of ____ of GSL0.
    
    # DEFINITION CALL: L0GS_pdist_nedgesextv = build_GSL0_pdstr_nedgesextv()
    Input
    ------
        1
        
    Return
    ------
        1
        
    Definition call
    ---------------
        x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

    Note
    ----
        note here
    """
    L0GS_pdist_nedgesextv = {'nbin=%d'%binCount: hist_data for binCount in BIN_N}
    L0GS_pdist_nedgesextv['nbin=nbin'] = hist_data # Will allways be added to the end
    return L0GS_pdist_nedgesextv
# - - - - - - - - - - - - - -
def build_GSL0_pdstr_mindiag(hist_data = build_hist_data(), BIN_N = [10, 50, 100, 500]):
    """
    # SUMMARY: Probability distribution data of number of ____ of GSL0.
    
    # DEFINITION CALL: L0GS_pdist_mindiag = build_GSL0_pdstr_mindiag()
    Input
    ------
        1
        
    Return
    ------
        1
        
    Definition call
    ---------------
        x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

    Note
    ----
        note here
    """
    L0GS_pdist_mindiag = {'nbin=%d'%binCount: hist_data for binCount in BIN_N}
    L0GS_pdist_mindiag['nbin=nbin'] = hist_data # Will allways be added to the end
    return L0GS_pdist_mindiag
# - - - - - - - - - - - - - -
def build_GSL0_pdstr_maxdiag(hist_data = build_hist_data(), BIN_N = [10, 50, 100, 500]):
    """
    # SUMMARY: Probability distribution data of number of ____ of GSL0.
    
    # DEFINITION CALL: L0GS_pdist_maxdiag = build_GSL0_pdstr_maxdiag()
    Input
    ------
        1
        
    Return
    ------
        1
        
    Definition call
    ---------------
        x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

    Note
    ----
        note here
    """
    L0GS_pdist_maxdiag = {'nbin=%d'%binCount: hist_data for binCount in BIN_N}
    L0GS_pdist_maxdiag['nbin=nbin'] = hist_data # Will allways be added to the end
    return L0GS_pdist_maxdiag
# - - - - - - - - - - - - - -
def build_GSL0_pdstr_extvintangle(hist_data = build_hist_data(), BIN_N = [10, 50, 100, 500]):
    """
    # SUMMARY: Probability distribution data of number of ____ of GSL0.
    
    # DEFINITION CALL: L0GS_pdist_extvintangle = build_GSL0_pdstr_extvintangle()
    Input
    ------
        1
        
    Return
    ------
        1
        
    Definition call
    ---------------
        x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

    Note
    ----
        note here
    """
    L0GS_pdist_extvintangle = {'nbin=%d'%binCount: hist_data for binCount in BIN_N}
    L0GS_pdist_extvintangle['nbin=nbin'] = hist_data # Will allways be added to the end
    return L0GS_pdist_extvintangle
# - - - - - - - - - - - - - -
def build_GSL0_pdstr_ctovdist(hist_data = build_hist_data(), BIN_N = [10, 50, 100, 500]):
    """
    # SUMMARY: Probability distribution data of number of ____ of GSL0.
    
    # DEFINITION CALL: L0GS_pdist_ctovdist = build_GSL0_pdstr_ctovdist()
    Input
    ------
        1
        
    Return
    ------
        1
        
    Definition call
    ---------------
        x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

    Note
    ----
        note here
    """
    L0GS_pdist_ctovdist = {'nbin=%d'%binCount: hist_data for binCount in BIN_N}
    L0GS_pdist_ctovdist['nbin=nbin'] = hist_data # Will allways be added to the end
    return L0GS_pdist_ctovdist
################################################################################
def pop_GSL0_idname(L0GS_gridnames):
    """
    POPULATE: ID and string name of the grain in the GSL0.
    
    GRn: grainCount
    L0GS_gridnames: L0 Grain Structure grain id names
    
    NOTE: THIS IS PRE-POPULATED. nothing more to work on this
    
    DEFINITION CALL:: L0GS_gridnames = pop_GSL0_idname()
    Input
    ------
        1
        
    Return
    ------
        1
        
    Definition call
    ---------------
        x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

    Note
    ----
        note here
    """
    return L0GS_gridnames
# - - - - - - - - - - - - - -
def pop_GSL0_edgeid(GRn, vid, L0GS_edgeid):
    """
    POPULATE: ID and string name of the grain in the GSL0.
    
    GRn: grainCount
    L0GS_edgeid: L0 Grain Structure grain id names
    DEFINITION CALL:
        L0GS_edgeid = pop_GSL0_edgeid(GRn, edgedata)
    Input
    ------
        1
        
    Return
    ------
        1
        
    Definition call
    ---------------
        x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

    Note
    ----
        note here
    """
    vidtemp = vid.copy()
    vidtemp.append(vidtemp[0])
    edgedata = []
    for i in range(len(vid)):
        edgedata.append([vidtemp[i], vidtemp[i+1]])
    L0GS_edgeid[GRn][1] = edgedata
    #np.squeeze(np.asarray(L0GS_edgeid[GRn][1].copy()))
    return L0GS_edgeid
# - - - - - - - - - - - - - -
def pop_GSL0_vid(GRn, viddata, L0GS_vid):
    """
    POPULATE: IDs vertices of the grain in GSL0. As returned by shapely.
    
    GRn: grainCount
    L0GS_vid: L0 Grain Structure vertices id
    
    # DEFINITION CALL::    L0GS_vid = pop_GSL0_vid(GRn, viddata)
    Input
    ------
        1
        
    Return
    ------
        1
        
    Definition call
    ---------------
        x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

    Note
    ----
        note here
    """
    # L0GS_vid[GRn][1].append(viddata)
    L0GS_vid[GRn][1] = viddata
    #L0GS_vid[GRn][1].pop(1)
    return L0GS_vid
# - - - - - - - - - - - - - -
def pop_GSL0_vcoord(GRn, vxycoord, L0GS_vcoord):
    """
    POPULATE: X-Y Coordinates of the vertices of the grain in GSL0.
    
    GRn: grainCount
    L0GS_vcoord: L0 Grain Structure vertices coordinates of grains
    # DEFINITION CALL
        L0GS_vcoord = pop_GSL0_vcoord(GRn, vxycoord)
    Input
    ------
        1
        
    Return
    ------
        1
        
    Definition call
    ---------------
        x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

    Note
    ----
        note here
    """
    L0GS_vcoord[GRn][1] = vxycoord
    return L0GS_vcoord
# - - - - - - - - - - - - - -
def pop_GSL0_ccoord(GRn, cxycoord, L0GS_ccoord):
    """
    POPULATE: X-Y Coordinates of the centroid of the grain in GSL0.
    
    GRn: grainCount
    L0GS_ccoord: L0 Grain Structure centroid coordinates of grains
    # DEFINITION CALL
        L0GS_ccoord = pop_GSL0_ccoord(GRn, cxycoord)
    Input
    ------
        1
        
    Return
    ------
        1
        
    Definition call
    ---------------
        x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

    Note
    ----
        note here
    """
    L0GS_ccoord[GRn][1] = cxycoord
    return L0GS_ccoord
# - - - - - - - - - - - - - -
def pop_GSL0_area(GRn, thisGrainArea, L0GS_area):
    """
    POPULATE: Geometric area of the grain in GSL0.
    
    GRn: grainCount
    L0GS_area: L0 Grain Structure area
    DEFINITION CALL:
        L0GS_area = pop_GSL0_area(GRn, L0GS_area = L0GS_area)
    Input
    ------
        1
        
    Return
    ------
        1
        
    Definition call
    ---------------
        x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

    Note
    ----
        note here
    """
    L0GS_area[GRn][1] = thisGrainArea
    return L0GS_area
# - - - - - - - - - - - - - -
def pop_GSL0_extperim(GRn, GrainExtPerim, L0GS_extperim):
    """
    POPULATE: Geometric external-perimeter of the grain in GSL0.
    
        If the grain has no fully enveloped holes or fully submerged precipitates,
        then the grain will only have external perimeter
    GRn: grainCount
    L0GS_extperim: L0 Grain Structure external perimeter
    DEFINITION CALL:
        L0GS_extperim = pop_GSL0_extperim(GRn, GrainExtPerim)
    Input
    ------
        1
        
    Return
    ------
        1
        
    Definition call
    ---------------
        x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

    Note
    ----
        note here
    """
    L0GS_extperim[GRn][1] = GrainExtPerim
    return L0GS_extperim
# - - - - - - - - - - - - - -
def pop_GSL0_intperim(L0GS_intperim):
    """
    POPULATE: Geometric internal-perimeter(s) of the grain in GSL0.
    
        If the grain has fully enveloped holes or fully submerged precipitates,
        then the grain will have internal perimeter(s) also.
        
        For each hole/precipitate, there will be a seperate internal perimeter
        in the list data structure. Just append inside the list, as needed, with
        correct feature id (either hole or precipitate) appropriately
        
    GRn: grainCount
    L0GS_intperim: L0 Grain Structure external perimeter
    Input
    ------
        1
        
    Return
    ------
        1
        
    Definition call
    ---------------
        x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

    Note
    ----
        note here
    """
    return L0GS_intperim
# - - - - - - - - - - - - - -
def pop_GSL0_nextedges(GRn, nextedges, L0GS_nextedges):
    """
    POPULATE: Number of edges on the external boundary in the grain in GSL0.
    
    GRn: grainCount
    L0GS_nextedges: L0 Grain Structure number external edges
    
    # DEFINITION CALL:
    # L0GS_nextedges = pop_GSL0_nextedges(GRn, nextedges)
    Input
    ------
        1
        
    Return
    ------
        1
        
    Definition call
    ---------------
        x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

    Note
    ----
        note here
    """
    L0GS_nextedges[GRn][1] = nextedges
    return L0GS_nextedges
# - - - - - - - - - - - - - -
def pop_GSL0_nedgesextv(L0GS_nedgesextv):
    """
    POPULATE: Number of edges for every vertex.
    
    GRn: grainCount
    L0GS_nedgesv: L0 Grain Structure
    Input
    ------
        1
        
    Return
    ------
        1
        
    Definition call
    ---------------
        x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

    Note
    ----
        note here
    """
    return L0GS_nedgesextv
# - - - - - - - - - - - - - -
def pop_GSL0_mindiag(GRn, minDiag, L0GS_mindiag):
    """
    POPULATE: Minimum diagonal length of the grain in the L0GS.
    
    GRn: grainCount
    L0GS_mindiag: L0 Grain Structure minimum diagonal
    
    DEFINITION CALL:
        L0GS_mindiag = pop_GSL0_mindiag(GRn, mindiag, L0GS_mindiag)
    Input
    ------
        1
        
    Return
    ------
        1
        
    Definition call
    ---------------
        x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

    Note
    ----
        note here
    """
    L0GS_mindiag[GRn][1] = minDiag
    return L0GS_mindiag
# - - - - - - - - - - - - - -
def pop_GSL0_maxdiag(GRn, maxDiag, L0GS_maxdiag):
    """
    POPULATE: Maximum diagonal length of the grain in the L0GS.
    
    GRn: grainCount
    L0GS_maxdiag: L0 Grain Structure maximum diagonal 
    
    DEFINITION CALL:
        L0GS_maxdiag = pop_GSL0_maxdiag(GRn, maxdiag, L0GS_maxdiag)
    Input
    ------
        1
        
    Return
    ------
        1
        
    Definition call
    ---------------
        x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

    Note
    ----
        note here
    """
    L0GS_maxdiag[GRn][1] = maxDiag
    return L0GS_maxdiag
# - - - - - - - - - - - - - -
def pop_GSL0_meandiag(GRn, meanDiag, L0GS_meandiag):
    """
    POPULATE: Mean diagonal length of the grain in the L0GS.
    
    GRn: grainCount
    L0GS_meandiag: L0 Grain Structure mean diagonal length
    
    DEFINITION CALL:
        L0GS_meandiag = pop_GSL0_meandiag(GRn, maxdiag, L0GS_meandiag)
    Input
    ------
        1
        
    Return
    ------
        1
        
    Definition call
    ---------------
        x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

    Note
    ----
        note here
    """
    L0GS_meandiag[GRn][1] = meanDiag
    return L0GS_meandiag
# - - - - - - - - - - - - - -
def pop_GSL0_stddiag(GRn, stdDiag, L0GS_meandiag):
    """
    POPULATE: standard deviation of the diagonal length of the grain in the L0GS.
    
    GRn: grainCount
    L0GS_stddiag: L0 Grain Structure standard deviation of the diagonal length
    
    DEFINITION CALL:
        L0GS_stddiag = pop_GSL0_stddiag(GRn, stdDiag, L0GS_stddiag)
    Input
    ------
        1
        
    Return
    ------
        1
        
    Definition call
    ---------------
        x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

    Note
    ----
        note here
    """
    L0GS_stddiag[GRn][1] = stdDiag
    return L0GS_stddiag
# - - - - - - - - - - - - - -
def pop_GSL0_extvintangle(L0GS_extvintangle):
    """
    POPULATE: Internal angles at all external vertices of the grain in GSL0.
    
    GRn: grainCount
    L0GS_extvintangle: L0 Grain Structure external vertice internal angle
    Input
    ------
        1
        
    Return
    ------
        1
        
    Definition call
    ---------------
        x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

    Note
    ----
        note here
    """
    return L0GS_extvintangle
# - - - - - - - - - - - - - -
def pop_GSL0_ctovdist(L0GS_ctovdist):
    """
	POPULATE: write here.

       DEFINITION: ctovdist:: For a GRn, 
        C: Centroid of the grain
        V: Vertices of the grain
            Calculation:
            1. Find Eucleadian distance between centroid and all vertices of the grain
            2. Populate GRSTR_L0_parameters['C_to_V_dist'][grainCount]
        One value set for every grain
    Input
    ------
        1
        
    Return
    ------
        1
        
    Definition call
    ---------------
        x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

    Note
    ----
        note here
    """
    #TODO: ctovdist: VALUES, STATISTICS
    return L0GS_ctovdist
################################################################################
def extract_vcoord_POU(POU):
    """
    Extract the coordinates of the vertices of exterior boundaries of the Polygonal Object Unit (POU).
    
    # DEFINITION CALL:
        Vxycoord = extract_Vcoord_POU(POU)
    """
    vxycoord = np.hstack((np.array(POU.exterior.xy[0])[np.newaxis].T,
                          np.array(POU.exterior.xy[1])[np.newaxis].T))
    #TODO: Write here
    return vxycoord
# - - - - - - - - - - - - - -
def extract_ccoord_POU(POU):
    """
    Extract the coordinates of the centroid of the Polygonal Object Unit (POU).
    
    # DEFINITION CALL:
        Cxycoord = extract_Ccoord_POU(POU)
    """
    cxycoord = np.array([POU.centroid.x, POU.centroid.y])[np.newaxis]
    #TODO: Write here
    return cxycoord
# - - - - - - - - - - - - - -
def extract_area_POU(POU):
    """
    Extract the coordinates of the centroid of the Polygonal Object Unit (POU).
    
    #TODO: Write here
    
    # DEFINITION CALL:
        GrainArea = extract_area_POU(POU)
    """
    GrainArea = POU.area
    
    return GrainArea
# - - - - - - - - - - - - - -
def extract_extperim_POU(POU):
    """
    Extract the coordinates of the centroid of the Polygonal Object Unit (POU).
    
    #TODO: Write here
    
    DEFINITION CALL:
        GrainExtPerim = extract_extperim_POU(POU)
    """
    GrainExtPerim = POU.length
    
    return GrainExtPerim
# - - - - - - - - - - - - - -
def extract_edges_POU(vid):
    """
    Get the edges of the POU in the form of end vertex ID list as: [start end].
    
    DEFINITION CALL:
        
    """
    #sunil
    #return edgedata
# - - - - - - - - - - - - - -
def calc_distance_matrix(numpy_coordinate_array_xy):
    """
    Calculate the distance matrix for the list of coordinates provided.
    
    #TODO: Write here

    DEFINITION CALL:
        DistMat = calc_distance_matrix(vxycoord)
    """
    DistMat = sp.spatial.distance_matrix(numpy_coordinate_array_xy,
                                         numpy_coordinate_array_xy)
    return DistMat
# - - - - - - - - - - - - - -
def calc_edgelengths_POU(POU):
    """
    Write description.
    
    #TODO: Write description for this defintiion
    
    DEFINITION CALL:
        edgeLengths = calc_edgelengths_POU(POU)
    """
    # Extract the coordinates of the vertices
    vxycoord = extract_vcoord_POU(POU)
    # Get the edge lengths of this grain
    edgeLengths = []
    for vcount in range(len(vxycoord)-1):
        x_this, y_this = vxycoord[vcount, 0], vxycoord[vcount, 1]
        x_next, y_next = vxycoord[vcount+1, 0], vxycoord[vcount+1, 1]
        edgeLengths.append(np.sqrt((x_next-x_this)**2+(y_next-y_this)**2))
    return np.array(edgeLengths)
# - - - - - - - - - - - - - -
def calc_diaglengths_POU(POU):
    """
    Get the polygon maximum diagonal length.
    
    CALCULATION:
        # maximum of all pair-wise distances amongst all polygon vertices
    
    PHYSICAL INTERPRETATION:
        # More regular the polygon is and more the number of edges, this values represents
        # the polygon's aspect ratio more and more accurately
    
    USAGE LIMITATION:
        # See physical interpretation and interpret
        # Best suited to understand polygons with all-convex boundary
        # What would happen for polygon with hole?: TO DO
    
    WARNING: No intersection checks are made in its calcualtion
    
    DEFINITION CALL:
        diagLengths = calc_diaglengths_POU(POU)
    """
    # Extract the coordinates of the vertices
    vxycoord = extract_vcoord_POU(POU)
    
    # Estimate the unique of distance matrix for POU vertices
    DistMat = np.unique(calc_distance_matrix(vxycoord))
    
    # Remove self-distance calculations
    DistMat = np.delete(DistMat, DistMat == 0)
    
    # find the unique of edge lengths of POU external boundary
    edgeLengths = np.unique(calc_edgelengths_POU(POU))
    
    # find the diagonal lengths
    # it must be the xor of DistMat and edgeLengths
    # As all elements of edgeLengths are sure to be there in DistMat, 
    # they WILL be removed. Remaining will be the lengths of the diagonals!
    diagLengths = np.setxor1d(DistMat, edgeLengths)
    if diagLengths.shape[0] == 0:
        # This happens when the POU is a triangle and there are no diagonals
        diagLengths = np.array(['NoDiag'])
    return diagLengths
# - - - - - - - - - - - - - -
def calc_maxdiag_POU(diagLengths):
    """
    Calculate the maximum of the diagonal lengths.
    
    Can also be calculated as the maximum of the distance matrix
    
    DEFINITION CALL:
        maxDiag = calc_maxdiag_POU()
    """
    maxDiag = np.asarray(diagLengths).max()
    return maxDiag
# - - - - - - - - - - - - - -
def calc_mindiag_POU(diagLengths):
    """
    Calculate the minimum of the diagonal lengths.
    
    Cannot be calculated from distance matrix
    
    DEFINITION CALL:
        minDiag = calc_mindiag_POU()
    """
    minDiag = np.asarray(diagLengths).min()
    return minDiag
# - - - - - - - - - - - - - -
def calc_meandiag_POU(diagLengths):
    """
    Calculate the mean of the diagonal lengths.
    
    Cannot be calculated from distance matrix
    
    DEFINITION CALL:
        minDiag = calc_mindiag_POU()
    """
    meanDiag = np.asarray(diagLengths).mean()
    return meanDiag
# - - - - - - - - - - - - - -
def calc_stddiag_POU(diagLengths):
    """
    Calculate the standard deviation of the diagonal lengths.
    
    Cannot be calculated from distance matrix
    
    DEFINITION CALL:
        stdDiag = calc_stddiag_POU()
    """
    stdDiag = np.asarray(diagLengths).std()
    return stdDiag
# - - - - - - - - - - - - - -
def calc_diaglength_stats(POU):
    """
    Contains defintiion calls to get stats of diagonal length of the POU.
    
    DEFINITION CALL:
        
    """
    # print(len(POU.exterior.xy[0]))
    # Calculate diagonal lengths of this grain
    diagLengths = calc_diaglengths_POU(POU)
    # If the POU is triangle, there will be no diagonals
    # In these cases, diagLengths has been populated with -99
    # If so, then, use -99 in all diagonal parameters below
    
    if diagLengths[0]=='NoDiag':
        maxDiag  = -99
        minDiag  = -99
        meanDiag = -99
        stdDiag  = -99
    else:
        # Get maximum of the diagonal lengths
        maxDiag     = calc_maxdiag_POU(diagLengths)
        # print('Maximum diagonal length is'+str(maxDiag))
        # Get minimum of the diagonal lengths
        minDiag     = calc_mindiag_POU(diagLengths)
        # Get the mean of the diagonal lengths
        meanDiag    = calc_meandiag_POU(diagLengths)
        # Get the standard deviation of the diagonal lengths
        stdDiag     = calc_stddiag_POU(diagLengths)
        
    # print(diagLengths)
    
    return maxDiag, minDiag, meanDiag, stdDiag 
################################################################################
x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)
x, y = add_Perturb_coordinates(x, y, "RandomUniform", factors=(0.05, 0.05))
x, y = form_2D_Coord_Array(x, y)
vseeds = form_2D_VorTess_Seeds(x, y)
vo = form_2D_VorTess_Object(vseeds)
#vis_Vor(vo)
VGrains, VGrainVertices = voronoi_finite_polygons_2d(vo)
GrainPar = ini_Grain_param(paramSize = len(VGrains))
PolygBBox_VT = make_Super_Bounding_Polygon(VoronoiObject = vo,
                                           METHOD = 'useVObounds_and_offsets')
L0GS_NGrains = L0GS_calc_num_VorGrains(VGrains)

GRAINS_KeyValDetails_dict = {'keyName'      : 'countValue', # Grain ID will be the key
                             'keyNameType'  : 'int',        # Integer type
                             'keySize'      : L0GS_NGrains,   # Total number of grains
                             }
GSL0_bare = GRAINS_KeyValDetails_dict.copy()
#++++++++++++++++++++++++++++
L0GS_pdist_area         = build_GSL0_pdstr_area() # TO DO
L0GS_pdist_extperim     = build_GSL0_pdstr_extperim() # TO DO
L0GS_pdist_intperim     = build_GSL0_pdstr_intperim() # TO DO
L0GS_pdist_nextedges    = build_GSL0_pdstr_nextedges() # TO DO
L0GS_pdist_mindiag      = build_GSL0_pdstr_mindiag() # TO DO
L0GS_pdist_maxdiag      = build_GSL0_pdstr_maxdiag() # TO DO
L0GS_pdist_extvintangle = build_GSL0_pdstr_extvintangle() # TO DO
L0GS_pdist_ctovdist     = build_GSL0_pdstr_ctovdist() # TO DO
#++++++++++++++++++++++++++++
L0GS_gridnames    = build_GSL0_idname(L0GS_NGrains)
L0GS_edgeid       = build_GSL0_edgeid(L0GS_NGrains)
L0GS_vid          = build_GSL0_vid(L0GS_NGrains)
L0GS_vcoord       = build_GSL0_vcoord(L0GS_NGrains)
L0GS_ccoord       = build_GSL0_ccoord(L0GS_NGrains)

L0GS_area         = build_GSL0_area(L0GS_NGrains)
L0GS_extperim     = build_GSL0_extperim(L0GS_NGrains)
L0GS_intperim     = build_GSL0_intperim(L0GS_NGrains) # TO DO
L0GS_nextedges    = build_GSL0_nextedges(L0GS_NGrains)
L0GS_nedgesextv   = build_GSL0_nedgesextv(L0GS_NGrains) # TO DO
L0GS_mindiag      = build_GSL0_mindiag(L0GS_NGrains)
L0GS_maxdiag      = build_GSL0_maxdiag(L0GS_NGrains)
L0GS_meandiag     = build_GSL0_meandiag(L0GS_NGrains)
L0GS_stddiag      = build_GSL0_stddiag(L0GS_NGrains)
L0GS_extvintangle = build_GSL0_extvintangle(L0GS_NGrains) # TO DO
L0GS_ctovdist     = build_GSL0_ctovdist(L0GS_NGrains) # TO DO
#++++++++++++++++++++++++++++
GRn = 0
# vid = VGrains[GRn]
for vid in VGrains:
    # Get the vertices coordinate array of this grain
    thisGrainVertices = VGrainVertices[vid]
    
    
    # Make POU out of the above grain vertices
    thisGrain_POU = Polygon(thisGrainVertices)
    # Clip this polygon with the boundary of the bounding box
    thisGrain_POU_clipped_BB = thisGrain_POU.intersection(PolygBBox_VT)
    POU = thisGrain_POU_clipped_BB
    
    
    # Name the grains
    L0GS_gridnames = pop_GSL0_idname(L0GS_gridnames)
    
    
    # Populate "grain vertices index data: this grain"
    L0GS_vid       = pop_GSL0_vid(GRn, vid, L0GS_vid)
    # Populate "Number of edges: this grain" with "len(vid)-1"
    L0GS_nextedges = pop_GSL0_nextedges(GRn, len(vid)-1, L0GS_nextedges)
    # Populate "Edge data: This grain"
    
    L0GS_edgeid = pop_GSL0_edgeid(GRn, vid, L0GS_edgeid)
    
    # Get the coordinates of the vertices of this POU
    vxycoord = extract_vcoord_POU(POU)
    # Get the coordinates of the centroid of this POU
    cxycoord = extract_ccoord_POU(POU)
    
    
    # Populate "grain vertices coordinate data: this grain"
    L0GS_vcoord = pop_GSL0_vcoord(GRn, vxycoord, L0GS_vcoord)
    # Populate "grain centroid coordinate data: this grain"
    L0GS_ccoord = pop_GSL0_ccoord(GRn, cxycoord, L0GS_ccoord)
    
    
    # Extract and populate "geoemtric grain area data: this grain"
    L0GS_area = pop_GSL0_area(GRn, extract_area_POU(POU), L0GS_area)
    # Extract and populate "grain external perimeter data: this grain"
    L0GS_extperim = pop_GSL0_extperim(GRn, extract_extperim_POU(POU), L0GS_extperim)
    
    # Calculate diagonal length stats
    maxDiag, minDiag, meanDiag, stdDiag = calc_diaglength_stats(POU)
    # Populate "minimum diagonal length data: this grain"
    L0GS_mindiag  = pop_GSL0_mindiag(GRn, minDiag, L0GS_mindiag)
    # Populate "maximum diagonal length data: this grain"
    L0GS_maxdiag  = pop_GSL0_maxdiag(GRn, maxDiag, L0GS_maxdiag)
    # Populate "mean diagonal length data: this grain"
    L0GS_meandiag = pop_GSL0_meandiag(GRn, meanDiag, L0GS_meandiag)
    # Populate "standard deviation of the diagonal length data: this grain"
    L0GS_stddiag  = pop_GSL0_stddiag(GRn, stdDiag, L0GS_stddiag)
    # - - - - - - - - - - - - - - - - - - - - -
    # if POU has holes, then update internal perimeters related
    # dictionaries appropriately
    # - - - - - - - - - - - - - - - - - - - - -
    # Find distance matrix b/w vxycoord and cxycoord
    # and find the maximum, minimum and standard deviations, and
    # appropriately updates c-to-v distance dictionaries
    # - - - - - - - - - - - - - - - - - - - - -
    # Find and populate vangle related dictionaries
    # vangle means angles at individual vertices
    # - - - - - - - - - - - - - - - - - - - - -
    GRn += 1
    # - - - - - - - - - - - - - - - - - - - - -
###############################################################################
# NOW CALCULATE HISTOGRAMS
# EXTRACT THE BINS AND COUNT
# EXPRESS COUNT AS PROBABILITY
# FIND MODALITY
# FIND SKEWNESS
# FIT PDF
# GET THE PDF PARAMETERS
# FIT CDF
# GET THE CDF PARAMETERS
# POPULATE THE DICTIONARIES APPROPRIATELY
###############################################################################
def calc_L0GS_histogram_data_1d(dataname, Data_Hist_Specifier, L0GSParamData):
    """
    To calculate histogramming data of the input "data" for nbin in "nbins" data.
    
    Input
    -----
    data: dictionary of grain structure parameter:
            'area'        
            'extperim'    
            'intperim'    
            'nextedges'   
            'nedgesextv'  
            'mindiag'     
            'maxdiag'     
            'meandiag'    
            'stddiag'     
            'extvintangle'
            'ctovdist'    

    #TODO: add actual parameter names in the above examples + 1 liner for each
    
    Return
    ------
    ThisHistogramData: Dictionary of the followling key-value pairs:
        {
        'bins': bins,
        'bin_edges': bin_edges,
        'bin_widths': bin_widths,
        'binC1d': binC1d,
        'binP1d': binP1d,
        }

    DEFINITION CALL
    ---------------
        ThisHistogramData = calc_L0GS_histogram_data_1d(dataname, Data_Hist_Specifier, L0GSParamData)
    """
    # Get the number of bins
    NumBins = Data_Hist_Specifier[dataname]['NumBins']
    # Extract data from the L0GSParamData
    data     = np.asarray(list(zip(*L0GSParamData.values()))[1])
        # Next line not needed anymore!
        # if data.shape[1]==1: data = np.squeeze(data, axis = 1)
    # Make dictionaries to collect bins, binC1d and binP1d
    bins       = [] # Bin values
    bin_edges  = [] # Bin-edge pairs
    bin_widths = [] # Bin width
    binC1d     = [] # Count values for every bin width
    binP1d     = [] # Unit normalized value (probability) for every bin width
    # Convert to list
    #data = list(data)
    for histcount in range(len(NumBins)):
        # Get this nbin
        # dkey = 'nbin=%d'%NumBins[histcount]

        # Make histograms and collect count and bin edge values
        binC1d_temp, bins_temp = np.histogram(data, bins = NumBins[histcount], density = False)
        
        # L0GS_pdist_mindiag = {'nbin=%d'%binCount: hist_data for binCount in BIN_N}
        
        # Unit normalize count to get probabiltity
        binP1d_temp = binC1d_temp/binC1d_temp.sum()

        # Calculate bin edges
        bin_edges_temp = []
        for i in range(len(bins_temp)-1):
            bin_edges_temp.append([bins_temp[i], bins_temp[i+1]])

        # Calculate bin widths from bin edges
        bin_widths_temp = np.abs(bins_temp[:-1] - bins_temp[1:])
        
        # Populate lists
        bins.append(list(bins_temp))
        bin_edges.append(list(bin_edges_temp))
        bin_widths.append(list(bin_widths_temp))
        binC1d.append(list(binC1d_temp))
        binP1d.append(list(binP1d_temp))
        # Make the dictionary
        ThisHistogramData = {'dataname'  : dataname,
                             'NumBins'   : NumBins,
                             'bins'      : bins,
                             'bin_edges' : bin_edges,
                             'bin_widths': bin_widths,
                             'binC1d'    : binC1d,
                             'binP1d'    : binP1d,
                             }
    if np.asarray(binP1d[histcount]).sum()!=1:
        difference = 1-np.asarray(binP1d[histcount]).sum()
        print('NOTE: sum(bin prob)!= 1. The, diff is '+'{:.2e}'.format(difference))
        if difference<1.0e-4:
            print('This difference is < 0.01 %. Will move on!')
    return ThisHistogramData
###############################################################################
L0GS_Hist_Specifier = {'area'        : {'NumBins': [2, 5]},
                       'extperim'    : {'NumBins': [2, 4]},
                       'intperim'    : {'NumBins': [2, 4]},
                       'nextedges'   : {'NumBins': [2, 4]},
                       'nedgesextv'  : {'NumBins': [2, 4]},
                       'mindiag'     : {'NumBins': [2, 4]},
                       'maxdiag'     : {'NumBins': [2, 4]},
                       'meandiag'    : {'NumBins': [2, 4]},
                       'stddiag'     : {'NumBins': [2, 4]},
                       'extvintangle': {'NumBins': [2, 4]},
                       'ctovdist'    : {'NumBins': [2, 4]},
                       }
###############################################################################
#HistogramData = calc_L0GS_histogram_data_1d('area', L0GS_Hist_Specifier, L0GS_area)
###############################################################################
L0GS_KDE_Specifier = {'area'        : {'bandwidth': [0.25, 0.50, 0.75, 1.00]},
                      'extperim'    : {'bandwidth': [0.25, 0.50, 0.75, 1.00]},
                      'intperim'    : {'bandwidth': [0.25, 0.50, 0.75, 1.00]},
                      'nextedges'   : {'bandwidth': [0.25, 0.50, 0.75, 1.00]},
                      'nedgesextv'  : {'bandwidth': [0.25, 0.50, 0.75, 1.00]},
                      'mindiag'     : {'bandwidth': [0.25, 0.50, 0.75, 1.00]},
                      'maxdiag'     : {'bandwidth': [0.25, 0.50, 0.75, 1.00]},
                      'meandiag'    : {'bandwidth': [0.25, 0.50, 0.75, 1.00]},
                      'stddiag'     : {'bandwidth': [0.25, 0.50, 0.75, 1.00]},
                      'extvintangle': {'bandwidth': [0.25, 0.50, 0.75, 1.00]},
                      'ctovdist'    : {'bandwidth': [0.25, 0.50, 0.75, 1.00]},
                      }
###############################################################################
def build_GSL0_STAT_KDEdata(L0GS_KDE_Specifier):
    """
    To build dictionary for storing KDE data of GSL0 parameters.

    Input
    -----
        param1

    Return
    ------
        L0GS_statKDE: The dictionary with following keys for every parameter of GSL0
            'kde'
            'kde_grids'
            'kde_values'
    """
    L0GS_statKDE = dict()
    for kde_key in L0GS_KDE_Specifier.keys():

        # Get bandwidth data for this parameter
        bandwidth = L0GS_KDE_Specifier[kde_key]['bandwidth']
        # Make dictionaries to store kde_objects
        kde_objects = {'bw=%0.3f'%bandwidth[bwCount]: [] for bwCount in range(len(bandwidth))}
        # Make dictionaries to store kde_grids
        kde_grids   = kde_objects.copy()
        # Make dictionaries to store kde_values
        kde_values  = kde_objects.copy()

        # Build the dictionary of data
        L0GS_statKDE_param = {'bandwidth'  : bandwidth,
                              'kde'        : kde_objects,
                              'kde_grids'  : kde_grids,
                              'kde_values' : kde_values,
                              }

        L0GS_statKDE[kde_key] = L0GS_statKDE_param

    return L0GS_statKDE
 ###############################################################################
def build_GSL0_STAT_Peaks_paramDistr(L0GS_KDE_Specifier):
    """
    To build dictionary for storing KDE data of GSL0 parameters.
    
    build_GSL0_STAT_Peaks_paramDistr:
                                     build
                                     Grain Structure Level 0
                                     Statistics
                                     Peaks
                                     Parameter Distribution
    
    Input
    -----
        param1

    Return
    ------
        L0GS_distr_PEAKS: The dictionary with following keys for every parameter of GSL0
    """
    L0GS_distr_PEAKS = dict()
    for kde_key in L0GS_KDE_Specifier.keys():

        # Get bandwidth data for this parameter
        bandwidth = L0GS_KDE_Specifier[kde_key]['bandwidth']
        # Make dictionaries to store kde_objects
        kde_PEAKS = {'bw=%0.3f'%bandwidth[bwCount]: [] for bwCount in range(len(bandwidth))}

        # Build the dictionary of data
        L0GS_statKDE_param = {'bandwidth' : bandwidth,
                              'kde_PEAKS' : kde_PEAKS,
                              }

        L0GS_distr_PEAKS[kde_key] = L0GS_statKDE_param

    return L0GS_distr_PEAKS
###############################################################################
def build_GSL0_STAT_SkewKurt_paramDistr(L0GS_KDE_Specifier):
    """
    To build dictionary for storing KDE data of GSL0 parameters.

    build_GSL0_STAT_SkewKurt_paramDistr:
                                     build
                                     Grain Structure Level 0
                                     Statistics
                                     Skewness and Kurtosis
                                     Parameter Distribution

    Input
    -----
        param1

    Return
    ------
        L0GS_distr_Stat_Skew_Kurt: The dictionary with following keys for every parameter of GSL0
    """
    L0GS_distr_Stat_Skew_Kurt = dict()
    for kde_key in L0GS_KDE_Specifier.keys():

        # Get bandwidth data for this parameter
        #bandwidth = L0GS_KDE_Specifier[kde_key]['bandwidth']
        # Make dictionaries to store kde_objects
        #kde_skew = {'bw=%0.3f'%bandwidth[bwCount]: [] for bwCount in range(len(bandwidth))}
        #kde_kurt = {'bw=%0.3f'%bandwidth[bwCount]: [] for bwCount in range(len(bandwidth))}
        kde_skew = []
        kde_kurt = []

        # Build the dictionary of data
        L0GS_statKDE_param = {'skewness' : kde_skew,
                              'kurtosis' : kde_kurt,
                              }

        L0GS_distr_Stat_Skew_Kurt[kde_key] = L0GS_statKDE_param

    return L0GS_distr_Stat_Skew_Kurt

###############################################################################
def stat_GSL0_estimate_SciPi_KDE_param(dataname, paramData, L0GS_statKDE):
    """
    Scipy based Gaussian Kernel Density Estimation.

    stat_GSL0_estimate_SciPi_KDE_param: Statistics
                                        Grain Structure Level 0
                                        Estimate
                                        use SciPi library
                                        Kernel Density Estimation
                                        of parameterData

    Inputs
    ------
        dataname: str value of name of data being passed
        paramData: actual data being analyzed. Must be 0-dimensional
        bandwidth: Bandwidth to use in kernel density estimation

    Return
    ------
        thisKDE: A dictionary packing the following data:
                     kde
                     kde_grid_1d
                     kde_values

    Definition call
    ---------------
        thisKDE = stat_GSL0_estimate_KDE_param('area', paramData, bandwidth)
    """
    # Get the bandwidth data out from L0GS_statKDE
    bandwidth = L0GS_statKDE[dataname]['bandwidth']

    # Get data bounds
    kde_grid_min, kde_grid_max = paramData.min(), paramData.max()

    # Make data grid from data bounds and length
    kde_grid_1d = np.linspace(kde_grid_min, kde_grid_max, len(paramData))

    # Calculate, evaluate and store data pertaining KDE and paramData.
    for bwCount in range(len(bandwidth)):
        # Evaluate Gaussian KDE to get the kde_objects
        this_kde_object = KDE_scipy_gaussian(paramData, bw_method = bandwidth[bwCount])
        # Evaluate this_kde_object over kde_grid_1d to get this_kde_value
        this_kde_value = this_kde_object.evaluate(kde_grid_1d)
        # Build the bandwidth string
        bwString = 'bw=%0.3f'%bandwidth[bwCount]
        # Build the dictionary of data
        L0GS_statKDE[dataname]['kde'][bwString]        = this_kde_object
        L0GS_statKDE[dataname]['kde_grids'][bwString]  = kde_grid_1d
        L0GS_statKDE[dataname]['kde_values'][bwString] = this_kde_value

    return L0GS_statKDE
###############################################################################
HistogramData             = calc_L0GS_histogram_data_1d('area', L0GS_Hist_Specifier, L0GS_area)
L0GS_statKDE              = build_GSL0_STAT_KDEdata(L0GS_KDE_Specifier)
L0GS_distr_PEAKS          = build_GSL0_STAT_Peaks_paramDistr(L0GS_KDE_Specifier)
L0GS_distr_Stat_Skew_Kurt = build_GSL0_STAT_SkewKurt_paramDistr(L0GS_KDE_Specifier)
###############################################################################
###############################################################################
###############################################################################
###############################################################################
def vis_stat_hist_kde(axis_hist_kde,
                      dataname,
                      histData, histBins,
                      kdeBW, kdeGrids, kdeValues):
    """
    To overlay hist and kde line plots.

    Input
    -----
        param 1

    Return
    ------
        axis_hist_kde
    """
    # A set of colors for histogram
    histFaceColors = ['orangered', 'navy', 'khaki', 'darkslategray']
    histFaceAlpha  = np.flip(np.linspace(0.20, 0.40, len(histFaceColors)))

    # Plot the histogram
    histcount = 0
    for histBinValue in histBins:
         plt.hist(histData,
                  histBinValue,
                  density = True,
                  alpha   = histFaceAlpha[histcount],
                  color   = histFaceColors[histcount],
                  #ec = 'black',
                  label   = 'nbin=%d'%histBinValue)
         histcount += 1

    # Plot the KDE data
    LineStyles = ['-', ':', '--', '-.', ':']
    LineWidths = [3, 3, 3, 3, 3, 3]
    LineColors = ['#20b2aa', 'black', '#005a9c', '#013220', 'black']
    # Medium Violet Red: #c71585
    # Dodger Blue: #005a9c
    # Light Sea Green: #20b2aa
    # Dark green: #013220
    bwCount = 0
    for bwKeyString in kdeValues.keys():
        plt.plot(kdeGrids[bwKeyString], kdeValues[bwKeyString],
                 LineStyles[bwCount],
                 lw = LineWidths[bwCount],
                 color = LineColors[bwCount],
                 label = bwKeyString)
        bwCount += 1

    # Display the legend information
    plt.legend(loc='upper right')

    # Set axis labels
    axis_hist_kde.set_xlabel(dataname)
    axis_hist_kde.set_ylabel('Number of occurances')

    # Set axis title
    axis_hist_kde.set_title('Hist. & Gaussian KDE of GSL0 param: %s'%dataname)
    return axis_hist_kde
###############################################################################
# PEAK IDENTIFICATION
def stat_GSL0_identify_PEAK_KDEparam(dataname, L0GS_statKDE, L0GS_distr_PEAKS):
    """
    Identify the peaks in the paramData.

    Inputs
    ------
        dataname:
        L0GS_statKDE:

    Returns
    -------
        L0GS_distr_PEAKS
    
    Definition call
    ---------------
        L0GS_distr_PEAKS = stat_GSL0_identify_PEAK_KDEparam(dataname, L0GS_statKDE, L0GS_distr_PEAKS)
    """
    # NOTE: data will be used under the "signals" names
    # Get the grid and values for all bandwidths for the "data" specified by dataname
    allsignals_x = L0GS_statKDE[dataname][ 'kde_grids']
    allsignals_y = L0GS_statKDE[dataname]['kde_values']
    # Get the key names of the dictionary
    KEYNAMES     = L0GS_statKDE[dataname][ 'kde_grids'].keys()
    # Looping over key-value pairs, calculate peaks for KDE value distributions
    for bandwidth_name in KEYNAMES:
        # Get the grid
        signal_x = allsignals_x[bandwidth_name]

        # Get the kde values at the grid
        signal_y = allsignals_y[bandwidth_name]

        # Identify peaks
        signal_peaks, _ = find_peaks(signal_y)

        # Get the number of peaks
        npeaks = len(signal_peaks)

        # Get the data corresponding to the peaks
        signal_x_at_peaks = [signal_x[signal_peaks[i]] for i in range(npeaks)]
        signal_y_at_peaks = [signal_y[signal_peaks[i]] for i in range(npeaks)]

        L0GS_distr_PEAKS[dataname]['kde_PEAKS'][bandwidth_name] = [signal_x_at_peaks, signal_y_at_peaks]
    
    return L0GS_distr_PEAKS
###############################################################################
def vis_stat_kde_Peaks(axis_hist_kde, dataname, L0GS_distr_PEAKS):
    """
    Visualize the data peaks overlaid on KDE and histogram

    Input
    -----
        param 1

    Return
    ------
        axis_hist_kde
    
    Definition call
    ---------------
        axis_hist_kde = vis_stat_kde_Peaks(axis_hist_kde,
                                           dataname,
                                           L0GS_distr_PEAKS)
    """

    # Get the key names of the dictionary
    KEYNAMES = L0GS_distr_PEAKS[dataname][ 'kde_PEAKS'].keys()
    # Looping over key-value pairs, calculate peaks for KDE value distributions
    bwCount = 0
    for bandwidth_name in KEYNAMES:
        # Get the Peak Data
        PeakData = L0GS_distr_PEAKS[dataname]['kde_PEAKS'][bandwidth_name]
        # Plot the peak data
        if len(PeakData[0])==1:
            plt.plot(PeakData[0], PeakData[1],
                     marker = 'o',
                     markerfacecolor = 'yellow',
                     markeredgecolor = 'yellow',
                     markersize = 8)
        else:
            for peakCount in range(len(PeakData[0])):
                plt.plot(PeakData[0][peakCount], PeakData[1][peakCount],
                         marker = 's',
                         markerfacecolor = 'lime',
                         markeredgecolor = 'lime',
                         markersize = 8)
        bwCount += 1
    return axis_hist_kde
###############################################################################
def stat_GSL0_estimate_skew_kurt_param(dataname, paramData, L0GS_distr_Stat_Skew_Kurt):
    """
    Estimation of skewness and kurtosis of data.

    stat_GSL0_estimate_skew_kurt_param: Statistics
                                        Grain Structure Level 0
                                        Estimate
                                        Skewness
                                        Kurtosis
                                        of parameterData

    Inputs
    ------
        dataname: str value of name of data being passed
        paramData: actual data being analyzed. Must be 0-dimensional
        bandwidth: Bandwidth to use in kernel density estimation

    Return
    ------
        thisKDE: A dictionary packing the following data:
                     kde
                     kde_grid_1d
                     kde_values

    Definition call
    ---------------
        L0GS_distr_Stat_Skew_Kurt = stat_GSL0_estimate_skew_kurt_param(dataname,
                                                                       paramData,
                                                                       L0GS_distr_Stat_Skew_Kurt)
    
    Examples
    --------
    EXAMPLE 1: Positive skewness
    sampledata = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2]
    plt.hist(sampledata)
    |
    | #
    | #
    | #                  #
    | #                  #               #
    |______________________________________________
    0    0.25          1.00           1.75     2.00
    skew(sampledata)
    # +1.54 >>>
        # Tail of the data is to the right
        # MEAN is to the right of MEDIAN
    example1kurtosis = kurtosis(sampledata)
    
    EXAMPLE 2: Negative skewness
    sampledata = [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3]
    plt.hist(sampledata)
    |
    |                            #
    |                            #
    |                            #
    | #          #               #              #
    |______________________________________________
    0           1.00            2.00           3.00
    skew(sampledata)
    # -1.20 >>>
        # Tail of the data is to the left
        # MEAN is to the left of MEDIAN
    
    EXAMPLE 3: Zero skewness
    sampledata = [0, 1,1, 2,2,2, 3,3,3,3,3, 4,4,4, 5,5, 6]
    plt.hist(sampledata)
    |             #
    |             #
    |         #   #   #
    |     #   #   #   #   #
    | #   #   #   #   #   #   #
    |__________________________
      0   1   2   3   4   5   6
    skew(sampledata)
    # 0.00 >>>
        # MEAN is the MEDIAN
    
    kurtosis(sampledata)
    
    EXAMPLE 4: w.r.t EXAMPLE 1: understanding kurtosis
    sampledata = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2]
    plt.hist(sampledata)
    | #
    | #
    | #
    | #                  #
    | #                  #               #
    |______________________________________________
    0    0.25          1.00           1.75     2.00
    skew(sampledata)
    # +1.81 >>>
        # Tail to the right and MEAN is MEDIAN's right side
    example4kurtosis = kurtosis(sampledata)
    
    example1kurtosis was 1.066 ---- example4kurtosis was 2.001
    INTERPRETATION: Data in example1, has fewer outliers than in example2
    """
    #plt.hist(paramData)
    L0GS_distr_Stat_Skew_Kurt[dataname]['skewness'] = skew(paramData)
    L0GS_distr_Stat_Skew_Kurt[dataname]['kurtosis'] = kurtosis(paramData)
    print("||||{:s}||||  SKEWNESS = {:4.4f}.....KURTOSIS = {:4.4f}".format(dataname,
                                                                           L0GS_distr_Stat_Skew_Kurt[dataname]['skewness'],
                                                                           L0GS_distr_Stat_Skew_Kurt[dataname]['kurtosis']
                                                                           )
          )
    #return skewness_value, kurtosis_value
    return L0GS_distr_Stat_Skew_Kurt
###############################################################################
def vis_annotation_stat_kde_Peaks(axis_hist_kde,
                                  dataname,
                                  L0GS_distr_Stat_Skew_Kurt, L0GS_NGrains):
    """
    Annotate data Skewness and Kurtosis value data on the plot overlaid on KDE and histogram

    Input
    -----
        param 1

    Return
    ------
        axis_hist_kde
    
    Definition call
    ---------------
        axis_hist_kde = vis_annotation_stat_kde_Peaks(axis_hist_kde,
                                          dataname,
                                          L0GS_distr_Stat_Skew_Kurt)
    """

    # Get the Skewness to S and Kurtosis to K of the data pertaining "dataname"
    S = L0GS_distr_Stat_Skew_Kurt[dataname]['skewness']
    K = L0GS_distr_Stat_Skew_Kurt[dataname]['kurtosis']
    
    # Get the x limits of the figure plot domain
    xlim          = axis_hist_kde.get_xlim()
    xmin, xheight = xlim[0], xlim[1]-xlim[0]
    text_Ng_x     = xmin + 0.05*xheight
    text_S_x      = xmin + 0.05*xheight
    text_K_x      = xmin + 0.05*xheight
    # Get the y limits of the figure plot domain
    ylim          = axis_hist_kde.get_ylim()
    ymin, yheight = ylim[0], ylim[1]-ylim[0]
    text_Ng_y     = ymin + 0.9*yheight
    text_S_y      = ymin + 0.8*yheight
    text_K_y      = ymin + 0.7*yheight
    
    # overlay the text annotation of S and K
    plt.text(text_Ng_x, text_Ng_y, 'Ng = %d'%L0GS_NGrains)
    plt.text(text_S_x, text_S_y, 'S = %1.2f'%S)
    plt.text(text_K_x, text_K_y, 'K = %1.2f'%K)
    
    
    return axis_hist_kde
###############################################################################
###############################################################################
# TODO: write lmfit calls to estimate peak width
# TODO: write lmfit calls to fit multiple Gaussians
# TODO: get the distribution fit parameters
# TODO: Get the display of L0GS
###############################################################################
###############################################################################
     # LEVEL1 GRAIN STRUCTURE
# TODO: Perturb the Voronoi Grain Structure boundaries as in PPT
# TODO: Perturb the Voronoi Grain Structure boundaries with geometric perturbations as in a CuCrZr grain structure
###############################################################################
     # LEVEL2 GRAIN STRUCTURE
# TODO: Perturb the Voronoi Grain Structure
# TODO: 
###############################################################################
def vis_L0GS_GrainStructure(axis_GSL0_vis):
    """
    Visualization of the grain structure.
    
    Input
    -----
        axis_GSL0_vis: Axis object of the current figure
        
    Return
    ------
        axis_GSL0_vis: Updated axis object of the current figure
        
    Definition call
    ---------------
        vis_L0GS_GrainStructure(axis_GSL0_vis)
    """
    grainFaceColors = ['orangered', 'navy', 'khaki', 'darkslategray']
    for i in L0GS_vcoord.keys():
        thisGrainXY = L0GS_vcoord[i][1]
        thisGrainX  = thisGrainXY[:,0]
        thisGrainY  = thisGrainXY[:,1]
        plt.fill(thisGrainX, thisGrainY,
                 facecolor  = grainFaceColors[0],
                 edgecolor  = 'black',
                 linestyle  = '-',
                 linewidth  = 1,
                 alpha      = 1.0)
    plt.axis('equal')  # square, equal
    #fig.tight_layout()
    plt.show()
    
    return axis_GSL0_vis
###############################################################################
axis_GSL0_vis = plt.figure(dpi = 100)
axis_GSL0_vis = vis_L0GS_GrainStructure(axis_GSL0_vis)
###############################################################################
L0GS = {'area'        : L0GS_area,
        'extperim'    : L0GS_extperim,
        'intperim'    : L0GS_intperim,
        'nextedges'   : L0GS_nextedges,
        'nedgesextv'  : L0GS_nedgesextv,
        'mindiag'     : L0GS_mindiag,
        'maxdiag'     : L0GS_maxdiag,
        'meandiag'    : L0GS_meandiag,
        'stddiag'     : L0GS_stddiag,
        'extvintangle': L0GS_extvintangle,
        'ctovdist'    : L0GS_ctovdist,
        }
###############################################################################
histBins = [10, 50]
for dataname in L0GS.keys():
    paramData = L0GS[dataname]
    paramData = np.asarray(list(zip(*paramData.values()))[1])
    if len(paramData.flatten())!=0:

        # DATA PREPERATION
        # Sub-select the paramData by rejecting invalid entries
            # NOTE: Invalud entries were populated with -99
        paramData        = paramData[paramData!=-99]
        # Estimate the KDE parameters for each KDE
        L0GS_statKDE     = stat_GSL0_estimate_SciPi_KDE_param(dataname, paramData, L0GS_statKDE)
        # Get the bandwidth values used to compute KDE
        kdeBW            = L0GS_statKDE[dataname]['bandwidth']
        # Get the 1D-grid used to compute the KDE
        kdeGrids         = L0GS_statKDE[dataname]['kde_grids']
        # Get the KDE values computed on the 1D-grid
        kdeValues        = L0GS_statKDE[dataname]['kde_values']
        # Find peaks in the KDE data for different bandwidths considered
        L0GS_distr_PEAKS = stat_GSL0_identify_PEAK_KDEparam(dataname, L0GS_statKDE, L0GS_distr_PEAKS)
        # Estimate Skewness and Kurtosis of the data
        L0GS_distr_Stat_Skew_Kurt = stat_GSL0_estimate_skew_kurt_param(dataname, paramData, L0GS_distr_Stat_Skew_Kurt)

        # DATA VISUALIZATION
        # Visualize the overlaid histogram and kde for different bandwidth values
        plt.figure(dpi   = 200)
        axis_hist_kde    = plt.axes()
        axis_hist_kde    = vis_stat_hist_kde(axis_hist_kde, dataname, paramData, histBins, kdeBW, kdeGrids, kdeValues)
        # Visualize the peak locations on each KDE
        axis_hist_kde    = vis_stat_kde_Peaks(axis_hist_kde, dataname, L0GS_distr_PEAKS)
        # Annotate text for skewness and kurtosis of the parameter data
        axis_hist_kde    = vis_annotation_stat_kde_Peaks(axis_hist_kde, dataname, L0GS_distr_Stat_Skew_Kurt, L0GS_NGrains)





print('Maximum y is at %2.2f'%y.max())






#import DefDap as DP



