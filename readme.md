# INFORMATION
## Major capabilities
1. Complex 2D and 3D grain structure generation, analysis, meshing and export
    a. Voronoi tessellations grain structures
    b.	Monte-Carlo simulation based grain structures with grain growth kinetics
    c.	(Voronoi : Monte-Carlo simulation) based grain structures
    d.	(Experimental : Monte-Carlo simulation) based grain structures
2. Qualification of morphological and crystallographic textures for representativeness
3. Qualification of mechanical behaviour of CPFE simulation for representativeness
## Post-processing capabilities:
We should be able to post-process the following from UPXO:
1. Calculate grain volume averaged quantities
2. Create texture components based element sets, grain volume binned element sets, grain aspect ratio binned element sets, internal grains element sets, boundary grains element sets
3. Identify grains within element sets with field variable values within a certain bound
4. Calculate and display grain boundary and core zones
5. Calculate and display grain boundary edge zones
6. Calculate and display grain boundary edge junction point zones
## Core modules
1. Polyxtal
2. mcgs
3. Point2d
4. MultiPoint2d
5. Edge2d
6. MulEdge2d
7. Eops
## Structure
Object oriented.
## Dependencies
The major dependencies of UPXO are: Numpy, Scipy, gmsh, PyVista, vedo, Shapely, VTK, Scikit-Image, OpenCV, Pandas, Matplotlib, Seaborn, numba
The minor dependencies


# Finite element export details
## Abaqus export
1. FE Types supported: Standard ABAQUS elements
2. FE Mesh types supported:
    a. Pixel/voxel based non-conformal FE mesh with standard Quad, Tri, Hex and Tet elements for both Voronoi tessellation based and Monte-Carlo grain structure
    b.	Conformal Quad and Tri
## Moose
To do
## Damask
To do


# TRACKING
## Development
1. Available in 2D: Generation, analysis, meshing (conformal), export (.ctf, .inp)
2. Available in 3D: Generation
3. Available in 2D: Generation, analysis, meshing (conformal, non-conformal), export (.ctf, .inp)
4. Available in 3D: Generation, analysis, meshing (non-conformal), export
5. Available in 2D: Generation, analysis, meshing, export
6. Available in 3D: None
7. Available in 2D: Generation, analysis, meshing, export
