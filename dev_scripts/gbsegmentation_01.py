import cv2
import numpy as np
import gmsh
import pyvista as pv
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter
from meshpy.triangle import MeshInfo, build
from upxo.ggrowth.mcgs import mcgs
# =====================================================================
pxt = mcgs()
pxt.simulate()
pxt.detect_grains()
tslice = 8  # Temporal slice number
pxt.char_morph_2d(tslice)
hgrid = pxt.gs[tslice].xgr
vgrid = pxt.gs[tslice].ygr
mcstates = pxt.gs[tslice].s
nstates = pxt.uisim.S


pxt.gs[8].g[1]['grain']


# pxt.gs[tslice].scale(sf=2)
pxt.gs[tslice].export_ctf(r'D:\export_folder', 'sunil')
pxt.gs[tslice].find_grain_boundary_junction_points()
# -----------------------------
#plt.figure()
#plt.imshow(pxt.gs[tslice].lgi)
#for r, c in zip(np.where(pxt.gs[tslice].gbjp)[0],
#                np.where(pxt.gs[tslice].gbjp)[1]):
#	plt.plot(c, r, 'k.')
# =====================================================================
pxt.gs[tslice].xomap_set(map_type='ebsd',
                         path=r"D:/export_folder/",
                         file_name_with_ext=r"sunil.ctf")

pxt.gs[tslice].xomap_prepare()
pxt.gs[tslice].xomap_extract_features()

pxt.gs[tslice].find_grain_boundary_junction_points(xorimap=True)
pxt.gs[tslice].xomap.gbjp

# ---------------------------------
# TEST SCRIPTS TO TEST EQAUALITY OF GRAIN ARTEAS
# TEST - 1: test against upxo grains
samples = [pxt.gs[8].g[i]['grain'] for i in pxt.gs[8].g.keys()]
[_.npixels for _ in samples]
upxo_sample = samples[0]
upxo_sample == samples
upxo_sample != samples
# TEST - 2: test against numbers
upxo_sample == [16, 17, 8, 16, 2]
upxo_sample != [16, 17, 8, 16, 2]
# TEST - 3: test against defDap grains
samples = pxt.gs[tslice].xomap.map.grainList
[len(_.coordList) for _ in samples]
upxo_sample == samples
upxo_sample != samples
# ---------------------------------

pxt.gs[tslice].xomap.map.grains
pxt.gs[tslice].xomap.map.eulerAngleArray
pxt.gs[tslice].xomap.map.quatArray
pxt.gs[tslice].xomap.map.grainList[0].coordList
# Above is equivalent to:
# COde here
plt.figure()
plt.imshow(pxt.gs[tslice].xomap.map.grains)
for r, c in zip(np.where(pxt.gs[tslice].xomap.gbjp)[0],
                np.where(pxt.gs[tslice].xomap.gbjp)[1]):
	plt.plot(c, r, 'k.')
# =====================================================================
#    INTEGRATION SUCCESSFULL TILL HERE:  "MCGS2_TEMPORAL_SLICE.PY"
# =====================================================================
from upxo.interfaces.defdap.importebsd import ebsd_data as ebsd
fileName = r"C:\Development\M2MatMod\upxo_packaged\upxo_private\src\upxo\_written_data\_ctf_export_2dmcgs\sunil"
fileName = r"D:\export_folder\sunil"
# fileName = r"C:\Development\M2MatMod\mtex\EBSD_scan_data\map"
gs = ebsd(fileName)
# gs.map.filterData(misOriTol=5) # Kuwahara filter
gs.map.buildQuatArray()
gs.map.findBoundaries(boundDef=10)
gs.map.findGrains(minGrainSize=1)
# gs.map.plotGrainMap()
# gs.map.plotBoundaryMap()
# gs.map.plotPhaseMap()
# gs.map.plotBandContrastMap()
# gs.map.plotEulerMap()
gs.map.buildNeighbourNetwork()
gs.map.findBoundaries()
# # DATA in DefDAP:
gs.map.grains # ---- > Equivalent to pxt.gs[8].lgi
gs.map.quatArray
gs.map.eulerAngleArray
# ACCESS GRAINS:
gs.map.grainList[0].coordList
# =====================================================================
# FIND JUNCTION POINTS:

gids = np.unique(pxt.gs[tslice].xomap.map.grains)
BJP = {gid: None for gid in gids}  # Boundary Junction Points
for gid in gids:
	BJP[gid] = np.argwhere(pxt.gs[tslice].xomap.gbjp*(pxt.gs[tslice].xomap.map.grains==gid))

plt.figure()
plt.imshow(pxt.gs[tslice].xomap.map.grains)
for gid in gids:
	bjpy, bjpx = BJP[gid].T
	plt.plot(bjpx, bjpy, 'ro', mfc='none')
# =====================================================================
# FIND THE NEIGHBOURING GRAIN IDs OF EVERY GRAIN
from scipy.ndimage import binary_dilation, generate_binary_structure
gids = gids[gids != 0]  # Exclude background or border if labeled as 0 or another specific value
# Dictionary to hold the neighbors for each grain ID
grain_neighbors = {gid: None for gid in gids}
# Generate a binary structure for dilation (connectivity)
struct = generate_binary_structure(2, 1)  # 2D connectivity, direct neighbors
for gid in gids:
    # Create a binary mask for the current grain
    mask = gs.map.grains == gid
    # Dilate the mask to include borders with neighbors
    dilated_mask = binary_dilation(mask, structure=struct)
    # Find unique neighboring grain IDs in the dilated area, excluding the current grain ID
    neighbors = np.unique(gs.map.grains[dilated_mask & ~mask])
    # Update the dictionary, excluding the current grain ID from its neighbors if present
    grain_neighbors[gid] = list(set(neighbors) - {gid})
# =====================================================================
# CALCULATE GRAIN AREAS
areas = np.array([len(gs.map.grainList[gid-1].coordList) for gid in gids])
# FIND OUIT  GRAIN OF INTEREST, WHCH IS THE LARGEST GRAIN
GOI_amax = int(np.argwhere(areas == areas.max()).squeeze())
GOI = GOI_amax
GOI  # GRAIN OF INTEREST
NEIGS = grain_neighbors[GOI+1]  # nEIGHBOURS LIST OF LARGEST GRAIN
# =====================================================================
# IDENTIFY THE GRAIN BOUNDARY SEGMENTS FOR EVERY GID-NEIGH PAIR
gbseg = {gid: {} for gid in gids}
for gid in gids:
	for neigh in grain_neighbors[gid]:
		gbseg[gid][neigh] = None

gbseg = {gid: {} for gid in gids}
struct = generate_binary_structure(2, 1)  # 2D connectivity, direct neighbors
for gid in gids:
    # Binary mask for the current grain
    gid_mask = gs.map.grains == gid

    for neigh in grain_neighbors[gid]:
        # Binary mask for the neighbor
        neigh_mask = gs.map.grains == neigh
        # Dilate each mask
        dilated_gid_mask = binary_dilation(gid_mask, structure=struct)
        dilated_neigh_mask = binary_dilation(neigh_mask, structure=struct)
        # Intersection of dilated masks with the original of the other to find boundary
        boundary_gid_to_neigh = np.where((dilated_gid_mask & neigh_mask))
        boundary_neigh_to_gid = np.where((gid_mask & dilated_neigh_mask))
        # Store the boundary locations as a list of tuples (y, x) positions
        # Choose boundary_gid_to_neigh or boundary_neigh_to_gid based on your specific needs or combine them
        gbseg[gid][neigh] = list(zip(boundary_gid_to_neigh[0], boundary_gid_to_neigh[1]))

# Plot to validate
A = deepcopy(gs.map.grains)
locs = gs.map.grainList[GOI].coordList
centroid = np.array(locs).T.sum(axis=1)/len(locs)
_ = A==GOI+1  # Mask of largest grain
_ = np.logical_or(_, A==grain_neighbors[GOI+1][0])
if len(grain_neighbors[GOI+1])>1:
	for neigh in grain_neighbors[GOI+1][1:]:
		_ = np.logical_or(_, A==neigh)

NEIGHBOURHOOD = gs.map.grains*_

plt.imshow(NEIGHBOURHOOD)
plt.plot(centroid[0], centroid[1], 'ks', ms=8, mfc='black')
for neigh in gbseg[GOI+1]:
	locs = np.array(gbseg[GOI+1][neigh]).T
	plt.plot(locs[1], locs[0], 'r.')
# =====================================================================
binary_image = np.where(gs.map.grains == GOI+1, 255, 0).astype(np.uint8)
contours, __ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
gb_points = contours[0].T.squeeze()

plt.imshow(gs.map.grains*_)
plt.plot(centroid[0], centroid[1], 'ks', ms=8, mfc='black')
for neigh in gbseg[GOI+1]:
	locs = np.array(gbseg[GOI+1][neigh]).T
	plt.plot(locs[1], locs[0], 'r.')
plt.plot(gb_points[0], gb_points[1], 'k.', mfc='none')
# =====================================================================
# EXTRACT DENSE GRAIN BOUNDARY POINTS - ARRANGED ANTI-CLOCKWISE
gb_points_list = list(zip(gb_points[0], gb_points[1]))
def inout_line(xstart, ystart, xend, yend, x1, y1, tolerance=1e-6):
    # Define the start and end points of the line segment
    p1 = (xstart, ystart)
    p2 = (xend, yend)
    p = (x1, y1)  # The point to check

    # Calculate distances
    d1 = np.sqrt((x1 - xstart)**2 + (y1 - ystart)**2)
    d2 = np.sqrt((x1 - xend)**2 + (y1 - yend)**2)
    d_line = np.sqrt((xend - xstart)**2 + (yend - ystart)**2)

    # Check if the point lies on the line segment
    return abs((d1 + d2) - d_line) < tolerance and d1 < d_line and d2 < d_line, d1
# =====================================================================
# =====================================================================
locs = gs.map.grainList[GOI-1].coordList
locs_extra = [point for point in locs if point not in gb_points_list]
segments = [[gbp] for gbp in gb_points_list]

for n in range(len(gb_points_list)):
	# ------------------------------------
	# GET THE SEARCH LINE
	xstart, ystart = gb_points_list[n][0], gb_points_list[n][1]
	if n == len(gb_points_list)-1:
		xend, yend = gb_points_list[0][0], gb_points_list[0][1]
	else:
		xend, yend = gb_points_list[n+1][0], gb_points_list[n+1][1]
	# ------------------------------------
	# INITIATE AND POPULATE
	inside = [False for _ in locs_extra]
	dist_start_loc = [None for _ in locs_extra]
	# ------------------------------------
	# FIND POINTS INSIDE AND THE DISTANCE TO STARTING POINT
	for i, (_x_, _y_) in enumerate(locs_extra):
		inside[i], dist_start_loc[i] = inout_line(xstart, ystart, xend, yend, _x_, _y_)
	# ------------------------------------
	# FIND INDICES OF ALL POINTS WHICH ARE INSIDE START AND END
	inside_index = np.where(inside)[0]
	# ------------------------------------
	# GET COORDINATES OF POINTS INSIDE START AND END POINTS.
	# Coordinates may not be sorted, distance wise from START
	if len(inside_index) > 0:
		locs_inside = [locs_extra[i] for i in inside_index]
	# ------------------------------------
	# GET SORTED COORDINATES OF POINTS INSIDE START AND END POINTS.
	if len(inside_index)>0:
		# GET DISTANCES TO START OF ALL INSIDE POINTS
		dist_inside = [dist_start_loc[i] for i in inside_index]
		# GET INDICES OF SORTED DISTANCES
		dist_sort_index = np.argsort(dist_inside)
		# BUILD THE SORTED LIST OF POINTS INSIDE THE LINE
		locs_inside_sorted = [locs_inside[i] for i in dist_sort_index]
		#++++++++++++++++++
		# UPDATE THE NEXT SEARCH BASE OF POINTS
		locs_extra = [point for point in locs_extra if point not in locs_inside_sorted]
		for _ in locs_inside_sorted:
			segments[n].append(_)
	else:
		locs_inside_sorted = None
# =====================================================================
plt.imshow(NEIGHBOURHOOD)
prime_count = 0
for seg in segments:
	x, y = np.array(seg).T
	sec_count = 0
	# ========================
	if len(x) == 1:
		plt.plot(x, y, 'o', ms=10)
		plt.text(x, y, str(prime_count), fontsize=12)
	else:
		plt.plot(x[0], y[0], 'o', ms=10)
		plt.plot(x[1:], y[1:], 'o', ms=6)
		# --------------------
		plt.text(x[0], y[0], str(prime_count), fontsize=12)
		for _x, _y in zip(x[1:], y[1:]):
			plt.text(_x, _y, str(sec_count), fontsize=8)
			sec_count += 1
	prime_count += 1
# =====================================================================
gbp = []
for seg in segments:
	for sg in seg:
		gbp.append(sg)

gbp = np.array(gbp).T
# =====================================================================
plt.imshow(NEIGHBOURHOOD)
gbp_listx, gbp_listy = list(gbp[0]), list(gbp[1])
gbp_listx.append(gbp[0][0])
gbp_listy.append(gbp[1][0])
plt.plot(gbp_listx, gbp_listy, '-k', lw=3)
# =====================================================================
# SMOOTHING THE GRAIN BOUNDARIES
points = np.array([gbp_listx, gbp_listy]).T
# METHGOD - 1
from scipy.interpolate import splprep, splev
tck, u = splprep([points[:,0], points[:,1]], s=0.00, per=True)
# S: amount of smoothig. Smaller value makes curve closer to original data
new_points = splev(np.linspace(0, 1, 1000), tck)
plt.figure(figsize=(10, 5))
plt.plot(points[:,0], points[:,1], 'ro-', label='Original Points')
plt.plot(new_points[0], new_points[1], 'b-', label='Smoothed Curve')
plt.legend()
plt.show()
# METHOD - 2
'''
Moving Average (Simple or Weighted)
A moving average smooths the data by replacing each point with the average of it and its neighbors. A weighted moving average assigns different weights to the points, usually giving more importance to the central points in the window.
'''
def moving_average(points, window_size=3):
    """Smooth points by applying a simple moving average."""
    smoothed_points = np.convolve(points, np.ones(window_size)/window_size, mode='same')
    return smoothed_points

# Assuming points is a 1D numpy array of your data
smoothed_x = moving_average(points[:, 0], window_size=5)
smoothed_y = moving_average(points[:, 1], window_size=5)
plt.plot(points[:, 0], points[:, 1], 'ro-', label='Original Points')
plt.plot(smoothed_x, smoothed_y, 'b-', label='Smoothed Curve')
plt.legend()
plt.show()
# METHOD - 3
'''
Gaussian Filter
A Gaussian filter smooths the data by applying a Gaussian kernel, effectively performing a weighted average where points closer to the center have higher weights. This method is useful for smoothing without drastically altering the data's shape.
'''
from scipy.ndimage import gaussian_filter
smoothed_x = gaussian_filter(points[:, 0], sigma=0.5)
smoothed_y = gaussian_filter(points[:, 1], sigma=0.5)
plt.plot(points[:, 0], points[:, 1], 'ro-', label='Original Points')
plt.plot(smoothed_x, smoothed_y, 'b-', label='Smoothed Curve')
plt.legend()
plt.show()
# METHOD - 4
'''
Savitzky-Golay Filter
The Savitzky-Golay filter smooths the data by fitting successive subsets of adjacent data points with a low-degree polynomial. It's particularly effective for preserving features like peaks or valleys.
'''
from scipy.signal import savgol_filter
# Window size must be odd, degree of polynomial must be less than window size
smoothed_x = savgol_filter(points[:, 0], window_length=11, polyorder=4)
smoothed_y = savgol_filter(points[:, 1], window_length=11, polyorder=4)
plt.plot(points[:, 0], points[:, 1], 'ro-', label='Original Points')
plt.plot(smoothed_x, smoothed_y, 'b-', label='Smoothed Curve')
plt.legend()
plt.show()
# METHOD - 5
'''
Kernel smoothing is another technique for smoothing data, often used to create a smooth curve from a set of discrete points. It works by averaging the points in the vicinity of each target point, weighted by a kernel function that decreases with distance from the target point. A common choice for the kernel function is the Gaussian kernel.
'''
def gaussian_kernel_smoothing(x, y, bandwidth):
    """
    Apply Gaussian kernel smoothing to the curve defined by (x, y).

    Parameters:
    - x: x-coordinates of the points.
    - y: y-coordinates of the points.
    - bandwidth: bandwidth of the Gaussian kernel.

    Returns:
    - smooth_x: x-coordinates of the smoothed curve.
    - smooth_y: y-coordinates of the smoothed curve.
    """
    # Create fine grid for the output curve
    smooth_x = np.linspace(np.min(x), np.max(x), 1000)
    smooth_y = np.zeros(smooth_x.shape)
    # Calculate the Gaussian kernel for each point on the fine grid
    for i, xi in enumerate(smooth_x):
        weights = np.exp(-0.5 * ((xi - x) / bandwidth) ** 2)
        weights /= weights.sum()
        smooth_y[i] = np.sum(weights * y)
    return smooth_x, smooth_y
x, y = points[:, 0], points[:, 1]
# Apply Gaussian kernel smoothing
bandwidth = 0.2  # Adjust the bandwidth to control the level of smoothing
smooth_x, smooth_y = gaussian_kernel_smoothing(x, y, bandwidth)
# Plotting
plt.figure(figsize=(10, 5))
plt.plot(x, y, 'ro', label='Original Points')
plt.plot(smooth_x, smooth_y, 'b-', label='Smoothed Curve')
plt.legend()
plt.show()
# METHOD - 6
'''
Kernel Smoothing using Statsmodels
The statsmodels package offers non-parametric kernel smoothing functionality. The KernelReg class can be used for this purpose.
'''

# METHOD - 7
'''
https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-for-a-dataset

It uses least squares to regress a small window of your data onto a polynomial, then uses the polynomial to estimate the point in the center of the window. Finally the window is shifted forward by one data point and the process repeats. This continues until every point has been optimally adjusted relative to its neighbors. It works great even with noisy samples from non-periodic and non-linear sources.

https://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay:
The Savitzky Golay filter is a particular type of low-pass filter, well adapted for data smoothing. For further information see: http://www.wire.tu-bs.de/OLDWEB/mameyer/cmr/savgol.pdf (or http://www.dalkescientific.com/writings/NBN/data/savitzky_golay.py for a pre-numpy implementation).
'''

# =====================================================================
plt.imshow(NEIGHBOURHOOD)
for i, seg in enumerate(segments, start=0):
	i = 9
	seg = segments[i]
	if len(seg) > 1:
		X, Y = np.array(seg).T
		plt.plot(X[:2], Y[:2], 'k', linewidth=3)
		plt.plot(X[1:], Y[1:], 'y', linewidth=1)
		if i != len(segments)-1:
			seg_next = segments[i+1]
		else:
			seg_next = segments[0]
		X, Y = np.array([seg[-1], seg_next[0]]).T
		plt.plot(X, Y, 'r', linewidth=1)
	elif len(seg) == 1:
		if i != len(segments)-1:
			seg_next = segments[i+1]
		else:
			seg_next = segments[0]
		X, Y = np.array([seg[0], seg_next[0]]).T
		plt.plot(X, Y, 'r', linewidth=1)
# =====================================================================
from shapely.geometry import Polygon
coordinates = list(zip(gbp_listx, gbp_listy))
polygon = Polygon(coordinates)
polygon
# =====================================================================
# MESH THE SHAPELY POLYGON
# Initialize Gmsh
gmsh.initialize()
gmsh.model.add("polygon")
# Coordinates from gbp_listx and gbp_listy
coordinates = list(zip(gbp_listx, gbp_listy))
# Adding points for polygon vertices
for i, (x, y) in enumerate(coordinates, start=1):
    gmsh.model.geo.addPoint(x, y, 0, meshSize=0.1, tag=i)
# Connecting points with lines
for i in range(1, len(coordinates)):
    gmsh.model.geo.addLine(i, i+1, tag=i)
gmsh.model.geo.addLine(len(coordinates), 1, tag=len(coordinates))  # Closing the loop
# Creating a curve loop and a plane surface
loop = gmsh.model.geo.addCurveLoop(list(range(1, len(coordinates) + 1)))
surface = gmsh.model.geo.addPlaneSurface([loop])
gmsh.model.geo.synchronize()
# Generating the mesh
gmsh.model.mesh.generate(2)
# Optional: Save the mesh to a file
gmsh.write("polygon.msh")
# Launch the Gmsh GUI to view the mesh
visualize_in_gmsh_GUI = False
if visualize_in_gmsh_GUI:
    gmsh.fltk.run()
gmsh.finalize()
# =====================================================================
# Path to the mesh file generated by Gmsh
mesh_file_path = "polygon.msh"
# Read the mesh file
mesh = pv.read(mesh_file_path)
# Create a plotter
plotter = pv.Plotter()
# Add the mesh to the plotter
plotter.add_mesh(mesh, show_edges=True, color="lightblue")
# Display the plotter
plotter.show()
# =====================================================================
