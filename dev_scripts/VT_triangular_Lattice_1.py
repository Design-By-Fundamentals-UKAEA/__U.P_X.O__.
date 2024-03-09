# =============================================================================
# #-----------------------------------------------------------------
# =============================================================================
import random
from shapely.ops import voronoi_diagram
from shapely.geometry import LineString, MultiPolygon, Polygon, Point, MultiPoint, LinearRing
from shapely.ops import polygonize, split, SplitOp, voronoi_diagram
from shapely import affinity
import numpy as np
import matplotlib.pyplot as plt
#-----------------------------------------------------------------
def clip_Voronoi_Tess_BoundBox(VGrains, Boundary_OBJ_VT):
    # FUNCTION IN TESS_SCIPY
    from shapely.ops import voronoi_diagram
    from shapely.geometry import LineString, MultiPolygon, Polygon, Point, MultiPoint, LinearRing
    from shapely.ops import polygonize, split, SplitOp, voronoi_diagram
    L0GS_PXTAL_units = []
    GRn_actual = 0
    for grain in VGrains:
        # Clip this polygon with the boundary of the bounding box
        thisGrain_POU_clipped_BB = grain.intersection(Boundary_OBJ_VT)
        POU = thisGrain_POU_clipped_BB
        if POU.area > 0:
            GRn_actual += 1
            L0GS_PXTAL_units.append(POU)
    return L0GS_PXTAL_units
#-----------------------------------------------------------------
# Set bounding limits and make bounding rectangle
xmin = -1
xmax = 2
ymin = -3
ymax = 2
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
# Number of points along x and y
nx = 10
ny = 10
# Make grid
x = np.linspace(xmin, xmax, nx)
xincr = x[1] - x[0]
y = np.linspace(ymin, ymax, ny)
__xy = np.meshgrid(x, y)
x, y = __xy[0], __xy[1]
for incr_loc in np.arange(1, np.shape(x)[0], 2):
    x[incr_loc] += xincr*0.5
x_flat = x.flatten()
y_flat = y.flatten()
mpl.scatter(x_flat, y_flat)
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------

#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
xy = [(i, j) for i, j in zip(x_flat, y_flat)]
#-----------------------------------------------------------------
seeds_ip = [Point(_xy) for _xy in xy]
seeds_mp = MultiPoint(seeds_ip)
#-----------------------------------------------------------------
pxtal = voronoi_diagram(seeds_mp)
#-----------------------------------------------------------------
VGrains = [grain for grain in pxtal.geoms]
#-----------------------------------------------------------------
xmin, xmax = min(x_flat), max(x_flat)
ymin, ymax = min(y_flat), max(y_flat)
bound_rect = Polygon([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
#-----------------------------------------------------------------
xtal_list_xtal = clip_Voronoi_Tess_BoundBox(VGrains, bound_rect)
pxtal = MultiPolygon(xtal_list_xtal)
#-----------------------------------------------------------------
plt.figure(figsize = (3.5, 3.5), dpi = 100)
gcount = 0
for xtal in xtal_list_xtal:
    plt.fill(xtal.boundary.xy[0],
             xtal.boundary.xy[1],
             color = 'white',
             edgecolor = 'black',
             linewidth = 1)
    xc = xtal.centroid.x
    yc = xtal.centroid.y
    #plt.text(xc, yc, str(gcount), fontsize = 10, fontweight = 'bold', color = 'red')
    gcount += 1
plt.plot(x_flat, y_flat, '+')
maximum = max([xmax, ymax])
plt.xlim([xmin, xmin + maximum])
plt.ylim([ymin, ymin + maximum])
