import numpy as np
#--------------------------------------------------------------------------
from GrainStructure import gstr
#--------------------------------------------------------------------------------------------------------------------------------
class vtess(gstr):
    vobj_base = None
    vobj_purt = None
    #--------------------------------------------------------------------------
    def __init__(self):
        super().__init__()
    #--------------------------------------------------------------------------
    def vogen2(self):
        """Generate 2D Voronoi object"""
        # make seed array:
        vseeds_base = np.hstack((gstr.xbase, gstr.ybase))
        vseeds_purt = np.hstack((gstr.x    , gstr.y))
        # Make Voronoi object
        from scipy.spatial import Voronoi
        vtess.vobj_base = Voronoi(vseeds_base)
        vtess.vobj_purt = Voronoi(vseeds_purt)
    #--------------------------------------------------------------------------
    def tessellate2(self, vo):
        if vo.points.shape[1] != 2:
            raise ValueError("Requires 2D input")
        new_regions = []
        new_vertices = vo.vertices.tolist()
        center = vo.points.mean(axis=0)
        radius = None
        if radius is None:
            radius = vo.points.ptp().max()*2
        # Construct a map containing all ridges for a given point
        all_ridges = {}
        for (p1, p2), (v1, v2) in zip(vo.ridge_points, vo.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))
        # Reconstruct infinite regions
        for p1, region in enumerate(vo.point_region):
            vertices = vo.regions[region]
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
                t = vo.points[p2] - vo.points[p1] # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal
                midpoint = vo.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = vo.vertices[v2] + direction * radius
                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())
            # sort region counterclockwise
            vs = np.asarray([new_vertices[v] for v in new_region])
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
            new_region = np.array(new_region)[np.argsort(angles)]
            # finish
            new_regions.append(new_region.tolist())
            cells = new_regions
            #Ng0 = len(new_regions)
            points = np.asarray(new_vertices)
        return cells, points
        #--------------------------------------------------------------------------
    def boundingPolygon(self, vo, method):
        if method == 'auto':
            min_x = vo.min_bound[0] - 0.00
            max_x = vo.max_bound[0] + 0.00
            min_y = vo.min_bound[1] - 0.00
            max_y = vo.max_bound[1] + 0.00
            # Polygonal bounding box for Voronoi tessellation
            from shapely.geometry import Polygon
            gstr.boundDomain = Polygon([[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]])