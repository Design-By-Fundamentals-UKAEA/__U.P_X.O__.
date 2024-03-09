'''

'''
import numpy as np
areas = np.array([_xtal_.area for _xtal_ in pxtal.L0.xtals])
pxtal.L0.mpo_xtals_reppoints.tree.data
for _ in pxtal.L0.xtals:
    print(_.representative_point().xy)

cut_off_radius = np.sqrt(areas.mean()/np.pi)
TREE = pxtal.L0.mpo_xtals_reppoints.tree
from scipy.spatial import cKDTree
OTHER = cKDTree([TREE.data[0]])
TREE.count_neighbors(OTHER, cut_off_radius)

TREE.query_ball_point(TREE.data[0], cut_off_radius)