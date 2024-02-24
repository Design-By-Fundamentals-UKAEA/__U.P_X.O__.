'''
In this tutorial, we will create an N number of point sets, with all points 
bounded by [0.0, 1.0] on x and y. Each point will be a UPXO point2d object.
Coordinates will have a random uniform distributi90onl.

We will then create a single reference point at (0, 0).

For this point, we will then find its neighbours in each of the 
point-sets within a certain cut-off-radius specific to each point set.

We will calcuolate the following:
    1. Neighbouring points for eqach point set
    2. Number of neighbopuring points for each point set
    3. Indices of neighbours from the parent point dataset
    4. Distances of the neighbours from the reference point

In most cases, user will not want to querry all the above four results, as
some can be calculated using others. However, user can replacve the 
un-necessary output by a dummy underscore. For example, if in the example
below, if the npoints if not needed in the find_neigh_points output, then 
the listy of outputs coule be "neigh_points, _, indices, distances"
oinstead of "neigh_points, npoints, indices, distances".

NOTES: For really biGGG data sets (say around 10,000 points), do issue, a 
number greater than 1 for argument "ckdtree_workers".
'''
import numpy as np
from point2d_04 import point2d
randd = np.random.uniform
# Assign the Number of point sets
n_pointdatasets = 2
# Assign the Number of points to create
n_points = 100
# Create the point data-sets
points = [[point2d(x=randd(), y=randd()) for _ in range(n_points)] for _ in range(n_pointdatasets)]
# Assign the cut-off-radii
cut_off_radii = [0.5 for _ in range(len(points))]
# Make the reference point of the UPXO data-set
ref_point = point2d(x=0.0, y=0.0)
# Calculate the neighbour details for each point-data-set
# neigh_points, npoints, indices, distances
NP, N, I, D = ref_point.find_neigh_points(method='points', points=points,
                                          point_type='upxo',
                                          cutoffshape='circle',
                                          cut_off_radii=cut_off_radii,
                                          ckdtree_workers=1
                                          )
# =============================================================================
# print(len(neigh_points))
# print(len(indices))
# print(len(distances))
# 
# print(neigh_points)
# print(indices)
# print(distances)
# 
# =============================================================================
