import cProfile
import gops, pops
from point2d import point2d
from edge2d import edge2d
#####################################################################################
gops.PROFILE_up2d_INST(10000, 'ignore')
#####################################################################################
p_lean_1, p_lean_2 = point2d(0, 0, lean='ignore'), point2d(1, 2, lean='ignore')
gops.PROFILE_this_method(pops.CMPEQ_points, 10000, (p_lean_1, p_lean_2), sort='cumulative')
gops.PROFILE_this_method(pops.CMPEQ_pnt_fast_exact, 10000, (p_lean_1, p_lean_2), sort='cumulative')
gops.PROFILE_this_method(pops.CMPEQ_pnt_fast_EPS, 10000, (p_lean_1, p_lean_2), sort='cumulative')
gops.PROFILE_this_method(pops.CMPEQ_pnt_fast_tdist, 10000, (p_lean_1, p_lean_2), sort='cumulative')
#####################################################################################
p_lean_1, p_lean_2 = point2d(0, 0, lean='highest'), point2d(1, 2, lean='highest')
gops.PROFILE_this_method(pops.CMPEQ_points, 10000, (p_lean_1, p_lean_2), sort='cumulative')
gops.PROFILE_this_method(pops.CMPEQ_pnt_fast_exact, 10000, (p_lean_1, p_lean_2), sort='cumulative')
gops.PROFILE_this_method(pops.CMPEQ_pnt_fast_EPS, 10000, (p_lean_1, p_lean_2), sort='cumulative')
gops.PROFILE_this_method(pops.CMPEQ_pnt_fast_tdist, 10000, (p_lean_1, p_lean_2), sort='cumulative')
#####################################################################################
point = point2d(0, 0, lean='ignore')
edge = edge2d(method='up2d',
              pnta=point2d(0.5, 0.5, lean='ignore'),
              pntb=point2d(1.0, 1.0, lean='ignore'),
              edge_lean='ignore')

gops.PROFILE_this_method(pops.CMPEQ_up2d_edge, 10000, (point, edge), sort='cumulative')

edges = [edge for _ in range(10)]
gops.PROFILE_this_method(pops.CMPEQ_up2d_edges, 10000, (point, edges, 'large_data_set'), sort='cumulative')
gops.PROFILE_this_method(pops.CMPEQ_up2d_edges, 10000, (point, edges, 'very_large_data_set'), sort='cumulative')

edges = [edge for _ in range(200)]
gops.PROFILE_this_method(pops.CMPEQ_up2d_edges, 10000, (point, edges, 'large_data_set'), sort='cumulative')
gops.PROFILE_this_method(pops.CMPEQ_up2d_edges, 10000, (point, edges, 'very_large_data_set'), sort='cumulative')
#####################################################################################
edges = [edge for _ in range(10)]
gops.PROFILE_this_method(pops.DIST_point_edges, 10000, (point, edges, 'large_data_set'), sort='cumulative')
gops.PROFILE_this_method(pops.DIST_point_edges, 10000, (point, edges, 'very_large_data_set'), sort='cumulative')

edges = [edge for _ in range(200)]
gops.PROFILE_this_method(pops.DIST_point_edges, 10000, (point, edges, 'large_data_set'), sort='cumulative')
gops.PROFILE_this_method(pops.DIST_point_edges, 10000, (point, edges, 'very_large_data_set'), sort='cumulative')
#####################################################################################
point = point2d(0, 0, lean='ignore')
points = [point for _ in range(100)]
gops.PROFILE_this_method(pops.RELPOS_point_points_above, 10000, (point, points), sort='cumulative')
gops.PROFILE_this_method(pops.RELPOS_point_points_below, 10000, (point, points), sort='cumulative')
gops.PROFILE_this_method(pops.RELPOS_point_points_left, 10000, (point, points), sort='cumulative')
gops.PROFILE_this_method(pops.RELPOS_point_points_right, 10000, (point, points), sort='cumulative')
#####################################################################################
