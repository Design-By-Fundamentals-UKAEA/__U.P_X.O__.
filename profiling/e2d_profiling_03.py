n = 1_000_000
for _ in range(0, n):
    e = edge2d(method = 'points',
               end_points = [p1, p2],
               point_lean = 'lowest')