from point2d_04 import point2d
p1 = point2d(x = 0, y = 0)
p2 = point2d(x = 1, y = 0)
p3 = point2d(x = 1, y = 1)
p4 = point2d(x = 0, y = 1)
p5 = point2d(x = 0.5, y = 0.5)
p6 = p1 + p2
p7 = p1 - p2
p8 = p4*0.4 + p2/2.2 - p2.negxy() + p3.negx()

from mulpoint2d_3 import mulpoint2d
m1 = mulpoint2d(method = 'points', point_objects = [p1, p2, p3, p4, p5, p6, p7, p8])
m1.npoints
m1.plot()

p8.displace_by(delx = 10, dely = 10)
m1.points[6]
m1.centroid
m1.plot(dpi = 50)

m1.recompute_basics()
m1.centroid
m1.plot(dpi = 50)

