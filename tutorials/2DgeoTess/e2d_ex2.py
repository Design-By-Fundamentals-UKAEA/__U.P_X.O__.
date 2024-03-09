from point2d_04 import point2d
p1 = point2d(x=0, y=0)
p2 = point2d(x=1, y=0)
p3 = point2d(x=1, y=1)
p4 = point2d(x=1, y=0.1)
p5 = point2d(x=1, y=0.01)
p6 = point2d(x=1, y=0.001)
p7 = point2d(x=1, y=0.2)
p8 = point2d(x=-1, y=0.0)
p9 = point2d(x=0, y=1.0)

from edge2d_05 import edge2d
e1 = edge2d(method='points', pnta=p1, pntb=p2)
e2 = edge2d(method='points', pnta=p2, pntb=p3)
e3 = edge2d(method='points', pnta=p3, pntb=p1)
e4 = edge2d(method='points', pnta=p1, pntb=p4)
e5 = edge2d(method='points', pnta=p1, pntb=p5)
e6 = edge2d(method='points', pnta=p1, pntb=p6)
e7 = edge2d(method='points', pnta=p1, pntb=p7)
e8 = edge2d(method='points', pnta=p8, pntb=p9)

print([e1.length, e2.length, e3.length, e4.length, e5.length, e6.length, e7.length])
print(e1.compare_length(e1))
print(e2.compare_length(e2))
print(e3.compare_length(e3))

print(e1.compare_length(e2))
print(e1.compare_length(e3))
print(e1.compare_length([e1, e2, e3]))
print(e2.compare_length([e1, e2, e3]))
print(e3.compare_length([e1, e2, e3]))

print([e1.length, e2.length, e3.length, e4.length, e5.length, e6.length, e7.length])
print([e1.EPS_e2dl_low, e1.EPS_e2dl_high])
print(e1.compare_length([e1, e2, e3, e4, e5, e6, e7]))


print([e1.slope, e2.slope, e3.slope, e4.slope, e5.slope, e6.slope])

print(e1.compare_slope(e1))
print(e2.compare_slope(e2))
print(e3.compare_slope(e3))

print(e1.compare_slope(e2))
print(e1.compare_slope(e3))

print(e2.compare_slope(e1))
print(e2.compare_slope(e3))

print(e1.compare_slope([e1, e2, e3]))

print(e4.slope)
print([e4.EPS_e2ds_lowest, e4.EPS_e2ds_highest])
print([e4.EPS_e2dl_low, e4.EPS_e2dl_high])

print(e1.compare_slope(e4))

print(e1.compare_slope(e4))
print(e1.compare_slope(e5))
print(e1.compare_slope(e6))
