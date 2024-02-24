import numpy as np
from point2d import point2d

p1 = point2d(x = 0, y = 0)
p2 = point2d(x = 1, y = 1)

p3 = p1 + p2
p3
p4 = p1 - p2
p4
p5 = p4*0.4 + p2.negxy()
p5

from edge2d import edge2d
e = edge2d(method = 'points', pnta = p1, pntb = p2)
e.pnta
e.pntb
e.length

e.pnta.x
e.pnta.y

[e._xa_, e._ya_]
e.pnta
e._xa_ = 10
[e._xa_, e._ya_]
e.pnta






print(f'Edge object: {e}')
print(f'X-coords: {e._xa_}, Y-coords:{e._ya_}')
p1.x = 0.5
print(f'X-coords: {[e._xa_, e._xb_]}, Y-coords{[e._ya_, e._yb_]}')
print(f'End points: {e.pnta}, {e.pntb}')
print(f'Edge object{e}')
e.update_xy()
e.update_centre()
print(f'X-coords: {[e._xa_, e._xb_]}, Y-coords{[e._ya_, e._yb_]}')

e._xa_ = 0.75

print(f'X-coords: {[e._xa_, e._xb_]}, Y-coords{[e._ya_, e._yb_]}')
print(f'End points: {e.pnta}, {e.pntb}')
e.calc_centre(saa = False, throw = True)
print(e.pnta, e.xycen, e.pntb)
e.displace(xyincr=[1, 1])
print(e.pnta, e.xycen, e.pntb)
print(f'End points: {e.pnta} and {e.pntb}')
print(f'X-coords: {[e._xa_, e._xb_]}, Y-coords{[e._ya_, e._yb_]}')
e.calc_center(saa = True, throw = True)
print(f'Centre coord: {e.xycen}')
e.calc_length(saa = True, throw = False)
print(f'Edge length: {e.length}')
e.make_m()
print(f'mulpoint2d object: {e.m}')

e.xycen
e.length
[e.pnta, e.pntb]
e._xa_
e._xa_ = 50
e.length
e.xycen
[e.pnta, e.pntb]
[e._xa_, e._xb_]
[e._ya_, e._yb_]

e.plot()

e.make_m()
print(f'mulpoint2d object: {e.m}')
e.calc_center_from_m()
e.xycen
######################################################################
from point2d import point2d
p1 = point2d(x = 10, y = -5)
p2 = point2d(x = 8, y = 12)
p3 = p1 + p2
p4 = p1 - p2
p5 = p4*0.4 + p2.negxy()


from edge2d import edge2d
e1 = edge2d(method = 'points',
            pnta = p1-p2,
            pntb = p2
            )
e2 = edge2d(method = 'points',
            pnta = p1,
            pntb = p2
            )
e1
e2
e1+e2
e2-e1
e2*2
e1/-5
e2.negx()
e2
e1
e1.negy()
e1
e1.negxy()
e1
######################################################################
from point2d import point2d
p1 = point2d(x = 10.46854, y = -5.846)
p2 = point2d(x = 8, y = 12)
p3 = point2d(x = 20, y = 0)
p4 = point2d(x = -5, y = 20)
from edge2d import edge2d
e1 = edge2d(method = 'points', pnta = p1, pntb = p2)
e2 = edge2d(method = 'points', pnta = p3, pntb = p4)
e3 = edge2d(method = 'points', pnta = p2, pntb = p3)
e4 = edge2d(method = 'points', pnta = p1, pntb = p4)

print(e1) # upxo.e2d[(10.0,-5.0)-(8.0, 12.0)]
print(e2) # upxo.e2d[(20.0,0.0)-(-5.0, 20.0)]


print(e1.length) # 17.11724276862369
print(e2.length) # 32.01562118716424

e1==e2 # False
e1!=e2 # False
e1>e2 # False
e1<e2 # True
e1>=e2 # False
e1<=e2 # True

e2.pntb.negx()
e2 # upxo.e2d[(20.0,0.0)-(5.0, 20.0)]
e3 = e1 + e2
print(e1)
print(e2)
print(e3) # upxo.e2d[(30.0,-5.0)-(13.0, 32.0)]

bool(e1)

e4 = e1 + e2*2
e4

# int(e1) #<<<----- NEEDS DEBUGGING !

e1
round(e1, 1)
e1._xb_= 100
e1.pnta
e1.pntb
e1
ceil(e1)

e5 = edge2d(method = 'points', pnta = p1, pntb = p1)

edges = [e1, e2, e3, e4, e5]



"""Check if any of the edges have length less than equal
to their respective tolerance length value"""
any(edges)
"""Check if all of the edges have length less than equal
to eachj of the edge's tolerance length value"""
all(edges)


e1.tol_len
e1.calc_slope()
e1.slope
e1

######################################################################
from point2d import point2d
p1 = point2d(x=0.0, y=0.0)
p2 = point2d(x=1.0, y=0.0)
p3 = point2d(x=0.0, y=1.0)
p4 = point2d(x=0.5, y=0.5)
p5 = point2d(x=1.0, y=1.0)
p6 = point2d(x=1.2, y=4.5)
p7 = point2d(x=-0.5, y=-0.1)
p8 = point2d(x=0.2, y=-6.6)
p9 = point2d(x=-9.6, y=4.3)
p10 = point2d(x=-0.5, y=-0.5)
p11 = point2d(x=2.0, y=0.0)
from edge2d import edge2d
e1 = edge2d(pnta=p1, pntb=p2)
e2 = edge2d(pnta=p1, pntb=p3)
e3 = edge2d(pnta=p1, pntb=p4)
e4 = edge2d(pnta=p1, pntb=p5)
e5 = edge2d(pnta=p1, pntb=p6)
e6 = edge2d(pnta=p1, pntb=p7)
e7 = edge2d(pnta=p1, pntb=p8)
e8 = edge2d(pnta=p1, pntb=p9)
e9 = edge2d(pnta=p1, pntb=p10)
e10 = edge2d(pnta=p1, pntb=p11)

edges = [e1, e2, e3, e4, e5, e6, e7, e8, e9, e10]
e1.__eql__(edges)
e1.__eqs__(edges)

print(e1 == edges)
print(e1 != edges)

e1 < edges
e1 > edges
e1 <= edges
e1 >= edges
e1 == edges

map(lambda e: e.length**2, edges)
map(lambda e: e + e1, edges)
list(map(lambda e: e1 < e, edges))
list(map(lambda e: e1 > e, edges))



from point2d_04 import point2d
p1 = point2d(x=0.0, y=0.0)
p2 = point2d(x=1.0, y=0.0)
p3 = point2d(x=0.0, y=1.0)

from edge2d_05 import edge2d
e1 = edge2d(pnta=p1, pntb=p2)
e2 = edge2d(pnta=p1, pntb=p3)
edges = [e1, e2]

e1 == e1
