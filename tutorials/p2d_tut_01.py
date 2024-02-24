from point2d_04 import point2d
p1 = point2d()
p1.set_dim()
p1.dim
p1._status_

sp = p1.make_shapely()
p1.plot()
#p1.make_rid()
#p1.rid
p1.make_mid()
p1.mid
p1.set_ptype()
p1.ptype
p1.set_location()
p1.loc
p1.set_tcname()
p1.tcname
p1.set_sfv_repr_ea()
p1.sfv_repr_ea
p1.set_sfv_ea(ea = [20, 40, 50])
p1.sfv_ea
p1.set_sfv_ea1(ea1=45)
p1.sfv_ea
p1.set_jn()

p1.has('rid')

p1._status_

p1.has('jn')
p1.has('ptype')
p1.has('mid')

p1.has('tcname')
p1.has('earepr')
p1.has('ea')
p1.has('area')
p1.has('ar')
p1.has('phid')
p1.has('phname')

p1 = point2d(x=1.0, y=2.0)
p2 = point2d(x=3.0, y=4.0)
p3 = point2d(x=2.5, y=3.2)
p1.set_tdist(tdist=0.0)
p1.has('tdist')
p1.tdist
p1.set_tdist(tdist=5.0)
p2.set_tdist(tdist=2.0)
p2.tdist
p2.borrow_tdist(p1)
p12 = p1 + p2

p1.align_to(method='point2d', ref_point_object=p12)

p = point2d(x=1.1, y=2.2,
            lean='lowest',
            set_mid=True,
            )
p.has('rid')
p.has('mid')


p = point2d(x=1.1, y=2.2,
            set_dim=True, dim=2,
            )
p.has('dimensionality')

p = point2d(x=1.1, y=2.2,
            set_ptype=True, ptype='vt2dseed',
            set_loc=True, loc='internal',
            )
p.has('pointtype')
p.has('location_in_pxtal')

p = point2d(x=1.1, y=2.2)
p.set_jn(jn=3)
p.has('jn')


p = point2d(x=1.1, y=2.2,
            set_phase=True, phid=1, phname='Cu',
            )
p.has('phase_id')
p.has('phase_name')


p = point2d(x=1.1, y=2.2,
            set_sfv_pol_area=True, sfv_pol_area=0.1234,
            )
p.has('area')

p = point2d(x=1.1, y=2.2,
            set_tcname=True, tcname='B',
            )
p.has('tcname')


p = point2d(x=1.1, y=2.2, set_sfv_ea=True, sfv_repr_ea='Bunge', sfv_ea=[45, 35, 0],)
p.has('ea_repr')

p = point2d(x=1.1, y=2.2, set_orientation_object=False, orientation_object=False,)

p = point2d(x=1.1, y=2.2,set_tdist=False, tdist=0.0000000000001,)

p = point2d(x=1.1, y=2.2,store_vis_prop=False,)
