# from orix import data, plot
from orix.vector import Vector3d
import numpy as np
from orix.quaternion import Misorientation, Orientation
import matplotlib.pyplot as plt
from orix.crystal_map import Phase
# from diffpy.structure import Atom, Lattice, Structure
from diffpy.structure import Atom, Lattice, Structure
# from orix.crystal_map import create_coordinate_arrays, CrystalMap, PhaseList
# from orix.quaternion import Rotation
from orix.io import loadctf
from defdap.quat import Quat
# --------------------------------------------------
fn = r"D:\export_folder\sunil.ctf"
upxo_map = loadctf(fn)

vec_sample = Vector3d([0, 0, 1])

upxo_map = upxo_map * vec_sample

# Specify a crystal structure and symmetry
phase = Phase(name="Cu",
              space_group=225,
              structure=Structure( atoms=[Atom("al", [0, 0, 0])],
                                  lattice=Lattice(0.405, 0.405, 0.405,
                                                  90, 90, 90)
                                  ), )
#phase = Phase(name="Cu",
#    point_group="6/mmm",
#    structure=Structure(lattice=Lattice(1, 1, 2, 90, 90, 120)),
#)

pg_laue = phase.point_group.laue

fig = plt.figure(figsize=(20, 20))
ax0 = fig.add_subplot(221, direction=vec_sample, projection="ipf", symmetry=pg_laue)
ax0.pole_density_function(Vector3d(upxo_map), log=False, resolution=1, sigma=2.5)

fig = plt.figure(figsize=(20, 20))
ax0 = fig.add_subplot(221, direction=vec_sample, projection="ipf")
ax0.scatter(Vector3d(upxo_map), alpha=0.5)

fig = plt.figure(figsize=(9, 8))
ax1 = fig.add_subplot(222, projection="stereographic")
ax1.pole_density_function(Vector3d(upxo_map), log=False, resolution=1, sigma=10)



o1 = Orientation.from_euler([0*np.pi/180, 0, 0], symmetry=phase.point_group)
o2 = Orientation.from_euler([45*np.pi/180, 0, 0], symmetry=phase.point_group)
m_ref = Misorientation(o2 * (~o1), symmetry=(o1.symmetry, o2.symmetry))
[m_ref.a[0], m_ref.b[0], m_ref.c[0], m_ref.d[0]]

a = Quat([m_ref.a[0], m_ref.b[0], m_ref.c[0], m_ref.d[0]])
2*np.arccos(a.eulerAngles())*180/np.pi



ori1 = Quat.fromEulerAngles(0*np.pi/180, 0, 0)
ori2 = Quat.fromEulerAngles(45*np.pi/180, 0, 0)
mori12 = round(ori1.misOri(ori2, 'cubic'), 5)

round(2*np.arccos(mori12)*180/np.pi, 10)
