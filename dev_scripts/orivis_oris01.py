from orix import data, plot
from orix.vector import Vector3d
import numpy as np
from orix.quaternion import Orientation
import matplotlib.pyplot as plt
from orix.crystal_map import Phase
# from diffpy.structure import Atom, Lattice, Structure
from diffpy.structure import Lattice, Structure
from orix.crystal_map import create_coordinate_arrays, CrystalMap, PhaseList
from orix.quaternion import Rotation
from orix.io import loadctf

fn = r"D:\export_folder\sunil.ctf"
upxo_map = loadctf(fn)


vec_sample = Vector3d([1, 1, 1])

upxo_map = upxo_map * vec_sample

# Specify a crystal structure and symmetry
phase = Phase(
    point_group="6/mmm",
    structure=Structure(lattice=Lattice(1, 1, 2, 90, 90, 120)),
)



fig = plt.figure(figsize=(8, 8))
ax0 = fig.add_subplot(111, direction=vec_sample, projection="ipf")
ax0.pole_density_function(Vector3d(upxo_map), log=False, resolution=1, sigma=10)
ax0.scatter(Vector3d(upxo_map), alpha=1.0)
