from orix import data, plot
from orix.vector import Vector3d
import numpy as np
from orix.quaternion import Orientation
import matplotlib.pyplot as plt
from orix.crystal_map import Phase
from diffpy.structure import Atom, Lattice, Structure
from orix.crystal_map import create_coordinate_arrays, CrystalMap, PhaseList
from orix.quaternion import Rotation

# Select sample direction
vec_sample = Vector3d([0, 0, 1])
vec_title = "Z"


euler_angles = np.random.random((1000, 3))

# Convert Euler angles to Orientation
orientations = Orientation.from_euler(euler_angles)

orientations = orientations * vec_sample


fig = plt.figure(figsize=(9, 8))
ax0 = fig.add_subplot(221, direction=vec_sample, projection="ipf")
ax0.scatter(orientations, alpha=0.05)
_ = ax0.set_title(f"Ferrite, {vec_title}")


fig = plt.figure(figsize=(9, 8))
ax0 = fig.add_subplot(221, direction=vec_sample, projection="ipf")
ax0.pole_density_function(orientations)
_ = ax0.set_title(f"Ferrite, {vec_title}")


fig = plt.figure(figsize=(9, 8))
ax1 = fig.add_subplot(222, projection="stereographic")
ax1.pole_density_function(orientations)


# ======================================
from orix.io import loadctf
fn = r"C:\Development\M2MatMod\upxo_packaged\upxo_private\src\upxo\_written_data\_ctf_export_2dmcgs\sunil.ctf"
fn = r"D:\export_folder\sunil.ctf"
upxo_map = loadctf(fn)


vec_sample = Vector3d([1, 1, 1])

upxo_map = upxo_map * vec_sample

# Specify a crystal structure and symmetry
phase = Phase(
    point_group="6/mmm",
    structure=Structure(lattice=Lattice(1, 1, 2, 90, 90, 120)),
)

fig = plt.figure(figsize=(20, 20))
ax0 = fig.add_subplot(221, direction=vec_sample, projection="ipf")
ax0.pole_density_function(Vector3d(upxo_map), log=False, resolution=1, sigma=10)




fig = plt.figure(figsize=(20, 20))
ax0 = fig.add_subplot(221, direction=vec_sample, projection="ipf")
ax0.scatter(Vector3d(upxo_map), alpha=0.5)
