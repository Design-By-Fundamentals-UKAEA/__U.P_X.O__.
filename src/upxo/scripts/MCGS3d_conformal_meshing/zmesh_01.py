# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 16:59:30 2024

@author: rg5749
"""

from zmesh import Mesher

labels = gstslice.lgi
mesher = Mesher( (10, 10, 10) ) # anisotropy of image
dir(mesher)
# initial marching cubes pass
# close controls whether meshes touching
# the image boundary are left open or closed
mesher.mesh(labels, close=True)

meshes = []
for obj_id in mesher.ids():
  meshes.append(
    mesher.get(
      obj_id,
      normals=False, # whether to calculate normals or not

      # tries to reduce triangles by this factor
      # 0 disables simplification
      reduction_factor=100,

      # Max tolerable error in physical distance
      # note: if max_error is not set, the max error
      # will be set equivalent to one voxel along the
      # smallest dimension.
      max_error=8,
      # whether meshes should be centered in the voxel
      # on (0,0,0) [False] or (0.5,0.5,0.5) [True]
      voxel_centered=False,
    )
  )
  mesher.erase(obj_id) # delete high res mesh

mesher.clear() # clear memory retained by mesher

mesh = meshes[0]
mesh = mesher.simplify(
  mesh,
  # same as reduction_factor in get
  reduction_factor=100,
  # same as max_error in get
  max_error=40,
  compute_normals=False, # whether to also compute face normals
) # apply simplifier to a pre-existing mesh

# compute normals without simplifying
mesh = mesher.compute_normals(mesh)

mesh.vertices
mesh.faces
mesh.normals
mesh.triangles() # compute triangles from vertices and faces

# Extremely common obj format
with open(r'C:\Development\M2MatMod\upxo_packaged\upxo_private\src\upxo\scripts\MCGS2d_characterization\iconic_doge.obj', 'wb') as f:
  f.write(mesh.to_obj())
