# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 15:48:36 2024

@author: rg5749
"""
import numpy as np
import h5py
import subprocess
import dream3d
from dream3d import simplpy
from dream3d import simpl
from dream3d import simpl_helpers as sc

# Define file path variables
hdf5_file_path = 'grain_structure_with_orientations.h5'
output_surface_file = 'grain_structure_surface_mesh.stl'
output_volumetric_mesh_file = 'grain_structure_volumetric_mesh.1.node'
