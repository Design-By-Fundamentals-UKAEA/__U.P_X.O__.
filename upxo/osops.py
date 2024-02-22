# from polycrystal import pxtal
#------------------------------------------------------------------
"""
GENERAL INFORMATION

This module has the following classes:
    1. compatibility
    2. folder
    3. file
    4. rw
"""
def get_pythnEnvName():
    import os
    import sys

    env_path = sys.prefix
    env_name = os.path.basename(env_path)  # Gets the last part of the path, which is typically the environment name

    print(f"Current environment path: {env_path}")
    print(f"Current environment name: {env_name}")

#00000000000000000000000000000000000000000000000000000000000000000000000000000
class compatibility():
    def __init__(self):
        pass
#00000000000000000000000000000000000000000000000000000000000000000000000000000
class folder():
    def __init__(self):
        pass
    def make(self):
        pass
    def delete(self):
        pass
    def move(self):
        pass
    def exist(self):
        pass
#000000000000000000000000000000000000000000000000000000000000000000000000000000
class file():
    def __init__(self):
        pass
    def make(self):
        pass
    def delete(self):
        pass
    def move(self):
        pass
    def exist(self):
        pass
#000000000000000000000000000000000000000000000000000000000000000000000000000000
class rw():
    def __init__(self):
        pass
    #------------------------------------------------------------------
    def w_par(self, **kwargs):
        # Write pxtal parameters to file as directed
        pass
    #------------------------------------------------------------------
    def w_xyz(self, **kwargs):
        # Write co-ordinate data to file as directed
        pass
    #------------------------------------------------------------------
    def r_par(self, **kwargs):
        # Read pxtal parameters from file as directed
        pass
    #------------------------------------------------------------------
    def r_xyz(self, **kwargs):
        # Read co-ordinate data from file as directed
        pass
    #------------------------------------------------------------------
    def w_id(self, **kwargs):
        # Write ID data to file as directed
        pass
    #------------------------------------------------------------------
    def w_field(self, **kwargs):
        # Write field data to file as directed
        pass
    #------------------------------------------------------------------
#000000000000000000000000000000000000000000000000000000000000000000000000000000
