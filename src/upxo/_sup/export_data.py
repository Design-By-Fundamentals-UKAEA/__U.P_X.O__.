"""
This module stores data used to export grain structures to different
formats.
"""
__name__ = "UPXO"
__author__ = "Dr. Sunil Anandatheertha"
__version__ = "0.0.1"
__doc__ = "Store data used to export grain structures"


import os
import re
from upxo.interfaces.os.osops import get_path_UPXOwriterdataDIR
from upxo.interfaces.os.osops import get_file_path
from upxo._sup.validation_values import _validation


class ctf_header():
    """
    from upxo._sup.export_data import ctf_header
    ctf = ctf_header()
    ctf.load_header_file(hfile='_ctf_header_CuCrZr_1.txt')
    ctf.set_metadata(projectname='UKAEA: UPXO grain structure: CPFEM',
                     username='Dr. Sunil Anandatheertha')
    ctf.make_header_from_lines()
    """
    __slots__ = ('root', 'path', 'header', 'header_lines', 'data_format',
                 'val', 'phase_name', 'hgrid', 'vgrid')

    def __init__(self,
                 hfile='_ctf_header_CuCrZr_1.txt',
                 phase_name='Copper'):
        self.root = os.getcwd()
        self.path = None
        self.header = None
        self.header_lines = None
        self.data_format = None
        self.val = _validation()
        self.phase_name = phase_name
        self.hgrid = None
        self.vgrid = None

    def load_header_file(self, hfile='_ctf_header_CuCrZr_1.txt'):
        """
        hfile: HEader filename, including the extention.
        """
        try:
            self.path = get_path_UPXOwriterdataDIR()
            self.path += hfile
        except Exception as e:
            print(f'Attempting to locate {hfile} using '
                  f'alternate method due to error: {e}')
            self.path = get_file_path(target_filename=hfile)
            if not self.path:
                raise FileNotFoundError(f"{hfile} could not be found.")

        if self.path:
            # Load the file
            try:
                with open(self.path, 'r', encoding='utf-8') as file:
                    self.header_lines = file.readlines()
            except FileNotFoundError:
                print(f"The file {self.path} does not exist.")
            except Exception as e:
                print(f"An error occurred while reading the file: {e}")

    def set_metadata(self, **kwargs):
        """
        Permitted input arguments:
            1. projectname
            2. username
        EXAMPLE:
            ctf.set_metadata(projectname='UKAEA: UPXO grain structure: CPFEM',
                             username='Dr. Sunil Anandatheertha')
        """
        # Permitted key: (line starts in header info lines, str to replace)
        _permitted_keys_starts_ = {'projectname': ('Prj', '__UPXO_Grain_Structure__'),
                                   'username': ('Author', '__USERNAME__')}
        # Validate if all inputs are permitted inputs
        if set(kwargs.keys())-set(_permitted_keys_starts_.keys()):
            raise TypeError('One or more inputs invalid.'
                            f'permitted arguments: {_permitted_keys_starts_}')
        # Validate if all input argument values atr strings
        self.val.valstrs(kwargs.values())
        # Replace the values in ctf header infgormation
        for idx, _ in enumerate(self.header_lines):
            for k, v in _permitted_keys_starts_.items():
                if v[0] in self.header_lines[idx]:
                    self.header_lines[idx] = self.header_lines[idx].replace(v[1],
                                                                            kwargs[k])

    def make_header_from_lines(self):
        self.header = ''.join(self.header_lines)

    def gen_fake_grid_field_data(self):
        if self.hgrid and self.vgrid and
        pass

    def format_header(self, xsz, ysz):
        self.header = self.header.format(xsz, ysz)

    def set_grid_data(self,
                      hgrid=None,
                      vgrid=None,
                      bands=None,
                      error=None,
                      euler1=None,
                      euler2=None,
                      euler3=None,
                      mad=None,
                      bc=None,
                      bs=None
                      ):
        """
        hgrid: Horizontal grid points
        vgrid: Vertical grid points
        bands: Number of Kikuchi Bands detected
        error: Error
        euler1: First Bunge's Euler angle
        euler2: Second Bung'e Euler angle
        euler3: Third Bunge's Euler angle'
        mad: Mean Angular Deviation
        bc: Band contrast
        bs: Band slope
        """
        # Validate for existance, type and shape of hgrid and vgrid
        self.val.valnparrs_shapes(hgrd, vgrd)
        # Validate for existance, type and shape of bands, error,...
        self.val.valnparrs_shapes(bands, error, euler1, euler2, euler3,
                                  mad, bc, bs)
