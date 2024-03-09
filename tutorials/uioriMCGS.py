from termcolor import colored
from color_specs import CPSPEC as CPS
import datatype_handlers as dth
import numpy as np
import re
import gops


class _UIORI_MCGS_():
    __slots__ = ('xtal_symmetry',
                 'sample_symmetry',
                 'ntc',
                 'nori',
                 'noritc1',
                 'noritc2',
                 'noritc3',
                 'noritc4',
                 'noritc5',
                 'noritc6',
                 'noritc7',
                 'noritc8',
                 'noritc9',
                 'ori_sampling_technique',
                 'auto_peak_identification',
                 'texture_model_parameters_cellBounds',
                 'exper_EA_Data_cellBounds',
                 'tc1_EA_Data_cellBounds',
                 'tc2_EA_Data_cellBounds',
                 'tc3_EA_Data_cellBounds',
                 'tc4_EA_Data_cellBounds',
                 'tc5_EA_Data_cellBounds',
                 'tc6_EA_Data_cellBounds',
                 'tc7_EA_Data_cellBounds',
                 'tc8_EA_Data_cellBounds',
                 'tc9_EA_Data_cellBounds',
                 'exper_EA_Data',
                 'tc1_EA',
                 'tc2_EA',
                 'tc3_EA',
                 'tc4_EA',
                 'tc5_EA',
                 'tc6_EA',
                 'tc7_EA',
                 'tc8_EA',
                 'tc9_EA',
                 '__uiORI__',
                 'gsi',
                 '__WS__'
                 )

    def __init__(self, WS, uiORI, gsi=None):
        self.gsi = gsi
        # --------------------------------------------------------------
        self.__uiORI__ = uiORI
        # --------------------------------------------------------------
        self.__WS__ = WS  # Work-Sheet in excel file
        # ----------------------------------------
        self.nori = None
        self.noritc1 = None
        self.noritc2 = None
        self.noritc3 = None
        self.noritc4 = None
        self.noritc5 = None
        self.noritc6 = None
        self.noritc7 = None
        self.noritc8 = None
        self.noritc9 = None
        # ----------------------------------------
        self.set_crystal_coordinate_system()
        self.set_sample_coordinate_system()
        self.set_number_of_tex_components()
        self.set_ori_sampling_technique()
        self.set_auto_peak_identification()
        self.set_orientations()

    def show_slots(self):
        return gops.slots(self)

    def show_methods(self):
        return gops.methods(self)

    def show_dunders(self):
        return gops.dunders(self)

    def print_dict(self):
        print(self.__uiORI__)

    def reload(self):
        print("Please use ui.load_ori()")

    def convert_strListOfNum_to_Numlist(self, strListOfNum):
        return list(map(int, re.findall(r'\d+', strListOfNum)))

    def CPRINTN(self,
                STR1,
                STR2,
                START=' '*5,
                SEP12=': ',
                CLR1='red',
                CLR2='cyan',
                ATTRIBUTES1 = ["bold", "dark"],
                ATTRIBUTES2 = [],
                ):
        # ---------------------------------
        t1 = START + "\033[%dm%s" % (CPS.COLORS[CLR1], STR1) + "\033[0m"
        for attr in ATTRIBUTES1:
            t1 = "\033[%dm%s" % (CPS.ATTRIBUTES[attr], t1) + "\033[0m"
        # ---------------------------------
        t2 = "\033[%dm%s" % (CPS.COLORS[CLR2], STR2) + "\033[0m"
        for attr in ATTRIBUTES2:
            t2 = "\033[%dm%s" % (CPS.ATTRIBUTES[attr], t2) + "\033[0m"
        # ---------------------------------
        return t1 + SEP12 + t2 + "\n"

    def __repr__(self):
        _ = ' '*5
        retstr = "Attributes of ORI: \n"
        retstr += self.CPRINTN('GSI', self.gsi)
        retstr += self.CPRINTN('XTAL Symmetry', self.xtal_symmetry)
        retstr += self.CPRINTN('Sample Symmetry', self.sample_symmetry)
        retstr += self.CPRINTN('No. of tex. components', self.ntc)
        retstr += self.CPRINTN('Orientation sampling technique', self.ori_sampling_technique)
        if self.ori_sampling_technique in ('mtex_sampling_ebsd'):
            retstr += self.CPRINTN('MTEX Auto peak orientation', self.auto_peak_identification)
            retstr += self.CPRINTN('CellBounds: Texture model parameters', self.texture_model_parameters_cellBounds)

        if self.ori_sampling_technique in ('exper_sampled_set',
                                           'mtex_sampling_ebsd',
                                           'defdap_sampling_ebsd',
                                           'pyebsd_sampling_ebsd',
                                           'mtex_model_tex_sampling'
                                           ):
            retstr += self.CPRINTN('CellBounds: Euler angle list', self.exper_EA_Data_cellBounds)

        if self.ori_sampling_technique == 'mtex_model_tex_sampling_tc_wise':
            retstr += self.CPRINTN('CellBounds: Tex Comp. 1 Euler angles', self.tc1_EA_Data_cellBounds)
            retstr += self.CPRINTN('CellBounds: Tex Comp. 2 Euler angles', self.tc2_EA_Data_cellBounds)
            retstr += self.CPRINTN('CellBounds: Tex Comp. 3 Euler angles', self.tc3_EA_Data_cellBounds)
            retstr += self.CPRINTN('CellBounds: Tex Comp. 4 Euler angles', self.tc4_EA_Data_cellBounds)
            retstr += self.CPRINTN('CellBounds: Tex Comp. 5 Euler angles', self.tc5_EA_Data_cellBounds)
            retstr += self.CPRINTN('CellBounds: Tex Comp. 6 Euler angles', self.tc6_EA_Data_cellBounds)
            retstr += self.CPRINTN('CellBounds: Tex Comp. 7 Euler angles', self.tc7_EA_Data_cellBounds)
            retstr += self.CPRINTN('CellBounds: Tex Comp. 8 Euler angles', self.tc8_EA_Data_cellBounds)
            retstr += self.CPRINTN('CellBounds: Tex Comp. 9 Euler angles', self.tc9_EA_Data_cellBounds)
        return retstr

    def cellname_to_rowcol(self, cell_name):
        col_name = ''.join(filter(str.isalpha, cell_name))
        row_name = ''.join(filter(str.isdigit, cell_name))

        col_idx = 0
        for i, char in enumerate(reversed(col_name)):
            col_idx += (ord(char) - ord('A') + 1) * (26 ** i)
        row_idx = int(row_name) - 1  # Convert 1-based index to 0-based index

        return row_idx, col_idx - 1  # Convert 1-based index to 0-based index

    def read_cell_range(self, start_cell, end_cell):
        start_row, start_col = self.cellname_to_rowcol(start_cell)
        end_row, end_col = self.cellname_to_rowcol(end_cell)

        values = []
        for row in range(start_row, end_row + 1):
            row_values = []
            for col in range(start_col, end_col + 1):
                cell_value = self.__WS__.cell_value(row, col)
                row_values.append(cell_value)
            values.append(row_values)

        return values

    def set_crystal_coordinate_system(self):
        self.xtal_symmetry = self.__uiORI__['main']['xtal_symmetry']

    def set_sample_coordinate_system(self):
        self.sample_symmetry = self.__uiORI__['main']['sample_symmetry']

    def set_number_of_tex_components(self):
        self.ntc = int(self.__uiORI__['main']['ntc'])

    def set_ori_sampling_technique(self):
        self.ori_sampling_technique = self.__uiORI__['main']['ori_sampling_technique']

    def set_auto_peak_identification(self):
        self.auto_peak_identification = self.__uiORI__['main']['auto_peak_identification']

    def set_texture_model_parameters_cellBounds(self):
        self.texture_model_parameters_cellBounds = self.__uiORI__['main']['texture_model_parameters_cellBounds']

    def set_exper_EA_Data_cellBounds(self):
        self.exper_EA_Data_cellBounds = self.__uiORI__['main']['exper_EA_Data_cellBounds']

    def set_tc1_EA_Data_cellBounds(self):
        self.tc1_EA_Data_cellBounds = self.__uiORI__['main']['tc1_EA_Data_cellBounds']

    def set_tc2_EA_Data_cellBounds(self):
        self.tc2_EA_Data_cellBounds = self.__uiORI__['main']['tc2_EA_Data_cellBounds']

    def set_tc3_EA_Data_cellBounds(self):
        self.tc3_EA_Data_cellBounds = self.__uiORI__['main']['tc3_EA_Data_cellBounds']

    def set_tc4_EA_Data_cellBounds(self):
        self.tc4_EA_Data_cellBounds = self.__uiORI__['main']['tc4_EA_Data_cellBounds']

    def set_tc5_EA_Data_cellBounds(self):
        self.tc5_EA_Data_cellBounds = self.__uiORI__['main']['tc5_EA_Data_cellBounds']

    def set_tc6_EA_Data_cellBounds(self):
        self.tc6_EA_Data_cellBounds = self.__uiORI__['main']['tc6_EA_Data_cellBounds']

    def set_tc7_EA_Data_cellBounds(self):
        self.tc7_EA_Data_cellBounds = self.__uiORI__['main']['tc7_EA_Data_cellBounds']

    def set_tc8_EA_Data_cellBounds(self):
        self.tc8_EA_Data_cellBounds = self.__uiORI__['main']['tc8_EA_Data_cellBounds']

    def set_tc9_EA_Data_cellBounds(self):
        self.tc9_EA_Data_cellBounds = self.__uiORI__['main']['tc9_EA_Data_cellBounds']

    def set_orientations(self):
        if self.ori_sampling_technique in ('exper_sampled_set',
                                           'mtex_sampling_ebsd',
                                           'defdap_sampling_ebsd',
                                           'pyebsd_sampling_ebsd',
                                           'mtex_model_tex_sampling'
                                           ):
            self.set_exper_EA_Data_cellBounds()
            self.set_exper_EA()
        elif self.ori_sampling_technique == 'mtex_model_tex_sampling_tc_wise':
            self.set_tc1_EA_Data_cellBounds()
            self.set_tc2_EA_Data_cellBounds()
            self.set_tc3_EA_Data_cellBounds()
            self.set_tc4_EA_Data_cellBounds()
            self.set_tc5_EA_Data_cellBounds()
            self.set_tc6_EA_Data_cellBounds()
            self.set_tc7_EA_Data_cellBounds()
            self.set_tc8_EA_Data_cellBounds()
            self.set_tc9_EA_Data_cellBounds()
            self.set_tc1_EA()
            self.set_tc2_EA()
            self.set_tc3_EA()
            self.set_tc4_EA()
            self.set_tc5_EA()
            self.set_tc6_EA()
            self.set_tc7_EA()
            self.set_tc8_EA()
            self.set_tc9_EA()
        self.set_texture_model_parameters_cellBounds()

    def set_exper_EA(self):
        startingCell, endingCell = self.exper_EA_Data_cellBounds.split(":")
        self.exper_EA_Data = np.array(self.read_cell_range(startingCell,
                                                           endingCell))

    def set_tc1_EA(self):
        startingCell, endingCell = self.tc1_EA_Data_cellBounds.split(":")
        self.tc1_EA = np.array(self.read_cell_range(startingCell,
                                                    endingCell))

    def set_tc2_EA(self):
        startingCell, endingCell = self.tc2_EA_Data_cellBounds.split(":")
        self.tc2_EA = np.array(self.read_cell_range(startingCell,
                                                    endingCell))

    def set_tc3_EA(self):
        startingCell, endingCell = self.tc3_EA_Data_cellBounds.split(":")
        self.tc3_EA = np.array(self.read_cell_range(startingCell,
                                                    endingCell))

    def set_tc4_EA(self):
        startingCell, endingCell = self.tc4_EA_Data_cellBounds.split(":")
        self.tc4_EA = np.array(self.read_cell_range(startingCell,
                                                    endingCell))

    def set_tc5_EA(self):
        startingCell, endingCell = self.tc5_EA_Data_cellBounds.split(":")
        self.tc5_EA = np.array(self.read_cell_range(startingCell,
                                                    endingCell))

    def set_tc6_EA(self):
        startingCell, endingCell = self.tc6_EA_Data_cellBounds.split(":")
        self.tc6_EA = np.array(self.read_cell_range(startingCell,
                                                    endingCell))

    def set_tc7_EA(self):
        startingCell, endingCell = self.tc7_EA_Data_cellBounds.split(":")
        self.tc7_EA = np.array(self.read_cell_range(startingCell,
                                                    endingCell))

    def set_tc8_EA(self):
        startingCell, endingCell = self.tc8_EA_Data_cellBounds.split(":")
        self.tc8_EA = np.array(self.read_cell_range(startingCell,
                                                    endingCell))

    def set_tc9_EA(self):
        startingCell, endingCell = self.tc9_EA_Data_cellBounds.split(":")
        self.tc9_EA = np.array(self.read_cell_range(startingCell,
                                                    endingCell))
