from color_specs import CPSPEC as CPS
from termcolor import colored
import datatype_handlers as dth
import numpy as np
import re
import gops


class _UIEXPORT_MCGS_():
    __slots__ = ('ctf',
                 'pickle',
                 'abaqus',
                 'moose',
                 'gsi',
                 '__uiEXPORT__',
                 )

    def __init__(self, uiEXPORT, gsi=None):
        self.gsi = gsi
        # -------------------------------------------------
        self.__uiEXPORT__ = uiEXPORT
        # -------------------------------------------------
        self.set_ctf()
        self.set_pickle()
        self.set_abaqus()
        self.set_moose()

    def show_slots(self):
        return gops.slots(self)

    def show_methods(self):
        return gops.methods(self)

    def show_dunders(self):
        return gops.dunders(self)

    def print_dict(self):
        print(self.__uiEXPORT__)

    def reload(self):
        print("Please use ui.load_export()")

    def convert_strListOfNum_to_Numlist(self, strListOfNum):
        return list(map(int, re.findall(r'\d+', strListOfNum)))

    def set_ctf(self):
        self.ctf = self.__uiEXPORT__['ctf']
        self.ctf['export_ctf'] = bool(self.ctf['export_ctf'])
        self.ctf['export_all_gs_instances'] = bool(self.ctf['export_all_gs_instances'])
        self.ctf['export_all_temporal_slices'] = bool(self.ctf['export_all_temporal_slices'])
        self.ctf['export_full_field'] = bool(self.ctf['export_full_field'])

    def set_pickle(self):
        self.pickle = self.__uiEXPORT__['pickle']
        self.pickle['export_pickle'] = bool(self.pickle['export_pickle'])
        self.pickle['export_all_gs_instances'] = bool(self.pickle['export_all_gs_instances'])
        self.pickle['export_all_temporal_slices'] = bool(self.pickle['export_all_temporal_slices'])
        self.pickle['export_full_field'] = bool(self.pickle['export_full_field'])

    def set_abaqus(self):
        self.abaqus = self.__uiEXPORT__['abaqus']
        self.abaqus['export_abaqus_input_file'] = bool(self.abaqus['export_abaqus_input_file'])
        self.abaqus['export_all_gs_instances'] = bool(self.abaqus['export_all_gs_instances'])
        self.abaqus['export_all_temporal_slices'] = bool(self.abaqus['export_all_temporal_slices'])
        self.abaqus['export_full_field'] = bool(self.abaqus['export_full_field'])

    def set_moose(self):
        self.moose = self.__uiEXPORT__['moose']
        self.moose['export_moose_input_file'] = bool(self.moose['export_moose_input_file'])
        self.moose['export_all_gs_instances'] = bool(self.moose['export_all_gs_instances'])
        self.moose['export_all_temporal_slices'] = bool(self.moose['export_all_temporal_slices'])
        self.moose['export_full_field'] = bool(self.moose['export_full_field'])

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
        retstr = "Attributes of EXPORT options: \n"
        retstr += self.CPRINTN('GSI', self.gsi)
        retstr += self.CPRINTN('Export to CTF', self.ctf['export_ctf'])
        retstr += self.CPRINTN('Pickle the data', self.pickle['export_pickle'])
        retstr += self.CPRINTN('Export to ABAQUS', self.abaqus['export_abaqus_input_file'])
        retstr += self.CPRINTN('Export to Moose', self.moose['export_moose_input_file'])
        return retstr
