from termcolor import colored
from color_specs import CPSPEC as CPS
import datatype_handlers as dth
import numpy as np
import re
import gops


class _UIGCS_MCGS_:
    """
    Parameters for grain strcuture characterisation
    """
    __slots__ = ('char_grains',  # from excel file - main
                 'char_stage',  # from excel file - main
                 'gs_characterisation_library',  # from excel file - main
                 'pxl_connectivity_length',  # from excel file - main
                 'enable_parallel_computation',  # from excel file - main
                 'char_gbseg',  # from excel file - main
                 'calc_g_area',  # from excel file - main
                 'calc_gb_length',  # from excel file - main
                 'calc_gb_length_crofton',  # from excel file - main
                 'calc_gb_njp_order',  # from excel file - main
                 'calc_g_eq_dia',  # from excel file - main
                 'calc_g_feq_dia',  # from excel file - main
                 'calc_g_solidity',  # from excel file - main
                 'calc_g_circularity',  # from excel file - main
                 'calc_g_mjaxis',  # from excel file - main
                 'calc_g_mnaxis',  # from excel file - main
                 'calc_g_morph_ori',  # from excel file - main
                 'calc_g_en',  # from excel file - main
                 'calc_g_ecc',  # from excel file - main
                 '__uiGSC__',
                 'gsi',
                 )

    def __init__(self,
                 uiGSC,
                 gsi=None
                 ):
        '''
        char_grains=True,
        char_stage='postsim',
        gs_characterisation_library='scikit-image',
        enable_parallel_computation=True,
        char_gbseg=True,
        calc_g_area=True,
        calc_gb_length=True,
        calc_gb_length_crofton=True,
        calc_gb_njp_order=True,
        calc_g_eq_dia=True,
        calc_g_feq_dia=True,
        calc_g_solidity=True,
        calc_g_circularity=True,
        calc_g_mjaxis=True,
        calc_g_mnaxis=True,
        calc_g_morph_ori=True,
        calc_g_en=True,
        calc_g_ecc=True,
        '''
        self.gsi = gsi
        # -------------------------------------------------
        self.__uiGSC__ = uiGSC
        # -------------------------------------------------
        self.set_char_grains()
        self.set_char_stage()
        self.set_gs_characterisation_library()
        self.set_enable_parallel_computation()
        self.set_char_gbseg()
        self.set_calc_g_area()
        self.set_calc_gb_length()
        self.set_calc_gb_length_crofton()
        self.set_calc_gb_njp_order()
        self.set_calc_g_eq_dia()
        self.set_calc_g_feq_dia()
        self.set_calc_g_solidity()
        self.set_calc_g_circularity()
        self.set_calc_g_mjaxis()
        self.set_calc_g_mnaxis()
        self.set_calc_g_morph_ori()
        self.set_calc_g_en()
        self.set_calc_g_ecc()

    def show_slots(self):
        return gops.slots(self)

    def show_methods(self):
        return gops.methods(self)

    def show_dunders(self):
        return gops.dunders(self)

    def print_dict(self):
        print(self.__uiGSC__)

    def reload(self):
        print("Please use ui.load_gsc()")

    def convert_strListOfNum_to_Numlist(self, strListOfNum):
        return list(map(int, re.findall(r'\d+', strListOfNum)))

    def set_char_grains(self):
        self.char_grains = bool(self.__uiGSC__['main']['char_grains'])

    def set_char_stage(self):
        self.char_stage = bool(self.__uiGSC__['main']['char_stage'])

    def set_gs_characterisation_library(self):
        self.gs_characterisation_library = self.__uiGSC__['main']['gs_characterisation_library']

    def set_enable_parallel_computation(self):
        self.enable_parallel_computation = bool(self.__uiGSC__['main']['enable_parallel_computation'])

    def set_char_gbseg(self):
        self.char_gbseg = bool(self.__uiGSC__['main']['char_gbseg'])

    def set_calc_g_area(self):
        self.calc_g_area = bool(self.__uiGSC__['main']['calc_g_area'])

    def set_calc_gb_length(self):
        self.calc_gb_length = bool(self.__uiGSC__['main']['calc_gb_length'])

    def set_calc_gb_length_crofton(self):
        self.calc_gb_length_crofton = bool(self.__uiGSC__['main']['calc_gb_length_crofton'])

    def set_calc_gb_njp_order(self):
        self.calc_gb_njp_order = bool(self.__uiGSC__['main']['calc_gb_njp_order'])

    def set_calc_g_eq_dia(self):
        self.calc_g_eq_dia = bool(self.__uiGSC__['main']['calc_g_eq_dia'])

    def set_calc_g_feq_dia(self):
        self.calc_g_feq_dia = bool(self.__uiGSC__['main']['calc_g_feq_dia'])

    def set_calc_g_solidity(self):
        self.calc_g_solidity = bool(self.__uiGSC__['main']['calc_g_solidity'])

    def set_calc_g_circularity(self):
        self.calc_g_circularity = bool(self.__uiGSC__['main']['calc_g_circularity'])

    def set_calc_g_mjaxis(self):
        self.calc_g_mjaxis = bool(self.__uiGSC__['main']['calc_g_mjaxis'])

    def set_calc_g_mnaxis(self):
        self.calc_g_mnaxis = bool(self.__uiGSC__['main']['calc_g_mnaxis'])

    def set_calc_g_morph_ori(self):
        self.calc_g_morph_ori = bool(self.__uiGSC__['main']['calc_g_morph_ori'])

    def set_calc_g_en(self):
        self.calc_g_en = bool(self.__uiGSC__['main']['calc_g_en'])

    def set_calc_g_ecc(self):
        self.calc_g_ecc = bool(self.__uiGSC__['main']['calc_g_ecc'])

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
        retstr = "Grain structure characterisation parameters: \n"
        retstr += self.CPRINTN('GSI', self.gsi)
        retstr += self.CPRINTN('CHAR_GRAINS', self.char_grains)
        retstr += self.CPRINTN('CHAR_STAGE', self.char_stage)
        retstr += self.CPRINTN('gs_characterisation_library', self.gs_characterisation_library)
        retstr += self.CPRINTN('enable_parallel_computation', self.enable_parallel_computation)
        retstr += self.CPRINTN('calc_g_area', self.calc_g_area)
        retstr += self.CPRINTN('calc_gb_length', self.calc_gb_length)
        retstr += self.CPRINTN('char_gbseg', self.char_gbseg)
        retstr += self.CPRINTN('calc_gb_length_CROFTON', self.calc_gb_length_crofton)
        retstr += self.CPRINTN('calc_gb_njp_order', self.calc_gb_njp_order)
        retstr += self.CPRINTN('calc_g_eq_dia', self.calc_g_eq_dia)
        retstr += self.CPRINTN('calc_g_feq_dia', self.calc_g_feq_dia)
        retstr += self.CPRINTN('calc_g_solidity', self.calc_g_solidity)
        retstr += self.CPRINTN('calc_g_circularity', self.calc_g_circularity)
        retstr += self.CPRINTN('calc_g_mjaxis', self.calc_g_mjaxis)
        retstr += self.CPRINTN('calc_g_mnaxis', self.calc_g_mnaxis)
        retstr += self.CPRINTN('calc_g_morph_ori', self.calc_g_morph_ori)
        retstr += self.CPRINTN('calc_g_en', self.calc_g_en)
        retstr += self.CPRINTN('calc_g_ecc', self.calc_g_ecc)
        return retstr
