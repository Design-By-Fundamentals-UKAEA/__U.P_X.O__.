from termcolor import colored
from color_specs import CPSPEC as CPS
import datatype_handlers as dth
import numpy as np
import re
import gops


class _UIGSVIS_MCGS_():
    __slots__ = ("grain_clr",
                 "gb_clr",
                 "gbz_clr",
                 "gcore_clr",
                 "future_name_3",
                 "future_name_4",
                 "future_name_5",
                 "future_name_6",
                 "future_name_7",
                 "future_name_8",
                 "future_name_9",
                 "future_name_10",
                 "gsi",
                 "__uiGSVIS__",
                 )

    def __init__(self,
                 uiGSVIS,
                 gsi=None
                 ):
        self.gsi = gsi
        self.__uiGSVIS__ = uiGSVIS
        self.set_grainRegion_clr()
        self.set_gb_clr()
        self.set_future_names()

    def show_slots(self):
        return gops.slots(self)

    def show_methods(self):
        return gops.methods(self)

    def show_dunders(self):
        return gops.dunders(self)

    def print_dict(self):
        print(self.__uiGSVIS__)

    def reload(self):
        print("Please use ui.load_gsvis()")

    def set_grainRegion_clr(self):
        self.grain_clr = self.__uiGSVIS__['main']['grain_clr']
        self.set_gbz_clr()
        self.set_gcore_clr()

    def set_gb_clr(self):
        self.gb_clr = self.__uiGSVIS__['main']['gb_clr']

    def set_future_names(self):
        self.set_future_name_3()
        self.set_future_name_4()
        self.set_future_name_5()
        self.set_future_name_6()
        self.set_future_name_7()
        self.set_future_name_8()
        self.set_future_name_9()
        self.set_future_name_10()

    def set_gbz_clr(self):
        self.gbz_clr = self.__uiGSVIS__['main']['gbz_clr']

    def set_gcore_clr(self):
        self.gcore_clr = self.__uiGSVIS__['main']['gcore_clr']

    def set_future_name_3(self):
        self.future_name_3 = self.__uiGSVIS__['main']['future_name_3']

    def set_future_name_4(self):
        self.future_name_4 = self.__uiGSVIS__['main']['future_name_4']

    def set_future_name_5(self):
        self.future_name_5 = self.__uiGSVIS__['main']['future_name_5']

    def set_future_name_6(self):
        self.future_name_6 = self.__uiGSVIS__['main']['future_name_6']

    def set_future_name_7(self):
        self.future_name_7 = self.__uiGSVIS__['main']['future_name_7']

    def set_future_name_8(self):
        self.future_name_8 = self.__uiGSVIS__['main']['future_name_8']

    def set_future_name_9(self):
        self.future_name_9 = self.__uiGSVIS__['main']['future_name_9']

    def set_future_name_10(self):
        self.future_name_10 = self.__uiGSVIS__['main']['future_name_10']

    def convert_strListOfNum_to_Numlist(self, strListOfNum):
        if type(strListOfNum) == str:
            return list(map(int, re.findall(r'\d+', strListOfNum)))
        else:
            return strListOfNum

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
        retstr = "Attributes of Grain Structure Visualization: \n"
        retstr += self.CPRINTN('GSI', self.gsi)
        retstr += _ + "--/"*12 + "\n"
        retstr += "Colour descriptions privided:" + "\n"
        retstr += self.CPRINTN('Grain region', self.grain_clr)
        retstr += self.CPRINTN('Grain boundary', self.gb_clr)
        retstr += self.CPRINTN('Grain boundary zone (GBZ)', self.gbz_clr)
        retstr += self.CPRINTN('Grain core (GCORE)', self.gcore_clr)
        retstr += _ + "   -"*8 + "\n"
        retstr += self.CPRINTN('Future_name_3', self.future_name_3)
        retstr += self.CPRINTN('Future_name_4', self.future_name_4)
        retstr += self.CPRINTN('Future_name_5', self.future_name_5)
        retstr += self.CPRINTN('Future_name_6', self.future_name_6)
        retstr += self.CPRINTN('Future_name_7', self.future_name_7)
        retstr += self.CPRINTN('Future_name_8', self.future_name_8)
        retstr += self.CPRINTN('Future_name_9', self.future_name_9)
        retstr += self.CPRINTN('Future_name_10', self.future_name_10)
        return retstr
