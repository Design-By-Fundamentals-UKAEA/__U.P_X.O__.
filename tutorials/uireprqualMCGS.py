from termcolor import colored
from color_specs import CPSPEC as CPS
import datatype_handlers as dth
import numpy as np
import re
import gops


class _UIREPRQUAL_MCGS_():
    __slots__ = ('target_type',
                 'sample_types',
                 'GrainArea',
                 'AspectRatio',
                 'GBLength',
                 'NNeighbours',
                 'GBSegLengths',
                 'GBSegCounts',
                 'GBJunctionPointOrder',
                 'EquivalentDiameter',
                 'FeretEquivalentDiameter',
                 'MinAxisLength',
                 'MaxAxisLength',
                 'GBJunctionPointAngles',
                 'GBCurvatures',
                 'GBRoughness',
                 'future_name_1',
                 'future_name_2',
                 'future_name_3',
                 'future_name_4',
                 'future_name_5',
                 'future_name_6',
                 'future_name_7',
                 'future_name_8',
                 'future_name_9',
                 'future_name_10',
                 'rqm1_metrics',
                 'rqm2_metrics',
                 'rqm2_MWUTest',
                 'rqm2_KWTest',
                 'rqm2_KSTest',
                 'gsi',
                 '__uiRQUAL__'
                 )


    def __init__(self,
                 uiRQUAL,
                 gsi=None
                 ):
        self.gsi = gsi
        # -------------------------------------------------
        self.__uiRQUAL__ = uiRQUAL
        # -------------------------------------------------
        self.set_target_type()
        self.set_sample_types()
        self.set_GrainArea()
        self.set_AspectRatio()
        self.set_GBLength()
        self.set_NNeighbours()
        self.set_GBSegLengths()
        self.set_GBSegCounts()
        self.set_GBJunctionPointOrder()
        self.set_EquivalentDiameter()
        self.set_FeretEquivalentDiameter()
        self.set_MinAxisLength()
        self.set_MaxAxisLength()
        self.set_GBJunctionPointAngles()
        self.set_GBCurvatures()
        self.set_GBRoughness()
        # -------------------------------------------------
        self.set_future_name_1()
        self.set_future_name_2()
        self.set_future_name_3()
        self.set_future_name_4()
        self.set_future_name_5()
        self.set_future_name_6()
        self.set_future_name_7()
        self.set_future_name_8()
        self.set_future_name_9()
        self.set_future_name_10()
        # -------------------------------------------------
        self.set_rqm1_metrics()
        self.set_rqm2_metrics()
        self.set_rqm2_MWUTest()
        self.set_rqm2_KWTest()
        self.set_rqm2_KSTest()

    def show_slots(self):
        return gops.slots(self)

    def show_methods(self):
        return gops.methods(self)

    def show_dunders(self):
        return gops.dunders(self)

    def print_dict(self):
        print(self.__uiRQUAL__)

    def reload(self):
        print("Please use ui.load_rqual()")

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
        retstr = "Attributes of REPRESENTATIVENESS QUALIFICATION: \n"
        retstr += self.CPRINTN('GSI', self.gsi)
        retstr += self.CPRINTN('Target type', self.target_type)
        retstr += self.CPRINTN('Sample types', self.sample_types)
        retstr += self.CPRINTN('Grain area consideration', self.GrainArea)
        retstr += self.CPRINTN('Grain aspect ration consideration', self.AspectRatio)
        retstr += self.CPRINTN('Grain boundary length consideration', self.GBLength)
        retstr += self.CPRINTN('Number of neaighbouring grains consideration', self.NNeighbours)
        retstr += self.CPRINTN('Grain boundary length consideration', self.GBSegLengths)
        retstr += self.CPRINTN('Grain boundary segment count consideration', self.GBSegCounts)
        retstr += self.CPRINTN('Grain boundary junction point order consideration', self.GBJunctionPointOrder)
        retstr += self.CPRINTN('Grain eq. dia. consideration', self.EquivalentDiameter)
        retstr += self.CPRINTN('Grain Feret eq. dia. consideration', self.FeretEquivalentDiameter)
        retstr += self.CPRINTN('Grain minor axis length consideration', self.MinAxisLength)
        retstr += self.CPRINTN('Grain major axis length consideration', self.MaxAxisLength)
        retstr += self.CPRINTN('Grain boundary junction point angle consideration', self.GBJunctionPointAngles)
        retstr += self.CPRINTN('Grain boundary curvatures consideration', self.GBCurvatures)
        retstr += self.CPRINTN('Grain boundary roughness consideration', self.GBRoughness)
        return retstr

    def set_target_type(self):
        self.target_type = self.__uiRQUAL__['main']['target_type']

    def set_sample_types(self):
        self.sample_types = self.__uiRQUAL__['main']['sample_types']

    def set_GrainArea(self):
        self.GrainArea = bool(self.__uiRQUAL__['main']['GrainArea'])

    def set_AspectRatio(self):
        self.AspectRatio = bool(self.__uiRQUAL__['main']['AspectRatio'])

    def set_GBLength(self):
        self.GBLength = bool(self.__uiRQUAL__['main']['GBLength'])

    def set_NNeighbours(self):
        self.NNeighbours = bool(self.__uiRQUAL__['main']['NNeighbours'])

    def set_GBSegLengths(self):
        self.GBSegLengths = bool(self.__uiRQUAL__['main']['GBSegLengths'])

    def set_GBSegCounts(self):
        self.GBSegCounts = bool(self.__uiRQUAL__['main']['GBSegCounts'])

    def set_GBJunctionPointOrder(self):
        self.GBJunctionPointOrder = bool(self.__uiRQUAL__['main']['GBJunctionPointOrder'])

    def set_EquivalentDiameter(self):
        self.EquivalentDiameter = bool(self.__uiRQUAL__['main']['EquivalentDiameter'])

    def set_FeretEquivalentDiameter(self):
        self.FeretEquivalentDiameter = bool(self.__uiRQUAL__['main']['FeretEquivalentDiameter'])

    def set_MinAxisLength(self):
        self.MinAxisLength = bool(self.__uiRQUAL__['main']['MinAxisLength'])

    def set_MaxAxisLength(self):
        self.MaxAxisLength = bool(self.__uiRQUAL__['main']['MaxAxisLength'])

    def set_GBJunctionPointAngles(self):
        self.GBJunctionPointAngles = bool(self.__uiRQUAL__['main']['GBJunctionPointAngles'])

    def set_GBCurvatures(self):
        self.GBCurvatures = bool(self.__uiRQUAL__['main']['GBCurvatures'])

    def set_GBRoughness(self):
        self.GBRoughness = bool(self.__uiRQUAL__['main']['GBRoughness'])

    def set_future_name_1(self):
        self.future_name_1 = self.__uiRQUAL__['main']['future_name_1']

    def set_future_name_2(self):
        self.future_name_2 = self.__uiRQUAL__['main']['future_name_2']

    def set_future_name_3(self):
        self.future_name_3 = self.__uiRQUAL__['main']['future_name_3']

    def set_future_name_4(self):
        self.future_name_4 = self.__uiRQUAL__['main']['future_name_4']

    def set_future_name_5(self):
        self.future_name_5 = self.__uiRQUAL__['main']['future_name_5']

    def set_future_name_6(self):
        self.future_name_6 = self.__uiRQUAL__['main']['future_name_6']

    def set_future_name_7(self):
        self.future_name_7 = self.__uiRQUAL__['main']['future_name_7']

    def set_future_name_8(self):
        self.future_name_8 = self.__uiRQUAL__['main']['future_name_8']

    def set_future_name_9(self):
        self.future_name_9 = self.__uiRQUAL__['main']['future_name_9']

    def set_future_name_10(self):
        self.future_name_10 = self.__uiRQUAL__['main']['future_name_10']

    def set_rqm1_metrics(self):
        self.rqm1_metrics = self.__uiRQUAL__['rqm1_metrics']

    def set_rqm2_metrics(self):
        self.rqm2_metrics = self.__uiRQUAL__['rqm2_metrics']

    def set_rqm2_MWUTest(self):
        self.rqm2_MWUTest = self.__uiRQUAL__['rqm2_MWUTest']

    def set_rqm2_KWTest(self):
        self.rqm2_KWTest = self.__uiRQUAL__['rqm2_KWTest']

    def set_rqm2_KSTest(self):
        self.rqm2_KSTest = self.__uiRQUAL__['rqm2_KSTest']
