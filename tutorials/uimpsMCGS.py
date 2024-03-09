from termcolor import colored
from color_specs import CPSPEC as CPS
import datatype_handlers as dth
import numpy as np
import re
import gops


class _UIMPS_MCGS_():
    _keys_reinstateBOOL_ = ('partitioning',
                            'distrPlot',
                            'kde',
                            'distrPlotKdeOverlay',
                            'distrPlotKdePeaks_Markers',
                            'distrPlotKdePeaks_Text',
                            'distrPlotKdePeaks_Vlines',
                            'distr_plot_legend',
                            'distr_plot_title',
                            'distrPlotTitle_ps_gsi',
                            'distrPlotTitle_epochNumber',
                            'find_key_statDescriptors',
                            'distrPlotText_statDescriptors'
                            )

    __slots__ = ('area_flag',
                 'eqDia_flag',
                 'feqDia_flag',
                 'length_flag',
                 'segmentLengths_flag',
                 'segmentCounts_flag',
                 'njp_flag',
                 'jpAngles_flag',
                 'solidity_flag',
                 'circularity_flag',
                 'majorAxis_flag',
                 'minorAxis_flag',
                 'morphOri_flag',
                 'eulerNumber_flag',
                 'eccentricity_flag',
                 'aspectRatio_flag',
                 "area_type",
                 "gb_length_type",
                 'area_details',
                 'eqDia_details',
                 'feqDia_details',
                 'length_details',
                 'segmentLength_details',
                 'segmentCount_details',
                 'njp_details',
                 'jpAngle_details',
                 'majorAxis_details',
                 'minorAxis_details',
                 'aspectRatio_details',
                 'morphOri_details',
                 'circularity_details',
                 'solidity_details',
                 'eccentricity_details',
                 '__uiMPS__',
                 'gsi'
                 )

    def __init__(self,
                 uiMPS,
                 gsi=None
                 ):
        self.gsi = gsi
        # -------------------------------------------------
        self.__uiMPS__ = uiMPS
        # --------------------------------
        self.set_area_flag()
        self.set_eqDia_flag()
        self.set_feqDia_flag()
        self.set_length_flag()
        self.set_segmentLengths_flag()
        self.set_segmentCounts_flag()
        self.set_njp_flag()
        self.set_jpAngles_flag()
        self.set_solidity_flag()
        self.set_circularity_flag()
        self.set_majorAxis_flag()
        self.set_minorAxis_flag()
        self.set_morphOri_flag()
        self.set_eulerNumber_flag()
        self.set_eccentricity_flag()
        self.set_aspectRatio_flag()
        # --------------------------------
        self.set_area_type()
        self.set_gb_length_type()
        # --------------------------------
        self.set_area_details()
        self.set_eqDia_details()
        self.set_feqDia_details()
        self.set_length_details()
        self.set_segmentLength_details()
        self.set_segmentCount_details()
        self.set_njp_details()
        self.set_jpAngle_details()
        self.set_majorAxis_details()
        self.set_minorAxis_details()
        self.set_aspectRatio_details()
        self.set_morphOri_details()
        self.set_circularity_details()
        self.set_solidity_details()
        self.set_eccentricity_details()

    def show_slots(self):
        return gops.slots(self)

    def show_methods(self):
        return gops.methods(self)

    def show_dunders(self):
        return gops.dunders(self)

    def print_dict(self):
        print(self.__uiMPS__)

    def reload(self):
        print("Please use ui.load_mps()")

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
        retstr = "Attributes of ORI: \n"

        retstr += self.CPRINTN('GSI', self.gsi)
        retstr += self.CPRINTN('Grain area characterisation', self.area_flag)
        retstr += self.CPRINTN('Grain eq. dia. characterisation',self.eqDia_flag)
        retstr += self.CPRINTN('Grain Feret eq. dia. characterisation', self.feqDia_flag)
        retstr += self.CPRINTN('Grain bound. length characterisation', self.length_flag)
        retstr += self.CPRINTN('Grain bound. segment length characterisation', self.segmentLengths_flag)
        retstr += self.CPRINTN('Grain bound. segment counts', self.segmentCounts_flag)
        retstr += self.CPRINTN('Grain bound. Junction Point (JP) Order characterisation', self.njp_flag)
        retstr += self.CPRINTN('Grain bound. JP angle characterisation', self.jpAngles_flag)
        retstr += self.CPRINTN('Grain solidity characterisation', self.solidity_flag)
        retstr += self.CPRINTN('Grain circularity characterisation', self.circularity_flag)
        retstr += self.CPRINTN('Grain major axis characterasation', self.majorAxis_flag)
        retstr += self.CPRINTN('Grain minor axis characterasation', self.minorAxis_flag)
        retstr += self.CPRINTN('Grain morph. ori. characterisation', self.morphOri_flag)
        retstr += self.CPRINTN('Grain Euler number characterisation', self.eulerNumber_flag)
        retstr += self.CPRINTN('Grain eccentricity characterisation', self.eccentricity_flag)
        retstr += self.CPRINTN('Grain aspect ratio characterisation', self.aspectRatio_flag)
        return retstr

    def set_area_flag(self):
        self.area_flag = bool(self.__uiMPS__['main']['area_flag'])

    def set_eqDia_flag(self):
        self.eqDia_flag = bool(self.__uiMPS__['main']['eqDia_flag'])

    def set_feqDia_flag(self):
        self.feqDia_flag = bool(self.__uiMPS__['main']['feqDia_flag'])

    def set_length_flag(self):
        self.length_flag = bool(self.__uiMPS__['main']['length_flag'])

    def set_segmentLengths_flag(self):
        self.segmentLengths_flag = bool(self.__uiMPS__['main']['segmentLengths_flag'])

    def set_segmentCounts_flag(self):
        self.segmentCounts_flag = bool(self.__uiMPS__['main']['segmentCounts_flag'])

    def set_njp_flag(self):
        self.njp_flag = bool(self.__uiMPS__['main']['njp_flag'])

    def set_jpAngles_flag(self):
        self.jpAngles_flag = bool(self.__uiMPS__['main']['jpAngles_flag'])

    def set_solidity_flag(self):
        self.solidity_flag = bool(self.__uiMPS__['main']['solidity_flag'])

    def set_circularity_flag(self):
        self.circularity_flag = bool(self.__uiMPS__['main']['circularity_flag'])

    def set_majorAxis_flag(self):
        self.majorAxis_flag = bool(self.__uiMPS__['main']['majorAxis_flag'])

    def set_minorAxis_flag(self):
        self.minorAxis_flag = bool(self.__uiMPS__['main']['minorAxis_flag'])

    def set_morphOri_flag(self):
        self.morphOri_flag = bool(self.__uiMPS__['main']['morphOri_flag'])

    def set_eulerNumber_flag(self):
        self.eulerNumber_flag = bool(self.__uiMPS__['main']['eulerNumber_flag'])

    def set_eccentricity_flag(self):
        self.eccentricity_flag = bool(self.__uiMPS__['main']['eccentricity_flag'])

    def set_aspectRatio_flag(self):
        self.aspectRatio_flag = bool(self.__uiMPS__['main']['aspectRatio_flag'])

    def set_area_type(self):
        self.area_type = self.__uiMPS__['main']['area_type']

    def set_gb_length_type(self):
        self.gb_length_type = self.__uiMPS__['main']['gb_length_type']
    # -------------------------------------------------------------------------

    def set_area_details(self):
        self.area_details = self.__uiMPS__['area_details']
        # Excel import removes the bool type, so lets reinstate them
        for key in self._keys_reinstateBOOL_:
            if key in self.area_details.keys():
                self.area_details[key] = bool(self.area_details[key])
        # Convert user list input in Excel to python list format
        _ = self.area_details['partitioningPar_mcstatesValues']
        self.area_details['partitioningPar_mcstatesValues'] = self.convert_strListOfNum_to_Numlist(_)

    def set_eqDia_details(self):
        self.eqDia_details = self.__uiMPS__['eqDia_details']
        # Excel import removes the bool type, so lets reinstate them
        for key in self._keys_reinstateBOOL_:
            if key in self.eqDia_details.keys():
                self.eqDia_details[key] = bool(self.eqDia_details[key])
        # Convert user list input in Excel to python list format
        _ = self.eqDia_details['partitioningPar_mcstatesValues']
        self.eqDia_details['partitioningPar_mcstatesValues'] = self.convert_strListOfNum_to_Numlist(_)

    def set_feqDia_details(self):
        self.feqDia_details = self.__uiMPS__['feqDia_details']
        # Excel import removes the bool type, so lets reinstate them
        for key in self._keys_reinstateBOOL_:
            if key in self.feqDia_details.keys():
                self.feqDia_details[key] = bool(self.feqDia_details[key])
        # Convert user list input in Excel to python list format
        _ = self.feqDia_details['partitioningPar_mcstatesValues']
        self.feqDia_details['partitioningPar_mcstatesValues'] = self.convert_strListOfNum_to_Numlist(_)

    def set_length_details(self):
        self.length_details = self.__uiMPS__['gblength_details']
        # Excel import removes the bool type, so lets reinstate them
        for key in self._keys_reinstateBOOL_:
            if key in self.length_details.keys():
                self.length_details[key] = bool(self.length_details[key])
        # Convert user list input in Excel to python list format
        _ = self.length_details['partitioningPar_mcstatesValues']
        self.length_details['partitioningPar_mcstatesValues'] = self.convert_strListOfNum_to_Numlist(_)

    def set_segmentLength_details(self):
        self.segmentLength_details = self.__uiMPS__['segmentLength_details']
        # Excel import removes the bool type, so lets reinstate them
        for key in self._keys_reinstateBOOL_:
            if key in self.segmentLength_details.keys():
                self.segmentLength_details[key] = bool(self.segmentLength_details[key])
        # Convert user list input in Excel to python list format
        _ = self.segmentLength_details['partitioningPar_mcstatesValues']
        self.segmentLength_details['partitioningPar_mcstatesValues'] = self.convert_strListOfNum_to_Numlist(_)

    def set_segmentCount_details(self):
        self.segmentCount_details = self.__uiMPS__['segmentCount_details']
        # Excel import removes the bool type, so lets reinstate them
        for key in self._keys_reinstateBOOL_:
            if key in self.segmentCount_details.keys():
                self.segmentCount_details[key] = bool(self.segmentCount_details[key])
        # Convert user list input in Excel to python list format
        _ = self.segmentCount_details['partitioningPar_mcstatesValues']
        self.segmentCount_details['partitioningPar_mcstatesValues'] = self.convert_strListOfNum_to_Numlist(_)

    def set_njp_details(self):
        self.njp_details = self.__uiMPS__['njp_details']
        # Excel import removes the bool type, so lets reinstate them
        for key in self._keys_reinstateBOOL_:
            if key in self.njp_details.keys():
                self.njp_details[key] = bool(self.njp_details[key])
        # Convert user list input in Excel to python list format
        _ = self.njp_details['partitioningPar_mcstatesValues']
        self.njp_details['partitioningPar_mcstatesValues'] = self.convert_strListOfNum_to_Numlist(_)

    def set_jpAngle_details(self):
        self.jpAngle_details = self.__uiMPS__['jpAngle_details']
        # Excel import removes the bool type, so lets reinstate them
        for key in self._keys_reinstateBOOL_:
            if key in self.jpAngle_details.keys():
                self.jpAngle_details[key] = bool(self.jpAngle_details[key])
        # Convert user list input in Excel to python list format
        _ = self.jpAngle_details['partitioningPar_mcstatesValues']
        self.jpAngle_details['partitioningPar_mcstatesValues'] = self.convert_strListOfNum_to_Numlist(_)

    def set_majorAxis_details(self):
        self.majorAxis_details = self.__uiMPS__['majorAxis_details']
        # Excel import removes the bool type, so lets reinstate them
        for key in self._keys_reinstateBOOL_:
            if key in self.majorAxis_details.keys():
                self.majorAxis_details[key] = bool(self.majorAxis_details[key])
        # Convert user list input in Excel to python list format
        _ = self.majorAxis_details['partitioningPar_mcstatesValues']
        self.majorAxis_details['partitioningPar_mcstatesValues'] = self.convert_strListOfNum_to_Numlist(_)

    def set_minorAxis_details(self):
        self.minorAxis_details = self.__uiMPS__['minorAxis_details']
        # Excel import removes the bool type, so lets reinstate them
        for key in self._keys_reinstateBOOL_:
            if key in self.minorAxis_details.keys():
                self.minorAxis_details[key] = bool(self.minorAxis_details[key])
        # Convert user list input in Excel to python list format
        _ = self.minorAxis_details['partitioningPar_mcstatesValues']
        self.minorAxis_details['partitioningPar_mcstatesValues'] = self.convert_strListOfNum_to_Numlist(_)

    def set_aspectRatio_details(self):
        self.aspectRatio_details = self.__uiMPS__['aspectRatio_details']
        # Excel import removes the bool type, so lets reinstate them
        for key in self._keys_reinstateBOOL_:
            if key in self.aspectRatio_details.keys():
                self.aspectRatio_details[key] = bool(self.aspectRatio_details[key])
        # Convert user list input in Excel to python list format
        _ = self.aspectRatio_details['partitioningPar_mcstatesValues']
        self.aspectRatio_details['partitioningPar_mcstatesValues'] = self.convert_strListOfNum_to_Numlist(_)

    def set_morphOri_details(self):
        self.morphOri_details = self.__uiMPS__['morphOri_details']
        # Excel import removes the bool type, so lets reinstate them
        for key in self._keys_reinstateBOOL_:
            if key in self.morphOri_details.keys():
                self.morphOri_details[key] = bool(self.morphOri_details[key])
        # Convert user list input in Excel to python list format
        _ = self.morphOri_details['partitioningPar_mcstatesValues']
        self.morphOri_details['partitioningPar_mcstatesValues'] = self.convert_strListOfNum_to_Numlist(_)

    def set_circularity_details(self):
        self.circularity_details = self.__uiMPS__['circularity_details']
        # Excel import removes the bool type, so lets reinstate them
        for key in self._keys_reinstateBOOL_:
            if key in self.circularity_details.keys():
                self.circularity_details[key] = bool(self.circularity_details[key])
        # Convert user list input in Excel to python list format
        _ = self.circularity_details['partitioningPar_mcstatesValues']
        self.circularity_details['partitioningPar_mcstatesValues'] = self.convert_strListOfNum_to_Numlist(_)

    def set_solidity_details(self):
        self.solidity_details = self.__uiMPS__['solidity_details']
        # Excel import removes the bool type, so lets reinstate them
        for key in self._keys_reinstateBOOL_:
            if key in self.solidity_details.keys():
                self.solidity_details[key] = bool(self.solidity_details[key])
        # Convert user list input in Excel to python list format
        _ = self.solidity_details['partitioningPar_mcstatesValues']
        self.solidity_details['partitioningPar_mcstatesValues'] = self.convert_strListOfNum_to_Numlist(_)

    def set_eccentricity_details(self):
        self.eccentricity_details = self.__uiMPS__['eccentricity_details']
        # Excel import removes the bool type, so lets reinstate them
        for key in self._keys_reinstateBOOL_:
            if key in self.eccentricity_details.keys():
                self.eccentricity_details[key] = bool(self.eccentricity_details[key])
        # Convert user list input in Excel to python list format
        _ = self.eccentricity_details['partitioningPar_mcstatesValues']
        self.eccentricity_details['partitioningPar_mcstatesValues'] = self.convert_strListOfNum_to_Numlist(_)
