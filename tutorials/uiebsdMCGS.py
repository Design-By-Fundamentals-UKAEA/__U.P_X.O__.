from color_specs import CPSPEC as CPS
from termcolor import colored
import datatype_handlers as dth
import numpy as np
import re
import gops


class _UIEBSD_MCGS_():
    __slots__ = ("ebsd_map_folder",
                 "ebsd_map_filename",
                 "ebsd_file_type",
                 "crc_file_availability",
                 "xtal_structure",
                 "make_subdomains",
                 "subdomain_nx",
                 "subdomain_ny",
                 "future_name1",
                 "future_name2",
                 "future_name3",
                 "future_name4",
                 "future_name5",
                 "future_name6",
                 "future_name7",
                 "future_name8",
                 "future_name9",
                 "future_name10",
                 "future_name11",
                 "future_name12",
                 "pylibrary",
                 "identify_grains",
                 "identify_gb",
                 "identify_tpj",
                 "prior_austenite_reconstruction",
                 "gb_identification_technique",
                 "tjp_identification_technique",
                 "generate_mc_lattice",
                 "write_mc_lattice",
                 "simulate_further_grain_growth",
                 "identify_border_grains",
                 "calc_mean_ori_grains",
                 "identify_ori_clusters",
                 "export_mean_ori_ALL",
                 "export_mean_ori_clusters",
                 "gsi",
                 "__uiEBSD__",
                 )

    def __init__(self, uiEBSD, gsi=None):
        # -------------------------------------------------
        self.gsi = gsi
        self.__uiEBSD__ = uiEBSD
        self.set_ebsd_file_details()
        self.set_xtal_structure()
        self.set_subdomain_details()
        self.set_future_names()
        self.set_pylibrary()
        self.set_morph_identification_details()
        self.set_mc_coupling_details()
        self.set_orientation_handling_details()
        self.set_export_details()
        # -------------------------------------------------

    def show_slots(self):
        return gops.slots(self)

    def show_methods(self):
        return gops.methods(self)

    def show_dunders(self):
        return gops.dunders(self)

    def print_dict(self):
        print(self.__uiEBSD__)

    def reload(self):
        print("Please use ui.load_ebsd()")

    def convert_strListOfNum_to_Numlist(self, strListOfNum):
        return list(map(int, re.findall(r'\d+', strListOfNum)))

    def set_ebsd_file_details(self):
        self.set_ebsd_map_folder()
        self.set_ebsd_map_filename()
        self.set_ebsd_file_type()
        self.set_crc_file_availability()

    def set_ebsd_map_folder(self):
        self.ebsd_map_folder = self.__uiEBSD__['main']['ebsd_map_folder']

    def set_ebsd_map_filename(self):
        self.ebsd_map_filename = self.__uiEBSD__['main']['ebsd_map_filename']

    def set_ebsd_file_type(self):
        self.ebsd_file_type = self.__uiEBSD__['main']['ebsd_file_type']

    def set_crc_file_availability(self):
        self.crc_file_availability = bool(self.__uiEBSD__['main']['crc_file_availability'])

    def set_xtal_structure(self):
        self.xtal_structure = self.__uiEBSD__['main']['xtal_structure']

    def set_subdomain_details(self):
        self.set_make_subdomains()
        self.set_subdomain_nx()
        self.set_subdomain_ny()

    def set_make_subdomains(self):
        self.make_subdomains = bool(self.__uiEBSD__['main']['make_subdomains'])

    def set_subdomain_nx(self):
        self.subdomain_nx = self.__uiEBSD__['main']['subdomain_nx']

    def set_subdomain_ny(self):
        self.subdomain_ny = self.__uiEBSD__['main']['subdomain_ny']

    def set_future_names(self):
        self.set_future_name1()
        self.set_future_name2()
        self.set_future_name3()
        self.set_future_name4()
        self.set_future_name5()
        self.set_future_name6()
        self.set_future_name7()
        self.set_future_name8()
        self.set_future_name9()
        self.set_future_name10()
        self.set_future_name11()
        self.set_future_name12()

    def set_future_name1(self):
        self.future_name1 = self.__uiEBSD__['main']['future_name1']

    def set_future_name2(self):
        self.future_name2 = self.__uiEBSD__['main']['future_name2']

    def set_future_name3(self):
        self.future_name3 = self.__uiEBSD__['main']['future_name3']

    def set_future_name4(self):
        self.future_name4 = self.__uiEBSD__['main']['future_name4']

    def set_future_name5(self):
        self.future_name5 = self.__uiEBSD__['main']['future_name5']

    def set_future_name6(self):
        self.future_name6 = self.__uiEBSD__['main']['future_name6']

    def set_future_name7(self):
        self.future_name7 = self.__uiEBSD__['main']['future_name7']

    def set_future_name8(self):
        self.future_name8 = self.__uiEBSD__['main']['future_name8']

    def set_future_name9(self):
        self.future_name9 = self.__uiEBSD__['main']['future_name9']

    def set_future_name10(self):
        self.future_name10 = self.__uiEBSD__['main']['future_name10']

    def set_future_name11(self):
        self.future_name11 = self.__uiEBSD__['main']['future_name11']

    def set_future_name12(self):
        self.future_name12 = self.__uiEBSD__['main']['future_name12']

    def set_pylibrary(self):
        self.pylibrary = self.__uiEBSD__['details']['pylibrary']

    def set_morph_identification_details(self):
        self.set_identify_grains()
        self.set_identify_border_grains()
        self.set_identify_gb()
        self.set_identify_tpj()
        self.set_prior_austenite_reconstruction()
        self.set_gb_identification_technique()
        self.set_tjp_identification_technique()


    def set_identify_grains(self):
        self.identify_grains = bool(self.__uiEBSD__['details']['identify_grains'])

    def set_identify_border_grains(self):
        self.identify_border_grains = bool(self.__uiEBSD__['details']['identify_border_grains'])

    def set_identify_gb(self):
        self.identify_gb = bool(self.__uiEBSD__['details']['identify_gb'])

    def set_identify_tpj(self):
        self.identify_tpj = bool(self.__uiEBSD__['details']['identify_tpj'])

    def set_prior_austenite_reconstruction(self):
        self.prior_austenite_reconstruction = bool(self.__uiEBSD__['details']['prior_austenite_reconstruction'])

    def set_gb_identification_technique(self):
        self.gb_identification_technique = self.__uiEBSD__['details']['gb_identification_technique']

    def set_tjp_identification_technique(self):
        self.tjp_identification_technique = self.__uiEBSD__['details']['tjp_identification_technique']

    def set_mc_coupling_details(self):
        self.set_generate_mc_lattice()
        self.set_write_mc_lattice()
        self.set_simulate_further_grain_growth()

    def set_generate_mc_lattice(self):
        self.generate_mc_lattice = bool(self.__uiEBSD__['details']['generate_mc_lattice'])

    def set_write_mc_lattice(self):
        self.write_mc_lattice = bool(self.__uiEBSD__['details']['write_mc_lattice'])

    def set_simulate_further_grain_growth(self):
        self.simulate_further_grain_growth = bool(self.__uiEBSD__['details']['simulate_further_grain_growth'])

    def set_orientation_handling_details(self):
        self.set_calc_mean_ori_grains()
        self.set_identify_ori_clusters()

    def set_calc_mean_ori_grains(self):
        self.calc_mean_ori_grains = bool(self.__uiEBSD__['details']['calc_mean_ori_grains'])

    def set_identify_ori_clusters(self):
        self.identify_ori_clusters = bool(self.__uiEBSD__['details']['identify_ori_clusters'])

    def set_export_details(self):
        self.set_export_mean_ori_ALL()
        self.set_export_mean_ori_clusters()

    def set_export_mean_ori_ALL(self):
        self.export_mean_ori_ALL = self.__uiEBSD__['details']['export_mean_ori_ALL']

    def set_export_mean_ori_clusters(self):
        self.export_mean_ori_clusters = self.__uiEBSD__['details']['export_mean_ori_clusters']

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
        retstr = "Attributes of Simulation parameters:\n"
        retstr += self.CPRINTN('GSI', self.gsi)
        retstr += _ + "--/"*12 + "\n"
        retstr += self.CPRINTN('ebsd_map_folder', self.ebsd_map_folder)
        retstr += self.CPRINTN('ebsd_map_filename', self.ebsd_map_filename)
        retstr += self.CPRINTN('ebsd_file_type', self.ebsd_file_type)
        retstr += self.CPRINTN('crc_file_availability', self.crc_file_availability)
        retstr += _ + "   -"*8 + "\n"
        retstr += self.CPRINTN('xtal_structure', self.xtal_structure)
        retstr += _ + "   -"*8 + "\n"
        retstr += self.CPRINTN('make_subdomains', self.make_subdomains)
        retstr += self.CPRINTN('subdomain_nx', self.subdomain_nx)
        retstr += self.CPRINTN('subdomain_ny', self.subdomain_ny)
        retstr += _ + "--/"*12 + "\n"
        retstr += self.CPRINTN('future_name1', self.future_name1)
        retstr += (_ + _ + ".\n")*2
        retstr += self.CPRINTN('future_name12', self.future_name12)
        retstr += _ + "--/"*12 + "\n"
        retstr += self.CPRINTN('pylibrary', self.pylibrary)
        retstr += self.CPRINTN('identify_grains', self.identify_grains)
        retstr += self.CPRINTN('identify_border_grains', self.identify_border_grains)
        retstr += self.CPRINTN('identify_gb', self.identify_gb)
        retstr += self.CPRINTN('identify_tpj', self.identify_tpj)
        retstr += self.CPRINTN('gb_identification_technique', self.gb_identification_technique)
        retstr += self.CPRINTN('tjp_identification_technique', self.tjp_identification_technique)
        retstr += _ + "   -"*8 + "\n"
        retstr += self.CPRINTN('prior_austenite_reconstruction', self.prior_austenite_reconstruction)
        retstr += _ + "   -"*8 + "\n"
        retstr += self.CPRINTN('generate_mc_lattice', self.generate_mc_lattice)
        retstr += self.CPRINTN('write_mc_lattice', self.write_mc_lattice)
        retstr += self.CPRINTN('simulate_further_grain_growth', self.simulate_further_grain_growth)
        retstr += _ + "   -"*8 + "\n"
        retstr += self.CPRINTN('calc_mean_ori_grains', self.calc_mean_ori_grains)
        retstr += self.CPRINTN('identify_ori_clusters', self.identify_ori_clusters)
        retstr += _ + "   -"*8 + "\n"
        retstr += self.CPRINTN('export_mean_ori_ALL', self.export_mean_ori_ALL)
        retstr += self.CPRINTN('export_mean_ori_clusters', self.export_mean_ori_clusters)
        return retstr
