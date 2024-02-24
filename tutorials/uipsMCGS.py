from termcolor import colored
from color_specs import CPSPEC as CPS
import datatype_handlers as dth
import numpy as np
import re
import gops


class _UIPS_MCGS_():
    __slots__ = ('target_fe_sw',
                 'enforce_RQUAL',
                 'base_GS_type',
                 'kbTFactor',
                 'coarse_fine_fac_bfMesh',
                 'nGBZ_variations',
                 'GBZ_t_spec',
                 'GBZ_t_factor_max',
                 'mesh_conformity',
                 'meshing_platform',
                 'meshing_algorithm',
                 'element_type',
                 'eltype_dstr',
                 'opt_mesh',
                 'grain_int_el_grad',
                 'grain_int_el_grad_par',
                 'global_elsize_min',
                 'global_elsize_max',
                 'MeshSizeFac',
                 'opt_parA',
                 'opt_parB',
                 'minAngle_DEF',
                 'maxAR_DEF',
                 'min_Jacobian_DEF',
                 'max_Jacobian_DEF',
                 'eseta',
                 'esetb',
                 'esetc',
                 'esetd',
                 'esete',
                 'esetf',
                 'esetg',
                 'nseta',
                 'nsetb',
                 'gsi',
                 '__uiPS__'
                 )

    def __init__(self, uiPS, gsi=None):
        self.gsi = gsi
        # -------------------------------------------------
        self.__uiPS__ = uiPS
        # -------------------------------------------------
        self.set_target_fe_sw()
        self.set_enforce_RQUAL()
        self.set_base_GS_type()
        self.set_kbTFactor()
        self.set_coarse_fine_fac_bfMesh()
        self.set_nGBZ_variations()
        self.set_GBZ_t_spec()
        self.set_GBZ_t_factor_max()
        # -------------------------------------------------
        self.set_mesh_conformity()
        self.set_meshing_platform()
        self.set_meshing_algorithm()
        # -------------------------------------------------
        self.set_element_type()
        self.set_eltype_dstr()
        self.set_grain_int_el_grad()
        self.set_grain_int_el_grad_par()
        self.set_global_elsize_min()
        self.set_global_elsize_max()
        self.set_MeshSizeFac()
        # -------------------------------------------------
        self.set_opt_mesh()
        self.set_opt_parA()
        self.set_opt_parB()
        self.set_minAngle_DEF()
        self.set_maxAR_DEF()
        self.set_min_Jacobian_DEF()
        self.set_max_Jacobian_DEF()
        # -------------------------------------------------
        self.set_eseta()
        self.set_esetb()
        self.set_esetc()
        self.set_esetd()
        self.set_esete()
        self.set_esetf()
        self.set_esetg()
        self.set_nseta()
        self.set_nsetb()

    def show_slots(self):
        return gops.slots(self)

    def show_methods(self):
        return gops.methods(self)

    def show_dunders(self):
        return gops.dunders(self)

    def print_dict(self):
        print(self.__uiPS__)

    def reload(self):
        print("Please use ui.load_ps()")

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
        retstr += self.CPRINTN("target_fe_sw", "Target FE software of all instances")
        retstr += self.CPRINTN("enforce_RQUAL", "Enfore representativeness qualification for all instances")
        retstr += self.CPRINTN("base_GS_type", "Base grain structure type")
        retstr += self.CPRINTN("kbTFactor", "Boltzmann temperature factor,float,[0,1]. Use to control GB roughness")
        retstr += self.CPRINTN("coarse_fine_fac_bfMesh", "Coarseness or fineness factor, float, > 0, to use before meshing")
        retstr += self.CPRINTN("nGBZ_variations", "Number of variations in grain boundary zone representations")
        retstr += self.CPRINTN("GBZ_t_spec", "Grain boundary zone thickness specticication")
        retstr += self.CPRINTN("GBZ_t_factor_max", "Maximum grain boudnary zone thickness")
        retstr += self.CPRINTN("mesh_conformity", "Conformityy of finite elements to local grain boundary contour")
        retstr += self.CPRINTN("meshing_platform", "Meshing platform to use")
        retstr += self.CPRINTN("meshing_algorithm", "FE meshing algorithm specification")
        retstr += self.CPRINTN("element_type", "Type of finite element")
        retstr += self.CPRINTN("eltype_dstr", "Element type distribution")
        retstr += self.CPRINTN("opt_mesh", "Flag to optimize mesh")
        retstr += self.CPRINTN("grain_int_el_grad", "Element gradient inside grain")
        retstr += self.CPRINTN("grain_int_el_grad_par", "Element gradient inside grain parameter")
        retstr += self.CPRINTN("global_elsize_min", "Global element size - minimum")
        retstr += self.CPRINTN("global_elsize_max", "Global element size - maximum")
        retstr += self.CPRINTN("MeshSizeFac", "Mesh size factor (i.e. times maximum global element size)")
        retstr += self.CPRINTN("opt_parA", "Mesh optimization parameter A")
        retstr += self.CPRINTN("opt_parB", "Mesh optimization parameter B")
        retstr += self.CPRINTN("minAngle_DEF", "Default element minimum angle")
        retstr += self.CPRINTN("maxAR_DEF", "Default element aspect ratio")
        retstr += self.CPRINTN("min_Jacobian_DEF", "Default element Jacobian - minimum")
        retstr += self.CPRINTN("max_Jacobian_DEF", "Default element Jacobian - maximum")
        retstr += self.CPRINTN("eseta", "lement set - a")
        retstr += self.CPRINTN("esetb", "Element set - b")
        retstr += self.CPRINTN("esetc", "Element set - c")
        retstr += self.CPRINTN("esetd", "Element set - d")
        retstr += self.CPRINTN("esete", "Element set - e")
        retstr += self.CPRINTN("esetf", "Element set - f")
        retstr += self.CPRINTN("esetg", "Element set - g")
        retstr += self.CPRINTN("nseta", "Node set - a")
        retstr += self.CPRINTN("nsetb", "Node set - b")
        return retstr


    def set_target_fe_sw(self):
        self.target_fe_sw = self.__uiPS__['target_fe_sw']

    def set_enforce_RQUAL(self):
        self.enforce_RQUAL = self.__uiPS__['enforce_RQUAL']

    def set_base_GS_type(self):
        self.base_GS_type = self.__uiPS__['base_GS_type']

    def set_kbTFactor(self):
        self.kbTFactor = self.__uiPS__['kbTFactor']

    def set_coarse_fine_fac_bfMesh(self):
        self.coarse_fine_fac_bfMesh = self.__uiPS__['coarse_fine_fac_bfMesh']

    def set_nGBZ_variations(self):
        self.nGBZ_variations = self.__uiPS__['nGBZ_variations']

    def set_GBZ_t_spec(self):
        self.GBZ_t_spec = self.__uiPS__['GBZ_t_spec']

    def set_GBZ_t_factor_max(self):
        self.GBZ_t_factor_max = self.__uiPS__['GBZ_t_factor_max']

    def set_mesh_conformity(self):
        self.mesh_conformity = self.__uiPS__['mesh_conformity']

    def set_meshing_platform(self):
        self.meshing_platform = self.__uiPS__['meshing_platform']

    def set_meshing_algorithm(self):
        self.meshing_algorithm = self.__uiPS__['meshing_algorithm']

    def set_element_type(self):
        self.element_type = self.__uiPS__['element_type']

    def set_eltype_dstr(self):
        self.eltype_dstr = self.__uiPS__['eltype_dstr']

    def set_opt_mesh(self):
        self.opt_mesh = self.__uiPS__['opt_mesh']

    def set_grain_int_el_grad(self):
        self.grain_int_el_grad = self.__uiPS__['grain_int_el_grad']

    def set_grain_int_el_grad_par(self):
        self.grain_int_el_grad_par = self.__uiPS__['grain_int_el_grad_par']

    def set_global_elsize_min(self):
        self.global_elsize_min = self.__uiPS__['global_elsize_min']

    def set_global_elsize_max(self):
        self.global_elsize_max = self.__uiPS__['global_elsize_max']

    def set_MeshSizeFac(self):
        self.MeshSizeFac = self.__uiPS__['MeshSizeFac']

    def set_opt_parA(self):
        self.opt_parA = self.__uiPS__['opt_parA']

    def set_opt_parB(self):
        self.opt_parB = self.__uiPS__['opt_parB']

    def set_minAngle_DEF(self):
        self.minAngle_DEF = self.__uiPS__['minAngle_DEF']

    def set_maxAR_DEF(self):
        self.maxAR_DEF = self.__uiPS__['maxAR_DEF']

    def set_min_Jacobian_DEF(self):
        self.min_Jacobian_DEF = self.__uiPS__['min_Jacobian_DEF']

    def set_max_Jacobian_DEF(self):
        self.max_Jacobian_DEF = self.__uiPS__['max_Jacobian_DEF']

    def set_eseta(self):
        self.eseta = self.__uiPS__['eseta']

    def set_esetb(self):
        self.esetb = self.__uiPS__['esetb']

    def set_esetc(self):
        self.esetc = self.__uiPS__['esetc']

    def set_esetd(self):
        self.esetd = self.__uiPS__['esetd']

    def set_esete(self):
        self.esete = self.__uiPS__['esete']

    def set_esetf(self):
        self.esetf = self.__uiPS__['esetf']

    def set_esetg(self):
        self.esetg = self.__uiPS__['esetg']

    def set_nseta(self):
        self.nseta = self.__uiPS__['nseta']

    def set_nsetb(self):
        self.nsetb = self.__uiPS__['nsetb']
