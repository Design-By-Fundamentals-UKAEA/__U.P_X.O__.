from termcolor import colored, cprint
import datatype_handlers as dth
from color_specs import CPSPEC as CPS
import numpy as np
import re
import gops


class _UIMESH_MCGS_():
    __slots__ = ("target_fe_sw",
                 "base_GS_type",
                 "coarse_fine_fac_bfMesh",
                 "mesh_conformity",
                 "meshing_platform",
                 "meshing_algorithm",
                 "element_type",
                 "eltype_dstr",
                 "opt_mesh",
                 "grain_int_el_grad",
                 "grain_int_el_grad_par",
                 "global_elsize_min",
                 "global_elsize_max",
                 "MeshSizeFac",
                 "opt_parA",
                 "opt_parB",
                 "minAngle_DEF",
                 "maxAR_DEF",
                 "min_Jacobian_DEF",
                 "max_Jacobian_DEF",
                 "eseta",
                 "esetb",
                 "esetc",
                 "esetd",
                 "esete",
                 "esetf",
                 "esetg",
                 "nseta",
                 "nsetb",
                 "gsi",
                 "__uiMESH__"
                 )

    def __init__(self, uiMESH, gsi=None):
        self.gsi = gsi
        # -------------------------------------------------
        self.__uiMESH__ = uiMESH
        # -------------------------------------------------
        self.set_target_fe_sw()
        self.set_base_GS_type()
        self.set_coarse_fine_fac_bfMesh()
        self.set_mesh_conformity()
        self.set_meshing_platform()
        self.set_meshing_algorithm()
        self.set_element_type()
        self.set_eltype_dstr()
        self.set_opt_mesh()
        self.set_grain_int_el_grad()
        self.set_grain_int_el_grad_par()
        self.set_global_elsize_min()
        self.set_global_elsize_max()
        self.set_MeshSizeFac()
        self.set_opt_parA()
        self.set_opt_parB()
        self.set_minAngle_DEF()
        self.set_maxAR_DEF()
        self.set_min_Jacobian_DEF()
        self.set_max_Jacobian_DEF()
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
        print(self.__uiMESH__)

    def reload(self):
        print("Please use ui.load_mesh()")

    def convert_strListOfNum_to_Numlist(self, strListOfNum):
        return list(map(int, re.findall(r'\d+', strListOfNum)))

    def set_target_fe_sw(self):
        self.target_fe_sw = self.__uiMESH__['main']['target_fe_sw']

    def set_base_GS_type(self):
        self.base_GS_type = self.__uiMESH__['main']['base_GS_type']

    def set_coarse_fine_fac_bfMesh(self):
        self.coarse_fine_fac_bfMesh = self.__uiMESH__['main']['coarse_fine_fac_bfMesh']

    def set_mesh_conformity(self):
        self.mesh_conformity = self.__uiMESH__['main']['mesh_conformity']

    def set_meshing_platform(self):
        self.meshing_platform = self.__uiMESH__['main']['meshing_platform']

    def set_meshing_algorithm(self):
        self.meshing_algorithm = self.__uiMESH__['main']['meshing_algorithm']

    def set_element_type(self):
        self.element_type = self.__uiMESH__['main']['element_type']

    def set_eltype_dstr(self):
        self.eltype_dstr = self.__uiMESH__['main']['eltype_dstr']

    def set_opt_mesh(self):
        self.opt_mesh = bool(self.__uiMESH__['main']['opt_mesh'])

    def set_grain_int_el_grad(self):
        self.grain_int_el_grad = self.__uiMESH__['main']['grain_int_el_grad']

    def set_grain_int_el_grad_par(self):
        self.grain_int_el_grad_par = self.__uiMESH__['main']['grain_int_el_grad_par']

    def set_global_elsize_min(self):
        self.global_elsize_min = self.__uiMESH__['main']['global_elsize_min']

    def set_global_elsize_max(self):
        self.global_elsize_max = self.__uiMESH__['main']['global_elsize_max']

    def set_MeshSizeFac(self):
        self.MeshSizeFac = self.__uiMESH__['main']['MeshSizeFac']

    def set_opt_parA(self):
        self.opt_parA = self.__uiMESH__['main']['opt_parA']

    def set_opt_parB(self):
        self.opt_parB = self.__uiMESH__['main']['opt_parB']

    def set_minAngle_DEF(self):
        self.minAngle_DEF = self.__uiMESH__['main']['minAngle_DEF']

    def set_maxAR_DEF(self):
        self.maxAR_DEF = self.__uiMESH__['main']['maxAR_DEF']

    def set_min_Jacobian_DEF(self):
        self.min_Jacobian_DEF = self.__uiMESH__['main']['min_Jacobian_DEF']

    def set_max_Jacobian_DEF(self):
        self.max_Jacobian_DEF = self.__uiMESH__['main']['max_Jacobian_DEF']

    def set_eseta(self):
        self.eseta = self.__uiMESH__['main']['eseta']

    def set_esetb(self):
    	self.esetb = self.__uiMESH__['main']['esetb']

    def set_esetc(self):
    	self.esetc = self.__uiMESH__['main']['esetc']

    def set_esetd(self):
    	self.esetd = self.__uiMESH__['main']['esetd']

    def set_esete(self):
    	self.esete = self.__uiMESH__['main']['esete']

    def set_esetf(self):
    	self.esetf = self.__uiMESH__['main']['esetf']

    def set_esetg(self):
    	self.esetg = self.__uiMESH__['main']['esetg']

    def set_nseta(self):
    	self.nseta = self.__uiMESH__['main']['nseta']

    def set_nsetb(self):
    	self.nsetb = self.__uiMESH__['main']['nsetb']

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
        retstr = "Attributes of meshing: \n"
        retstr += self.CPRINTN('GSI', self.gsi)
        retstr += _ + "--/"*12 + "\n"
        retstr += self.CPRINTN('NAME', self.target_fe_sw)
        retstr += _ + "--/"*12 + "\n"
        retstr += _ + "PRIMARY PARAMETERS OF MESHING" + "\n"
        retstr += self.CPRINTN('coarse_fine_fac_bfMesh', self.coarse_fine_fac_bfMesh)
        retstr += self.CPRINTN('meshing_platform', self.meshing_platform)
        retstr += self.CPRINTN('meshing_algorithm', self.meshing_algorithm)
        retstr += self.CPRINTN('mesh_conformity', self.mesh_conformity)
        retstr += self.CPRINTN('element_type', self.element_type)
        retstr += self.CPRINTN('eltype_dstr', self.eltype_dstr)
        retstr += self.CPRINTN('global_elsize_min', self.global_elsize_min)
        retstr += self.CPRINTN('MeshSizeFac', self.MeshSizeFac)
        retstr += _ + "--/"*12 + "\n"
        retstr += _ + "MESH GRADIENT SPECIFICATIONS" + "\n"
        retstr += self.CPRINTN('grain_int_el_grad', self.grain_int_el_grad)
        retstr += self.CPRINTN('grain_int_el_grad_par', self.grain_int_el_grad_par)
        retstr += _ + "--/"*12 + "\n"
        retstr += _ + "MESH OPTIMIZATION PARAMETERS" + "\n"
        retstr += self.CPRINTN('opt_mesh', self.opt_mesh)
        retstr += self.CPRINTN('global_elsize_max', self.global_elsize_max)
        retstr += self.CPRINTN('opt_parA', self.opt_parA)
        retstr += self.CPRINTN('opt_parB', self.opt_parB)
        retstr += self.CPRINTN('minAngle_DEF', self.minAngle_DEF)
        retstr += self.CPRINTN('maxAR_DEF', self.maxAR_DEF)
        retstr += self.CPRINTN('min_Jacobian_DEF', self.min_Jacobian_DEF)
        retstr += self.CPRINTN('max_Jacobian_DEF', self.max_Jacobian_DEF)
        retstr += _ + "--/"*12 + "\n"
        retstr += _ + "ELEMENT SET SPECIFICATIONS" + "\n"
        retstr += self.CPRINTN('ELEMENT SET - A', self.eseta)
        retstr += self.CPRINTN('ELEMENT SET - B', self.esetb)
        retstr += self.CPRINTN('ELEMENT SET - C', self.esetc)
        retstr += self.CPRINTN('ELEMENT SET - D', self.esetd)
        retstr += self.CPRINTN('ELEMENT SET - E', self.esete)
        retstr += self.CPRINTN('ELEMENT SET - F', self.esetf)
        retstr += self.CPRINTN('ELEMENT SET - G', self.esetg)
        retstr += _ + "--/"*12 + "\n"
        retstr += _ + "NODAL SET SPECIFICATIONS" + "\n"
        retstr += self.CPRINTN('NODAL SET - A', self.nseta)
        retstr += self.CPRINTN('NODAL SET - B', self.nsetb)
        return retstr
