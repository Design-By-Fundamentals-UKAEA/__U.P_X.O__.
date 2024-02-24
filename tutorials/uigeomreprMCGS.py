from termcolor import colored
from color_specs import CPSPEC as CPS
import datatype_handlers as dth
import numpy as np
import re
import gops


class _UIGEOMREPR_MCGS_():
    """
    make_mp_grain_centroids: bool :: Make MP of grain_centroids
    make_mp_grain_points: bool :: Grains as multi-point
    make_ring_grain_boundaries: bool :: GB as UPXO ring
    make_xtal_grain: bool :: Grains as UPXO XTAL object
    make_chull_grain: bool :: Make convex hull for each grain
    create_gbz: bool :: create_grain_boundary_zone

    CALL:
        from mcgs import _uidata_mcgs_generate_geom_reprs_
        uidata_georepr = _uidata_mcgs_generate_geom_reprs_(uidata)
    """
    __slots__ = ('make_mp_grain_centroids',
                 'make_mp_grain_points',
                 'make_ring_grain_boundaries',
                 'make_xtal_grain',
                 'make_chull_grain',
                 'create_gbz',
                 'gbz_thickness_factor',
                 '__uiGREP__',
                 'gsi'
                 )

    def __init__(self,
                 uiGREP,
                 gsi=None
                 ):
        self.gsi = gsi
        # -------------------------------------------------
        self.__uiGREP__ = uiGREP
        # -------------------------------------------------
        self.set_make_mp_grain_centroids_FLAG()
        self.set_make_mp_grain_points_FLAG()
        self.set_make_ring_grain_boundaries_FLAG()
        self.set_make_xtal_grain_FLAG()
        self.set_make_chull_grain_FLAG()
        self.set_create_gbz_FLAG()
        self.set_gbz_thickness_factor()

    def show_slots(self):
        return gops.slots(self)

    def show_methods(self):
        return gops.methods(self)

    def show_dunders(self):
        return gops.dunders(self)

    def print_dict(self):
        print(self.__uiGREP__)

    def reload(self):
        print("Please use ui.load_grep()")

    def convert_strListOfNum_to_Numlist(self, strListOfNum):
        return list(map(int, re.findall(r'\d+', strListOfNum)))

    def set_make_mp_grain_centroids_FLAG(self):
        self.make_mp_grain_centroids = bool(self.__uiGREP__['main']['make_mp_grain_centroids'])

    def set_make_mp_grain_points_FLAG(self):
        self.make_mp_grain_points = bool(self.__uiGREP__['main']['make_mp_grain_points'])

    def set_make_ring_grain_boundaries_FLAG(self):
        self.make_ring_grain_boundaries = bool(self.__uiGREP__['main']['make_ring_grain_boundaries'])

    def set_make_xtal_grain_FLAG(self):
        self.make_xtal_grain = bool(self.__uiGREP__['main']['make_xtal_grain'])

    def set_make_chull_grain_FLAG(self):
        self.make_chull_grain = bool(self.__uiGREP__['main']['make_chull_grain'])

    def set_create_gbz_FLAG(self):
        self.create_gbz = bool(self.__uiGREP__['main']['create_gbz'])

    def set_gbz_thickness_factor(self):
        self.gbz_thickness_factor = self.__uiGREP__['main']['gbz_thickness_factor']

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
        retstr = "Geometric representation parameters: \n"
        retstr += self.CPRINTN('GSI', self.gsi)
        retstr += self.CPRINTN('MAKE_MP_GRAIN_CENTOIDS', self.make_mp_grain_centroids)
        retstr += self.CPRINTN('MAKE_MP_GRAIN_POINTS', self.make_mp_grain_points)
        retstr += self.CPRINTN('MAKE_RING_GRAIN_BOUNDARIES', self.make_ring_grain_boundaries)
        retstr += self.CPRINTN('MAKE_XTAL_GRAIN', self.make_xtal_grain)
        retstr += self.CPRINTN('MAKE_CHULL_GRAIN', self.make_chull_grain)
        retstr += self.CPRINTN('CREATE_GBZ', self.create_gbz)
        return retstr
