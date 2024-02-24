from termcolor import colored
from color_specs import CPSPEC as CPS
import datatype_handlers as dth
import numpy as np
import re
import gops


class _UIGRID_MCGS_:
    """
    type : str :: Type of underlying grid
    dim: int :: Physical dimensionality of the domain
    xmin : float :: X-coordinate of the start of the simulation domain
    xmax : float :: X-coordinate of the end of the simulation domain
    xinc : float :: X-coordinate increments in the simulation domain
    ymin : float :: Y-coordinate of the start of the simulation domain
    ymax : float :: Y-coordinate of the end of the simulation domain
    yinc : float :: Y-coordinate increments in the simulation domain
    zmin : float :: Z-coordinate of the start of the simulation domain
    zmax : float :: Z-coordinate of the end of the simulation domain
    zinc : float :: Z-coordinate increments in the simulation domain
    px_size: float :: Pixel size in the grid
    transformation: str :: Geometric transformation operation for the grid
    __lock__: dict :: Sub-locks (type, npixx, npixy, npix) and summary lock (_)

    CALL:
        from mcgs import _manual_uidata_mcgs_gridding_definitions_ as imname
        uidata_gridpar = imname(domainsize=Value,
                                read_from_file=Value,
                                filename=Value)
    """
    _def_Type = 'square'
    _def_dim = 2
    _def_xmin, _def_xmax, _def_xinc = 0.0, 50.0, 1.0
    _def_ymin, _def_ymax, _def_yinc = 0.0, 50.0, 1.0
    _def_zmin, _def_zmax, _def_zinc = 0.0, 50.0, 1.0
    _def_px_size = 1.0
    _def_transformation = 'none'
    __slots__ = ('type', 'dim',
                 'xmin', 'xmax', 'xinc', 'xlim',
                 'ymin', 'ymax', 'yinc', 'ylim',
                 'zmin', 'zmax', 'zinc', 'zlim',
                 'px_size', 'transformation',
                 '__uiGRID__',
                 'gsi',
                 )

    def __init__(self, uiGRID, gsi=None):
        self.gsi = gsi
        # --------------------------------
        self.__uiGRID__ = uiGRID
        # --------------------------------
        self.type = self.__uiGRID__['main']['type']
        self.dim = int(self.__uiGRID__['main']['dim'])
        # --------------------------------
        self.px_size = self.__uiGRID__['main']['px_size']
        # --------------------------------
        self.xmin = self.__uiGRID__['main']['xmin']
        self.xmax = self.__uiGRID__['main']['xmax']
        self.xinc = self.__uiGRID__['main']['xinc']
        self.xlim = (self.xmin, self.xmax)
        # --------------------------------
        self.ymin = self.__uiGRID__['main']['ymin']
        self.ymax = self.__uiGRID__['main']['ymax']
        self.yinc = self.__uiGRID__['main']['yinc']
        self.ylim = (self.ymin, self.ymax)
        # --------------------------------
        self.zmin = self.__uiGRID__['main']['zmin']
        self.zmax = self.__uiGRID__['main']['zmax']
        self.zinc = self.__uiGRID__['main']['zinc']
        self.zlim = (self.zmin, self.zmax)
        # --------------------------------
        self.transformation = uiGRID['main']['transformation']

    def show_slots(self):
        return gops.slots(self)

    def show_methods(self):
        return gops.methods(self)

    def show_dunders(self):
        return gops.dunders(self)

    def print_dict(self):
        print(self.__uiGRID__)

    def reload(self):
        print("Please use ui.load_grid()")

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
        retstr = 'Attribues of gridding definitions: \n'
        retstr += self.CPRINTN('GSI', self.gsi)
        retstr += self.CPRINTN('TYPE', self.type)
        retstr += self.CPRINTN('DIMENSIONALITY', self.dim)
        retstr += self.CPRINTN('X', (self.xmin, self.xmax, self.xinc))
        retstr += self.CPRINTN('Y', (self.ymin, self.ymax, self.yinc))
        retstr += self.CPRINTN('Z', (self.zmin, self.zmax, self.zinc))
        retstr += self.CPRINTN('PIXEL SIZE', str(self.px_size) + 'µm')
        retstr += self.CPRINTN('TRANSFORMATION', self.transformation)
        return retstr

    @property
    def x(self):
        return (self.xmin, self.xmax, self.xinc)

    @property
    def y(self):
        return (self.ymin, self.ymax, self.yinc)

    @property
    def z(self):
        return (self.zmin, self.zmax, self.zinc)

    @property
    def xls(self):
        # Make the linear space for x
        return np.linspace(self.xmin,
                           self.xmax,
                           int((self.xmax-self.xmin)/self.xinc+1))

    @property
    def yls(self):
        # Make the linear space for y
        return np.linspace(self.ymin,
                           self.ymax,
                           int((self.ymax-self.ymin)/self.yinc+1))

    @property
    def zls(self):
        # Make the linear space for y
        return np.linspace(self.ymin,
                           self.ymax,
                           int((self.ymax-self.ymin)/self.yinc+1))

    @property
    def grid(self):
        if self.dim == 2:
            # make the grid from linear spaces of x and y
            x, y = np.meshgrid(self.xls, self.yls)
            npixels = x.size*y.size
        return x, y, npixels

    @property
    def lock_status(self):
        if self.__lock__['_']:
            print(f"{colored('Locked sub-locks exist','red',attrs=['bold'])}")
        return self.__lock__['_']

    @property
    def locks(self):
        return self.__lock__

    @property
    def lock_update(self):
        """
        Updates the lock. Outcome could be leave lock either in open or
        locked state. Returns None.
        """
        keys = list(self.locks.keys())
        keys[list(self.locks.keys()).index('_')] = -1
        keys.remove(-1)
        sublocks = [self.locks[key] for key in keys]
        if True in set(sublocks):
            self.__lock__['_'] = True
        else:
            self.__lock__['_'] = False
        return None
