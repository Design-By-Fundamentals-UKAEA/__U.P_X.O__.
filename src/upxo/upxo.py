import random
import numpy as np
import string
from scipy.spatial import Voronoi
from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
class pxtal():
    GS     = None
    TEX    = None
    FIELDS = None
    def __init__(self):
        pass

class gstr(pxtal):
    def __init__(self):
        # PXTAL material and model dimensionality
        self.matname     = None
        self.dimen       = None
        # PXTAL model domain size and location
        self.origin      = None
#        self.bounds      = None
#        self.stepSize    = None
        self.xLength     = None
        self.yLength     = None
        self.zLength     = None
        # PXTAL model surface area and volume
 #       self.surfaceArea = None
 #       self.totalVolume = None
        # PXTAL generating technique
        self.technique   = None
        # House keeping
        self.cmdSep      = None
        self.cmdSepCount = None
        # L0GS parameters
    
    # SETTER DEFINITIONS
    def setUID(self, *args):
        """Unique ID string of the poly-xtal."""
        """Uasge: gs.setUID(), gs.setUID('ID_string')"""
        if len(args) == 0:
            self.gsid = 'GSID__{}'.format(''.join(random.choice(string.ascii_letters + string.digits+string.punctuation) for i in range(12)))
            print("Unique grain structure ID is   {}".format(self.gsid))
        elif len(args) >= 1:
            self.gsid = 'GSID__' + args[0]
            print("You ID'd this grain structure as {}".format(self.gsid))

    @property
    def setHouseKeepingRules(self):
        """House keeping rules related values."""
        """Usage: gs.setHouseKeepingRules"""
        self.cmdSep      = '_'
        self.cmdSepCount = 40

    def setMaterialName(self, *args):
        ## MOVE TO PXTAL
        """Name the material which the poly-xtal will represent."""
        """Usage: gs.setMaterialName(), gs.setMaterialName('NameString')"""
        if len(args) == 0:
            self.matname  = input("Material name (default cu-cr-zr) >>> ") or "cu-cr-zr"
        elif len(args) >= 1:
            self.matname  = args[0]
            print("Material name is {}".format(self.matname))

    def setGenTech(self, *args):
        """Set generating technique to use for making poly-xtal."""
        """Usage: """
        if len(args) == 0:
            self.gentech = input("Generating technique: vt (default), mc or cb >>> ") or 'vt'
        elif len(args) >= 1:
            self.gentech = args[0]

    def setDimen(self, *args):
        """Set dimensionality of the poly-xtal domain."""
        """Usage: """
        if len(args) == 0:
            self.dimen = int(input("GS dimen: 2 (default) or 3 >>> ") or 2)
        elif len(args) >= 1:
            self.dimen = args[0]
        print("dimen is {}".format(self.dimen))

    def setLevel(self, *args):
        """Set level of the synthetic grain structure."""
        """Usage: """
        # print(self.cmdSepCount*self.cmdSep)
        if len(args) == 0:
            #temp = input("Maximum GS level: 0, 1 or 2 >>> ") or 1
            #if type(temp) == str:
            #    print("Please enter again below")
            #    self.setLevel()
            #else:
            #    self.gslevel = int(temp)
            temp = input("Maximum GS level: 0 (default), 1 or 2 >>> ") or 0
            if int(temp) in (0, 1, 2):
                self.gslevel = temp
            else:
                print("Value should be 0, 1 or 2. Please re-enter below.")
                self.setLevel()
        elif len(args) >= 1:
            if int(args[0]) in (0, 1, 2):
                self.gslevel = args[0]
            else:
                print("Value should be 0, 1 or 2. Please re-enter below.")
                self.setLevel()

    def prep00(self, **kwargs):
        """Collects defs together. See code below for which definitions are called collectively."""
        """kwargs keys: (id / gsid), (mat / material), (d / dim / dimen / dimensionality)"""
        """Usages: 
                 gs.prep00()
                 gs.prep00(gsid = 'xxxxxxxxx')
                 gs.prep00(mat = 'mmmmmmm')
                 gs.prep00(dimen = 3)
                 gs.prep00(gntech = 'vt')
                 all combinations of the above. See example:
                     gs.prep00(mat = 'mmmmmmm', dimen = 3)
                     gs.prep00(dimen = 3, mat = 'mmmmmmm')
                     gs.prep00(gsid = 'GrStr_identity', mat = 'name of the material', dimen = 3, level = 0, gentech = 'vt')
        """
        self.setHouseKeepingRules
        if 'gsid'    in kwargs.keys()    : self.setUID(kwargs['gsid'])
        if 'gsid'    not in kwargs.keys(): self.setUID()
        
        if 'mat'     in kwargs.keys()    : self.setMaterialName(kwargs['mat'])
        if 'mat'     not in kwargs.keys(): self.setMaterialName()
        
        if 'dimen'   in kwargs.keys()    : self.setDimen(kwargs['dimen'])
        if 'dimen'   not in kwargs.keys(): self.setDimen()
        
        if 'level'   in kwargs.keys():     self.setLevel(kwargs['level'])
        if 'level'   not in kwargs.keys(): self.setLevel()
        
        if 'gentech' in kwargs.keys()    : self.setGenTech(kwargs['gentech'])
        if 'gentech' not in kwargs.keys(): self.setGenTech()

    def setOrigin(self, *args):
        """Set location of the poly-xtal domain in Eucledean space."""
        """Usage:
                 gs.setOrigin()
                 gs.setOrigin('z')
                 gs.setOrigin('zeros')
                 gs.setOrigin([0, 0])
                 gs.setOrigin([0, 0, 0])
                 gs.setOrigin([0, 0.0, 0])
        """
        if len(args) == 0:
            self.origin = np.asfarray(input("Origin: [0.0, 0.0, 0.0] (default) >>> ") or [0.0, 0.0, 0.0])
        elif len(args) == 1:
            if type(args[0]) == str:
                if args[0] in ('z' or 'zeros'):
                    self.origin = np.array([0.0, 0.0, 0.0])
            elif type(args[0]) == list:
                if len(args[0]) == 1:
                    print("Coordinates of both or at least x and y are needed")
                    print("Please enter origin coordinates again below")
                    self.setOrigin()
                elif len(args[0]) == 2:
                    self.origin = np.array([args[0][0], args[0][1], 0.0])
                elif len(args[0]) == 3:
                    self.origin = np.array([args[0][0], args[0][1], args[0][2]]).astype(float)
            else:
                print("Please enter correct values below")
                self.setOrigin()
        elif len(args) == 2:
            self.origin = np.array([args[0][0], args[0][1], args[0][2]])

    def setDomainSize(self, *args):
        """Set size of the poly-xtal domain in Eucledean space."""
        """Usage: """
        if len(args) == 0:
            self.xLength = float(input("Domain length along x, 1.0 (default) >>> ") or 1.0)
            self.yLength = float(input("Domain length along y, 1.0 (default) >>> ") or 1.0)
            if self.dimen == 3: self.zLength = float(input("Domain length along z, 1.0 (default) >>> ") or 1.0)
        elif len(args) == 1:
            self.xLength = float(args[0][0])
            self.yLength = float(args[0][1])
            if self.dimen == 3: self.zLength = float(args[0][2])
        else:
            print("Please enter correct values for domain size below")
            self.setDomainSize()

    def prep10(self, **kwargs):
        """Collects defs together. See code below which definitions are called collectively."""
        """kwargs keys: (origin), ()"""
        """Usages:
                  gs.prep10()
                  gs.prep10(origin = [3, 3, 2])
                  gs.prep10(domsize = [1, 1.0, 2])
                  gs.prep10(origin = [3, 3, 2], domsize = [1, 1.0, 2])
        """
        if 'origin' in kwargs.keys(): self.setOrigin(kwargs['origin'])
        if 'origin' not in kwargs.keys(): self.setOrigin()
        if 'domsize' in kwargs.keys(): self.setDomainSize(kwargs['domsize'])
        if 'domsize' not in kwargs.keys(): self.setDomainSize()

    @property
    def setupL0GS(self):
        """Set details of the Level 0 poly-xtal."""
        """Usage: """
        print(self.cmdSepCount*self.cmdSep)
        print('SET L0-GS DETAILS')
        self.L0_nInstances     = int(input("Number of L0 instances (default 1) >>>") or 1)
        self.L0_InstanceNames  = input("Instance name array [L0Instance1, L0Instance2, etc]") or 'L0Instance1'
        self.L0_nPhases        = int(input("Number of phases (default 1) >>> ") or 1)
        self.L0_namesPhases    = input("Phase name array [pname1, pname2, etc]. default cu >>> ") or ['cu']
        self.L0_phaseFractions = np.asfarray(input("Phase fraction array [VfP1, VfP2, etc]. default [1.0] >>> " ) or [1.0])
        self.L0_twinVolumeFrac = input("Phase wise twin volume fraction array [TVfP1, TVfP2, etc]. default [0.05]") or [0.05]

    @property
    def setupL1GS(self):
        """Set details of the Level 1 poly-xtal."""
        """Usage: """
        if int(self.gslevel) == 1:
            print(self.cmdSepCount*self.cmdSep)
            print('SET L1-GS DETAILS')
            self.L1_nInstances    = input("Number of L1 instances (default 1) >>>") or 1
            self.L1_InstanceNames = input("Instance name array [L1Instance1, L1Instance2, etc] >>> ") or 'L1Instance1'
            
            self.L1_TwinFlag = input("Make twins? [0: no (default), 1: yes]>>> ") or 0
            if int(self.L1_TwinFlag) == 1:
                self.L1_twinWidth_min_fraction  = input("Min.  width of twin as Frac. of host grain mean edge length (default 0.05) >>> ") or '0.05'
                self.L1_twinWidth_mean_fraction = input("Mean. width of twin as Frac. of host grain mean edge length (default 0.20) >>> ") or '0.20'
                self.L1_twinWidth_max_fraction  = input("Max.  width of twin as Frac. of host grain mean edge length (default 0.30) >>> ") or '0.30'
                self.L1_twinGrain_ori_relation  = input("Type of twin orientation relationship (default: ks) small case >>> ") or 'ks'
            
            self.L1_LathsFlag = input("Lathe grain structure? [0: no (default), 1: yes] >>> ") or '0'
            if int(self.L1_LathsFlag) == 1:
                self.L1_n_min_PriorAustPockets  = input("Min num. of prior autenite pockets (default 3) >>> ") or '3'
                self.L1_n_mean_PriorAustPockets = input("Mean num. of prior autenite pockets (default 3) >>> ") or '3'
                self.L1_n_max_PriorAustPockets  = input("Max num. of prior autenite pockets (default 5) >>> ") or '5'
                self.L1_lathWidth_min_fraction  = input("Min.  width of lathe as Frac. of host grain mean pocket edge length (default 0.05) >>> ") or '0.05'
                self.L1_lathWidth_mean_fraction = input("Mean. width of lathe as Frac. of host grain mean pocket edge length (default 0.20) >>> ") or '0.20'
                self.L1_lathWidth_max_fraction  = input("Max.  width of lathe as Frac. of host grain mean pocket edge length (default 0.30) >>> ") or '0.30'

    # GETTERS DEFINITIONS
    @property
    def getUID(self):
        """Get unique ID string of the poly-xtal."""
        """Usage: """
        return self.gsid
    @property
    def getMatName(self):
        """Get name of the material which the pxtal is representing."""
        """Usage: """
        return self.matname
    @property
    def getDimen(self):
        """Get dimen of the pxtal domain."""
        """Usage: """
        return self.dimen
    @property
    def getOrigin(self):
        """Get origin location of the pxtal."""
        return self.origin
    @property
    def getOriginX(self):
        """Get origin X location of the pxtal."""
        return self.origin[0]
    @property
    def getOriginY(self):
        """Get origin Y location of the pxtal."""
        return self.origin[1]
    @property
    def getOriginZ(self):
        """Get origin Z location of the pxtal."""
        return self.origin[2]
    @property
    def getStepSize(self):
        """Get stepSize of the pixellated or voxellated pxtal domain."""
        return self.stepSize
    @property
    def getSizeX(self):
        """Get xLength of the pxtal domain."""
        return self.xLength
    @property
    def getSizeY(self):
        """Get yLength of the pxtal domain."""
        return self.yLength
    @property
    def getSizeZ(self):
        """Get zLength of the pxtal domain."""
        return self.zLength
    @property
    def getBounds(self):
        """Get bounds of the pxtal domain."""
        if self.dimen == 2:
            bounds = np.array([[self.origin[0], self.origin[0]+self.xLength],
                               [self.origin[1], self.origin[1]+self.yLength]
                              ])
        elif self.dimen == 3:
            bounds = np.array([[self.origin[0], self.origin[0]+self.xLength],
                               [self.origin[1], self.origin[1]+self.yLength],
                               [self.origin[2], self.origin[2]+self.zLength]
                              ])
        return bounds

    @property
    def getLevel(self):
        """Return level of the grain structure."""
        pxtLevel = self.gslevel
        return pxtLevel

    @property
    def getArea0(self):
        """Return area of the poly-xtal as per initial user definition."""
        if self.dimen == 2:
            self.PXTLsurfaceArea = self.xLength*self.yLength
            print("Total surface area of the PXTAL domain = {}".format(self.PXTLsurfaceArea))
            return self.PXTLsurfaceArea
        else:
            print("Your GS model is 3D. Please get volume with [gs.getVolume0]")
            pass

    @property
    def getArea0_XY(self):
        if self.dimen == 2:
            self.PXTLsurfaceArea_XY = self.xLength*self.yLength
        elif self.dimen == 3:
            self.PXTLsurfaceArea_XY = self.xLength*self.yLength
        return self.PXTLsurfaceArea_XY

    @property
    def getArea0_YZ(self):
        if self.dimen == 2:
            print("Your GS model is 2D. Please get area XY with [gs.getArea0_XY] or [gs.getArea0]")
            print("Displaying XY area instead")
            return self.xLength*self.yLength
        elif self.dimen == 3:
            self.PXTLsurfaceArea_YZ = self.yLength*self.zLength
            return self.PXTLsurfaceArea_YZ

    @property
    def getArea0_ZX(self):
        if self.dimen == 2:
            print("Your GS model is 2D. Please get area XY with [gs.getArea0_XY] or [gs.getArea0]")
            print("Displaying XY area instead")
            return self.xLength*self.yLength
        elif self.dimen == 3:
            self.PXTLsurfaceArea_ZX = self.zLength*self.xLength
            return self.PXTLsurfaceArea_ZX

    @property
    def getVolume0(self):
        """Return volume of the poly-xtal as per initial user definition."""
        if self.dimen == 3:
            self.PXTLvolume = self.xLength*self.yLength*self.zLength
            print("Total volume of the PXTAL domain = {}".format(self.PXTLvolume))
            return self.PXTLvolume
        else:
            print("Your GS model is 2D. Displaying")
            pass
    
    @property
    def getNInst(self):
        """Return number of instances of the grain structure."""
        return self.L0_nInstances

    @property
    def getNamesInst(self):
        """Return names of the instances of the grain structure."""
        return self.L0_InstanceNames
    
    @property
    def getNPhases(self):
        """Return number of phases in the grain strucutre."""
        return self.L0_nPhases
    
    @property
    def getNamesPhases(self):
        """Return names of the indivisdual phases in the grain structure."""
        return self.L0_namesPhases
    
    @property
    def getVfPhase(self):
        """Return volume fraction of each phase in the grain structure."""
        return self.L0_phaseFractions
    
    @property
    def getFlagTwin(self):
        """Return Flag value of twins in the grain structure."""
        return self.L1_TwinFlag

    @property
    def getVfTwin(self):
        """Return Volume fraction of twins in the grain structure."""
        return self.L0_twinVolumeFrac
    
    @property
    def getGeomTwinWidthMin(self):
        """Return minimum value of the width fraction of twin."""
        return self.L1_twinWidth_min_fraction
    
    @property
    def getGeomTwinWidthMean(self):
        """Return mean value of the width fraction of twin."""
        return self.L1_twinWidth_mean_fraction
    
    @property
    def getGeomTwinWidthMax(self):
        """Return maximum value of the width fraction of twin."""
        return self.L1_twinWidth_max_fraction
    
    @property
    def getNameOriRelTwinGrain(self):
        """Return type of crystallographic orientation relation between twin and hosting grain."""
        return self.L1_twinGrain_ori_relation
    
    @property
    def getFlagLath(self):
        """Return Flag value of Laths in the grain structure."""
        return self.L1_LathsFlag
    
    @property
    def getNumAustPockMin(self):
        """Return minimium number of prior austenite pockets."""
        return self.L1_n_min_PriorAustPockets
    
    @property
    def getNumAustPockMean(self):
        """Return mean number of prior austenite pockets."""
        return self.L1_n_mean_PriorAustPockets
    
    @property
    def getNumAustPockMax(self):
        """Return max number of prior austenite pockets."""
        return self.L1_n_max_PriorAustPockets
    
    @property
    def getGeomLathWidthMin(self):
        """Return min lath width fraction."""
        return self.L1_lathWidth_min_fraction
    
    @property
    def getGeomLathWidthMean(self):
        """Return Mean Lath Width fraction."""
        return self.L1_lathWidth_mean_fraction
    
    @property
    def getGeomLathWidthMax(self):
        """Return Max Lath Width fraction."""
        return self.L1_lathWidth_max_fraction
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
gs = gstr()
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
class vtgs2dl0(gstr):
    L0_gr_ID = None # Grain
    L0_gb_ID = None # Grain boundaries
    L0_jp_ID = None # Junction points

    def __init__(self):
        self.VOR_CLIPPING_METHOD = 'useVObounds_and_offsets'
        # gs: grain structure object
        #super().origin
        #gs.__init__(self)
        #super().setOrigination()
        #super().setDomainSize()
        #super().setLevel()
        #super().setLevel0Details()

    def vtgs2_lattice(self):
        """ """
        #print(gs.cmdSepCount*gs.cmdSep)
        print('t: triangular')
        print('s: square')
        print('h: hexagonal')
        print('ru: random uniform')
        print('rn: random normal')
        print('ts: triangular and square interweaved')
        print('tru: traingular and uniform random')
        print('trn: traingular and normal random')
        print("Lattice: t(default), s, h, ru, rn, ts, tt")
        self.typeVTGSlattice = input("VTGS-2D base--lattice-point distribution >>> ") or 't'
        self.latticePertFlag = int(input("Perturbed lattice, 0: no, 1: yes (default) >>> ") or 1)
        if self.latticePertFlag == 1:
            self.perturbType = input("Perturbation type, RandomUniform (default) >>> ") or 'RandomUniform'
            self.pertFactorX = float(input("X perturbation factor, 0.02 (default)>>> ") or 0.02)
            self.pertFactorY = float(input("Y perturbation factor, 0.02 (default)>>> ") or 0.02)
            if gs.dimen == 3:
                self.pertFactorZ = float(input("Z perturbation factor, 0.02 (default)>>> ") or 0.02)

    def set_vtgs2DL0_Hex(self):
        """ """
        #print(gs.cmdSepCount*gs.cmdSep)
        print("HEXAGONAL GRAINS")
        self.a = float(input("Unit side length (Default, 0.1) >>> ") or 0.1)
        self.startx  = gs.origin[0]
        self.endx    = gs.origin[0] + gs.xLength
        self.starty  = gs.origin[1]
        self.nyunits = int(input("Number of unit copies along y (Default, 3) >>> ") or 3)
        self.angle   = float(input("Angle of the unit side (Default, 60 degrees) >>> ") or 60)
        self.UnitCellDIM = {'LengthAngle': {'UnitLength': self.a,
                                            'UnitAngle' : self.angle}}
        self.xdomain     = {'StartEnd'  : (self.startx, self.endx)}
        self.ydomain     = {'NumCopies' : (self.nyunits)}

    def display_vtgs_base_info(self):
        """ """
        #print(gs.cmdSepCount*gs.cmdSep)
        print("Making {}D VTGS for {}".format(gs.dimen, gs.matname))
        print("Lattice type: [-- {} --]".format(self.typeVTGSlattice))

    def gen_TRI_Lattice_Coord(self):
        """
        Generate triangular lattice coordinate data.
        
        Input
        ------
            UnitCellDIM: Dictionary to get inputs on lattice
            xdomain: data specifying x-bounds of lattice
            ydomain: data specifying y-bounds of lattice
        Return
        ------
            x: x-coordinate of the lattice
            y: y-coordinate of the lattice
            
        Definition call
        ---------------
            x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

        Note
        ----
            none now.
        """
        
        if list(self.UnitCellDIM.keys())[0]=='LengthAngle':
            data  = self.UnitCellDIM[list(self.UnitCellDIM.keys())[0]]
            a     = data['UnitLength']
            angle = data['UnitAngle']
        if list(self.xdomain.keys())[0]=='StartEnd':
            startx, endx = list(self.xdomain.values())[0]
        if list(self.ydomain.keys())[0]=='NumCopies':
            nyunits = list(self.ydomain.values())[0]
        # Row 1 - x
        x  = np.arange(startx, endx, a)
        x1 = np.copy(x)
        # Row 2 - x
        x2 = x1 + a*np.cos(np.deg2rad(angle))
        # Row 1 and Row 2 - x
        xi = np.vstack((x1, x2))
        # Row 1 - y
        y1 = np.zeros(np.shape(x1))
        # Row 2 - y
        y2 = y1 + a*np.sin(np.deg2rad(angle))
        # Row 1 and Row 2 - y
        yi = np.vstack((y1,y2))
        # Make pattern by Translation
        x = np.copy(xi)
        y = np.copy(yi)
        for count in range(nyunits):
            x = np.vstack((x, xi))
            y = np.vstack((y, yi+(count+1)*2*a*np.sin(np.deg2rad(angle))))
        self.x = x
        self.y = y

    def add_Perturb_coordinates_2d(self):
        """
        Add perturbations to the NpArray.
        
        Input
        ------
            x      : numpy.ndarray: (:, 1): x-coordinates of the lattice
            y      : numpy.ndarray: (:, 1): y-coordinates of the lattice
            method : str          : Which method to use to introduce lattice vertex distributions
            factors: Factors specifying the amount of lattice vertex perturbation
            
        Return
        ------
            x: perturbed x-coordinates of the lattice
            y: perturbed y-coordinates of the lattice
            
        Definition call
        ---------------
            x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

        Note
        ----
            none now.

        """
        if self.perturbType == 'RandomUniform':
            # Calculate the pertubation value
            self.xPerturbation = self.pertFactorX*np.random.random(np.shape(self.x))
            self.yPerturbation = self.pertFactorY*np.random.random(np.shape(self.y))
        # Calculate the perturbed coordinate
        self.x = self.x + self.xPerturbation
        self.y = self.y + self.yPerturbation
    
    def stats_Perturb_L02D(self):
        """ """
        # Get the statistics data of perturbation values
        self.stats_xPerturbation = self.xPerturbation.min(),\
                                   self.xPerturbation.mean(),\
                                   self.xPerturbation.std(),\
                                   self.xPerturbation.max()

        self.stats_yPerturbation = self.yPerturbation.min(),\
                                   self.yPerturbation.mean(),\
                                   self.yPerturbation.std(),\
                                   self.yPerturbation.max()
        # Get the statistics data of perturbed coordinates
        self.stats_xlatticeCoord = self.x.min(), self.x.mean(), self.x.std(), self.x.max()
        self.stats_ylatticeCoord = self.y.min(), self.y.mean(), self.y.std(), self.y.max()
        
    def form_2D_Coord_Array(self):
        """
        Prepare co-ordinate data format.
        
        Input
        -----
            x: x-coordinate of the data
            y: y-coordinate of the data
            
        Return
        ------
            x: x-coordinate of the data
            y: y-coordinate of the data
            
        Definition call
        ---------------
            x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

        Note
        ----
            note here
        """
        self.x = np.reshape(self.x, np.size(self.x))[np.newaxis].T
        self.y = np.reshape(self.y, np.size(self.y))[np.newaxis].T
    def form_2D_VorTess_Seeds(self):
        """
        Prepare the voronoi seed values.
        
        Input
        ------
            x: x-coordinaste data of the lattice coordinates
            y: y-coordinate data of the lattice coordinates
            
        Return
        ------
            vseeds: seed coordinate array needed to make the Voronoi tesseallation
            
        Definition call
        ---------------
            x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

        Note
        ----
            note here
        """
        self.vseeds = np.hstack((self.x, self.y))
        
        
    def form_2D_VorTess_Object(self):
        """
        Calculate the voronoi object (vo).
        Input
        ------
            vseeds: seed coordinate array needed to Voronoi tessellate 
        Return
        ------
            vo: Voronoi object
        Definition call
        ---------------
            x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)
        """
        self.vo = Voronoi(self.vseeds)        
    
    def voronoi_finite_polygons_2d(self):
        """
        Reconstruct infinite voronoi regions in a 2D diagram to finite regions.
        
        Parameters
        ----------
        vor : Voronoi
            Input diagram
        radius : float, optional
            Distance to 'points at infinity'.
            
        Returns
        -------
        regions : list of tuples
            Indices of vertices in each revised Voronoi regions.
        vertices : list of tuples
            Coordinates for revised Voronoi vertices. Same as coordinates
            of input vertices, with 'points at infinity' appended to the
            end.
        """
        if self.vo.points.shape[1] != 2:
            raise ValueError("Requires 2D input")

        new_regions = []
        new_vertices = self.vo.vertices.tolist()

        center = self.vo.points.mean(axis=0)
        radius = None
        if radius is None:
            radius = self.vo.points.ptp().max()*2

        # Construct a map containing all ridges for a given point
        all_ridges = {}
        for (p1, p2), (v1, v2) in zip(self.vo.ridge_points, self.vo.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))

        # Reconstruct infinite regions
        for p1, region in enumerate(self.vo.point_region):
            vertices = self.vo.regions[region]

            if all(v >= 0 for v in vertices):
                # finite region
                new_regions.append(vertices)
                continue

            # reconstruct a non-finite region
            ridges = all_ridges[p1]
            new_region = [v for v in vertices if v >= 0]

            for p2, v1, v2 in ridges:
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    # finite ridge: already in the region
                    continue

                # Compute the missing endpoint of an infinite ridge

                t = self.vo.points[p2] - self.vo.points[p1] # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal

                midpoint = self.vo.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = self.vo.vertices[v2] + direction * radius

                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())

            # sort region counterclockwise
            vs = np.asarray([new_vertices[v] for v in new_region])
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
            new_region = np.array(new_region)[np.argsort(angles)]

            # finish
            new_regions.append(new_region.tolist())
            self.VGrains = new_regions
            self.L0GS_NGrains = len(new_regions)
            self.VGrainVertices = np.asarray(new_vertices)

    def make_Super_Bounding_Polygon(self):
        """
        Generate bounding box for the VTGS.
        
        VoronoiObject: The voronoi tessellation object data
        METHOD:        (1) useVObounds_and_offsets: str
        BoundingData:  (1) For METHOD=useVObounds_and_offsets, it is the offset values
                           for left edge, right edge, bottom edge and top edge
                           arranged in this order, in a list
        
        Input
        ------
            VoronoiObject: Shapely object: Voronoi object of the grain structure. Unbounded Voronoi grain structure
            METHOD       : str           : Specifies how bounding box is constructed
                                           useVObounds_and_offsets

        # TODO: Rename definition name from "make_Super_Bounding_Polygon" to "make_L0GS_Super_Bounding_Polygon"

        Return
        ------
            PolygBBox_VT: Shapely polygon object. Represents the bounding box.
            
        Definition call
        ---------------
            PolygBBox_VT = make_Super_Bounding_Polygon(VoronoiObject, METHOD)

        Note
        ----
            note here
        """
        if self.VOR_CLIPPING_METHOD=='useVObounds_and_offsets':
            min_x = self.vo.min_bound[0] - 0.00
            max_x = self.vo.max_bound[0] + 0.00
            min_y = self.vo.min_bound[1] - 0.00
            max_y = self.vo.max_bound[1] + 0.00

            # PolygBBox_VT: Polygonal bounding box for Voronoi tessellation
            self.PolygBBox_VT = Polygon([[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]])
    
    def ini_Grain_param(self):
        """
        Write the summary line here.
        
        #TODO: Describe this definition
        #TODO: Investigate what is happening to centroid in thie definition codes
        
        Input
        ------
            paramSize: Size of hwe parameter dictionary
            
        Return
        ------
            GrainPar: Dictionary for storing the Grain structure parameters
            
        Definition call
        ---------------
            x, y = gen_TRI_Lattice_Coord(UnitCellDIM, xdomain, ydomain)

        Note
        ----
            note here
        """
        areas      = np.zeros(self.L0GS_NGrains)
        centroid   = areas.copy()#<<<<<<<<<<<<<<<<<<
        perimeters = areas.copy()
        # Collect to dict
        self.GrainPar   = {'areas'     : areas,
                           'centroid'  : centroid,
                           'perimeters': perimeters}
    
    def clip_Voronoi_Tess_BoundBox(self):
        """ """
        self.L0GS_PXTAL_units = []
        # GRn = 0
        # vid = VGrains[GRn]
        GRn_actual = 0
        for vid in self.VGrains:
            # Get the vertices coordinate array of this grain and make POU from it
            thisGrain_POU = Polygon(self.VGrainVertices[vid])
            # Clip this polygon with the boundary of the bounding box
            thisGrain_POU_clipped_BB = thisGrain_POU.intersection(self.PolygBBox_VT)
            POU = thisGrain_POU_clipped_BB
            if POU.area > 0:
                GRn_actual += 1
                self.L0GS_PXTAL_units.append(POU)
 
    def pxtal_from_xtals(self):
        """Usage: """
        self.PXTAL_L0GS = MultiPolygon(self.L0GS_PXTAL_units)

# =============================================================================
#     def build_GSL0_DATA(self):
#         """ """
#         print("Building grain id names")
#         self.L0GS_gridnames       = {GRn: ['L0GRAIN-%07d'%GRn, []] for GRn in range(self.L0GS_NGrains)}
#         print("Building grain neighbour list")
#         self.L0GS_neighList = {GRn: ['L0GRAIN-%07d'%GRn, 0] for GRn in range(self.L0GS_NGrains)}
#         print("Building grain vertices data")
#         self.L0GS_vertices  = []
#         print("Building vertices indices data")
#         self.L0GS_vid             = {GRn: ['L0GRAIN-%07d'%GRn, 0] for GRn in range(self.L0GS_NGrains)}
#         print("Building ID data of grain boundary edges")
#         self.L0GS_edgeid          = {GRn: ['L0GRAIN-%07d'%GRn, []] for GRn in range(self.L0GS_NGrains)}
#         print("Building grain vertices coordinate data")
#         self.L0GS_vcoord          = {GRn: ['L0GRAIN-%07d'%GRn, []] for GRn in range(self.L0GS_NGrains)}
#         print("Building grain centroid coordinates data")
#         self.L0GS_ccoord          = {GRn: ['L0GRAIN-%07d'%GRn, []] for GRn in range(self.L0GS_NGrains)}
#         print("Building grain area data")
#         self.L0GS_area            = {GRn: ['L0GRAIN-%07d'%GRn, 0] for GRn in range(self.L0GS_NGrains)}
#         print("Building grain external perimeter data")
#         self.L0GS_extperim        = {GRn: ['L0GRAIN-%07d'%GRn, 0] for GRn in range(self.L0GS_NGrains)}
#         print("Building grain internal perimeter data")
#         self.L0GS_intperim        = {GRn: ['L0GRAIN-%07d'%GRn, []] for GRn in range(self.L0GS_NGrains)}
#         print("Building number of external edges")
#         self.L0GS_nextedges       = {GRn: ['L0GRAIN-%07d'%GRn, 0] for GRn in range(self.L0GS_NGrains)}
#         print("")
#         self.L0GS_nedgesex        = {GRn: ['L0GRAIN-%07d'%GRn, []] for GRn in range(self.L0GS_NGrains)}
#         print("Building grain minimum diagonal")
#         self.L0GS_mindiag         = {GRn: ['L0GRAIN-%07d'%GRn, []] for GRn in range(self.L0GS_NGrains)}
#         print("Building grain maximum diagonal")
#         self.L0GS_maxdiag         = {GRn: ['L0GRAIN-%07d'%GRn, []] for GRn in range(self.L0GS_NGrains)}
#         print("Building grain mean diagonal")
#         self.L0GS_meandiag        = {GRn: ['L0GRAIN-%07d'%GRn, []] for GRn in range(self.L0GS_NGrains)}
#         print("?????????????????")
#         self.L0GS_stddiag         = {GRn: ['L0GRAIN-%07d'%GRn, []] for GRn in range(self.L0GS_NGrains)}
#         print("External verttices, internal angle")
#         self.L0GS_extvintangle    = {GRn: ['L0GRAIN-%07d'%GRn, []] for GRn in range(self.L0GS_NGrains)}
#         print("Centroid to vertices distance")
#         self.L0GS_ctovdist        = {GRn: ['L0GRAIN-%07d'%GRn, []] for GRn in range(self.L0GS_NGrains)}
# =============================================================================
    def findNeigh(self):
        """Find neighbouring grains"""
        """Usage: vt.findNeigh()"""
        for thisGrain in range(self.L0GS_NGrains):
            thisGrain_POU = self.PXTAL_L0GS.geoms[thisGrain]
            neighCount = 0 # Number of neighbours for this count
            for possible_neighGrain in range(self.L0GS_NGrains):
                if thisGrain != possible_neighGrain:  # Avoid self-checks
                    possible_neighGrain_POU = self.PXTAL_L0GS.geoms[possible_neighGrain]
                    if thisGrain_POU.touches(possible_neighGrain_POU):
                        # This means these two grains are neighbours
                        if neighCount == 0:
                            self.L0GS_neighList[thisGrain][1] = [possible_neighGrain]
                        else:
                            self.L0GS_neighList[thisGrain][1].append(possible_neighGrain)
                        neighCount += 1
        # Explanation: Say, GRn = 1
        # Then, L0GS_neighList[GRn] is something like: ['L0GRAIN-0000001', [1, 3, 6]]
        # This means, grains 1, 3 and 6 are neighbours of grain GRn = 1

    @property
    def getpxAreaGS1(self):
        self.pxAreaGS1 = self.PXTAL_L0GS.area
        return self.pxAreaGS1

    @property
    def getpxLengthGB1(self):
        self.pxLengthGB1 = self.PXTAL_L0GS.boundary.length
        return self.pxLengthGB1

    def pxL0GSvert(self):
        for grainCount in range(self.L0GS_NGrains):
            vertices_x = self.PXTAL_L0GS.geoms[grainCount].exterior.xy[0]
            vertices_y = self.PXTAL_L0GS.geoms[grainCount].exterior.xy[1]
        #    vertexCount = 0
            for vertexCount in list(range(len(vertices_x)-1)):
                xy = [vertices_x[vertexCount], vertices_y[vertexCount]]
                self.L0GS_vertices.append(xy)
        # Make the coordinate list unique
        self.L0GS_vertices = np.unique(self.L0GS_vertices, axis = 0)

    def pxL0GSvertID_a(self):
        # Make the list of IDs and the corresponding coordinates values
        self.PXTAL_L0GS_vertID_vertices = np.zeros((len(self.L0GS_vertices), 3))
        vcount = 0
        for vcount in range(len(self.L0GS_vertices)):
            self.PXTAL_L0GS_vertID_vertices[vcount][0] = vcount
            self.PXTAL_L0GS_vertID_vertices[vcount][1] = self.L0GS_vertices[vcount][0]
            self.PXTAL_L0GS_vertID_vertices[vcount][2] = self.L0GS_vertices[vcount][1]
            vcount += 1

    def pxL0GSvertID_b(self):
        # Go through the ppxtal, one grain at a time
        for grainCount in range(self.L0GS_NGrains):
            verticesGrain = np.hstack((np.array(self.PXTAL_L0GS.geoms[grainCount].exterior.xy[0][:-1])[np.newaxis].T,
                                       np.array(self.PXTAL_L0GS.geoms[grainCount].exterior.xy[1][:-1])[np.newaxis].T))
            # In this grain, match its vertices coordinates to the sorted collection of vertice coordinates
            for vCountGRAIN in range(len(verticesGrain)):
                # One vertex of the grain at a time
                xy = verticesGrain[vCountGRAIN]
                # See the ID of this vertex by comparing against the PXTAL coordinate list
                # After seeing it, itself becomes the ID number of the vertex
                vertexID = list(np.prod(xy == self.L0GS_vertices, axis = 1)).index(1)
                # Put this vertexID inside the vertex ID dictionaryt of the poly-xtal
                if vCountGRAIN == 0:
                    self.L0GS_vid[grainCount][1] = [vertexID]
                else:
                    self.L0GS_vid[grainCount][1].append(vertexID)

    def pxSubset(self, *args):
        if len(args) == 0:
            # iF NO ARGUMENTS ARE PASSES, A COPY OF THE ORIGINAL WILL BE CREATED
            # TOCHECK: will this copy be shallow or deep?
            self.pxsubset = self.PXTAL_L0GS
        elif len(args) == 1: 
            # Extract a single grain useing grain number ID
            self.pxsubset = self.PXTAL_L0GS.geoms[args[0]]
        elif len(args) == 2:
            # Extract multiple grains using start ID number, end ID number and a default uinit incr
            self.pxsubset = self.PXTAL_L0GS.geoms[args[0] : args[1] : 1]
        elif len(args) == 3:
            # Extract multiple grains using start ID number, end ID number and custom specified init incr
            self.pxsubset = self.PXTAL_L0GS.geoms[args[0] : args[1] : args[2]]

# =============================================================================
#     def build_GSL0_STAT(self):
#         """ """
#         self.hist_data = {'numbins' : [],
#                           'bins'    : [],
#                           'counts'  : [],
#                           'prob'    : [],
#                           'cumprob' : [],
#                           'pdf_data': (),
#                           'cdf_data': (),
#                           'modality': [],
#                           'skewness': [],
#                           'width'   : (),
#                           'peaks@'  : (),
#                          }
# 
#         self.BIN_N                                = [10, 50, 100, 500]
#         
#         self.L0GS_pdist_area                      = {'nbin=%d'%binCount: self.hist_data for binCount in self.BIN_N}
#         self.L0GS_pdist_area['nbin=nbin']         = self.hist_data # Will allways be added to the end
#         
#         self.L0GS_pdist_extperim                  = {'nbin=%d'%binCount: self.hist_data for binCount in self.BIN_N}
#         self.L0GS_pdist_extperim['nbin=nbin']     = self.hist_data # Will allways be added to the end
#         
#         self.L0GS_pdist_intperim                  = {'nbin=%d'%binCount: self.hist_data for binCount in self.BIN_N}
#         self.L0GS_pdist_intperim['nbin=nbin']     = self.hist_data # Will allways be added to the end
#         
#         self.L0GS_pdist_nextedges                 = {'nbin=%d'%binCount: self.hist_data for binCount in self.BIN_N}
#         self.L0GS_pdist_nextedges['nbin=nbin']    = self.hist_data # Will allways be added to the end
#         
#         self.L0GS_pdist_nedgesextv                = {'nbin=%d'%binCount: self.hist_data for binCount in self.BIN_N}
#         self.L0GS_pdist_nedgesextv['nbin=nbin']   = self.hist_data # Will allways be added to the end
#         
#         self.L0GS_pdist_mindiag                   = {'nbin=%d'%binCount: self.hist_data for binCount in self.BIN_N}
#         self.L0GS_pdist_mindiag['nbin=nbin']      = self.hist_data # Will allways be added to the end
#         
#         self.L0GS_pdist_maxdiag                   = {'nbin=%d'%binCount: self.hist_data for binCount in self.BIN_N}
#         self.L0GS_pdist_maxdiag['nbin=nbin']      = self.hist_data # Will allways be added to the end
#         
#         self.L0GS_pdist_extvintangle              = {'nbin=%d'%binCount: self.hist_data for binCount in self.BIN_N}
#         self.L0GS_pdist_extvintangle['nbin=nbin'] = self.hist_data # Will allways be added to the end
#         
#         self.L0GS_pdist_ctovdist                  = {'nbin=%d'%binCount: self.hist_data for binCount in self.BIN_N}
#         self.L0GS_pdist_ctovdist['nbin=nbin']     = self.hist_data # Will allways be added to the end
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
vt = vtgs2dl0()
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
class dataops(vtgs2dl0, gstr):
    # Please maintain: Please instantiate as dop
    # When def starts with B_, B means build
    # When B_ is followed by D_, D implies actual data
    def __init__(self):
        pass

    def B_D_gridnames(self, **kwargs):
        """Build grain id names for L0 grain structure"""
        if kwargs['gslevel'] == 0:
            print("Building grain id names")
            gs.L0GS_gridnames = {GRn: ['L0GRAIN-%07d'%GRn, []] for GRn in range(vt.L0GS_NGrains)}
        elif kwargs['gslevel'] == 1: pass
        elif kwargs['gslevel'] == 2: pass

    def B_neighList(self, **kwargs):
        """Build grain neighbour list for L0 grain structure"""
        if kwargs['gslevel'] == 0:
            print("Building grain neighbour list of L0 grain structure")
            gs.L0GS_neighList = {GRn: ['L0GRAIN-%07d'%GRn, 0] for GRn in range(vt.L0GS_NGrains)}
        elif kwargs['gslevel'] == 1: pass
        elif kwargs['gslevel'] == 2: pass

    def B_GrVert(self, **kwargs):
        """Build data structure to hold grain vertices data"""
        #TODO - write which vertices data will be stored?
        if kwargs['gslevel'] == 0:
            print("Building xx of L0 grain structure")
            gs.L0GS_vertices = []            
        elif kwargs['gslevel'] == 1: pass
        elif kwargs['gslevel'] == 2: pass

    def B_L0GS_vid(self, **kwargs):
        """Build """
        if kwargs['gslevel'] == 0:
            print("Building vertices ID of grains in L0 grain structure")
            gs.L0GS_vid = {GRn: ['L0GRAIN-%07d'%GRn, 0] for GRn in range(vt.L0GS_NGrains)}
        elif kwargs['gslevel'] == 1: pass
        elif kwargs['gslevel'] == 2: pass
        
    def B_L0GS_edgeid(self, **kwargs):
        """Build """
        if kwargs['gslevel'] == 0:
            print("Building grain boundary edges ID of L0 grain structure")
            gs.L0GS_edgeid = {GRn: ['L0GRAIN-%07d'%GRn, []] for GRn in range(vt.L0GS_NGrains)}
        elif kwargs['gslevel'] == 1: pass
        elif kwargs['gslevel'] == 2: pass
        
    def B_vcoord(self, **kwargs):
        """Build """
        if kwargs['gslevel'] == 0:
            print("Building vertices coordinates of grains of L0 grain structure")
            gs.L0GS_vcoord = {GRn: ['L0GRAIN-%07d'%GRn, []] for GRn in range(vt.L0GS_NGrains)}
        elif kwargs['gslevel'] == 1: pass
        elif kwargs['gslevel'] == 2: pass
        
    def B_ccoord(self, **kwargs):
        """Build """
        if kwargs['gslevel'] == 0:
            print("Building [grain centroid] coordinates of L0 grain structure")
            gs.L0GS_ccoord = {GRn: ['L0GRAIN-%07d'%GRn, []] for GRn in range(vt.L0GS_NGrains)}
        elif kwargs['gslevel'] == 1: pass
        elif kwargs['gslevel'] == 2: pass
        
    def B_area(self, **kwargs):
        """Build """
        if kwargs['gslevel'] == 0:
            print("Building [grain area] dictionary data structure of L0 grain structure")
            gs.L0GS_area = {GRn: ['L0GRAIN-%07d'%GRn, 0] for GRn in range(vt.L0GS_NGrains)}
        elif kwargs['gslevel'] == 1: pass
        elif kwargs['gslevel'] == 2: pass
        
    def B_extperim(self, **kwargs):
        """Build """
        if kwargs['gslevel'] == 0:
            print("Building [external perimeter] dictionary data structure of L0 grain structure")
            gs.L0GS_extperim = {GRn: ['L0GRAIN-%07d'%GRn, 0] for GRn in range(vt.L0GS_NGrains)}
        elif kwargs['gslevel'] == 1: pass
        elif kwargs['gslevel'] == 2: pass
        
    def B_intperim(self, **kwargs):
        """Build """
        if kwargs['gslevel'] == 0:
            print("Building [internal perimeter] dictionary data structure of L0 grain structure")
            gs.L0GS_intperim = {GRn: ['L0GRAIN-%07d'%GRn, []] for GRn in range(vt.L0GS_NGrains)}
        elif kwargs['gslevel'] == 1: pass
        elif kwargs['gslevel'] == 2: pass
        
    def B_nextedges(self, **kwargs):
        """Build """
        if kwargs['gslevel'] == 0:
            print("Building [number of external edges] dictionary data structure of L0 grain structure")
            gs.L0GS_nextedges = {GRn: ['L0GRAIN-%07d'%GRn, 0] for GRn in range(vt.L0GS_NGrains)}
        elif kwargs['gslevel'] == 1: pass
        elif kwargs['gslevel'] == 2: pass
        
    def B_nedgesex(self, **kwargs):
        """Build """
        if kwargs['gslevel'] == 0:
            print("Building [??????????] of L0 grain structure")
            gs.L0GS_nedgesex = {GRn: ['L0GRAIN-%07d'%GRn, []] for GRn in range(vt.L0GS_NGrains)}
        elif kwargs['gslevel'] == 1: pass
        elif kwargs['gslevel'] == 2: pass
        
    def B_mindiag(self, **kwargs):
        """Build """
        if kwargs['gslevel'] == 0:
            print("Building [grain minimum diagonal] dictionary data structure of L0 grain structure")
            gs.L0GS_mindiag = {GRn: ['L0GRAIN-%07d'%GRn, []] for GRn in range(vt.L0GS_NGrains)}
        elif kwargs['gslevel'] == 1: pass
        elif kwargs['gslevel'] == 2: pass
        
    def B_maxdiag(self, **kwargs):
        """Build """
        if kwargs['gslevel'] == 0:
            print("Building [grain maximum diagonal] dictionary data structure of L0 grain structure")
            gs.L0GS_maxdiag = {GRn: ['L0GRAIN-%07d'%GRn, []] for GRn in range(vt.L0GS_NGrains)}
        elif kwargs['gslevel'] == 1: pass
        elif kwargs['gslevel'] == 2: pass
        
    def B_meandiag(self, **kwargs):
        """Build """
        if kwargs['gslevel'] == 0:
            print("Building [grain mean diagonal] dictionary data structure of L0 grain structure")
            gs.L0GS_meandiag = {GRn: ['L0GRAIN-%07d'%GRn, []] for GRn in range(vt.L0GS_NGrains)}
        elif kwargs['gslevel'] == 1: pass
        elif kwargs['gslevel'] == 2: pass
        
    def B_stddiag(self, **kwargs):
        """Build """
        if kwargs['gslevel'] == 0:
            print("Building [grain diagoanl stndard deviation] dictionary data structure of L0 grain structure")
            gs.L0GS_stddiag = {GRn: ['L0GRAIN-%07d'%GRn, []] for GRn in range(vt.L0GS_NGrains)}
        elif kwargs['gslevel'] == 1: pass
        elif kwargs['gslevel'] == 2: pass
        
    def B_extvintangle(self, **kwargs):
        """Build """
        if kwargs['gslevel'] == 0:
            print("Building [internal angle for each external vertex] dictionary data structure of L0 grain structure")
            gs.L0GS_extvintangle = {GRn: ['L0GRAIN-%07d'%GRn, []] for GRn in range(vt.L0GS_NGrains)}
        elif kwargs['gslevel'] == 1: pass
        elif kwargs['gslevel'] == 2: pass
        
    def B_ctovdist(self, **kwargs):
        """Build """
        if kwargs['gslevel'] == 0:
            print("Building [centroid to vertex] dictionary data structure of L0 grain structure")
            gs.L0GS_ctovdist = {GRn: ['L0GRAIN-%07d'%GRn, []] for GRn in range(vt.L0GS_NGrains)}
        elif kwargs['gslevel'] == 1: pass
        elif kwargs['gslevel'] == 2: pass

####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
class statops(vtgs2dl0, gstr):
    """
    # Please maintain: Please instantiate as sop
    
    # When def starts with B: 
        # B means build
        # When B_ is followed by S, S implies Statistical parameters
        # When S is followed by XXXX, XXXX is the actual data on which the statistical parameter is taken
            # The data could be another parameter itself. See example below:
                    # example: A grin structure has many grains. Each grain has many edges, and one value for mean number of edges
                    # in such a case, this mean number of edges, itself, will have a distribution in the overall grain structure
                    # This understanding naturally extends to other parameters as required
        # The XXXX will be followed by the sub-name, which could be min, mean, max, std, etc.
    
    # 
    """
    def __init__(self):
        pass
    
    def setBinsArea(self):
        gs.BIN_N_global = [10, 50]

    def build_GSL0_STAT(self):
        """
        # 'numbins': Number of binning instances. Data basically is len(BIN_N). Represents number of histogramming instances.
        #            Could be specificed by using setBinsArea
        #            May also be explicitly specified by user on a case to case basis. See example below:
        #            Example: see user input field @ definition B_S_area. 
        #            NOTE: If not expliciytly specified, value by sop.setBinsArea() will be used 
        #                  from self.BIN_N (SEE this method for more details)
        #            DATA STRUCTURE: just a single number of type int

        # bins     : actual bins specified by BIN_N

        # counts   : Count values of occurances (y-axis) as a function of bin width location (x-axis)

        # prob     : Probability values of occurances (y-axis) as a function of bin width location (x-axis)

        # cumprob  : Cumulative probabilities

        # pdf_data : Probability distribution function data

        # cdf_data : Cumulative distribution fucntion data

        # modality : Modality of the distribution

        # skewness : Skewness of the distribution

        # width    : Width of the distribution. Mostly, it will be the Full Width at Half Maximum

        # peaks@   : The location information of peak(s) in the distribution
        """

        self.hist_data = {'numbins' : [],
                          'bins'    : [],
                          'counts'  : [],
                          'prob'    : [],
                          'cumprob' : [],
                          'pdf_data': (),
                          'cdf_data': (),
                          'modality': [],
                          'skewness': [],
                          'width'   : (),
                          'peaks@'  : (),
                         }

        def B_S_area(self, **kwargs):
            # Example usage: sop.B_S_area(binn = [10, 20, 100, 500], gslevel = 0)
            if 'binn' in kwargs.keys(): BIN_N = kwargs['binn']
            else: BIN_N = gs.BIN_N_global
            gs.L0GS_pdist_area                      = {'nbin=%d'%binCount: self.hist_data for binCount in self.BIN_N}
            gs.L0GS_pdist_area['nbin=nbin']         = self.hist_data # Will allways be added to the end

        def B_S_area(self, **kwargs):
            """Build XXXX data structure for statistical distribution data of YYYY """
            if 'binn' in kwargs.keys(): BIN_N = kwargs['binn']
            else: BIN_N = gs.BIN_N_global
            gs.L0GS_pdist_extperim                  = {'nbin=%d'%binCount: self.hist_data for binCount in self.BIN_N}
            gs.L0GS_pdist_extperim['nbin=nbin']     = self.hist_data # Will allways be added to the end

        def B_S_area(self, **kwargs):
            """Build XXXX data structure for statistical distribution data of YYYY """
            if 'binn' in kwargs.keys(): BIN_N = kwargs['binn']
            else: BIN_N = gs.BIN_N_global
            gs.L0GS_pdist_intperim                  = {'nbin=%d'%binCount: self.hist_data for binCount in self.BIN_N}
            gs.L0GS_pdist_intperim['nbin=nbin']     = self.hist_data # Will allways be added to the end

        def B_S_area(self, **kwargs):
            """Build XXXX data structure for statistical distribution data of YYYY """
            if 'binn' in kwargs.keys(): BIN_N = kwargs['binn']
            else: BIN_N = gs.BIN_N_global
            gs.L0GS_pdist_nextedges                 = {'nbin=%d'%binCount: self.hist_data for binCount in self.BIN_N}
            gs.L0GS_pdist_nextedges['nbin=nbin']    = self.hist_data # Will allways be added to the end

        def B_S_area(self, **kwargs):
            """Build XXXX data structure for statistical distribution data of YYYY """
            if 'binn' in kwargs.keys(): BIN_N = kwargs['binn']
            else: BIN_N = gs.BIN_N_global
            gs.L0GS_pdist_nedgesextv                = {'nbin=%d'%binCount: self.hist_data for binCount in self.BIN_N}
            gs.L0GS_pdist_nedgesextv['nbin=nbin']   = self.hist_data # Will allways be added to the end

        def B_S_area(self, **kwargs):
            """Build XXXX data structure for statistical distribution data of YYYY """
            if 'binn' in kwargs.keys(): BIN_N = kwargs['binn']
            else: BIN_N = gs.BIN_N_global
            gs.L0GS_pdist_mindiag                   = {'nbin=%d'%binCount: self.hist_data for binCount in self.BIN_N}
            gs.L0GS_pdist_mindiag['nbin=nbin']      = self.hist_data # Will allways be added to the end

        def B_S_area(self, **kwargs):
            """Build XXXX data structure for statistical distribution data of YYYY """
            if 'binn' in kwargs.keys(): BIN_N = kwargs['binn']
            else: BIN_N = gs.BIN_N_global
            gs.L0GS_pdist_maxdiag                   = {'nbin=%d'%binCount: self.hist_data for binCount in self.BIN_N}
            gs.L0GS_pdist_maxdiag['nbin=nbin']      = self.hist_data # Will allways be added to the end

        def B_S_area(self, **kwargs):
            """Build XXXX data structure for statistical distribution data of YYYY """
            if 'binn' in kwargs.keys(): BIN_N = kwargs['binn']
            else: BIN_N = gs.BIN_N_global
            gs.L0GS_pdist_extvintangle              = {'nbin=%d'%binCount: self.hist_data for binCount in self.BIN_N}
            gs.L0GS_pdist_extvintangle['nbin=nbin'] = self.hist_data # Will allways be added to the end

        def B_S_area(self, **kwargs):
            """Build XXXX data structure for statistical distribution data of YYYY """
            if 'binn' in kwargs.keys(): BIN_N = kwargs['binn']
            else: BIN_N = gs.BIN_N_global
            gs.L0GS_pdist_ctovdist                  = {'nbin=%d'%binCount: self.hist_data for binCount in self.BIN_N}
            gs.L0GS_pdist_ctovdist['nbin=nbin']     = self.hist_data # Will allways be added to the end
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################