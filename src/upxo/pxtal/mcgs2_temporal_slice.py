from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.measure import label as skim_label

class mcgs2_grain_structure():
    __slots__ = ('dim',  # Dimensionality of the grain structure
                 'uigrid',  # Copy of grid.uigrid datastructure
                 'xgr',  # min, incr, max of x-axis
                 'ygr',  # min, incr, max of y-axis
                 'zgr',  # min, incr, max of z-axis
                 'm',  # MC temporal step to which this GS belongs to.
                 's',  # State array
                 'S',  # Total number of states
                 'binaryStructure2D',  # 2D Binary Structure to identify grains
                 'binaryStructure3D',  # 3D Binary Structure to identify grains
                 'n',  # Number of grains
                 'lgi',  # Lattice of Grains Ids
                 'spart_flag',  # State wise partitioning
                 'gid',  # Grain numbers used as grain IDs
                 's_gid',  # DICT: {s: overall grain id i.e grain number}
                 'gid_s',  # LIST: [a, b, c, ...] see explanation below.
                 's_n',  # DICT: State partitioned number of grains
                 'g',  # DICT: grains
                 'gb',  # DICT: grains
                 'positions',  # DICT: gids as per spatial location string
                 'mp',  # DICT: UPXO mul-point objects
                 'vtgs',  # DICT: VTGS instances
                 'mesh',  # OBJECT: mesh data structure
                 'px_size',  # FLOAT: pixel area if dim=2 else volume of dim=3
                 'dim',  # INT: DImensionaality
                 'prop_flag',  # DICT: flags indicating variables to compute
                 'prop',  # PANDAS TABLE of properties
                 'are_properties_available',  # True if properties have been caculated
                 'prop_stat',  # PANDAS TABLE of property statistics
                 '__gi__',  # Grain index used for __iter__
                 '__ui',  # Stores original user inp used by grid() instance
                 'display_messages',
                 'info',
                 )
    '''
    Explanation of 'n':
        It is the total number of grains across all states

    Explantion of 'lgi':
        * It is the lattice of pixels with grain ID values.
        * A pixel belonging to nth grain is assigned a value of n.
        * Counting is global. This means that:
            Grain numbering is not state-wise but over all available states.
            Pixel numbers take the overall number of the grain it belongs to.
        * Example: If pxtal.gs[m=10].lgi has 3 states with 1, 4 and 3 grains
        belonging to 1st, 2nd and 3rd grains, then all pixels belonging
        to this single grain of the 1st state will be assigned a value 1,
        all pixels belomngoig to the first grain of the 2nd state will be
        assigned a value of 2 (if numbering is local and state-wise), then
        these pixels too would have recieved a value of 1. Similarly,
        all pixels belonging to the last grain of the 3rd state will be
        assigned a value of 8.
        * Benefit:
            * It is far better to store 1 single array with global numbering
            and using a mapper list between state value and lgi values,
            instead of having S number arrays with local state-wise numnbered
            pixels and no mapper list.
            * Reduces code complexity
            * Consumes less memory
            * One such array is enough to represent all (most) the data inside
            the grain structrure.
            * Avoides the requirement to store individual grains
        * Use:
            * extracting individual grains

    Explanation of 'gid_s':
        * LIST: [a, b, c, ...]
        * a is the state value of the 0th grain:
            - if grains exist in this state, then it will be s in S
            - if no grains bvelowng to state s, then a will be None
        * b is the state value of the 1st grain:
            - if grains exist in this state, then it will be s in S
            - if no grains bvelowng to state s, then a will be None
        * c is the state value of the 2nd grain:
            - if grains exist in this state, then it will be s in S
            - if no grains bvelowng to state s, then a will be None
        * and so on..
    '''
    EPS = 1e-12
    __maxGridSizeToIgnoreStoringGrids = 25**3

    def __init__(self,
                 dim=2,
                 m=None,
                 uidata=None,
                 S_total=None,
                 px_size=None,
                 xgr=None,
                 ygr=None,
                 zgr=None,
                 uigrid=None
                 ):
        """


        Parameters
        ----------
        dim : TYPE, optional
            DESCRIPTION. The default is 2.
        m : TYPE, optional
            DESCRIPTION. The default is None.
        uidata : TYPE, optional
            DESCRIPTION. The default is None.
        S_total : TYPE, optional
            DESCRIPTION. The default is None.
        px_size : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        self.dim = dim
        self.m = m
        self.S = S_total
        self.__ui = uidata
        self.px_size = px_size
        self.uigrid = uigrid
        self.set__spart_flag(S_total)
        self.set__s_gid(S_total)
        self.set__gid_s()
        self.set__s_n(S_total)
        self.g = {}
        self.gb = {}
        self.info={}
        # ------------------------------------
        '''
        gc: Grain Centroids
        gcpos: Grain Centroids for position segregated grains
        rp: Representative Points
        jp2: Double Junction Points
        jp3: Triple Junction Points
        jp4: Qadruple Point Junctions
        '''
        self.mp = {'gc': None,
                   'gcpos': {'in': None,
                             'boundary': None,
                             'corner': None,
                             'left': None,
                             'bottom': None,
                             'right': None,
                             'top': None,
                             'pure_left': None,
                             'pure_bottom': None,
                             'pure_right': None,
                             'pure_top': None,
                             },
                   'rp': None,
                   'jp2': None,
                   'jp3': None,
                   }
        # ------------------------------------
        if self.dim==2:
            self.xgr, self.ygr = xgr, ygr
        elif self.dim==3:
            if xgr.size>=self.__maxGridSizeToIgnoreStoringGrids:
                self.xgr, self.ygr, self.zgr = None, None, None
                self.info['grid'] = 'Large grid. Please use >> Grid_Object.(xgr/ygr/zgr) instead'
            elif xgr.size<self.__maxGridSizeToIgnoreStoringGrids:
                self.xgr, self.ygr, self.zgr = xgr, ygr, zgr
        # ------------------------------------
        self.are_properties_available = False
        self.display_messages = False
        self.__setup__positions__()

    def __iter__(self):
        self.__gi__ = 1
        return self

    def __next1__(self):
        if self.n:
            if self.__gi__ <= self.n:
                grain_pixel_indices = np.argwhere(self.lgi == self.__gi__)
                self.__gi__ += 1
                return grain_pixel_indices
            else:
                raise StopIteration

    def __next__(self):
        if self.n:
            if self.__gi__ <= self.n:
                thisgrain = self.g[self.__gi__]['grain']
                self.__gi__ += 1
                return thisgrain
            else:
                raise StopIteration


    def __str__(self):
        """


        Returns
        -------
        str
            DESCRIPTION.

        """

        return 'grains :: att : n, lgi, id, ind, spart'

    def __att__(self):
        return gops.att(self)

    @property
    def get_px_size(self):
        """


        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        return self.px_size

    def set__s_n(self,
                 S_total,
                 ):
        """
        nth value represents the number of grains in the nth state

        Parameters
        ----------
        S_total : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        self.s_n = [0 for s in range(1, S_total+1)]

    def set__s_gid(self,
                   S_total,
                   ):
        """


        Parameters
        ----------
        S_total : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        self.s_gid = {s: None for s in range(1, S_total+1)}

    def set__gid_s(self):
        """


        Returns
        -------
        None.

        """

        self.gid_s = []

    def set__spart_flag(self,
                        S_total,
                        ):
        """


        Parameters
        ----------
        S_total : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.spart_flag = {_s_: False for _s_ in range(1, S_total+1)}

    def get_binaryStructure3D(self):
        return self.binaryStructure3D

    def set_binaryStructure3D(self, n):
        if n in (1, 2, 3):
            self.binaryStructure3D = n
        else:
             print('Invalid binary structure-3D. n must be in (1, 2, 3). Value not set')

    def _check_lgi_dtype_uint8(self,
                               lgi,
                               ):
        """
        Validates and modifies (if needed) lgi user input data-type

        Parameters
        ----------
        lgi : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        if type(lgi) == np.ndarray and np.size(lgi) > 0 and np.ndim(lgi) == 2:
            if self.lgi.dtype.name != 'uint8':
                self.lgi = lgi.astype(np.uint8)
            else:
                self.lgi = lgi
        else:
            self.lgi = 'invalid mcgs 4685'

    def calc_num_grains(self,
                        throw=False,
                        ):
        """
        Calculate the total number of grains in this grain structure

        Parameters
        ----------
        throw : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        if self.lgi:
            self.n = self.lgi.max()
            if throw:
                return self.n

    def neigh(self):
        for _gid_ in self.gid:
            self.neigh_gid(_gid_)

    def neigh_gid(self, gid, throw=False):
        bounds = self.g[gid]['grain'].bbox_ex_bounds
        probable_grains_locs = self.lgi[bounds[0]:bounds[1]+1,
                                        bounds[2]:bounds[3]+1
                                        ]
        # probable_grains = np.unique(probable_grains_locs)
        temp = deepcopy(probable_grains_locs)
        """ For row, col of a location in probable_grains_locs with value = 2,
        replace the value immediate neighbourhood of row and col to be nan.
        If the immediate neighbourhood has value == 2, then ignore """
        for row in range(temp.shape[0]):
            for col in range(temp.shape[1]):
                if temp[row, col] == gid:
                    if row - 1 >= 0:
                        if temp[row - 1, col] != gid:
                            temp[row - 1, col] = -1
                    if row + 1 < temp.shape[0]:
                        if temp[row + 1, col] != gid:
                            temp[row + 1, col] = -1
                    if col - 1 >= 0:
                        if temp[row, col - 1] != gid:
                            temp[row, col - 1] = -1
                    if col + 1 < temp.shape[1]:
                        if temp[row, col + 1] != gid:
                            temp[row, col + 1] = -1
        """
        if values in probable_grains_locs not equal to -1,
        then replace them with 0
        """
        for row in range(temp.shape[0]):
            for col in range(temp.shape[1]):
                if temp[row, col] != -1:
                    temp[row, col] = 0
        """ Find out the gids of the neighbouring grains """
        neigh_pixel_locs = np.argwhere(temp == -1)
        neigh_pixel_grain_ids = probable_grains_locs[neigh_pixel_locs[:, 0],
                                                     neigh_pixel_locs[:, 1]]
        neighbour_ids = np.unique(neigh_pixel_grain_ids)
        """ Store the neighbnour_ids inside the grain object """
        self.g[gid]['grain'].neigh = tuple(neighbour_ids)
        """ Mark the locations of grain boundaries which woulsd be individual
        segments. Each segnment marks the grain boundary interface of the
        'gid' grain with its neighbouring grains. """
        gbsegs_pre = np.zeros_like(temp)
        for ni in neighbour_ids:
            gbsegs_pre[np.logical_and(temp == -1,
                                      probable_grains_locs == ni)] = ni
        """ Store the grain boundary segment locations inside the grain
        object's data structure """
        self.g[gid]['grain'].gbsegs_pre = gbsegs_pre



        plot_neighbourhood = 0

        if plot_neighbourhood == 1:
            plt.figure()
            plt.imshow(self.s[bounds[0] : bounds[1] + 1,
                              bounds[2] : bounds[3] + 1])
            plt.title("Local grain neighbourhood of \n Grain #= {}".format(gid))
            plt.xlabel("x")
            plt.ylabel("y")
            plt.show()

            plt.figure()
            plt.imshow(gbsegs_pre)
            plt.title("Local grain boundary neighbourhood of \n Grain #= {}".format(gid))
            plt.xlabel("x")
            plt.ylabel("y")
            plt.show()

            temp = np.zeros_like(self.lgi)
            for ni in neighbour_ids:
                temp[np.where(self.lgi == ni)] = ni

            plt.figure()
            plt.imshow(temp)
            # title showing grain number and its neighbouring grain numbers
            plt.title(f"Grain #= {gid} \n Neighbouring grain numbers: \n {neighbour_ids}")

            plt.xlabel("x")
            plt.ylabel("y")
            plt.show()

    def make_prop2d_df(self,
                       brec=True,
                       brec_ex=True,
                       npixels=True,
                       npixels_gb=True,
                       area=True,
                       eq_diameter=True,
                       perimeter=True,
                       perimeter_crofton=True,
                       compactness=True,
                       gb_length_px=True,
                       aspect_ratio=True,
                       solidity=True,
                       morph_ori=True,
                       circularity=True,
                       eccentricity=True,
                       feret_diameter=True,
                       major_axis_length=True,
                       minor_axis_length=True,
                       euler_number=True,
                       append=False,
                       ):
        """
        Construct empty pandas dataframe of properties

        Parameters
        ----------
        brec : bool
            Bounding rectangle
        brec_ex : bool
            Extended bounding rectangle
        npixels : bool
            Number of pixels in the grain.
        npixels_gb : bool
            Number of pixels on the grain boundary.
        area : bool
            Area of the grain: number of pixels in the
            grain * pixel area.
        eq_diameter : bool
            Equivalent circle diameter.
        perimeter : bool
            Perimeter of the grain boundary.
            DEF: `Total length of all lines  passing through the centres
            of grain boundary pixels taken in order`
        perimeter_crofton : bool
            Crofton type perimeter of the grain boundary.
        compactness : bool
            Compactness of the grain.
            DEF: `(pixel area) / (Area of circle with perimeter equal to
                                  grain perimeter)`.
        gb_length_px : bool
            Deprecated use. Not recommended.
        aspect_ratio : bool
            Aspect ratio of the grain.
            Calculated as the ratio of major axis length to minor axis
            length of a ellipse fit to the grain.
        solidity : bool
            Solidity of the grain.
            DEF: `(npixels) / (number of pixels falling inside
                               the convex hull computed on the grain)`
        morph_ori : bool
            Morphological orientation of the grain (in deg).
            DEF: `-pi/2 to +pi/2. Counter-clockwise from x-axis`
        circularity : bool
            Indicate how close the grain shape is to being circular.
        eccentricity : bool
            Eccentricity of the grain.
            DEF: `(distance between focal points) / (major axis length)`
        feret_diameter : bool
            Average Feret diameter of the grain. Also called Caliper
            diameter
            NOTE: The name Caliper diameter is not explicitly used inside
            UPXO.
            DEF: Feret, or Caliper diameter is essentially the perpendicular
            distance between the two parallel lines running parallel to the
            grain boundary. Consequently, it is bounded by a minimum and a
            maximum value. `<Df> = Df_max / Df_min`, where `Df_max` and
            `Df_min` are the maximum and minimum Feret diamater.
        major_axis_length : bool
            Major axis length of the ellipse fit to the grain.
        minor_axis_length : bool
            Minor axis length of the ellipse fit to the grain.
        euler_number : bool
            Euler number of the grain.
            Will be 1 for grains without island grains.
        append : bool
            DESCRIPTION


        Returns
        -------
        None.

        """
        if not append:
            import pandas as pd
            # Make new Pandas dataframe
            self.prop_flag = {'npixels': npixels,
                              'npixels_gb': npixels_gb,
                              'area': area,
                              'eq_diameter': eq_diameter,
                              'perimeter': perimeter,
                              'perimeter_crofton': perimeter_crofton,
                              'compactness': compactness,
                              'gb_length_px': gb_length_px,
                              'aspect_ratio': aspect_ratio,
                              'solidity': solidity,
                              'morph_ori': morph_ori,
                              'circularity': circularity,
                              'eccentricity': eccentricity,
                              'feret_diameter': feret_diameter,
                              'major_axis_length': major_axis_length,
                              'minor_axis_length': minor_axis_length,
                              'euler_number': euler_number
                              }
            _columns = [key for key in self.prop_flag.keys()
                        if self.prop_flag[key]]
            self.prop = pd.DataFrame(columns=_columns)
            self.prop_stat = pd.DataFrame(columns=_columns)

    def char_morph_2d(self,
                      brec=True,
                      brec_ex=True,
                      npixels=True,
                      npixels_gb=True,
                      area=True,
                      eq_diameter=True,
                      perimeter=True,
                      perimeter_crofton=True,
                      compactness=True,
                      gb_length_px=True,
                      aspect_ratio=True,
                      solidity=True,
                      morph_ori=True,
                      circularity=True,
                      eccentricity=True,
                      feret_diameter=True,
                      major_axis_length=True,
                      minor_axis_length=True,
                      euler_number=True,
                      append=False,
                      ):
        """
        This method allows user to calculate morphological parameters
        of a given grain structure slice.

        Parameters
        ----------
        brec : bool
            Bounding rectangle
        brec_ex : bool
            DESCRIPTION
        npixels : bool
            Number of pixels in the grain.
        npixels_gb : bool
            Number of pixels on the grain boundary.
        area : bool
            Area of the grain: number of pixels in the
            grain * pixel area.
        eq_diameter : bool
            Equivalent circle diameter.
        perimeter : bool
            Perimeter of the grain boundary.
            DEF: `Total length of all lines  passing through the centres
            of grain boundary pixels taken in order`
        perimeter_crofton : bool
            Crofton type perimeter of the grain boundary.
        compactness : bool
            Compactness of the grain.
            DEF: `(pixel area) / (Area of circle with perimeter equal to
                                  grain perimeter)`.
        gb_length_px : bool
            Deprecated use. Not recommended.
        aspect_ratio : bool
            Aspect ratio of the grain.
            Calculated as the ratio of major axis length to minor axis
            length of a ellipse fit to the grain.
        solidity : bool
            Solidity of the grain.
            DEF: `(npixels) / (number of pixels falling inside
                               the convex hull computed on the grain)`
        morph_ori : bool
            Morphological orientation of the grain (in deg).
            DEF: `-pi/2 to +pi/2. Counter-clockwise from x-axis`
        circularity : bool
            Indicate how close the grain shape is to being circular.
        eccentricity : bool
            Eccentricity of the grain.
            DEF: `(distance between focal points) / (major axis length)`
        feret_diameter : bool
            Average Feret diameter of the grain. Also called Caliper
            diameter
            NOTE: The name Caliper diameter is not explicitly used inside
            UPXO.
            DEF: Feret, or Caliper diameter is essentially the perpendicular
            distance between the two parallel lines running parallel to the
            grain boundary. Consequently, it is bounded by a minimum and a
            maximum value. `<Df> = Df_max / Df_min`, where `Df_max` and
            `Df_min` are the maximum and minimum Feret diamater.
        major_axis_length : bool
            Major axis length of the ellipse fit to the grain.
        minor_axis_length : bool
            Minor axis length of the ellipse fit to the grain.
        euler_number : bool
            Euler number of the grain.
            Will be 1 for grains without island grains.
        append : bool
            DESCRIPTION


        Returns
        -------
        None.

        Pre-requsites
        -------------
        Successfull grain detection, with following attributes to exist:
            n: number of grains
            g: s-partitioned dictionary for storing grain objects
            gs: s-partitioned dictionary for storing grain_boundary
            objects

        """
        # Make data holder for properties
        self.make_prop2d_df(brec=brec,
                            brec_ex=brec_ex,
                            npixels=npixels,
                            npixels_gb=npixels_gb,
                            area=area,
                            eq_diameter=eq_diameter,
                            perimeter=perimeter,
                            perimeter_crofton=perimeter_crofton,
                            compactness=compactness,
                            gb_length_px=gb_length_px,
                            aspect_ratio=aspect_ratio,
                            solidity=solidity,
                            morph_ori=morph_ori,
                            circularity=circularity,
                            eccentricity=eccentricity,
                            feret_diameter=feret_diameter,
                            major_axis_length=major_axis_length,
                            minor_axis_length=minor_axis_length,
                            euler_number=euler_number,
                            append=append,
                            )
        # Find one property at a time
        # npixels, area, eq_diameter, gb_length_px = [], [], [], []
        # aspect_ratio, solidity, morph_ori = [], [], []
        # circularity, eccentricity, feret_diameter = [], [], []
        # major_axis_length, minor_axis_length = [], []
        # euler_number, perimeter, perimeter_crofton = [], [], []
        # compactness, npixels_gb = [], []
        # ---------------------------------------------
        # from mcgs import grain2d
        from upxo.xtal.mcgrain2d_definitions import grain2d
        # ---------------------------------------------
        from skimage.measure import regionprops
        # ---------------------------------------------
        Rlab = self.lgi.shape[0]
        Clab = self.lgi.shape[1]
        # ---------------------------------------------
        print('////////////////////////////////')
        print('Extracting requested grain structure properties across all available states')
        for s in self.s_gid.keys():
            if self.display_messages:
                print(f"     State value: {s}")
            # Extract s values which contain grains
            s_gid_vals_npy = list(self.s_gid.values())
            nonNone = np.argwhere(np.array(list(self.s_gid.values())) != None)
            s_gid_vals_npy = [s_gid_vals_npy[i] for i in np.squeeze(nonNone)]
            s_gid_keys_npy = np.array(list(self.s_gid.keys()))
            s_gid_keys_npy = s_gid_keys_npy[np.squeeze(nonNone)]
            # ---------------------------------------------
            sn = 1
            for state, grains in zip(s_gid_keys_npy, s_gid_vals_npy):
                # Iterate through each grain of this state value
                for gn in grains:
                    _, lab = cv2.connectedComponents(np.array(self.lgi == gn,
                                                              dtype=np.uint8))
                    self.g[gn] = {'s': state,
                                  'grain': grain2d()}
                    self.g[gn]['grain'].gid = gn
                    locations = np.argwhere(lab == 1)
                    self.g[gn]['grain'].loc = locations
                    _ = locations.T
                    self.g[gn]['grain'].xmin = _[0].min()
                    self.g[gn]['grain'].xmax = _[0].max()
                    self.g[gn]['grain'].ymin = _[1].min()
                    self.g[gn]['grain'].ymax = _[1].max()
                    self.g[gn]['grain'].s = state
                    self.g[gn]['grain'].sn = sn
                    self.g[gn]['grain'].px_area = self.px_size
                    sn += 1
                    # ---------------------------------------------
                    # Extract grain boundary indices
                    mask = np.zeros_like(self.lgi)
                    mask[self.lgi == gn] = 255
                    mask = mask.astype(np.uint8)
                    contours, _ = cv2.findContours(mask,
                                                   cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_NONE)
                    gb = np.squeeze(contours[0], axis=1)
                    # Interchange the row and column to get into right
                    # indexing order
                    gb[:, [1, 0]] = gb[:, [0, 1]]
                    self.g[gn]['grain'].gbloc = deepcopy(gb)
                    # ---------------------------------------------
                    # Extract bounding rectangle
                    Rlab = lab.shape[0]
                    Clab = lab.shape[1]

                    # PXGS.gs[4].lgi
                    # labels = np.array(PXGS.gs[4].lgi==4, dtype = int)

                    rmin = np.where(lab == 1)[0].min()
                    rmax = np.where(lab == 1)[0].max()+1
                    cmin = np.where(lab == 1)[1].min()
                    cmax = np.where(lab == 1)[1].max()+1

                    rmin_ex = rmin - int(rmin != 0)
                    rmax_ex = rmax + int(rmin != Rlab)
                    cmin_ex = cmin - int(cmin != 0)
                    cmax_ex = cmax + int(cmax != Clab)
                    # Store the bounds of the bounding box
                    self.g[gn]['grain'].bbox_bounds = [rmin,
                                                       rmax,
                                                       cmin,
                                                       cmax]
                    # Store the bounds of the extended bounding box
                    self.g[gn]['grain'].bbox_ex_bounds = [rmin_ex,
                                                          rmax_ex,
                                                          cmin_ex,
                                                          cmax_ex]
                    # Store bounding box
                    self.g[gn]['grain'].bbox = np.array(lab[rmin:rmax,
                                                            cmin:cmax],
                                                        dtype=np.uint8)
                    # Store the extended bounding box
                    self.g[gn]['grain'].bbox_ex = np.array(lab[rmin_ex:rmax_ex,
                                                               cmin_ex:cmax_ex],
                                                           dtype=np.uint8)
                    # Store the scikit-image regionproperties generator
                    self.g[gn]['grain'].make_prop(regionprops, skprop=True)
                    self.g[gn]['grain'].coords=np.array([[self.xgr[ij[0], ij[1]],
                                                          self.ygr[ij[0], ij[1]]]
                                                         for ij in self.g[gn]['grain'].loc])
        print('////////////////////////////////\n\n\n')
        self.build_prop()
        self.are_properties_available = True
        self.char_grain_positions_2d()

    def make_prop3d_df(self,
                       bcub=True,
                       bcub_ex=True,
                       npixels=True,
                       npixels_gb=True,
                       npixels_gbe=True,
                       npixels_gbjp=True,
                       volume=True,
                       volumeGeo=True,
                       areas=True,
                       sphere_eq_diameter=True,
                       elfita=True,
                       elfitb=True,
                       elfitc=True,
                       aspect_ratio_ab=True,
                       aspect_ratio_bc=True,
                       aspect_ratio_ac=True,
                       solidity=True,
                       append=False,
                       ):
        """
        bcub
        bcub_ex
        npixels
        npixels_gb
        npixels_gbe
        npixels_gbjp
        volume
        volumeGeo
        areas
        sphere_eq_diameter
        elfita
        elfitb
        elfitc
        aspect_ratio_ab
        aspect_ratio_bc
        aspect_ratio_ac
        solidity

        Construct empty pandas dataframe of properties needed for 3D grain
        structure

        Parameters
        ----------
        bcub : bool
            Bounding cuboid. np.array
        bcub_ex : bool
            Extended bounding cuboid. np.array
        npixels : bool
            Number of pixels in the grain. int
        npixels_gb : bool
            Number of pixels on the grain boundary surface. int
        npixels_gbe : bool
            Number of pixels on grain boundary edge. int
        npixels_gbjp : bool
            Number of pixels on grain boundary junction points. int
        volume : bool
            Pixellated volume of the grains. float
            npixels * pixel_volume
        volumeGeo : bool
            Geometric volume of the grains. This is calculayed using
            boundary surface extraction, triangulation and smoothing
            operations. float
        areas : bool
            Areas of the grain boundary surfaces. This is calculated using
            boundary surface extraction, triangulation and smoothing
            operations.
        sphere_eq_diameter : bool
            Equivalent sphere diameter.
        ellfita : bool
            Maximum axis length a of ellipsoidal fit. float
        ellfitb : bool
            INtermediate axis length b of ellipsoidal fit. float
        ellfitc : bool
            Minimum axis length c of ellipsoidal fit. float
        ellori: bool
            Morphological orientation of the ellipsoidal fit. np.array
        aspect_ratio_ab : bool
            ellfita/ellfitb
        aspect_ratio_bc : bool
            ellfitb/ellfitc
        aspect_ratio_ac : bool
            ellfita/ellfitc
        solidity : bool
            Solidity of the grain calculated as the ratio of p[ixel volume to
            total convex hull volume

        Returns
        -------
        None.

        """
        if not append:
            import pandas as pd
            # Make new Pandas dataframe
            self.prop_flag = {'bcub': bcub,
                              'bcub_ex': bcub_ex,
                              'npixels': npixels,
                              'npixels_gb': npixels_gb,
                              'npixels_gbe': npixels_gbe,
                              'npixels_gbjp': npixels_gbjp,
                              'volume': volume,
                              'volumeGeo': volumeGeo,
                              'areas': areas,
                              'sphere_eq_diameter': sphere_eq_diameter,
                              'elfita': elfita,
                              'elfitb': elfitb,
                              'elfitc': elfitc,
                              'aspect_ratio_ab': aspect_ratio_ab,
                              'aspect_ratio_bc': aspect_ratio_bc,
                              'aspect_ratio_ac': aspect_ratio_ac,
                              'solidity': solidity,
                              }
            _columns = [key for key in self.prop_flag.keys()
                        if self.prop_flag[key]]
            self.prop = pd.DataFrame(columns=_columns)
            self.prop_stat = pd.DataFrame(columns=_columns)

    def __setup__positions__(self):
        self.positions = {'top_left': [], 'bottom_left': [],
                          'bottom_right': [], 'top_right': [],
                          'pure_right': [], 'pure_bottom': [],
                          'pure_left': [], 'pure_top': [],
                          'left': [], 'bottom': [], 'right': [], 'top': [],
                          'boundary': [], 'corner': [], 'internal': []
                          }

    def char_grain_positions_2d(self):
        row_max = self.lgi.shape[0]-1
        col_max = self.lgi.shape[1]-1
        for grain in self:
            # Calculate normalized centroids serving as numerical position
            # values
            grain.position = list(grain.centroid)
            # Determine the location strings for all grains
            all_pixel_locations = grain.loc.tolist()
            apl = np.array(all_pixel_locations).T  # all_pixel_locations
            if 0 in apl[0]:  # TOP
                '''
                grain touches either:
                    top and/or left boundary, OR, top and/or right boundary
                '''
                if 0 in apl[1]:  # TOP AND LEFT
                    '''
                    BRANCH.1.A. Grain touches top and left boundary: top_left
                    grain. This means the grain is TOP_LEFT CORNER GRAIN
                    '''
                    grain.position.append('top_left')
                elif col_max in apl[1]:  # TOP AND RIGHT
                    '''
                    BRANCH.1.B. Grain touches top and right boundary: top_right
                    grain This means the grain is a TOP_RIGHT CORNER GRAIN
                    '''
                    grain.position.append('top_right')
                else:  # TOP, NOT LEFT, NOT RIGHT: //PURE TOP//
                    '''
                    BRANCH.1.C. Grain touches top boundary only and not the
                    corners of the top boundary. This means the grain is a
                    TOP GRAIN
                    '''
                    grain.position.append('pure_top')
            if row_max in apl[0]:  # BOTTOM
                '''
                grain touches either:
                    * bottom and/or left boundary, OR,
                    * bottom and/or right boundary
                '''
                if 0 in apl[1]:  # BOTTOM AND LEFT
                    '''
                    BRANCH.2.A. Grain touches bottom and left boundary:
                    bot_left grain. This means the grain is BOTTOM_LEFT CORNER
                    GRAIN
                    '''
                    grain.position.append('bottom_left')
                elif col_max in apl[1]:  # BOTTOM AND RIGHT
                    '''
                    BRANCH.2.B. Grain touches bottom and right boundary:
                    bot_right grain. This means the grain is BOTTOM_RIGHT
                    CORNER GRAIN
                    '''
                    grain.position.append('bottom_right')
                else:  # BOTTOM, NOT LEFT, NOT RIGHT: //PURE BOTTOM//
                    '''
                    BRANCH.2.C. Grain touches only bottom boundary and not the
                    corners of the bottom boundary. This means the grain is a
                    BOTTOM GRAIN
                    '''
                    grain.position.append('pure_bottom')
            if 0 in apl[1]:  # LEFT
                '''
                grain touches either:
                    * left and/or top boundary, OR,
                    * left and/or bottom boundary
                '''
                if 0 in apl[0]:  # LEFT AND TOP
                    '''
                    BRANCH.3.A. Grain touches left and top boundary: top_left
                    grain. This means the grain is LEFT_TOP CORNER GRAIN
                    '''
                    # THIS BRANCH HAS ALREADY BEEN VISITED IN BRANCH.1.A
                    # NOTHING MORE TO DO HERE. SKIP.
                    pass
                elif row_max in apl[0]:  # LEFT AND BOTTOM
                    '''
                    BRANCH.3.B. Grain touches left and bottom boundary:
                    bot_left grain. This means the grain is a LEFT_BOTTOM
                    CORNER GRAIN
                    '''
                    # THIS BRANCH HAS ALREADY BEEN VISISITED IN BRANCH.2.A
                    # NOTHING MORE TO DO HERE. SKIP.
                    pass
                else:  # LEFT, NOT TOP, NOT BOTTOM: //PURE LEFT//
                    '''
                    BRANCH.3.C. Grain touches left boundary only and not the
                    corners of the left boundary. This means the grain is a #
                    LEFT GRAIN
                    '''
                    grain.position.append('pure_left')
            if col_max in apl[1]:  # RIGHT
                '''
                grain touches either:
                    * right and/or top boundary, OR,
                    * right and/or bottom boundary
                '''
                if 0 in apl[0]:  # RIGHT AND TOP
                    '''
                    BRANCH.4.A. Grain touches right and top boundary: top_right
                    grain. This means the grain is RIGHT_TOP CORNER GRAIN
                    '''
                    # THIS BRANCH HAS ALREADY BEEN VISITED IN BRANCH.1.B
                    # NOTHING MORE TO DO HERE. SKIP.
                    pass
                elif row_max in apl[0]:  # RIGHT AND BOTTOM
                    '''
                    BRANCH.4.B. Grain touches left and bottom boundary:
                    bot_left grain. This means the grain is a RIGHT_BOTTOM
                    CORNER GRAIN
                    '''
                    # THIS BRANCH HAS ALREADY VBEEN VISISITED IN BRANCH.2.B
                    # NOTHING MORE TO DO HERE. SKIP.
                    pass
                else:  # RIGHT, NOT TOP, NOT BOTTOM: //PURE RIGHT//
                    '''
                    BRANCH.4.C. Grain touches left boundary only and not the
                    corners of the left boundary. This means the grain is a
                    RIGHT GRAIN
                    '''
                    grain.position.append('pure_right')
            if 0 not in apl[0] and row_max not in apl[0]:
                # NOT TOP, NOT BOTTOM
                if 0 not in apl[1] and col_max not in apl[1]:
                    # NOT LEFT, NOT RIGHT
                    grain.position.append('internal')

        for grain in self:
            position = grain.position[2]
            gid = grain.gid
            _ = [position == 'top_left',
                 position == 'bottom_left',
                 position == 'bottom_right',
                 position == 'top_right',
                 position == 'pure_right',
                 position == 'pure_bottom',
                 position == 'pure_left',
                 position == 'pure_top',
                 position == 'left',
                 position == 'bottom',
                 position == 'right',
                 position == 'top',
                 position == 'boundary',
                 position == 'corner',
                 position == 'internal'
                 ]
            self.positions[[_*position
                            for _ in _ if _*position][0]].append(gid)

        for pos in ['top_left', 'bottom_left', 'pure_left']:
            if self.positions[pos]:
                for value in self.positions[pos]:
                    self.positions['left'].append(value)
        for pos in ['bottom_left', 'pure_bottom', 'bottom_right']:
            if self.positions[pos]:
                for value in self.positions[pos]:
                    self.positions['bottom'].append(value)
        for pos in ['bottom_right', 'pure_right', 'top_right']:
            if self.positions[pos]:
                for value in self.positions[pos]:
                    self.positions['right'].append(value)
        for pos in ['top_right', 'pure_top', 'top_left']:
            if self.positions[pos]:
                for value in self.positions[pos]:
                    self.positions['top'].append(value)
        for pos in ['top_left', 'bottom_left', 'bottom_right', 'top_right']:
            if self.positions[pos]:
                for value in self.positions[pos]:
                    self.positions['corner'].append(value)
        for pos in ['top_left', 'bottom_left', 'bottom_right', 'top_right',
                    'pure_left', 'pure_bottom', 'pure_right', 'pure_top'
                    ]:
            if self.positions[pos]:
                for value in self.positions[pos]:
                    self.positions['boundary'].append(value)

    def find_prop_npixels(self):
        # Get grain NUMBER OF PIXELS into pandas dataframe
        if self.prop_flag['npixels']:
            npixels = []
            for g in self.g.values():
                npixels.append(len(g['grain'].loc))
            self.prop['npixels'] = npixels
            if self.display_messages:
                print('    Number of Pixels making the grains: DONE')

    def find_prop_npixels_gb(self):
        # Get grain GRAIN BOUNDARY LENGTH (NO. PIXELS) into pandas dataframe
        if self.prop_flag['npixels_gb']:
            npixels_gb = []
            for g in self.g.values():
                npixels_gb.append(len(g['grain'].gbloc))
            self.prop['npixels_gb'] = npixels_gb
            if self.display_messages:
                print('    Number of Pixels in grain bound. of grains: DONE')

    def find_prop_gb_length_px(self):
        # Get grain GRAIN BOUNDARY LENGTH (NO. PIXELS) into pandas dataframe
        if self.prop_flag['gb_length_px']:
            gb_length_px = []
            for g in self.g.values():
                gb_length_px.append(len(g['grain'].gbloc))
            self.prop['gb_length_px'] = gb_length_px
            if self.display_messages:
                print('    Grain Boundary Lengths of grains: DONE')

    def find_prop_area(self):
        # Get grain AREA into pandas dataframe
        if self.prop_flag['area']:
            area = []
            for g in self.g.values():
                area.append(g['grain'].skprop.area)
            self.prop['area'] = area
            if self.display_messages:
                print('    Areas of grains: DONE')

    def find_prop_eq_diameter(self):
        # Get grain EQUIVALENT DIAMETER into pandas dataframe
        if self.prop_flag['eq_diameter']:
            eq_diameter = []
            for g in self.g.values():
                eq_diameter.append(g['grain'].skprop.equivalent_diameter_area)
            self.prop['eq_diameter'] = eq_diameter
            if self.display_messages:
                print('    Circle Equivalent Diameter of grains: DONE')

    def find_prop_perimeter(self):
        # Get grain PERIMETER into pandas dataframe
        if self.prop_flag['perimeter']:
            perimeter = []
            for g in self.g.values():
                perimeter.append(g['grain'].skprop.perimeter)
            self.prop['perimeter'] = perimeter
            if self.display_messages:
                print('    Perimeter of grains: DONE')

    def find_prop_perimeter_crofton(self):
        # Get grain CROFTON PERIMETER into pandas dataframe
        if self.prop_flag['perimeter_crofton']:
            perimeter_crofton = []
            for g in self.g.values():
                perimeter_crofton.append(g['grain'].skprop.perimeter_crofton)
            self.prop['perimeter_crofton'] = perimeter_crofton
            if self.display_messages:
                print('    Crofton Perimeters of grains: DONE')

    def find_prop_compactness(self):
        # Get grain COMPACTNESS into pandas dataframe
        if self.prop_flag['compactness']:
            compactness = []
            if self.prop_flag['area']:
                if self.prop_flag['perimeter']:
                    for i, g in enumerate(self.g.values()):
                        area = self.prop['area'][i]
                        # Calculate area of circle with the same perimeter
                        # P = pi*D --> D = P/pi
                        # A = pi*D**2/4 = pi*(P/pi)**2/4 = P/(4*pi)
                        circle_area = self.prop['perimeter'][i]**2/(4*np.pi)
                        if circle_area >= self.EPS:
                            compactness.append(area/circle_area)
                        else:
                            compactness.append(1)
                else:
                    for i, g in self.g.values():
                        area = self.prop['area'][i]
                        circle_area = g['grain'].skprop.perimeter**2/(4*np.pi)
                        if circle_area >= self.EPS:
                            compactness.append(area/circle_area)
                        else:
                            compactness.append(1)
            else:
                if self.prop_flag['perimeter']:
                    for i, g in self.g.values():
                        area = g['grain'].skprop.area
                        circle_area = self.prop['perimeter'][i]**2/(4*np.pi)
                        if circle_area >= self.EPS:
                            compactness.append(area/circle_area)
                        else:
                            compactness.append(1)
                else:
                    for i, g in self.g.values():
                        area = g['grain'].skprop.area
                        circle_area = g['grain'].skprop.perimeter**2/(4*np.pi)
                        if circle_area >= self.EPS:
                            compactness.append(area/circle_area)
                        else:
                            compactness.append(1)

            self.prop['compactness'] = compactness
            if self.display_messages:
                print('    Compactness of grains: DONE')

    def find_prop_aspect_ratio(self):
        # Get grain ASPECT RATIO into pandas dataframe
        if self.prop_flag['aspect_ratio']:
            aspect_ratio = []
            for g in self.g.values():
                maj_axis = g['grain'].skprop.major_axis_length
                min_axis = g['grain'].skprop.minor_axis_length
                if min_axis <= self.EPS:
                    aspect_ratio.append(np.inf)
                else:
                    aspect_ratio.append(maj_axis/min_axis)
            self.prop['aspect_ratio'] = aspect_ratio
            if self.display_messages:
                print('    Aspect Ratios of grains: DONE')

    def find_prop_solidity(self):
        # Get grain SOLIDITY into pandas dataframe
        if self.prop_flag['solidity']:
            solidity = []
            for g in self.g.values():
                solidity.append(g['grain'].skprop.solidity)
            self.prop['solidity'] = solidity
            if self.display_messages:
                print('    Solidity of grains: DONE')

    def find_prop_circularity(self):
        # Get grain CIRCULARITY into pandas dataframe
        if self.prop_flag['circularity']:
            circularity = []
            if self.display_messages:
                print('    Circularity of grains: DONE')
            pass

    def find_prop_major_axis_length(self):
        # Get grain MAJOR AXIS LENGTH into pandas dataframe
        if self.prop_flag['major_axis_length']:
            major_axis_length = []
            for g in self.g.values():
                major_axis_length.append(g['grain'].skprop.axis_major_length)
            self.prop['major_axis_length'] = major_axis_length
            if self.display_messages:
                print('    Major Axis Length of ellipse fits of grains: DONE')

    def find_prop_minor_axis_length(self):
        # Get grain MINOR AXIS LENGTH into pandas dataframe
        if self.prop_flag['minor_axis_length']:
            minor_axis_length = []
            for g in self.g.values():
                minor_axis_length.append(g['grain'].skprop.axis_minor_length)
            self.prop['minor_axis_length'] = minor_axis_length
            if self.display_messages:
                print('    Minor Axis Length of ellipse fits of grains: DONE')

    def find_prop_morph_ori(self):
        # Get grain MORPHOLOGICAL ORIENTATION into pandas dataframe
        if self.prop_flag['morph_ori']:
            morph_ori = []
            for g in self.g.values():
                morph_ori.append(g['grain'].skprop.orientation)
            self.prop['morph_ori'] = [mo*180/np.pi for mo in morph_ori]
            if self.display_messages:
                print('    Morph. Orientation angle (deg) of grains: DONE')

    def find_prop_feret_diameter(self):
        # Get grain FERET DIAMETER into pandas dataframe
        if self.prop_flag['feret_diameter']:
            feret_diameter = []
            for g in self.g.values():
                feret_diameter.append(g['grain'].skprop.feret_diameter_max)
            self.prop['feret_diameter'] = feret_diameter
            if self.display_messages:
                print('    Feret Diameter of grains: DONE')

    def find_prop_euler_number(self):
        # Get grain EULER NUMBER into pandas dataframe
        if self.prop_flag['euler_number']:
            euler_number = []
            for g in self.g.values():
                euler_number.append(g['grain'].skprop.euler_number)
            self.prop['euler_number'] = euler_number
            if self.display_messages:
                print('    Euler Number of grains: DONE')

    def find_prop_eccentricity(self):
        # Get grain ECCENTRICITY into pandas dataframe
        if self.prop_flag['eccentricity']:
            eccentricity = []
            for g in self.g.values():
                eccentricity.append(g['grain'].skprop.eccentricity)
            self.prop['eccentricity'] = eccentricity
            if self.display_messages:
                print('    Eccentricity of grains: DONE')
        print("\n")

    def build_prop(self):
        self.find_prop_npixels()
        self.find_prop_npixels_gb()
        self.find_prop_gb_length_px()
        self.find_prop_area()
        self.find_prop_eq_diameter()
        self.find_prop_perimeter()
        self.find_prop_perimeter_crofton()
        self.find_prop_compactness()
        self.find_prop_aspect_ratio()
        self.find_prop_solidity()
        self.find_prop_circularity()
        self.find_prop_major_axis_length()
        self.find_prop_minor_axis_length()
        self.find_prop_morph_ori()
        self.find_prop_feret_diameter()
        self.find_prop_euler_number()
        self.find_prop_eccentricity()
        # ------------------------------------------
        if self.display_messages:
            count = 1
            print('The following user requested PROP_NAME are available:')
            if any(self.prop_flag):
                for prop_name, prop_name_flag in zip(self.prop_flag.keys(),
                                                     self.prop_flag.values()):
                    if prop_name_flag:
                        print(f'     {count}. {prop_name}')
                    count += 1
                print("\n")
                print("Storing all requested grain structure properties to pandas dataframe")
            else:
                print("No properties calulated as none were requested. Skipped")

    def docu(self):
        print("ACCESS-1:")
        print("---------")
        print("You can access all properties across all states as: ")
        print("    >> PXGS.gs[M].prop['PROP_NAME']")
        print("ACCESS-2:")
        print("---------")
        print("You can access all state-partitioned properties as:")
        print("    >> PXGS.gs[M].s_prop(s, PROP_NAME)")
        print('    Here, M: requested requested nth temporal slice of grain structure\n')
        print("          s: Desired state value\n")

        print('BASIC STATS:')
        print('------------')
        print("You can readily extract some basic statstics as:")
        print("    >> PXGS.gs[M].prop['area'].describe()[STAT_PARAMETER_NAME]")
        print('    Here, M: requested requested nth temporal slice of grain structure\n')
        print("    Permitted STAT_PARAMETER_NAME are:")
        print("    'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'\n")

        print("DATA VISUALIZATION:")
        print("-------------------")
        print("You can quickly view the distribution of a property as:")
        print("    >> plt.hist(PXGS.gs[n].prop['PROP_NAME'])\n")
        print("You can quickly view the grain structure as:")
        print("    >> plt.imshow(PXGS.gs[M].s)")
        print("    >> PXGS.plotgs(M, cmap='jet')")

        print("    >> plt.imshow(PXGS.gs[M].lgi)\n")
        print("You can quickly view a Ngth single grain:")
        print("    >> plt.imshow(PXGS.gs[M].g[Ng]['grain'].bbox_ex)\n")

        print('FURTHER DATA EXTRACTION:')
        print('------------------------')
        print('You can extract further grain properties as permitted by: ''skimage.measure.regionprops'', as:')
        print("    >> PXGS.gs[M].g[Ng]['grain'].PROP_NAME")
        print("    Here, M: temporal slice")
        print("          Ng: nth grain")
        print("          PROP_NAME: as permitted by sckit-image")
        print("    REF: https://github.com/scikit-image/scikit-image/blob/v0.21.0/skimage/measure/_regionprops.py#L1046-L1329")

    def get_stat(self,
                 PROP_NAME,
                 saa=True,
                 throw=False,
                 ):
        """
        Calculates ths statistics of a property in the 'prop' attribute.

        NOTE
        ----
        Input data is not sanitised before calculating the statistics.
        Will results in an error if invalid entries are found.

        Parameters
        ----------
        PROP_NAME : str
            Name of the property, whos statistics is to be calculated. They
            could be from the following list:
                1. npixels
                2. npixels_gb
                3. area
                4. eq_diameter
                5. perimeter
                6. perimeter_crofton
                7. compactness
                8. gb_length_px
                9. aspect_ratio
                10. solidity
                11. morph_ori
                12. circularity
                13. eccentricity
                14. feret_diameter
                15. major_axis_length
                16. minor_axis_length
                17. euler_number
        saa : bool, optional
            Flag to save the statistics as attribute.
            The default is True.
        throw : bool, optional
            Flag to return the computed statistics.
            The default is False.

        Returns
        -------
        metrics : TYPE
            DESCRIPTION.

        Metrics calculated
        ---------------------
        Following stastical metrics will be calculated:
            count: Data count value
            mean: Mean of the data
            std: Standard deviation of the data
            min: Minimum value of the data
            25%: First quartile of the data
            50%: Second quartile of the data
            75%: Third quartile of the data
            max: Maximum value of the data
            median: Median value of the data
            mode: List of modes of the data
            var: Variance of the data
            skew: Skewness of the data
            kurt: Kurtosis of the data
            nunique: Number of unique values in the data
            sem: Standard error of the mean of the data

        Example call
        ------------
            PXGS.gs[4].extract_statistics_prop('area')
        """
        # Extract the values of the PROP_NAME
        values = np.array(self.prop[PROP_NAME])
        # Extract non-inf subset
        values = values[np.where(values != np.inf)[0]]
        # Make the values dataframe
        import pandas as pd
        values_df = pd.DataFrame(columns=['temp'])
        values_df['temp'] = values
        # Extract basic statistics
        values_stats = values_df.describe()
        metrics = {'PROP_NAME': PROP_NAME,
                   'count': values_stats['temp']['count'],
                   'mean': values_stats['temp']['mean'],
                   'std': values_stats['temp']['std'],
                   'min': values_stats['temp']['min'],
                   '25%': values_stats['temp']['25%'],
                   '50%': values_stats['temp']['50%'],
                   '75%': values_stats['temp']['75%'],
                   'max': values_stats['temp']['max'],
                   'median': values_df['temp'].median(),
                   'mode': [i for i in values_df['temp'].mode()],
                   'var': values_df['temp'].var(),
                   'skew': values_df['temp'].skew(),
                   'kurt': values_df['temp'].kurt(),
                   'nunique': values_df['temp'].nunique(),
                   'sem': values_df['temp'].sem(),
                   }
        if saa:
            self.prop_stat = metrics
        if throw:
            return metrics

    def make_valid_prop(self,
                        PROP_NAME='aspect_ratio',
                        rem_nan=True,
                        rem_inf=True,
                        PROP_df_column = None,
                        ):
        """
        Remove invalid entries from a column in a Pandas dataframe and
        returns sanitized pandas column with the PROP_NAME as column name

        Parameters
        ----------
        PROP_NAME : str, optional
            Property to be cleansed. The default is 'aspect_ratio'.
        rem_nan : TYPE, optional
            Boolean flag to remove np.nan. The default is True.
        rem_inf : TYPE, optional
            Boolean flag to remove np.inf. Both negative and positive inf
            will be removed. The default is True.

        Returns
        -------
        subset : pd.data_frame
            A single column pandas dataframe with cleansed values.#
        ratio : float
            Ratio of total number of values removed to the size of the property
            column in the self.prop dataframe

        """
        if not PROP_df_column:
            # TYhis means internal data in prop atrtribute is to be cleaned
            if hasattr(self, 'prop'):
                if PROP_NAME in self.prop.columns:
                    _prop_size_ = self.prop[PROP_NAME].size
                    subset = self.prop[PROP_NAME]
                    subset = subset.replace([-np.inf,
                                             np.inf],
                                            np.nan).dropna()
                    ratio = (_prop_size_-subset.size)/_prop_size_
                else:
                    subset, ratio = None, None
                    print(f"Property {PROP_NAME} has not been calculated in temporal slice {self.m}")
            else:
                subset, ratio = None, None
                print(f"Temporal slice {self.m} has no prop. Skipped")
        else:
            # This means the user provided single-colulmn pandas dataframe,
            # named "PROP_df_column" is to be cleaned
            # It will be assumed user has input valid dataframe column
            _prop_size_ = PROP_df_column.size
            PROP_df_column = PROP_df_column.replace([-np.inf,
                                                     np.inf],
                                                    np.nan).dropna()
            ratio = (_prop_size_-PROP_df_column.size)/_prop_size_

        return subset, ratio

    def s_prop(self,
               s=1,
               PROP_NAME='area'
               ):
        """
        Extract state wise partitioned property. Property name has to be
        specified by the user.

        Parameters
        ----------
        s : int, optional
            Value of the
            The default is 1.
        PROP_NAME : TYPE, optional
            DESCRIPTION. The default is 'area'.

        Returns
        -------
        TYPE
            DESCRIPTION.

        # TODO
            1: add validity checking layers for s and PROP_NAME
            2: if s = 0, then any of the available be selected at random and
                returned
            3: if s = -1, then the state with the minimum number of grains
                will be returned
            4: if s = -2, then the state with the maximum number of grains
                will be returned
        """
        if hasattr(self, 'prop'):
            if PROP_NAME in self.prop.columns:
                if s in self.s_gid.keys():
                    PROP_VALUES_VALID = self.make_valid_prop(rem_nan=True,
                                                             rem_inf=True,
                                                             PROP_df_column = self.prop[PROP_NAME],
                                                             )
                    subset = self.prop[PROP_NAME].iloc[[i-1 for i in self.s_gid[s]]]
                else:
                    subset = None
                    print(f"Temporal slice {self.m} has no grains in s: {s}. Skipped")
            else:
                subset, ratio = None, None
                print(f"Property {PROP_NAME} has not been calculated in temporal slice {self.m}")
        else:
            print(f"Temporal slice {self.m} has no prop. Skipped")
        return subset

    def get_gid_prop_range(self,
                           PROP_NAME='area',
                           reminf=True,
                           remnan=True,
                           range_type='percentage',
                           value_range=[1, 2],
                           percentage_range=[0, 20],
                           rank_range=[60, 90],
                           pivot=None):
        '''
        DATA AND SUB-SELERCTION PROCEDURE:
        PROP_min--inf--------A-----nan------B---nan----inf------PROP_max
            1. clean data for inf and nans
            2. Then subselect from A to PROP_max
            3. Then subselect from A to B, which is what we need

        Example-1
        ---------
        gid, value, df_loc = PXGS.gs[8].get_gid_prop_range(PROP_NAME='aspect_ratio',
                                                   range_type='rank',
                                                   rank_range=[80, 100]
                                                   )
        Example-2
        ---------
        gid, value, df_loc = PXGS.gs[8].get_gid_prop_range(PROP_NAME='area',
                                                   range_type='percentage',
                                                   rank_range=[80, 100]
                                                   )
        Example-3
        ---------
        gid, value, df_loc = PXGS.gs[8].get_gid_prop_range(PROP_NAME='aspect_ratio',
                                                   range_type='value',
                                                   rank_range=[2, 2.5]
                                                   )
        '''
        print(PROP_NAME)
        if PROP_NAME in self.prop.columns:
            PROPERTY = self.prop[PROP_NAME].replace([-np.inf,
                                                     np.inf], np.nan).dropna()
            if range_type in ('percentage', '%',
                              'perc', 'by_percentage',
                              'by_perc', 'by%'
                              ):
                # If the user chooses to use percentage to describe the range
                # Get the minimum and maximum of the property
                PROP_min = PROPERTY.min()
                PROP_max = PROPERTY.max()
                # Calculate the fuill range if the proiperty
                PROP_range_full = PROP_max - PROP_min
                # Calculate the Lower cut-off
                lco = min(percentage_range)*PROP_range_full/100
                # Caluclate the upper cut-off
                uco = max(percentage_range)*PROP_max/100
                # w.r.t the the illustration in the DocString, subselect between A
                # and PROP_max
                A_MAX = self.prop[PROP_NAME][self.prop[PROP_NAME].index[self.prop[PROP_NAME] >= lco]]
                A_B_indices = A_MAX.index[A_MAX <= uco]
                A_B_values = A_MAX[A_B_indices].to_numpy()
                gids = A_B_indices+1
            elif range_type == ('value', 'by_value'):
                # If the user chooses to use values to describe the range of
                # objects
                lco = min(value_range)
                uco = max(value_range)
                # w.r.t the the illustration in the DocString, subselect between A
                # and PROP_max
                A_MAX = self.prop[PROP_NAME][self.prop[PROP_NAME].index[self.prop[PROP_NAME] >= lco]]
                A_B_indices = A_MAX.index[A_MAX <= uco]
                A_B_values = A_MAX[A_B_indices].to_numpy()
                gids = A_B_indices+1
            elif range_type == ('rank', 'by_rank'):
                '''
                # TODO: debug for the case where two entered values are same
                # TODO: Handle invalud user data
                '''
                values = self.prop[PROP_NAME]
                _ = values.replace([-np.inf,
                                    np.inf],
                                   np.nan).dropna().sort_values(ascending=False)
                indices = _.index
                ptile_i, ptile_j = [100-max(rank_range), 100-min(rank_range)]
                A_B_values = _[indices[int(ptile_i*_.size/100):int(ptile_j*_.size/100)]]
                A_B_indices = A_B_values.index
                gids = A_B_values.index.to_numpy()+1

        return gids, A_B_values, A_B_indices

    def plot_largest_grain(self):
        """
        A humble method to just plot the largest grain in a temporal slice
        of a grain structure

        Returns
        -------
        None.

        # TODO: WRAP THIS INSIDE A FIND_LARGEST_GRAIN AND HAVE IT TRHOW
        THE GID TO THE USER

        """
        gid = self.prop['area'].idxmax()+1
        self.g[gid]['grain'].plot()

    def plot_longest_grain(self):
        """
        A humble method to just plot the longest grain in a temporal slice
        of a grain structure

        Returns
        -------
        None.

        # TODO: WRAP THIS INSIDE A FIND_LONGEST_GRAIN AND HAVE IT TRHOW
        THE GID TO THE USER
        """
        gid, _, _ = self.get_gid_prop_range(PROP_NAME='aspect_ratio',
                                            range_type='percentage',
                                            percentage_range=[100, 100],
                                            )
        # plt.imshow(self.g[gid[0]]['grain'].bbox_ex)
        for _gid_ in gid:
            plt.figure()
            self.g[gid]['grain'].plot()
            plt.show()

    def mask_lgi_with_gids(self,
                           gids,
                           masker=-10
                           ):
        """
        Mask the lgi (PXGS.gs[n] specific lgi array: lattice of grain IDs)
        against user input grain indices, with a default UPXO-reserved
        place-holder value of -10.

        Parameters
        ----------
        gids : int/list
            Either a single grain index number or list of them
        kwargs:
            masker:
                An int value, preferably -10, but compulsorily less than -5.
        Returns
        -------
        s_masked : np.ndarray(dtype=int)
            lgi masked against gid values

        Internal calls (@dev)
        ---------------------
        None
        """

        # -----------------------------------------
        lgi_masked = deepcopy(self.lgi).astype(int)
        for gid in gids:
            if gid in self.gid:
                lgi_masked[lgi_masked == gid] = masker
            else:
                print(f"Invalid gid: {gid}. Skipped")
        # -----------------------------------------
        return lgi_masked, masker

    def mask_s_with_gids(self,
                         gids,
                         masker=-10,
                         force_masker=False):
        """
        Mask the s (PXGS.gs[n] specific s array) against user input grain
        indices

        Parameters
        ----------
        gids : int/list
            Either a single grain index number or list of them
        kwargs:
            masker:
                An int value, preferably -10.
            force_masker:
                This is here to satisfy the tussle of future development needs
                and user-readiness!! Please go with it for now.

                If True, user value for masker will be forced to
                masker variable, else the defaultr value of -10 will be used.

        Returns
        -------
        lgi_masked : np.ndarray(dtype=int)
            lgi masked against gid values

        Internal calls (@dev)
        ---------------------
        self.mask_lgi_with_gids()

        """
        # Validate suer supplied masker
        masker = (-10*(not force_masker) + int(masker*(force_masker and type(masker)==int)))
        # -----------------------------------------
        lgi_masked, masker = self.mask_lgi_with_gids(gids, masker)
        # -----------------------------------------
        if masker != -10:
            '''
            Redundant branching !!

            ~~RETAIN~~ as an entry space for further development for needs
            of having different masker values, example using differnet#
            masker values for different phases like particles, voids, etc.
            '''
            __new_mask__ = -10
            lgi_masked[lgi_masked == masker] = __new_mask__
            s_masked = deepcopy(self.s)
            s_masked[lgi_masked != __new_mask__] = masker
        else:
            __new_mask__ = -10
            lgi_masked[lgi_masked == -10] = __new_mask__
            s_masked = deepcopy(self.s)
            s_masked[lgi_masked != -10] = masker
        # -----------------------------------------
        return s_masked, masker

    def plotgs(self, figsize=(6,6)):
        plt.figure(figsize=figsize)
        plt.imshow(self.s)
        plt.title(f"tslice={self.m}")
        plt.xlabel(r"X-axis, $\mu m$", fontsize=12)
        plt.ylabel(r"Y-axis, $\mu m$", fontsize=12)


    def plot_grains_gids(self,
                         gids,
                         gclr='color',
                         title="user grains",
                         cmap_name='CMRmap_r'
                         ):
        """


        Parameters
        ----------
        gids : int/list
            Either a single grain index number or list of them
        title : TYPE, optional
            DESCRIPTION. The default is "user grains".
        gclr :

        Returns
        -------
        None.

        Example-1
        ---------
            After acquiring gids for aspect_ratio between ranks 80 and 100,
            we will visualize those grains.
            . . . . . . . . . . . . . . . . . . . . . . . . . .
            As we are only interested in gid, we will not use the other
            two values returned by PXGS.gs[n].get_gid_prop_range() method:

            gid, _, __ = PXGS.gs[8].get_gid_prop_range(PROP_NAME='aspect_ratio',
                                                       range_type='rank',
                                                       rank_range=[80, 100]
                                                       )
            . . . . . . . . . . . . . . . . . . . . . . . . . .
            Now, pass gid as input for the PXGS.gs[n].plot_grains_gids(),
            which will then plot the grain strucure with only these values:

            PXGS.gs[8].plot_grains_gids(gid, cmap_name='CMRmap_r')
        """
        if gclr not in ('binary', 'grayscale'):
            s, _ = self.mask_s_with_gids(gids)
            plt.imshow(s, cmap=cmap_name, vmin=1)
            plt.colorbar()
        elif gclr in ('binary', 'grayscale'):
            s, _ = self.mask_s_with_gids(gids,
                                         masker=0,
                                         force_masker=True)
            s[s != 0] = 1
            plt.imshow(s, cmap='gray_r', vmin=0, vmax=1)
        plt.title(title)
        plt.xlabel(r"X-axis, $\mu m$", fontsize=12)
        plt.ylabel(r"Y-axis, $\mu m$", fontsize=12)
        plt.show()

    def plot_grains_prop_range(self,
                               PROP_NAME='area',
                               range_type='percentage',
                               value_range=[1, 2],
                               percentage_range=[0, 20],
                               rank_range=[60, 90],
                               pivot=None,
                               gclr='color',
                               title=None,
                               cmap_name='CMRmap_r'
                               ):
        """
        Method to plot grains having properties within the domain defined by
        the range description specified by the user.

        Parameters
        ----------
        PROP_NAME : str, optional
            Name of the grain structure property. The default is 'area'.
        range_type : str, optional
            Range descript9ion type. The default is 'percentage'.
        value_range : iterable, optional
            Range of the actual PROP_NAME values. The default is [1, 2].
        percentage_range : iterable, optional
            Percentage range defining the PROP_NAME values. The default is
            [0, 20].
        rank_range : iterable, optional
            Ranks defining the range of PROP_NAME values.
            If rank_range=[6, 10] and there are 20 grains, then
            those grains having 12th to 20th largest PROP_NAME values will
            be selected. The default is [60, 90].
        pivot : str, optional
            Describes the range location.
            Options: ('ends', 'mean', 'primary_mode'):
                - If 'ends' and percentage_range=[5, 8], then this means that
                PROP_NAME vaklues between 5% and 8% of vaklues will be used to
                select the grains.
                - If 'mean' and percentage_range=[5, 8], then this means that
                PROP_NAME values between 0.95*mean and 1.08*mean will be used
                to select the grains.
                - If 'primary_mode' and percentage_range=[5, 8], then this
                means that PROP_NAME values between 0.95*primary_mode and
                1.08*primary_mode will be used to select the grains.
            The default is None.
        gclr : str, optional
            Specify whether grains are to have colours or grayscale.
            Choose 'binary' or 'grayscale' for grayscale
            The default is 'color'.
        title : str, optional
            DESCRIPTION.
            The default is None.
        cmap_name : str, optional
            DESCRIPTION.
            The default is 'CMRmap_r'.

        Returns
        -------
        None.

        """
        if range_type in ('percentage', 'value', 'rank'):
            gid, value, _ = self.get_gid_prop_range(PROP_NAME=PROP_NAME,
                                                    range_type=range_type,
                                                    rank_range=rank_range
                                                    )
            _rdesc_ = {'percentage': percentage_range,
                       'value': value_range,
                       'rank': rank_range
                       }
            title = f"Grains by area. \n {range_type} bounds: {_rdesc_[range_type]}"
            self.plot_grains_gids(gid,
                                  gclr='color',
                                  title=title,
                                  cmap_name=cmap_name
                                  )
        else:
            print(f"Invalid range_type: {range_type}")
            print("range_type must be either of the follwonig:")
            print(".......(percentage, value, rank)")

    def plot_large_grains(self, extent=5):
        gids, _, _ = self.get_gid_prop_range(PROP_NAME='area',
                                             range_type='percentage',
                                             percentage_range=[100-extent,
                                                               100],
                                             )
        for gid in gids:
            plt.imshow(self.g[gid]['grain'].bbox_ex)
        plt.imshow

    def plot_neigh_grains(self,
                          gids=[None],
                          throw=True,
                          gclr="color",
                          title="Neigh grains",
                          cmap_name="CMRmap_r"
                          ):
        neighbours = [self.g[gid]["grain"].neigh for gid in gids]
        _neighbours_ = []
        for neighs in neighbours:
            for gid in neighs:
                _neighbours_.append(gid)
        self.plot_grains_gids(gids=_neighbours_,
                              gclr=gclr,
                              title=title+f" of \n grains: {gids}",
                              cmap_name=cmap_name
                              )
        if throw:
            return neighbours

    def plot_grains_with_holes(self):
        # Use Euler number here
        pass

    def plot_skeletons(self):
        # Use sciki-image skeletenoise command here
        pass

    def plot(self,
             PROP_NAME=None,
             title='auto',
             cmap='CMRmap_r',
             vmin = 1,
             vmax = 5,
             ):
        '''
        if no kwargs: plot the entire greain structure: just use plotgs()

        '''
        if not PROP_NAME:
            plt.imshow(self.s, cmap=cmap)
        elif PROP_NAME in ('npixels', 'area', 'aspect_ratio',
                           'perimeter', 'eq_diameter', 'solidity',
                           'eccentricity', 'compactness', 'circularity',
                           'major_axis_length', 'minor_axis_length'
                           ):
            PROP_LGI = deepcopy(self.lgi)
            for gid in self.gid:
                PROP_LGI[PROP_LGI==gid]=self.prop[PROP_NAME][gid-1]
            plt.imshow(PROP_LGI, cmap=cmap)
        elif PROP_NAME in ('phi1', 'psi', 'phi2'):
            pass
        elif PROP_NAME in ('gnd_avg'):
            pass
        if title == 'auto':
            title = f"Grain structure by {PROP_NAME}"
        plt.title(f"{title}")
        plt.xlabel("x-axis, $\mu m$")
        plt.ylabel("y-axis, $\mu m$")
        if PROP_NAME and PROP_NAME in ('aspect_ratio'):
            plt.colorbar(extend='both')
        else:
            plt.colorbar()
        plt.show()

    def plot_grain(self,
                   gid,
                   neigh=False,
                   neigh_hops=1,
                   save_png=False,
                   filename='auto',
                   field_variable=None,
                   throw=False
                   ):
        """
        Plots the nth grain.

        Parameters
        ----------
        Ng : int
            The grain number to plot. Grain number is global and not state
            specific.
        neigh : bool
            Flag to decide plotting of grains neighbouring to Ng
        neigh_hops : 1
            Non-locality of neighbours.
            If 1, only neighbours of Ng will be plotted along with Ng grain
            If 2, neighbours of neighbours of Ng will be plotted along with
            Ng grain
            NOTE: maximum number of hops permitted = 2
                  If a number greater than 2 is provided, then hops will be
                  restricted to 2.
        save_png : bool
            Flag to consider saving .png image to disk
        filename : str
            Use this filename for the .png imaage.
            If 'auto', then filename will be generated containing:
                * Grain structure temporal slice number
                * Global grain number
            If None or an invalid, image will not be saved to disk.
        field_variable : str
            Global field variable
            This is @ future development when SDVs can be re-mapped from
            CPFE simulation to UPXO.mcgs2d

        Returns
        -------
        grain_plot : bool
            matplotlib.plt.imshow object

        Example call
        ------------
            PXGS.gs[4].plot_grain(3, filename='t4_ng3.png'

        # TODO
            1. Add validity checking layer for gid
            2. Add validity check for save_png and filename
            2. Generate automatic filename
            3. Save image to file
            4. Add branching for dimensionality
            5. Add validity check for existence of data
        """
        operation_validity = False
        if self.g[gid]['grain']:
            if hasattr(self.g[gid]['grain'], 'bbox_ex'):
                if not neigh:
                    if not field_variable:
                        grain_plot = plt.imshow(self.g[gid]['grain'].bbox_ex)
                        operation_validity = True
                    else:
                        # 1. check field variable validity
                        # 2. check if the field variable data is available
                        # 3. Extract field data map relavant to current grain
                        #    only. No need to extract from remaining portions
                        #    of bbox_ex, whcih would be containing neighbouring
                        #    grains
                        # 4. PLot the data
                        pass
                else:
                    if hasattr(self.g[gid]['grain'], 'neigh'):
                        if len(self.g[gid]['grain'].neigh) > 0:
                            grain_plot = plt.imshow(self.g[gid]['grain'].bbox_ex)
                if save_png and type(filename) == str:
                    if filename == 'auto':
                        # Generate automatic filename
                        pass
                    else:
                        # Use the user input name for storing the filename.
                        pass
                    #  Save the image file
                elif save_png and type(filename) != str:
                    print("Invalid filename to store image")
                    pass

        if operation_validity and throw:
            return grain_plot


    def plot_grains_prop_bounds_s(self,
                                  s,
                                  PROP_NAME=None,
                                  prop_min=0,
                                  prop_max='',
                                  ):
        pass

    def plot_grains_at_position(self,
                                position='corner',
                                overlay_centroids=True,
                                markersize=6,
                                ):
        """
        Example-1
        PXGS.gs[tslice].plot_grains_at_position(position='boundary')
        """
        LGI = deepcopy(self.lgi)
        boundary_array = self.positions[position]
        pseudos = np.arange(-len(boundary_array), 0)
        for pseudo, ba in zip(pseudos, boundary_array):
            LGI[LGI==ba] = pseudo
        LGI[LGI > 0] = 0
        for i, pseudo in enumerate(pseudos):
            LGI[LGI==pseudo] = boundary_array[i]
        plt.figure()
        plt.imshow(LGI)
        if overlay_centroids:
            for grain in self:
                if grain.gid in boundary_array:
                    x, y = grain.position[0:2]
                    plt.plot(x, y,
                             'ko',
                             markersize=markersize,
                             )
        plt.title(f"Corner grains. Ng: {len(self.positions[position])}")
        plt.xlabel("x-axis, $\mu m$")
        plt.ylabel("y-axis, $\mu m$")
        plt.show()

    def detect_grain_boundaries(self):
        for label in np.unique(self.lgi):
            pass


    def hist(self,
             PROP_NAME=None,
             bins=20,
             kde=True,
             bw_adjust=None,
             stat='density',
             color='blue',
             edgecolor='black',
             alpha=1.0,
             line_kws={'color': 'k',
                        'lw': 2,
                        'ls': '-'
                        },
             auto_xbounds=True,
             auto_ybounds=True,
             xbounds=[0, 50],
             ybounds=[0, 0.2],
             peaks=False,
             height=0,
             prominance=0.2,
             __stack_call__=False,
             __tslice__=None
             ):
        if self.are_properties_available:
            if PROP_NAME in self.prop.columns:
                self.prop[PROP_NAME].replace([-np.inf, np.inf],
                                             np.nan,
                                             inplace=True
                                             )
                sns.histplot(self.prop[PROP_NAME].dropna(),
                             bins=bins,
                             kde=False,
                             stat=stat,
                             color=color,
                             edgecolor=edgecolor,
                             line_kws=line_kws
                             )
                if kde and bw_adjust:
                    if peaks:
                        x, y = (sns.kdeplot(data=self.prop[PROP_NAME].dropna(),
                                            bw_adjust=bw_adjust,
                                            color=line_kws['color'],
                                            linewidth=line_kws['lw'],
                                            fill=False,
                                            alpha=0.5,
                                            ).lines[0].get_data()
                                )
                        peaks, peaks_properties = find_peaks(y,
                                                             height=0,
                                                             prominence=0.02
                                                             )
                        plt.plot(x, y)
                        plt.plot(x[peaks],
                                 peaks_properties["peak_heights"],
                                 "o",
                                 markerfacecolor='black',
                                 markersize=8,
                                 markeredgewidth=1.5,
                                 markeredgecolor='black')
                        plt.vlines(x=x[peaks],
                                   ymin=y[peaks] - peaks_properties["prominences"],
                                   ymax=y[peaks],
                                   color="gray",
                                   linewidth=1,
                                   )
                        # Find the minima and plot it
                        minima_indices = argrelextrema(y, np.less)[0]
                        plt.plot(x[minima_indices],
                                 y[minima_indices],
                                 "s",
                                 markerfacecolor='white',
                                 markersize=8,
                                 markeredgewidth=1.5,
                                 markeredgecolor='black')
                    else:
                        sns.kdeplot(self.prop[PROP_NAME].dropna(),
                                    bw_adjust=bw_adjust,
                                    label='KDE',
                                    color=line_kws['color'],
                                    linewidth=line_kws['lw'],
                                    fill=False,
                                    alpha=0.5,
                                    )
                if kde and not bw_adjust:
                    if peaks:
                        x, y = (sns.kdeplot(data=self.prop[PROP_NAME].dropna(),
                                            color=line_kws['color'],
                                            linewidth=line_kws['lw'],
                                            fill=False,
                                            alpha=0.5,
                                            ).lines[0].get_data()
                                )
                        peaks, peaks_properties = find_peaks(y,
                                                             height=0,
                                                             prominence=0.02
                                                             )
                        plt.plot(x, y)
                        plt.plot(x[peaks],
                                 peaks_properties["peak_heights"],
                                 "o",
                                 markerfacecolor='black',
                                 markersize=8,
                                 markeredgewidth=1.5,
                                 markeredgecolor='black')
                        plt.vlines(x=x[peaks],
                                   ymin=y[peaks] - peaks_properties["prominences"],
                                   ymax=y[peaks],
                                   color="gray",
                                   linewidth=1,
                                   )
                        # Find the minima and plot it
                        minima_indices = argrelextrema(y, np.less)[0]
                        plt.plot(x[minima_indices],
                                 y[minima_indices],
                                 "s",
                                 markerfacecolor='white',
                                 markersize=8,
                                 markeredgewidth=1.5,
                                 markeredgecolor='black')
                if __stack_call__:
                    plt.title(f"Distribution of {PROP_NAME} @ tslice: {__tslice__}")
                else:
                    plt.title(f"Distribution of {PROP_NAME}")
                plt.xlabel(f'{PROP_NAME}')
                plt.ylabel(f'{stat}')
                if auto_xbounds == 'user':
                    plt.xlim(xbounds)
                if auto_ybounds == 'user':
                    plt.ylim(ybounds)
                plt.show()
            else:
                if not __stack_call__:
                    print(f"PROP_NAME: {PROP_NAME} has not yet been caluclated. Skipped")
        else:
            print(f"PROP_NAME: {PROP_NAME} has not yet been caluclated. Skipped")

    def kde(self,
            PROP_NAMES,
            bw_adjust,
            ):
        print(PROP_NAMES)
        for PROP_NAME in PROP_NAMES:
            if PROP_NAME in self.prop.columns:
                self.prop[PROP_NAME].replace([-np.inf, np.inf],
                                             np.nan,
                                             inplace=True
                                             )
                sns.kdeplot(self.prop[PROP_NAME].dropna(),
                            bw_adjust=bw_adjust,
                            label='KDE',
                            color='red', attrs=['bold'])
                plt.title(f"{PROP_NAME} distribution")
                plt.xlabel(f"{PROP_NAME}")
                plt.ylabel("Density")
                plt.legend()
            if PROP_NAME == PROP_NAMES[-1]:
                plt.show()

    def plot_histograms(self,
                        props=['area',
                               'perimeter',
                               'orientation',
                               'solidity',
                               ],
                        ncolumns=3):
        if self.prop:
            properties = []
            # Establish the validity of this property text
            for prop in props:
                if prop in dth.valid_region_properties.scikitimage_region_properties2d:
                    properties.append(prop)

            num_of_subplots = len(properties)
            nrows = num_of_subplots // ncolumns
            if num_of_subplots % ncolumns != 0:
                nrows += 1
            positions = list(range(1, num_of_subplots+1))

            fig = plt.figure(1)
            for prop, position in zip(properties, positions):
                # Establish the validity of this property text
                ax = fig.add_subplot(nrows, ncolumns, position)
                plt.hist([rp[prop] for rp in self.prop])
                plt.xlabel(f'{prop}')
                plt.ylabel(f'count')
                plt.axis('on')
            plt.show()

    def femesh(self,
               saa=True,
               throw=False,
               ):
        '''
        Set up finite element mesh of the poly-xtal
        Use saa=True to update grain structure mesh atttribute
        Use saa=True and throw=True to update and return mesh
        Use saa=False and throw=True to only return mesh
        '''
        from mcgs import _uidata_mcgs_gridding_definitions_
        uigrid = _uidata_mcgs_gridding_definitions_(self.__ui)
        from mcgs import _uidata_mcgs_mesh_
        uimesh = _uidata_mcgs_mesh_(self.__ui)

        from mcgs import mesh
        if saa:
            self.mesh = mesh(uimesh, uigrid, self.dim, self.m, self.lgi)
            if throw:
                return self.mesh
        if not saa:
            if throw:
                return mesh(uimesh, uigrid, self.dim, self.m, self.lgi)
            else:
                return 'Please enter valid saa and throw arguments'
    # --------------------------------------------------------------------
    @property
    def pxtal_length(self):
        return self.uigrid.xmax-self.uigrid.xmin+self.uigrid.xinc

    @property
    def pxtal_height(self):
        return self.uigrid.ymax-self.uigrid.ymin+self.uigrid.yinc

    @property
    def pxtal_area(self):
        return self.pxtal_length*self.pxtal_height
    # --------------------------------------------------------------------

    @property
    def centroids(self):
        return [grain.centroid for grain in self]

    # --------------------------------------------------------------------
    @property
    def bboxes(self):
        return [grain.bbox for grain in self]

    @property
    def bboxes_bounds(self):
        return [grain.bbox_bounds for grain in self]

    @property
    def bboxes_ex(self):
        return [grain.bbox_ex for grain in self]

    @property
    def bboxes_ex_bounds(self):
        return [grain.bbox_ex_bounds for grain in self]

    # --------------------------------------------------------------------
    @property
    def areas(self):
        return np.array([self.px_size*grain.loc.shape[0] for grain in self])

    @property
    def areas_min(self):
        return self.areas.min()

    @property
    def areas_mean(self):
        return self.areas.mean()

    @property
    def areas_std(self):
        return self.areas.std()

    @property
    def areas_var(self):
        return self.areas.var()

    @property
    def areas_max(self):
        return self.areas.max()

    @property
    def areas_stat(self):
        areas = self.areas
        return {'min': areas.min(),
                'mean': areas.mean(),
                'max': areas.max(),
                'std': areas.std(),
                'var': areas.var()
                }

    # --------------------------------------------------------------------
    @property
    def aspect_ratios(self):
        gid_stright_grains = self.straight_line_grains
        mj_axis = [grain.skprop.axis_major_length for grain in self]
        mn_axis = [grain.skprop.axis_minor_length for grain in self]
        npixels = [len(grain.loc) for grain in self]
        ar = []
        for i, (npx, mja, mna) in enumerate(zip(npixels, mj_axis, mn_axis)):
            if i+1 not in gid_stright_grains:
                ar.append(mja/mna)
            else:
                if npx == 1:
                    ar.append(1)
                else:
                    ar.append(len(self.g[i+1]['grain'].loc))
        return ar

    @property
    def aspect_ratios_min(self):
        return self.aspect_ratios.min()

    @property
    def aspect_ratios_mean(self):
        return self.aspect_ratios.mean()

    @property
    def aspect_ratios_std(self):
        return self.aspect_ratios.std()

    @property
    def aspect_ratios_var(self):
        return self.aspect_ratios.var()

    @property
    def aspect_ratios_max(self):
        return self.aspect_ratios.max()

    @property
    def aspect_ratios_stat(self):
        aspect_ratios = self.aspect_ratios
        return {'min': aspect_ratios.min(),
                'mean': aspect_ratios.mean(),
                'max': aspect_ratios.max(),
                'std': aspect_ratios.std(),
                'var': aspect_ratios.var()
                }

    @property
    def npixels(self):
        npx = np.array([len(grain.loc) for grain in self])
        return npx

    @property
    def single_pixel_grains(self):
        return np.where(self.npixels == 1)[0]+1

    @property
    def plot_single_pixel_grains(self):
        self.plot_grains_gids(self.single_pixel_grains)

    @property
    def straight_line_grains(self):
        # get the axis lengths of all availabel grains
        mja = [grain.skprop.axis_major_length for grain in self]
        mna = np.array([grain.skprop.axis_minor_length for grain in self])
        # retrieve the grains where minor axis is zero. These are the grains
        # where skimage is unable to fit ellipse, as they are unit pixel wide.
        # some of them could be for single pixel grains too.
        gid_mna0 = list(np.where(mna == 0)[0]+1)
        # Now, retrieve the single pixel grains.
        gid_npx1 = self.single_pixel_grains
        # Remove the single pixel grains
        if len(gid_npx1) > 0:
            # This means single pixel grains exist
            for _gid_npx1_ in gid_npx1:
                gid_mna0.remove(_gid_npx1_)
            gid_ar = np.array([len(self.g[_gid_mna0_]['grain'].loc)
                              for _gid_mna0_ in gid_mna0])
        return np.array(gid_mna0, dtype=int), gid_ar

    @property
    def locations(self):
        return [grain.position for grain in self]
    # --------------------------------------------------------------------
    @property
    def perimeters(self):
        characteristic_length = math.sqrt(self.px_size)
        return np.array([characteristic_length*grain.gbloc.shape[0]
                         for grain in self])

    @property
    def perimeters_min(self):
        return self.perimeters.min()

    @property
    def perimeters_mean(self):
        return self.perimeters.mean()

    @property
    def perimeters_std(self):
        return self.perimeters.std()

    @property
    def perimeters_var(self):
        return self.perimeters.var()

    @property
    def perimeters_stat(self):
        perimeters = self.perimeters
        return {'min': perimeters.min(),
                'mean': perimeters.mean(),
                'max': perimeters.max(),
                'std': perimeters.std(),
                'var': perimeters.var()
                }

    # --------------------------------------------------------------------------
    @property
    def ratio_p_a(self):
        return np.array([p/a for p, a in zip(self.perimeters, self.areas)])

    @property
    def AF_bgrains_igrains(self):
        areas = self.areas
        A_bgr = [areas[gid-1]
                 for gid in np.unique(self.positions['boundary'])]
        A_igr = [areas[gid-1]
                 for gid in np.unique(self.positions['internal'])]
        pxtal_area = self.pxtal_area
        AF = (np.array(A_bgr).sum()/pxtal_area,
              np.array(A_igr).sum()/pxtal_area)
        return AF

    @property
    def grains(self):
        return (_ for _ in self)

    # --------------------------------------------------------------------------
    def make_mulpoint2d_grain_centroids(self):
        self.mp['gc'] = mulpoint2d(method='xy_pair_list',
                                   coordxy=self.centroids
                                   )

    def plot_mcgs_mpcentroids(self):
        plt.figure()
        # Plot the grain structure
        plt.imshow(self.s)
        # Plot the grain mulpoints of the grain centroids
        plt.plot(self.mp['gc'].locx,
                 self.mp['gc'].locy,
                 'ko',
                 markersize=6)
        plt.xlabel('x-axis $\mu m$', fontdict={'fontsize':12} )
        plt.ylabel('y-axis $\mu m$', fontdict={'fontsize':12} )
        plt.title(f"MCGS tslice:{self.m}.\nUPXO.mulpoint2d of grain centroids", fontdict={'fontsize':12})
        plt.show()

    def vtgs2d(self, visualize=True):
        from polyxtal import polyxtal2d as polyxtal
        import upxo.pxtal.polyxtal as polyxtal
        self.make_mulpoint2d_grain_centroids()
        self.vtgs = polyxtal(gsgen_method = 'vt',
                             vt_base_tool = 'shapely',
                             point_method = 'mulpoints',
                             mulpoint_object = self.mp['gc'],
                             xbound = [self.uigrid.xmin,
                                       self.uigrid.xmax+self.uigrid.xinc],
                             ybound = [self.uigrid.ymin,
                                       self.uigrid.ymax+self.uigrid.yinc],
                             vis_vtgs = True
                             )
        if visualize:
            self.vtgs.plot(dpi = 100,
                           default_par_faces = {'clr': 'teal', 'alpha': 1.0, },
                           default_par_lines = {'width': 1.5, 'clr': 'black', },
                           xtal_marker_vertex = True,
                           xtal_marker_centroid = True
                           )

    def ebsd_write_ctf(self,
                       folder='upxo_ctf',
                       file='ctf.ctf'):

        x = np.arange(0, 100.1, 2.5)
        y = np.arange(0, 100.1, 2.5)
        X, Y = np.meshgrid(x, y)

        PHI1 = np.random.uniform(low=0, high=360, size=X.shape)
        PSI = np.random.uniform(low=0, high=360, size=X.shape)
        PHI2 = np.random.uniform(low=0, high=180, size=X.shape)

        os.makedirs(folder, exist_ok=True)
        file = file
        file_path = os.path.join(folder, file)

        with open(file_path, 'w') as f:
            f.write("Channel Text File\n")
            f.write("Prj	C:\CHANNEL5_olddata\Joe's Creeping Crud\Joes creeping crud on Cu\Cugrid_after 2nd_15kv_2kx_2.cpr\n")
            f.write("Author	[Unknown]\n")
            f.write("JobMode	Grid\n")
            f.write("XCells	550\n")
            f.write("YCells	400\n")
            f.write("XStep	0.1\n")
            f.write("YStep	0.1\n")
            f.write("AcqE1	0\n")
            f.write("AcqE2	0\n")
            f.write("AcqE3	0\n")
            f.write("Euler angles refer to Sample Coordinate system (CS0)!	Mag	2000	Coverage	100	Device	0	KV	15	TiltAngle	70	TiltAxis	0\n")
            f.write("Phases	1\n")
            f.write("3.6144;3.6144;3.6144	90;90;90	Copper	11	225	3803863129_5.0.6.3	-906185425	Ann. Acad. Sci. Fenn., Ser. A6 [AAFPA4], vol. 223A, pages 1-10\n")
            f.write('Phase X Y Euler1 Euler2 Euler3\n')
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    x = X[i, j]
                    y = Y[i, j]
                    phi1 = PHI1[i, j]
                    psi = PSI[i, j]
                    phi2 = PHI2[i, j]
                    f.write(f"1 {x} {y} {phi1} {psi} {phi2}\n")
        f.close()

    def export_vtk3d(self, grid: dict, grid_fields: dict, file_path: str, file_name: str, add_suffix: bool = True) -> None:
            """
            Export data to .vtk format.

            Parameters
            ----------
            grid : dict
                The grid dictionary containing the grid points.
                grid = {"x": xgr, "y": ygr, "z": zgr}
            grid_fields : dict
                The grid fields dictionary containing the grid fields.
                grid_fields = {"state_matrix": state_matrix,
                  "gid_matrix": gid_matrix}
            file_path : str
                The path where the .vtk file will be saved.
            file_name : str
                The name of the .vtk file.
            add_suffix : bool, optional
                If True, the suffix '_upxo' will be added at the end of the file name.
                This is advised to enable distinguishing any .vtk files you may create using
                applications such as Dream3D etc. The default is True.

            Returns
            -------
            None.

            """
            try:
                import pyvista as pv
            except ModuleNotFoundError:
                raise ModuleNotFoundError("pyvista is not installed. Please install it using 'pip install pyvista'.")
                return

            full_file_name = os.path.join(file_path, file_name + ("_upxo.vtk" if add_suffix else ".vtk"))

            try:
                grid = pv.StructuredGrid(grid['x'], grid['y'], grid['z'])
                grid["values"] = grid_fields['state_matrix'].flatten(order="F")  # Flatten in Fortran order to match VTK's indexing
                grid["gid_values"] = grid_fields['gid_matrix'].flatten(order="F")  # Flatten in Fortran order to match VTK's indexing
                grid.save(full_file_name)
            except IOError as e:
                print(f"Error saving VTK file: {e}")


    def export_vtk2d(self):
        pass

    def export_ctf(self,
                   filePath,
                   metaData
                   ):
        """


        Parameters
        ----------
        filePath : str
            Provide the full path to the file. str following the last filesep
            should be filename.ctf
        metaData : dict
            Dictionary of ctf file header information. Example:
            {'projectName':'UPXOProject',
             'author': 'UPXO',
             'nphases': 1,
             'phase1': 'Copper',
             }

        Returns
        -------
        None.

        """

        pass

    def extract_slice_from3d(self,
                             mstep,
                             sliceNormal=[0, 0, 1],
                             sliceLocation=0,
                             outputFormat='grid',
                             metaData={'projectName':'UPXOProject',
                                       'author': 'UPXO',
                                       'nphases': 1,
                                       'phase1': 'Copper',
                                       }
                             ):
        """
        This method helps extract a 2D slice from 3D grain structure database.

        Parameters
        ----------
        mstep : int
            Monte-Carlo time step
        sliceNormal : list/tuple, optional
            Normal vector to the slice plane. The default is [0, 0, 1],
            meaning slicing along plane normal to z-axis.
        sliceLocation : float, optional
            Bounds: [0, 100]. Value is percentage. If the grid size is 20x30x40
            , sliceNormal is [0, 0, 1], then a sliceLocation of 40%
            will create a slice at a location of z=16. The default is 50.
        outputFormat : 'str', optional
            Specify the data format needed. Options are 'grid', 'ctf', 'vtk'
            and 'upxo_gs'. The default is 'grid'.
            * If 'grid', return will be a dictionary with the keys, 'x',
            'y', 'z', 'S' and 'lgi', with corresponding values.
            * If 'ctf', return will be a dictionary having keys 'folderPath' and
            'filename', indicating the written .ctf file.
            * If 'upxo_gs', a upxo grain structure database will be created,
            grains will be identified afresh.
        metaData : TYPE, optional
            DESCRIPTION. The default is {'projectName':'UPXOProject',
                                         'author': 'UPXO',
                                         'nphases': 1,
                                         'phase1': 'Copper',
                                         }.
        Returns
        -------
        slice_2d : TYPE
            DESCRIPTION.
        """
        nonxyz = 1
        if sliceNormal[0]==1 and sliceNormal[1]==0 and sliceNormal[2]==0:
            # SLice normal is x
            nonxyz = 0
            pass
        elif sliceNormal[0]==0 and sliceNormal[1]==1 and sliceNormal[2]==0:
            # SLice normal is y
            nonxyz = 0
            pass
        elif sliceNormal[0]==0 and sliceNormal[1]==0 and sliceNormal[2]==1:
            # SLice normal is z
            nonxyz = 0
            pass
        else:
            nonxyz = 1
            # Use PyVista here
            # Step 1: Validate sliceNormal
            # Step 2: Use the PyVista model of 3D GS
            # Step 3: Extract the slice as (x, y, z, S, gid)
            pass
        #------------------------------
        if outputFormat=='grid':
            # Convert slice_2d to grid format and return
            pass
        elif outputFormat=='ctf':
            # Convert slice_2d to ctf format and return
            pass
        return slice_2d

    def export_slices(self,
                      xboundPer,
                      yboundPer,
                      zboundPer,
                      mlist,
                      sliceStepSize,
                      sliceNormal,
                      xoriConsideration,
                      resolution_factor,
                      exportDir,
                      fileFormats,
                      overwrite,
                      ):
        """
        Exports datafiles of slices through the grain structures.

        Parameters
        ----------
        xboundPer : list/tuple
            (min%, max%), min% < max%. min% shows the percentage length from
            grid's xmin, where the bound starts and the max% shows the
            percentage xlength from grid's xmin, where the bounds ends
        yboundPer : list/tuple
            (min%, max%), min% < max%. min% shows the percentage length from
            grid's ymin, where the bound starts and the max% shows the
            percentage ylength from grid's ymin, where the bounds ends
        zboundPer : list/tuple
            (min%, max%), min% < max%. min% shows the percentage length from
            grid's zmin, where the bound starts and the max% shows the
            percentage ylength from grid's zmin, where the bounds ends
        mlist : list/tuple of int values
            List of monte-carlo temporal time values, where slices are needed.
            For each entry, a seperate folder will be created.
        sliceStepSize : int
            Pixel-distance (number of pixels) between each individual slice.
            Minimum should be 1, in which case, the every adjqacent possible
            slice will be sliced and exported. If 2, slices 0, 2, 4, ... will
            be considered. If 5, slices, 0, 5, 10, ... will be considered.
        sliceNormal : str
            Options include x, y, z
        xoriConsideration : dict
            Xtal orientation consideration
            Mandatory key: 'method'. Options include:
                * 'ignore'. Only when crystallographical orientations have
                already been mapped to grains.
                * 'random'. Value could be a dummy value.
                * 'userValues'. Value to be a numpy array of 3 Bunge's Euler
                angles, shaped (nori, 3).
                * 'import'.
        resolution_factor : float
        exportDir : str
            Directory path string which would be parent directory for all
            exports made from this PXGS.export_slices(.). If directory does
            not exit, it will be created.
        fileFormats : dict
            Keys include txt, h5d, ctf, vtk.
            * Include txt or h5d to export for for further work in UPXO
            * Include ctf for export to MTEX or Dream3D's h5ebsd reconstruction
            pipeline
            * Include vtk2d for export to VTK format of each slice
            * Include vtk3d for export to VTK of entire grain structure
        overwrite : bool
            If True, any existing contents in all child directories inside
            exportDir will be overwritten
            If False, existing contents will not be altered.

        Returns
        -------
        None.

        Example-1
        ---------
        xboundPer = (0, 100)
        yboundPer = (0, 100)
        zboundPer = (0, 100)
        mlist = [0, 10, 20]
        sliceStepSize = 1
        sliceNormal = 'z'
        xoriConsideration = {'method': 'random'}
        exportDir = 'FULL PATH'
        fileFormats = {'.ctf': {},
                       '.vtk3d': {},
                       }
        overwrite = True
        PXGS.export_slices(xboundPer,
                           yboundPer,
                           zboundPer,
                           mlist,
                           sliceStepSize,
                           sliceNormal,
                           exportDir,
                           fileFormats,
                           overwrite)
        """
        from scipy.ndimage import label, generate_binary_structure
        import math
        xsz = math.floor((self.uigrid.xmax-self.uigrid.xmin)/self.uigrid.xinc);
        ysz = math.floor((self.uigrid.ymax-self.uigrid.ymin)/self.uigrid.yinc);
        zsz = math.floor((self.uigrid.zmax-self.uigrid.zmin)/self.uigrid.zinc);
        Smax = self.uisim.S;
        slices = list(range(0, 9, sliceStepSize))
        phase_name = 1;
        phi1 = np.random.rand(Smax)*180
        psi = np.random.rand(Smax)*90
        phi2 = np.random.rand(Smax)*180
        textureInstanceNumber = 1;

    def import_ctf(self,
                   filePath,
                   fileName,
                   convertUPXOgs=True):
        pass

    def import_crc(self,
                   filePath,
                   fileName,
                   convertUPXOgs=True):
        # Use DefDAP to get the job done here
        pass

    def clean_exp_gs(self,
                     minGrainSize=10
                     ):
        # Use DefDAP to get the job done here
        pass

    def import_dream3d(self,
                       filePath,
                       fileName,
                       convertUPXOgs=True):
        pass

    def import_vtk(self,
                   filePath,
                   fileName,
                   convertUPXOgs=True):
        pass

    def update_dream3d_ABQ_file(self):
        """
        Take Eralp's code Dream3D2Abaqus and update it to also write:
            * element sets (or make them as groups) for:
                . texture partitioned grains
                . grain area binned grains
                . aspect ratio binned grains
                . boundary grains
                . internal grains
                . grain boundary surface elements
                . grain boundary edge elements
                . grain boundary junction point elements
                .
        Returns
        -------
        None.

        """
        pass
