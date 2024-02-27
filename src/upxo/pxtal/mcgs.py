"""
Description
-----------
This is a core module of UPXO making use of Monte-Carlo (MC) simulation of
grain growth.

High level tasks
----------------
    * Generation and analyses of 2D and 3D MC grain structures (GS)
    * Representativeness qualification of 2D MCGS
    * Meshing and export of grain structure to ABAQUS

NOTE
----
Circulation restricted to the following until official release:
    * @UKAEA: Vaasu Anandatheertha, Chris Hardie, Vikram Phalke
    * @UKAEA:  Ben Poole, Allan Harte, Cori Hamelin
    * @OX,UKAEA:  Eralp Demir, Ed Tarleton
-------------------------------------------------------------------
GENERAL INFORMATION

1. Large simulation domain:
    Number of pixels in lattice >= 1E4
    Number of pixels in lattice * Total MC time >= 1E7
"""

# import time
# from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# import numpy.random as rand
# import matplotlib.colors as colors
# import matplotlib.ticker as ticker
# from random import sample as sample_rand
# from .._sup import stop_watch as sw
# import datatype_handlers as dth
# import gops
# import os
# import re
# import math
import numpy as np
from scipy.interpolate import griddata
# import scipy.stats as stats
import cv2
from skimage.measure import label as skim_label
# from point2d import point2d
# from mulpoint2d import mulpoint2d
# import xlrd
from prettytable import PrettyTable
# import pandas as pd
# from termcolor import colored
# import seaborn as sns
# from scipy.signal import argrelextrema
# from scipy.signal import find_peaks
from upxo.interfaces.user_inputs.gather_user_inputs import load_uidata
from upxo._sup import gops
# from ..interfaces.os import package_check as pkgChk
from upxo._sup import dataTypeHandlers as dth
# from ..geoEntities import mulpoint2d
__authors__ = ["Vaasu Anandatheertha"]
__lead_developer__ = ["Vaasu Anandatheertha"]
__emails__ = ["vaasu.anandatheertha@ukaea.uk", ]
__version__ = ["0.2: from.03072023.git.no", "0.3: from 06072023.git.no",
               "0.4: from 10072023.git.no", "0.5: from 19072023.git.no",
               "0.6: from 01082023.git.no", "0.7: from 05082023.git.no",
               "0.8: from 18082023.git.no", "0.9: from 22082023.git.no",
               "1.0: from 28082023.git.no"
               ]
__license__ = "GPL v3"

class grid():
    """
    Description
    -----------
    This is a core UPXO > mcgs class.

    Dependencies
    ------------
    Parent class for:
        mcgs class

    Slots
    -----
        __ui: DICT: User input (ui) dict
        uigrid:  CLASS: ui: gridding parameters
        uisim:  CLASS: ui: simulation par
        uigsc:  CLASS: ui: grain strucure characterisation par
        uiint:  CLASS: ui: intervals
        uigsprop:  CLASS: ui: grain str property calculation par
        uigeorep:  CLASS: ui: geometric representations cacl par
        _mcsteps_:  LIST: stores history of mcsteps
        __g__:  DICT: base dict template for grains
        __gprop__:  DICT: base dict template for grain properties
        __gb__:  DICT: base dict template for grain boundaries
        __gbprop__:  DICT: base dict template for grain boundary properties
        g:  DICT: Grains @latest mcstep
        m: LIST: available temporal slices
        xgr: np.ndarray:
        ygr: np.ndarray:
        zgr: np.ndarray:
        NL_dict: dict: Specifies Non-Locality detasils
        px_length: Iterable: Side lengths of the pixel
        px_size:, Area or volume of the pixel
        S:  np.ndarray: State matrix
        sa: State martix modified enable fast consideration of
            Wrapped Boundary Condition
        vis: Stores instant of awrtwork class
        AIA: np.ndarray: Appended Index Array (@dev)
        AIA0: np.ndarray: Appended Index Array (@dev)
        AIA1: np.ndarray: Appended Index Array (@dev)
        xind: np.ndarray: xindices (3D only)
        yind: np.ndarray: yindices (3D only)
        zind: np.ndarray: zindices (3D only)
        xinda: np.ndarray: appended xindices (3D only)
        yinda: np.ndarray: appended yindices (3D only)
        zinda: np.ndarray: appended zindices (3D only)
        NLM_nd: np.ndarray: Non-Locality matrix
        NLM: np.ndarray: Non-locality matrix
    """
    __slots__ = ('uigrid', 'uisim', 'uigsc', 'uiint', 'study',
                 'uigsprop', 'uimesh', 'uigeomrepr' '_mcsteps_',
                 'uidata_all', 'index',
                 '__ui', '__g__', '__gprop__', '__gb__', '__gbprop__',
                 'gs', 'xgr', 'ygr', 'zgr',
                 'NL_dict', 'px_length', 'px_size',
                 'S', 'sa', 'AIA', 'AIA0', 'AIA1',
                 'xind' 'yind', 'zind', 'xinda', 'yinda', 'zinda',
                 'NLM_nd', 'NLM',
                 'tslices_with_prop', 'vis', 'vizstyles', 'display_messages',
                 '__info_message_display_level__'
                 )

    def __init__(self,
                 study='independent',
                 input_dashboard='input_dashboard.xls',
                 ):
        self.study = study
        self.__info_message_display_level__ = 'detailed'
        if study == 'independent':
            uidata_all = load_uidata(input_dashboard)
            self.uigrid = uidata_all['uigrid']
            self.uisim = uidata_all['uisim']
            self.uigsc = uidata_all['uigsc']
            self.uiint = uidata_all['uiint']
            self.uigsprop = uidata_all['uigsprop']
            self.uigeomrepr = uidata_all['uigeorep']
            self.uimesh = uidata_all['uimesh']
            self.__ui = uidata_all
            self.uidata_all = uidata_all

            self.initiate()
        elif study in ('para_sweep'):
            # Parameters to be manually set
            pass

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.gs.keys()):
            _m_grain_str_pair_ = list(self.gs.values())[self.index]
            self.index += 1
            return _m_grain_str_pair_
        else:
            raise StopIteration

    @property
    def info_message_display_level(self):
        return self.__info_message_display_level__

    @property
    def info_message_display_level_simple(self):
        return self.__info_message_display_level__

    @info_message_display_level_simple.setter
    def info_message_display_level_simple(self):
        self.__info_message_display_level__ = 'simple'

    @property
    def info_message_display_level_detailed(self):
        return self.__info_message_display_level__

    @info_message_display_level_detailed.setter
    def info_message_display_level_detailed(self):
        self.__info_message_display_level__ = 'detailed'

    def __repr__(self):
        self.__info_message_display_level__ = 'detailed'
        if self.__info_message_display_level__ == 'simple':
            if self.uigrid.dim == 2:
                return 'UPXO 2D.MCGS\n'
            elif self.uigrid.dim == 3:
                return 'UPXO 3D.MCGS\n'
        elif self.__info_message_display_level__ == 'detailed':
            if self.uigrid.dim == 2:
                sep1 = 'UPXO 2D.MCGS\n'
            elif self.uigrid.dim == 3:
                sep1 = 'UPXO 3D.MCGS\n'
            GRID = "(A: GRID):: "
            if hasattr(self, 'uigrid'):
                GRID += f"  x:({self.uigrid.xmin},{self.uigrid.xmax},{self.uigrid.xinc}), "
                GRID += f"  y:({self.uigrid.ymin},{self.uigrid.ymax},{self.uigrid.yinc}), "
                GRID += f"  z:({self.uigrid.zmin},{self.uigrid.zmax},{self.uigrid.zinc})\n"
            else:
                GRID += 'Grid parameters not set.\n'
            # --------------------------------------
            SIMPAR = "(B: SIMPAR):: "
            if hasattr(self, 'uisim'):
                SIMPAR += f"  nstates: {self.uisim.S}  mcsteps: {self.uisim.mcsteps}"
                SIMPAR += f"  algorithms: {self.uisim.algo_hops}\n"
            else:
                SIMPAR += 'Simulation parameters not set.\n'
            # --------------------------------------
            MESHPAR = "(C: MESHPAR):: "
            if hasattr(self, 'uimesh'):
                MESHPAR += f"  GB Conformity: {self.uimesh.mesh_gb_conformity}\n"
                MESHPAR += ' '*15+f"Target FE Software: {self.uimesh.mesh_target_fe_software}"
                MESHPAR += f"  Element type: {self.uimesh.mesh_element_type}"
            else:
                MESHPAR += 'Mesh parameters not set yet.\n'
        # --------------------------------------
            return sep1 + GRID + SIMPAR + MESHPAR + '\n' + '-'*60

    def set_uigrid(self,
                   domain_size=None,
                   read_from_file=False,
                   filename=None
                   ):
        from ..interfaces.user_inputs import _manual_uidata_mcgs_gridding_definitions_
        self.uigrid = _manual_uidata_mcgs_gridding_definitions_(domain_size=domain_size,
                                                    read_from_file=read_from_file,
                                                    filename=filename
                                                    )

    def set_uisim(self,
                  n=1,
                  sim_parameters=None,
                  read_from_file=False,
                  filename=None
                  ):
        from ..interfaces.user_inputs import _manual_uidata_mcgs_simpar_ as _muisimpar_
        self.uisim = _muisimpar_(n=n,
                                 sim_parameters=sim_parameters,
                                 read_from_file=read_from_file,
                                 filename=filename)

    def set_uigsc(self,
                  char_grains=True,
                  char_stage='postsim',
                  library='scikit-image',
                  parallel=True,
                  find_gbseg=True,
                  g_area=True,
                  gb_length=True,
                  gb_length_crofton=True,
                  gb_njp_order=True,
                  g_eq_dia=True,
                  g_feq_dia=True,
                  g_solidity=True,
                  g_circularity=True,
                  g_mjaxis=True,
                  g_mnaxis=True,
                  g_morph_ori=True,
                  g_el=True,
                  g_ecc=True,
                  read_from_file=False,
                  filename=None
                  ):
        from ..interfaces.user_inputs import _manual_uidata_mcgs_gsc_par_
        self.uigsc = _manual_uidata_mcgs_gsc_par_(char_grains=char_grains,
                                                  char_stage=char_stage,
                                                  library=library,
                                                  parallel=parallel,
                                                  find_gbseg=find_gbseg,
                                                  g_area=g_area,
                                                  gb_length=gb_length,
                                                  gb_length_crofton=gb_length_crofton,
                                                  gb_njp_order=gb_njp_order,
                                                  g_eq_dia=g_eq_dia,
                                                  g_feq_dia=g_feq_dia,
                                                  g_solidity=g_solidity,
                                                  g_circularity=g_circularity,
                                                  g_mjaxis=g_mjaxis,
                                                  g_mnaxis=g_mnaxis,
                                                  g_morph_ori=g_morph_ori,
                                                  g_el=g_el,
                                                  g_ecc=g_ecc,
                                                  read_from_file=read_from_file,
                                                  filename=filename
                                                  )

    def set_uigeomrepr(self,
                       make_mp_grain_centoids=True,
                       make_mp_grain_points=True,
                       make_ring_grain_boundaries=True,
                       make_xtal_grain=True,
                       make_chull_grain=True,
                       create_gbz=True,
                       gbz_thickness=True,
                       read_from_file=False,
                       filename=None
                       ):
        from ..interfaces.user_inputs import _manual_uidata_mcgs_generate_geom_reprs_ as gr
        self.uigeomrepr = gr(make_mp_grain_centoids=make_mp_grain_centoids,
                             make_mp_grain_points=make_mp_grain_points,
                             make_ring_grain_boundaries=make_ring_grain_boundaries,
                             make_xtal_grain=make_xtal_grain,
                             make_chull_grain=make_chull_grain,
                             create_gbz=create_gbz,
                             gbz_thickness=gbz_thickness,
                             read_from_file=read_from_file,
                             filename=filename
                             )

    def set_uimesh(self,
                   generate_mesh=False,
                   target_fe_software='abaqus',
                   par_treatment='global',
                   mesher='upxo',
                   gb_conformities=('conformal', 'non_conformal'),
                   global_elsizes=(0.5, 1.0),
                   mesh_algos=(4, 6),
                   grain_internal_el_gradient=('constant', 'constant'),
                   grain_internal_el_gradient_par=(('automin', 'automax'),
                                                   ('automin', 'automax'),
                                                   ),
                   target_eltypes=('CSP4', 'CSP8'),
                   elsets=('grains', 'grains'),
                   nsets=('x-', 'x+', 'y-', 'y+', ),
                   optimize=(False, False),
                   opt_par=('min_angle', [45, 60],
                            'jacobian', [0.45, 0.6]),
                   read_from_file=False,
                   filename=None
                   ):
        """
        Please refer to parameter_sweep.set_param_mesh() for documentation.
        """
        from ..interfaces.user_inputs import _manual_uidata_mesh_
        self.uimesh = _manual_uidata_mesh_(generate_mesh=generate_mesh,
                                           target_fe_software=target_fe_software,
                                           par_treatment=par_treatment,
                                           mesher=mesher,
                                           gb_conformities=gb_conformities,
                                           global_elsizes=global_elsizes,
                                           mesh_algos=mesh_algos,
                                           grain_internal_el_gradient=grain_internal_el_gradient,
                                           grain_internal_el_gradient_par=grain_internal_el_gradient_par,
                                           target_eltypes=target_eltypes,
                                           elsets=elsets,
                                           nsets=nsets,
                                           optimize=optimize,
                                           opt_par=opt_par,
                                           read_from_file=read_from_file,
                                           filename=filename
                                           )

    def initiate(self):
        print('I am in')
        self._mcsteps_ = [self.uisim.S]
        # -------------------------------------------
        if self.uigrid.dim == 2:
            self.px_size = self.uigrid.xinc*self.uigrid.yinc
            self.px_length = (self.uigrid.xinc+self.uigrid.yinc)/2
        elif self.uigrid.dim == 3:
            self.px_size = self.uigrid.xinc * self.uigrid.yinc * self.uigrid.xinc
            self.px_length = (self.uigrid.xinc +
                              self.uigrid.yinc +
                              self.uigrid.zinc)/3
        # ----------------------------------------
        # Build original co-ordinate grSid
        self.build_original_coordinate_grid()
        # ----------------------------------------
        # Perform grid transformations
        self.coord_transform()
        # ----------------------------------------
        # Build original orientation state matrices
        self.build_original_state_matrix()
        self.m = list(np.arange(0,
                                self.uisim.mcsteps,
                                self.uiint.mcint_save_at_mcstep_interval,
                                dtype='int')
                      )
        self.tslices = list(np.arange(0,
                            self.uisim.mcsteps,
                            self.uiint.mcint_save_at_mcstep_interval,
                            dtype='int')
                            )
        # ----------------------------------------
        # Build the non-localcity parameter dictionary
        # TODO: This is to be made user input
        self.NL_dict = dict(NLM_b_dict=dict(flag="no",),
                            NLM_d_dict=dict(flag="no",
                                            func="mexpan",
                                            par=(5., 5., 5., 5.,
                                                 0., 0., 0., 0.,
                                                 0., 0., 0., 0.),
                                            normflag="yes",),
                            ARdetails=dict(teevrate=0,
                                           GrainAxis="90"),
                            )
        # ----------------------------------------
        # Calculate Non-Local Martix
        self.build_non_locality_matrix()
        # ----------------------------------------
        # Build appended index array
        self.AppIndArray()
        # ----------------------------------------
        # Square subset matrix
        # ssub = self.SquareSubsetMatrix()
        # ----------------------------------------
        from ..viz.artwork_definitions import artwork
        self.vis = artwork()
        self.vis.q_Col_Mat(self.uisim.S)
        # ----------------------------------------
        self.setup_transition_probability_rules()
        # self.vis.s_partitioned_tranition_probabilities(self.uisim.S,
        #                                                self.uisim.s_boltz_prob)
        # ----------------------------------------
        self.tslices_with_prop = []
        # ----------------------------------------
        self.vizstyles = {'hist_colors_fill': "#4CC9F0",
                          'hist_colors_edge': 'black',
                          'hist_colors_fill_alpha': 0.5,
                          'kde_color': 'crimson',
                          'kde_thickness': 1,
                          'bins': 25,
                          'hist_area_xbounds': [0, 100],
                          'hist_area_ybounds_density': [0, 0.2],
                          'hist_area_ybounds_freq': [0, 50],
                          'hist_area_ybounds_counts': [0, 50],
                          'hist_peri_xbounds': [0, 100],
                          'hist_peri_ybounds_density': [0, 0.2],
                          'hist_peri_ybounds_freq': [0, 50],
                          'hist_peri_ybounds_counts': [0, 50],
                          }
        self.display_messages = True

    def build_original_coordinate_grid(self):
        """
        Original Coordinate Grid: DESCRIPTION
        OCG inputs
            1. Grid_Data: dictionary
        OCG outputs
            1. cogrid : Coordinate grid data, nd np array.
                        plane1 - x, plane2 - y

        Returns
        -------
        None.

        """
        if self.uigrid.dim == 2:
            cogrid = np.meshgrid(np.arange(self.uigrid.xmin,
                                           self.uigrid.xmax+1,
                                           float(self.uigrid.xinc)),
                                 np.arange(self.uigrid.ymin,
                                           self.uigrid.ymax+1,
                                           float(self.uigrid.yinc)),
                                 copy=True, sparse=False, indexing='xy')
            self.xgr, self.ygr, self.zgr = cogrid[0], cogrid[1], 0
        # ----------------------------------------
        if self.uigrid.dim == 3:
            xmin, xmax = self.uigrid.xmin, self.uigrid.xmax
            ymin, ymax = self.uigrid.ymin, self.uigrid.ymax
            zmin, zmax = self.uigrid.zmin, self.uigrid.zmax
            xinc = self.uigrid.xinc
            yinc = self.uigrid.yinc
            zinc = self.uigrid.zinc
            xarr = np.arange(xmin, xmax, xinc)
            yarr = np.arange(ymin, ymax, yinc)
            zarr = np.arange(zmin, zmax, zinc)
            #nx = np.floor_divide(xmax-xmin, xinc)
            #ny = np.floor_divide(ymax-ymin, yinc)
            #nz = np.floor_divide(zmax-zmin, zinc)
            #self.xgr, self.ygr, self.zgr = np.mgrid[xmin:xmax:nx*1j,
            #                                        ymin:ymax:ny*1j,
            #                                        zmin:zmax:nz*1j]
            self.xgr, self.ygr, self.zgr = np.meshgrid(xarr, yarr, zarr)

    def coord_transform(self):
        """
        Summary line.

        OCG_transform: DESCRIPTION
        transformation parameter: in ["none", "affine", "curvilinear",
                                      "rand_pert_x", "rand_pert_y",
                                      "randpert_xy"]

        Returns
        -------
        None.

        """
        if self.uigrid.transformation == 'none':
            pass

    def build_original_state_matrix(self):
        """
        Summary line.

        Original State Matrix: DESCRIPTION
        OSM inputs
            1. S        : No. of orientation states
            2. OCG_Size : Size of the original
                          coordinate grid: a 3 element list
        OSM outputs
            1. S        : orientation state matrix
            2. S_sz0    : dim0 len of S
            3. S_sz1    : dim1 len of S
            4. S_sz2    : dim2 len of S
            5. Svec     : S in single row format. IS THIS STILL NEEDED???


        Returns
        -------
        None.

        """

        if self.uigrid.dim == 2:
            OCG_size = (self.xgr.shape[0],
                        self.xgr.shape[1])
            # @ 2D grain structure
            if self.uisim.mcalg[0] not in ('4', '5'):
                self.S = np.random.randint(1,
                                           self.uisim.S+1,
                                           size=(OCG_size[0],
                                                 OCG_size[1])).astype(int)
            else:
                self.S = np.random.randint(1,
                                           self.uisim.S+1,
                                           size=(OCG_size[0],
                                                 OCG_size[1])).astype(np.float64)
        elif self.uigrid.dim == 3:
            OCG_size = (self.xgr.shape[0],
                        self.xgr.shape[1],
                        self.xgr.shape[2])
            # @ 3D grain structure
            self.S = np.random.randint(1,
                                       self.uisim.S+1,
                                       size=(OCG_size[0],
                                             OCG_size[1],
                                             OCG_size[2])).astype(int)

    def build_non_locality_matrix(self):
        """
        Construct the non-locality matrix used in some Monte-Carlo simulation

        Returns
        -------
        None.

        """

        if self.uigrid.dim == 2:  # 2D GRAIN STRUCTURE
            # Calculate the size of non-local matrix
            NLM_sz0 = 2*self.uisim.NL+1
            NLM_sz1 = 2*self.uisim.NL+1
            # Calculate Non-Local Matrix of Booleans
            if self.NL_dict["NLM_b_dict"]['flag'] in set(['yes', 'y', '1']):
                NLM_bw = self.IntRules()
            else:
                NLM_bw = np.repeat([np.repeat([1.], NLM_sz0, 0)], NLM_sz1, 0)
            # Calculate Non-Local Matrix of Distance measures
            if self.NL_dict["NLM_d_dict"]['flag'] in set(['yes', 'y', '1']):
                NLM_d = self.NLM_dist()
            else:
                NLM_d = np.repeat([np.repeat([1.], NLM_sz0, 0)], NLM_sz1, 0)
            # Calculate the overall Non-Local Matrix
            self.NLM_nd = NLM_bw * NLM_d
            self.NLM = np.concatenate(self.NLM_nd)
        elif self.uigrid.dim == 3:  # 3D GRAIN STRUCTURE
            # Calculate the size of non-local matrix
            NLM_sz0 = 2*self.uisim.NL+1
            NLM_sz1 = 2*self.uisim.NL+1
            NLM_sz2 = 2*self.uisim.NL+1
            # Calculate Non-Local Matrix of Booleans
            if self.NL_dict["NLM_b_dict"]['flag'] in set(['yes', 'y', '1']):
                NLM_bw = self.IntRules()
            else:
                NLM_bw = np.repeat([np.repeat([np.repeat([1.],
                                                         NLM_sz0, 0)],
                                              NLM_sz1, 0)],
                                   NLM_sz2, 0)
            # Calculate Non-Local Matrix of Distance measures
            if self.NL_dict["NLM_d_dict"]['flag'] in set(['yes', 'y', '1']):
                NLM_d = self.NLM_dist()
            else:
                NLM_d = np.repeat([np.repeat([np.repeat([1.],
                                                        NLM_sz0, 0)],
                                             NLM_sz1, 0)],
                                  NLM_sz2, 0)
            # Calculate the overall Non-Local Matrix
            self.NLM_nd = NLM_bw * NLM_d
            self.NLM = np.concatenate(np.concatenate(self.NLM_nd))

    def IntRules(self):
        """
        Input arguments
        [*] ~~Kineticity~~
               Nature of partition evolution in Euclidean space
                   OPTIONS: all lower case
                       ("static", "s")
                       ("kinetic0", "k0") -- Default kinetic

        [*] ~~Dimensionality~~
            Number of fundamental axes of simulation Euclidean space
                OPTIONS: all lower case
                    (1d, 1)
                    (2d, 2)
                    (3d, 3)

        [*] ~~ARteevrate~~
            To describe the strength of partition aspect ratio
                OPTIONS: all float (following are examples)
                    0: Aims for equi-axed grains
                    1: Aims for non unit AR
                    2: Aims for a higher AR than 1
                    n: AR_(n) > AR_(n-1) > AR_0
                NOTE: Max value limited by no. of pxls across min Gr thickness

        [*] ~~NL~~
            Non-Locality parameter


        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        # Calculate the size of non-local matrix
        NLM_sz0, NLM_sz1 = 2*self.uisim.NL+1, 2*self.uisim.NL+1

        ARteevrate = self.NL_dict['ARdetails']['teevrate']
        GrainAxis = self.NL_dict['ARdetails']['GrainAxis']
        ones = np.repeat([np.repeat([1.], NLM_sz0, 0)], NLM_sz1, 0)
        arts = np.repeat([np.repeat([float(ARteevrate)],
                                    NLM_sz0, 0)],
                         NLM_sz1, 0)
        if self.uisim.kineticity in set(["static", "s",
                                         "sta", "kinetic",
                                         "k", "kin"]):
            if self.uisim.kineticity == "static":
                if self.uigrid.dim == 2:
                    if self.uisim.NL == 1:
                        # if ARteevrate == 0:
                        NLM_bw = ones + arts*np.array([[+1., +1., +1.],
                                                       [+1., +1., +1.],
                                                       [+1., +1., +1.]])
                        # elif ARteevrate == 1:
                        _vert_ = ['90', '270', 'V', 'vert', 'vertical']
                        _hor_ = ['0', '180', 'H', 'hor', 'horizontal']
                        if GrainAxis.lower() in set([string.lower()
                                                     for string in _vert_]):
                            NLM_bw = ones + arts*np.array([[+0., +0., +0.],
                                                           [+1., +0., +1.],
                                                           [+0., +0., +0.]])
                        elif GrainAxis.lower() in set([string.lower()
                                                       for string in ['+-45',
                                                                      'x']]):
                            NLM_bw = ones + arts*np.array([[+1., +0., +1.],
                                                           [+0., +0., +0.],
                                                           [+1., +0., +1.]])
                        elif GrainAxis.lower() == "+45":
                            NLM_bw = ones + arts*np.array([[+1., +0., +0.],
                                                           [+0., +1., +0.],
                                                           [+0., +0., +1.]])
                        elif GrainAxis.lower() == "-45":
                            NLM_bw = ones + arts*np.array([[+0., +0., +1.],
                                                           [+0., +0., +0.],
                                                           [+1., +0., +0.]])
                        elif GrainAxis.lower() in set([string.lower()
                                                       for string in _hor_]):
                            NLM_bw = ones + arts*np.array([[+0.0, +1., +0.0],
                                                           [+0.0, +0., +0.0],
                                                           [+0.0, +1., +0.0]])
                    elif self.uisim.NL == 2:
                        NLM_bw = ones+arts*np.array([[+1., +1., +1., +1., +1.],
                                                     [+1., +1., +1., +1., +1.],
                                                     [+1., +1., +1., +1., +1.],
                                                     [+1., +1., +1., +1., +1.],
                                                     [+1., +1., +1., +1., +1.]
                                                     ])
                    else:
                        None
            elif self.uisim.kineticity == "kinetic":
                None
            return NLM_bw
        else:
            return print("Please input Kin")

    def AppIndArray(self):
        """
        Summary line.

        Appended Index Array: DESCRIPTION
        AppIndArray inputs
            1. NL       : Non-Locality
            2. OCG_size : List of 3 values. Each is dim along axes 0, 1 and 2 respectively.      ~~~~TO BE DONE~~~~
        AppIndArray outputs
            1. AIA      : Appended Index Array (Matlab type array element numbers!) TO KEEP FOR THE MOMENT
            2. AIA0     : Appended dim0 Index Array
            3. AIA1     : Appended dim1 Index Array
            4. AIA2     : Appended dim2 Index Array     ~~~~TO BE DONE~~~~

        Returns
        -------
        None.

        """

        if self.uigrid.dim == 2:
            OCG_size = (self.xgr.shape[0],
                        self.xgr.shape[1])
            # Appended Index Array
            self.AIA = np.arange(np.prod(OCG_size)).reshape((OCG_size[0],
                                                             OCG_size[1]))
            dim0 = np.arange(OCG_size[0])
            dim1 = np.arange(OCG_size[1])
            # Appended Index Array along dimensions 1 & 0:
            DIM1, DIM0 = np.meshgrid(dim0, dim1,
                                     copy=True,
                                     sparse=False,
                                     indexing='xy')
            for NLcount in range(0,  self.uisim.NL):
                # Left Edge
                LE = self.AIA[:, [2*NLcount]]
                # ------------------------------------------------------------
                # Right edge
                RE = self.AIA[:, [len(self.AIA[0]) - 1 - 2*NLcount]]
                self.AIA = np.concatenate((self.AIA, LE), axis=1)
                self.AIA = np.concatenate((RE, self.AIA), axis=1)
                # ------------------------------------------------------------
                # Top edge
                TE = self.AIA[[2*NLcount], :]
                # ------------------------------------------------------------
                # Bottom edge
                BE = self.AIA[[len(self.AIA) - 1 - 2*NLcount], :]
                self.AIA = np.concatenate((BE, self.AIA), axis=0)
                self.AIA = np.concatenate((self.AIA, TE), axis=0)
                # ------------------------------------------------------------
                # Left Edge
                LE_dim0 = DIM0[:, [2*NLcount]]
                # ------------------------------------------------------------
                # Right edge
                RE_dim0 = DIM0[:, [len(DIM0[0]) - 1 - 2*NLcount]]
                DIM0 = np.concatenate((DIM0, LE_dim0), axis=1)
                DIM0 = np.concatenate((RE_dim0, DIM0), axis=1)
                # ------------------------------------------------------------
                # Top edge
                TE_dim0 = DIM0[[2*NLcount], :]
                # ------------------------------------------------------------
                # Bottom edge
                BE_dim0 = DIM0[[len(DIM0) - 1 - 2*NLcount], :]
                DIM0 = np.concatenate((BE_dim0, DIM0), axis=0)
                DIM0 = np.concatenate((DIM0, TE_dim0), axis=0)
                # ------------------------------------------------------------
                # Left Edge
                LE_dim1 = DIM1[:, [2*NLcount]]
                # ------------------------------------------------------------
                # Right edge
                RE_dim1 = DIM1[:, [len(DIM1[0]) - 1 - 2*NLcount]]
                DIM1 = np.concatenate((DIM1, LE_dim1), axis=1)
                DIM1 = np.concatenate((RE_dim1, DIM1), axis=1)
                # ------------------------------------------------------------
                # Top edge
                TE_dim1 = DIM1[[2*NLcount], :]
                # ------------------------------------------------------------
                # Bottom edge
                BE_dim1 = DIM1[[len(DIM1) - 1 - 2*NLcount], :]
                DIM1 = np.concatenate((BE_dim1, DIM1), axis=0)
                DIM1 = np.concatenate((DIM1, TE_dim1), axis=0)
                # ------------------------------------------------------------
            self.AIA1 = DIM0.T
            self.AIA0 = DIM1.T
        elif self.uigrid.dim == 3:
            OCG_size = (self.xgr.shape[0],
                        self.xgr.shape[1],
                        self.xgr.shape[2])
            self.sa = np.zeros((OCG_size[0]+2, OCG_size[1]+2, OCG_size[2]+2))
            self.sa[1:-1, 1:-1, 1:-1] = self.S
            # ------------------------------------------------------------
            # FRONT FACE
            self.sa[0][1:-1, 1:-1] = self.S[-1]
            # BACK FACE
            self.sa[-1][1:-1, 1:-1] = self.S[0]
            # TOP FACE
            self.sa[1:-1, 0, 1:-1] = self.S[:, -1, :]
            # BOTTOM FACE
            self.sa[1:-1, -1, 1:-1] = self.S[:, 0, :]
            # LEFT FACE
            self.sa[1:-1, 1:-1, 0] = self.S[:, :, -1]
            # RIGHT FACE
            self.sa[1:-1, 1:-1, -1] = self.S[:, :, 0]
            # ------------------------------------------------------------
            # EDGE @FRONT and TOP
            self.sa[0, 0, 1:-1] = self.S[-1, -1, :]
            # EDGE @FRONT and BOTTOM
            self.sa[0, -1, 1:-1] = self.S[-1, 0, :]
            # EDGE @FRONT and LEFT
            self.sa[0, 1:-1, 0] = self.S[-1, :, -1]
            # EDGE @FRONT and RIGHT
            self.sa[0, 1:-1, -1] = self.S[-1, :, 0]
            # ------------------------------------------------------------
            # EDGE @BACK and TOP
            self.sa[-1, 0, 1:-1] = self.S[0, -1, :]
            # EDGE @BACK and BOTTOM
            self.sa[-1, -1, 1:-1] = self.S[0, 0, :]
            # EDGE @BACK and LEFT
            self.sa[-1, 1:-1, 0] = self.S[0, :, -1]
            # EDGE @BACK and RIGHT
            self.sa[-1, 1:-1, -1] = self.S[0, :, 0]
            # ------------------------------------------------------------
            # EDGE @TOP and LEFT
            self.sa[1:-1, 0, 0] = self.S[:, -1, -1]
            # EDGE @TOP and RIGHT
            self.sa[1:-1, 0, 0] = self.S[:, -1, -1]
            # EDGE @BOTTOM and LEFT
            self.sa[1:-1, -1, 0] = self.S[:, 0, -1]
            # EDGE @BOTTOM and RIGHT
            self.sa[1:-1, -1, -1] = self.S[:, 0, 0]
            # ------------------------------------------------------------
            # VERTEX @FRONT-LEFT-TOP FACES
            self.sa[0, 0, 0] = self.S[-1, -1, -1]
            # VERTEX @FRONT-LEFT-BOTTOM FACES
            self.sa[0, -1, 0] = self.S[-1, 0, -1]
            # VERTEX @FRONT-RIGHT-BOTTOM FACES
            self.sa[0, -1, -1] = self.S[-1, 0, 0]
            # VERTEX @FRONT-RIGHT-TOP FACES
            self.sa[0, 0, -1] = self.S[-1, -1, 0]
            # ------------------------------------------------------------
            # VERTEX @BACK-LEFT-TOP FACES
            self.sa[-1, 0, 0] = self.S[0, -1, -1]
            # VERTEX @BACK-LEFT-BOTTOM FACES
            self.sa[-1, -1, 0] = self.S[0, 0, -1]
            # VERTEX @BACK-RIGHT-BOTTOM FACES
            self.sa[-1, -1, -1] = self.S[0, 0, 0]
            # VERTEX @FRONT-RIGHT-TOP FACES
            self.sa[-1, 0, -1] = self.S[0, -1, 0]
            # ------------------------------------------------------------
            self.xind = np.zeros((OCG_size[0],
                                  OCG_size[1],
                                  OCG_size[2]), dtype=int)
            self.yind = np.zeros((OCG_size[0],
                                  OCG_size[1],
                                  OCG_size[2]), dtype=int)
            self.zind = np.ones((OCG_size[0],
                                 OCG_size[1],
                                 OCG_size[2]), dtype=int)
            # ------------------------------------------------------------
            tempx = np.tile(np.arange(OCG_size[0]), (OCG_size[0], 1))
            for xaxiscount in range(OCG_size[0]):
                self.xind[xaxiscount] = tempx
            tempy = np.tile(np.array([np.arange(OCG_size[1])]).T,
                            (1, OCG_size[1]))
            for yaxiscount in range(OCG_size[1]):
                self.yind[yaxiscount] = tempy
            for zaxiscount in range(OCG_size[2]):
                self.zind[zaxiscount] = float(zaxiscount)*self.zind[zaxiscount]
            # ------------------------------------------------------------
            self.xinda = np.zeros((OCG_size[0] + 2,
                                   OCG_size[1] + 2,
                                   OCG_size[2] + 2), dtype=int)
            self.yinda = np.zeros((OCG_size[0] + 2,
                                   OCG_size[1] + 2,
                                   OCG_size[2] + 2), dtype=int)
            self.zinda = np.zeros((OCG_size[0] + 2,
                                   OCG_size[1] + 2,
                                   OCG_size[2] + 2), dtype=int)
            # ------------------------------------------------------------
            self.xinda[1:-1, 1:-1, 1:-1] = self.xind
            # FRONT FACE
            self.xinda[0][1:-1, 1:-1] = self.xind[-1]
            # BACK FACE
            self.xinda[-1][1:-1, 1:-1] = self.xind[0]
            # TOP FACE
            self.xinda[1:-1, 0, 1:-1] = self.xind[:, -1, :]
            # BOTTOM FACE
            self.xinda[1:-1, -1, 1:-1] = self.xind[:, 0, :]
            # LEFT FACE
            self.xinda[1:-1, 1:-1, 0] = self.xind[:, :, -1]
            # RIGHT FACE
            self.xinda[1:-1, 1:-1, -1] = self.xind[:, :, 0]
            # EDGE @FRONT and TOP
            self.xinda[0, 0, 1:-1] = self.xind[-1, -1, :]
            # EDGE @FRONT and BOTTOM
            self.xinda[0, -1, 1:-1] = self.xind[-1, 0, :]
            # EDGE @FRONT and LEFT
            self.xinda[0, 1:-1, 0] = self.xind[-1, :, -1]
            # EDGE @FRONT and RIGHT
            self.xinda[0, 1:-1, -1] = self.xind[-1, :, 0]
            # EDGE @BACK and TOP
            self.xinda[-1, 0, 1:-1] = self.xind[0, -1, :]
            # EDGE @BACK and BOTTOM
            self.xinda[-1, -1, 1:-1] = self.xind[0, 0, :]
            # EDGE @BACK and LEFT
            self.xinda[-1, 1:-1, 0] = self.xind[0, :, -1]
            # EDGE @BACK and RIGHT
            self.xinda[-1, 1:-1, -1] = self.xind[0, :, 0]
            # EDGE @TOP and LEFT
            self.xinda[1:-1, 0, 0] = self.xind[:, -1, -1]
            # EDGE @TOP and RIGHT
            self.xinda[1:-1, 0, 0] = self.xind[:, -1, -1]
            # EDGE @BOTTOM and LEFT
            self.xinda[1:-1, -1, 0] = self.xind[:, 0, -1]
            # EDGE @BOTTOM and RIGHT
            self.xinda[1:-1, -1, -1] = self.xind[:, 0, 0]
            # VERTEX @FRONT-LEFT-TOPFACES
            self.xinda[0, 0, 0] = self.xind[-1, -1, -1]
            # VERTEX @FRONT-LEFT-BOTFACES
            self.xinda[0, -1, 0] = self.xind[-1, 0, -1]
            # VERTEX @FRONT-RIGHT-BOTFACES
            self.xinda[0, -1, -1] = self.xind[-1, 0, 0]
            # VERTEX @FRONT-RIGHT-TOPFACES
            self.xinda[0, 0, -1] = self.xind[-1, -1, 0]
            # VERTEX @BACK-LEFT-TOP FACES
            self.xinda[-1, 0, 0] = self.xind[0, -1, -1]
            # VERTEX @BACK-LEFT-BOTFACES
            self.xinda[-1, -1, 0] = self.xind[0, 0, -1]
            # VERTEX @BACK-RIGHT-BOTFACES
            self.xinda[-1, -1, -1] = self.xind[0, 0, 0]
            # VERTEX @FRONT-RIGHT-TOPFACES
            self.xinda[-1, 0, -1] = self.xind[0, -1, 0]
            # ------------------------------------------------------------
            self.yinda[1:-1, 1:-1, 1:-1] = self.yind
            # FRONT FACE
            self.yinda[0][1:-1, 1:-1] = self.yind[-1]
            # BACK FACE
            self.yinda[-1][1:-1, 1:-1] = self.yind[0]
            # TOP FACE
            self.yinda[1:-1, 0, 1:-1] = self.yind[:, -1, :]
            # BOTTOM FACE
            self.yinda[1:-1, -1, 1:-1] = self.yind[:, 0, :]
            # LEFT FACE
            self.yinda[1:-1, 1:-1, 0] = self.yind[:, :, -1]
            # RIGHT FACE
            self.yinda[1:-1, 1:-1, -1] = self.yind[:, :, 0]
            # EDGE @FRONT and TOP
            self.yinda[0, 0, 1:-1] = self.yind[-1, -1, :]
            # EDGE @FRONT and BOTTOM
            self.yinda[0, -1, 1:-1] = self.yind[-1, 0, :]
            # EDGE @FRONT and LEFT
            self.yinda[0, 1:-1, 0] = self.yind[-1, :, -1]
            # EDGE @FRONT and RIGHT
            self.yinda[0, 1:-1, -1] = self.yind[-1, :, 0]
            # EDGE @BACK and TOP
            self.yinda[-1, 0, 1:-1] = self.yind[0, -1, :]
            # EDGE @BACK and BOTTOM
            self.yinda[-1, -1, 1:-1] = self.yind[0, 0, :]
            # EDGE @BACK and LEFT
            self.yinda[-1, 1:-1, 0] = self.yind[0, :, -1]
            # EDGE @BACK and RIGHT
            self.yinda[-1, 1:-1, -1] = self.yind[0, :, 0]
            # EDGE @TOP and LEFT
            self.yinda[1:-1, 0, 0] = self.yind[:, -1, -1]
            # EDGE @TOP and RIGHT
            self.yinda[1:-1, 0, 0] = self.yind[:, -1, -1]
            # EDGE @BOTTOM and LEFT
            self.yinda[1:-1, -1, 0] = self.yind[:, 0, -1]
            # EDGE @BOTTOM and RIGHT
            self.yinda[1:-1, -1, -1] = self.yind[:, 0, 0]
            # VERTEX @FRONT-LEFT-TOP FACES
            self.yinda[0, 0, 0] = self.yind[-1, -1, -1]
            # VERTEX @FRONT-LEFT-BOTFACES
            self.yinda[0, -1, 0] = self.yind[-1, 0, -1]
            # VERTEX @FRONT-RIGHT-BOTFACES
            self.yinda[0, -1, -1] = self.yind[-1, 0, 0]
            # VERTEX @FRONT-RIGHT-TOPFACES
            self.yinda[0, 0, -1] = self.yind[-1, -1, 0]
            # VERTEX @BACK-LEFT-TOP FACES
            self.yinda[-1, 0, 0] = self.yind[0, -1, -1]
            # VERTEX @BACK-LEFT-BOTFACES
            self.yinda[-1, -1, 0] = self.yind[0, 0, -1]
            # VERTEX @BACK-RIGHT-BOTFACES
            self.yinda[-1, -1, -1] = self.yind[0, 0, 0]
            # VERTEX @FRONT-RIGHT-TOPFACES
            self.yinda[-1, 0, -1] = self.yind[0, -1, 0]
            # ------------------------------------------------------------
            self.zinda[1:-1, 1:-1, 1:-1] = self.zind
            # FRONT FACE
            self.zinda[0][1:-1, 1:-1] = self.zind[-1]
            # BACK FACE
            self.zinda[-1][1:-1, 1:-1] = self.zind[0]
            # TOP FACE
            self.zinda[1:-1, 0, 1:-1] = self.zind[:, -1, :]
            # BOTTOM FACE
            self.zinda[1:-1, -1, 1:-1] = self.zind[:, 0, :]
            # LEFT FACE
            self.zinda[1:-1, 1:-1, 0] = self.zind[:, :, -1]
            # RIGHT FACE
            self.zinda[1:-1, 1:-1, -1] = self.zind[:, :, 0]
            # EDGE @FRONT and TOP
            self.zinda[0, 0, 1:-1] = self.zind[-1, -1, :]
            # EDGE @FRONT and BOTTOM
            self.zinda[0, -1, 1:-1] = self.zind[-1, 0, :]
            # EDGE @FRONT and LEFT
            self.zinda[0, 1:-1, 0] = self.zind[-1, :, -1]
            # EDGE @FRONT and RIGHT
            self.zinda[0, 1:-1, -1] = self.zind[-1, :, 0]
            # EDGE @BACK and TOP
            self.zinda[-1, 0, 1:-1] = self.zind[0, -1, :]
            # EDGE @BACK and BOTTOM
            self.zinda[-1, -1, 1:-1] = self.zind[0, 0, :]
            # EDGE @BACK and LEFT
            self.zinda[-1, 1:-1, 0] = self.zind[0, :, -1]
            # EDGE @BACK and RIGHT
            self.zinda[-1, 1:-1, -1] = self.zind[0, :, 0]
            # EDGE @TOP and LEFT
            self.zinda[1:-1, 0, 0] = self.zind[:, -1, -1]
            # EDGE @TOP and RIGHT
            self.zinda[1:-1, 0, 0] = self.zind[:, -1, -1]
            # EDGE @BOTTOM and LEFT
            self.zinda[1:-1, -1, 0] = self.zind[:, 0, -1]
            # EDGE @BOTTOM and RIGHT
            self.zinda[1:-1, -1, -1] = self.zind[:, 0, 0]
            # VERTEX @FRONT-LEFT-TOP FACES
            self.zinda[0, 0, 0] = self.zind[-1, -1, -1]
            # VERTEX @FRONT-LEFT-BOTFACES
            self.zinda[0, -1, 0] = self.zind[-1, 0, -1]
            # VERTEX @FRONT-RIGHT-BOTFACES
            self.zinda[0, -1, -1] = self.zind[-1, 0, 0]
            # VERTEX @FRONT-RIGHT-TOPFACES
            self.zinda[0, 0, -1] = self.zind[-1, -1, 0]
            # VERTEX @BACK-LEFT-TOP FACES
            self.zinda[-1, 0, 0] = self.zind[0, -1, -1]
            # VERTEX @BACK-LEFT-BOTFACES
            self.zinda[-1, -1, 0] = self.zind[0, 0, -1]
            # VERTEX @BACK-RIGHT-BOTFACES
            self.zinda[-1, -1, -1] = self.zind[0, 0, 0]
            # VERTEX @FRONT-RIGHT-TOPFACES
            self.zinda[-1, 0, -1] = self.zind[0, -1, 0]

    def SquareSubsetMatrix(self):
        """
        Summary line.

        SquareSubsetMatrix: DESCRIPTION
        SquareSubsetMatrix inputs:
            1. NL: Non-Locality parameter
        SquareSubsetMatrix outputs:
            1. ssub


        Returns
        -------
        ssub : TYPE
            DESCRIPTION.

        """

        ss_sz0 = 2*self.uisim.NL+1  # S matrix Subset SiZe axis 0, row
        ss_sz1 = 2*self.uisim.NL+1  # S matrix Subset SiZe axis 1, col
        # NOTE: For anisotropic sampling, just make the above
        # rectangular by using unqual sizes!! Something for future work!
        # constraint: must be "odd-length" on all dimensions
        ssub = np.zeros((ss_sz0, ss_sz1), dtype=float)
        return ssub

    def add_gs_data_structure_template(self,
                                       m=None,
                                       dim=None,
                                       study='independent'
                                       ):

        from upxo.pxtal.mcgs2_temporal_slice import mcgs2_grain_structure as grain_structure
        if study == 'independent':
            if m == 0:
                self.gs = {m: grain_structure(m=m,
                                              dim=dim,
                                              uidata=self.__ui,
                                              px_size=self.px_size,
                                              S_total=self.uisim.S,
                                              xgr=self.xgr,
                                              ygr=self.ygr,
                                              uigrid=self.uigrid,
                                              )}
            else:
                self.gs[m] = grain_structure(m=m,
                                             dim=dim,
                                             uidata=self.__ui,
                                             px_size=self.px_size,
                                             S_total=self.uisim.S,
                                             xgr=self.xgr,
                                             ygr=self.ygr,
                                             uigrid=self.uigrid
                                             )
        elif study == 'parameter_sweep':
            xgr, ygr, npixels = self.uigrid.grid
            if m == 0:
                self.gs = {m: grain_structure(m=m,
                                              dim=self.uigrid.dim,
                                              uidata=self.__ui,
                                              px_size=self.uigrid.px_size,
                                              S_total=self.uisim.S,
                                              xgr=self.xgr,
                                              ygr=self.ygr,
                                              uigrid=self.uigrid,
                                              )}
            else:
                self.gs[m] = grain_structure(m=m,
                                             dim=self.uigrid.dim,
                                             uidata=self.__ui,
                                             px_size=self.px_size,
                                             S_total=self.uisim.S,
                                             xgr=xgr,
                                             ygr=ygr,
                                             uigrid=self.uigrid
                                             )

    def _setup_grain_properties_dict_(self):
        """
        Store grain properties
            * areas: Areas (pixels) of all grains: s partitioned
            * Centroids of all grains: s partitioned
            * Neighbouring grain IDs (immediate ones)
            * IDs of immediate neighbours and IDs of neighbours of
              neighbouring grains


        Returns
        -------
        None.

        """

        self.__gprop__ = dict(areas=None,
                              centroids=None,
                              neigh_1=[],
                              neigh_2=[[]],
                              )
        self.gprop = {0: self.__gprop__}

    def _setup_grainboundaries_dict_(self):
        """
        Store grain boundaries
            * Grain boundary numbers used as ID
            * Compulsory: list of list of IDs of all grain boundaries
            * State wise partitioning
            * Grain boundary vertices (NOT grain boundary points)

        Returns
        -------
        None.

        """

        self.__gb__ = dict(ids=None,
                           ind=None,
                           spart=None,
                           vert=None,
                           )
        self.gb = {0: self.__gb__}

    def _setup_grainboundaries_properties_dict_(self):
        """
        Store grain boundary propeties
            * Non-pixel form of total length
            * Total length calculated from pixel side lengths
            * Total lengths of all straight lines between grain
              boundary vertices
            * Total lengths of all boundary segments between grain
              boundary vertices
            * IDs of shared grains
            * Grain boundary zone

        Returns
        -------
        None.

        """

        self.__gbprop__ = dict(length_curve=[],
                               length_pixels=[],
                               lengths_straight=[],
                               lengths=[],
                               shared_grains=[],
                               gbz=None,
                               )
        self.gbprop = {0: self.__gbprop__}

    def setup_transition_probability_rules(self):
        """
        Set up transition probability rules and estimate T.P

        Returns
        -------
        None.

        """

        if self.uisim.s_boltz_prob == 'q_unrelated':
            _a_ = np.random.random(size=self.simpar.S)
            kbf = self.uisim.boltzmann_temp_factor_max
            self.uisim.s_boltz_prob = np.exp(-kbf*_a_)
        elif self.uisim.s_boltz_prob == 'q_related':
            _a_ = np.arange(self.uisim.S)
            _a_ = self.uisim.boltzmann_temp_factor_max*_a_/_a_.max()
            _ = np.random.random(size=self.uisim.S)
            self.uisim.s_boltz_prob = np.exp(-_a_*_)

    def detect_grains(self,
                      M=None,
                      ):
        """
        Applies branching to grain identifiers based on user provided value of
        grain_identification_librar (which, could be UPXO (deprecated), opencv,
                                     scikit-image).
        M is the temporal slice number. If not provided, grains will be
        detected in all available temporal slices.

        Parameters
        ----------
        M : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        M_available = list(self.tslices)  # Available time slices
        if not M:
            M = M_available
        if type(M) == int:
            self.detect_grains(M = [M])
        elif type(M) in dth.dt.ITERABLES:
            print('////////////////////////////////')
            if self.uigrid.dim == 2:
                if self.uigsc.grain_identification_library == 'upxo':
                    for m in M:
                        if m in M_available:
                            self.identify_grains_upxo_2d(m)
                        else:
                            print(f'MC temporal slice no {m} invalid. Skipped')
                elif self.uigsc.grain_identification_library == 'opencv':
                    print("Using opencv for grain identification")
                    for m in M:
                        if m in M_available:
                            self.find_grains_opencv_2d(m)
                        else:
                            print(f'MC temporal slice no {m} invalid. Skipped')
                elif self.uigsc.grain_identification_library == 'scikit-image':
                    print("Using scikit-image for grain identification")
                    for m in M:
                        if m in M_available:
                            self.find_grains_scikitimage_2d(m, connectivity=2)
                        else:
                            print(f'MC temporal slice no {m} invalid. Skipped')
            elif self.uigrid.dim == 3:
                for m in M:
                    if m in M_available:
                        self.find_grains_scilab_ndimage_3d(m)
                    else:
                        print(f'MC temporal slice no {m} invalid. Skipped')
        else:
            print('Please enter valid M as list/tuple')

    def identify_grains_upxo_2d(self):
        """
        Until further development, this method is to remain deprecated.

        Returns
        -------
        None.

        """

        # NOT OPERATIONAL WITH NEW MODIFICATIONS
        Nx = self.S.shape[0]
        Ny = self.S.shape[1]
        grains = []
        visited = np.zeros_like(self.S, dtype=bool)
        while np.sum(visited) < Nx*Ny:
            # Get the indices of the first unvisited cell
            i, j = np.unravel_index(np.argmax(~visited), (Nx, Ny))
            # Flood-fill to find the grain
            grain_label, grain, stack = self.S[i, j], [], [(i, j)]
            while stack:
                x, y = stack.pop()
                if x < 0 or y < 0 or x >= Nx or y >= Ny or visited[x, y] or self.S[x, y] != grain_label:
                    continue
                visited[x, y] = True
                grain.append((x, y))
                stack.extend([(x-1, y), (x+1, y), (x, y-1), (x, y+1)])
            grains.append(grain)
        self.g['gid'] = grains

    def find_grains_opencv_2d(self,
                              m,
                              ):
        """
        Detect grains using the library 'opencv'.
        m is a single temporal slice number.
        Note:
            This method is for internal call. It is recommended that the user
            uses the method 'detect_grains(M)' instead.

        Parameters
        ----------
        m : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        _S_ = self.gs[m].s
        for i, _s_ in enumerate(np.unique(_S_)):
            # Mark the presence of this state
            self.gs[m].spart_flag[_s_] = True
            # Recognize the grains belonging to this state
            image = (_S_ == _s_).astype(np.uint8) * 255
            _, labels = cv2.connectedComponents(image)
            if i == 0:
                self.gs[m].lgi = labels
            else:
                labels[labels > 0] += self.gs[m].lgi.max()
                self.gs[m].lgi = self.gs[m].lgi + labels
            self.gs[m].s_gid[_s_] = tuple(np.delete(np.unique(labels), 0))
            self.gs[m].s_n[_s_-1] = len(self.gs[m].s_gid[_s_])
        # Get the total number of grains
        self.gs[m].n = np.unique(self.gs[m].lgi).size
        # Generate and store the gid-s mapping
        self.gs[m].gid = list(range(1, self.gs[m].n+1))
        _gid_s_ = []
        for _gs_, _gid_ in zip(self.gs[m].s_gid.keys(),
                               self.gs[m].s_gid.values()):
            if _gid_:
                for __gid__ in _gid_:
                    _gid_s_.append(_gs_)
            else:
                _gid_s_.append(0)
        self.gs[m].gid_s = _gid_s_
        # Make the output string to print on promnt
        optput_string_01 = f'Temporal slice number = {m}.'
        optput_string_02 = f' |||| No. of grains detected = {self.gs[m].n}'
        print(optput_string_01 + optput_string_02)

    def find_grains_scikitimage_2d(self,
                                   m,
                                   connectivity=2,
                                   ):
        """
        Detect grains using the library 'scikit-image'.

        Parameters
        ----------
        m : TYPE
            a single temporal slice number.
        connectivity : TYPE, optional
            recommended value = 2. The default is 2.
            Note:
                This method is for internal call. It is recommended that the user
                uses the method 'detect_grains(M)' instead.
            1-connectivity     2-connectivity     diagonal connection close-up
                 [ ]           [ ]  [ ]  [ ]             [ ]
                  |               \  |  /                 |  <- hop 2
            [ ]--[x]--[ ]      [ ]--[x]--[ ]        [x]--[ ]
                  |               /  |  \             hop 1
                 [ ]           [ ]  [ ]  [ ]
            REF: https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.label

        Returns
        -------
        None.

        """
        _S_ = self.gs[m].s
        for i, _s_ in enumerate(np.unique(_S_)):
            # Mark the presence of this state
            self.gs[m].spart_flag[_s_] = True
            # -----------------------------------------------
            # Recognize the grains belonging to this state
            bin_img = (_S_ == _s_).astype(np.uint8)
            labels = skim_label(bin_img, connectivity=connectivity)
            if i == 0:
                self.gs[m].lgi = labels
            else:
                labels[labels > 0] += self.gs[m].lgi.max()
                self.gs[m].lgi = self.gs[m].lgi + labels
            self.gs[m].s_gid[_s_] = tuple(np.delete(np.unique(labels), 0))
            self.gs[m].s_n[_s_-1] = len(self.gs[m].s_gid[_s_])
            # -----------------------------------------------
        # Get the total number of grains
        self.gs[m].n = np.unique(self.gs[m].lgi).size
        # Generate and store the gid-s mapping
        self.gs[m].gid = list(range(1, self.gs[m].n+1))
        _gid_s_ = []
        for _gs_, _gid_ in zip(self.gs[m].s_gid.keys(),
                               self.gs[m].s_gid.values()):
            if _gid_:
                for __gid__ in _gid_:
                    _gid_s_.append(_gs_)
            else:
                _gid_s_.append(0)
        self.gs[m].gid_s = _gid_s_
        # Make the output string to print on promnt
        optput_string_01 = f'Temporal slice number = {m}.'
        optput_string_02 = f' |||| No. of grains detected = {self.gs[m].n}'
        print(optput_string_01 + optput_string_02)

    def char_morph_2d(self,
                      M,
                      ):
        """
        In each of the temporal slice in M:
            [1] MAKE `grain` DATA-STRUCTURE for every detected grain and,
            [2] CHARACTERIZE the entire grain structure (morphological)

        Parameters
        ----------
        M : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        if type(M) == int and M in self.m:
            print('Making grain data-structures')
            print(f'Morphological characterization for tslice = {M}')
            self.gs[M].char_morph_2d()
            self.tslices_with_prop.append(M)
            print(f'....... tslice = {M}: |-| COMPLETE |-|')
        elif type(M) == int and M not in self.m:
            print('Please enter valid temporal slice from PXGS.m')
        elif type(M) in dth.dt.ITERABLES:
            for tslice in M:
                if tslice in self.m:
                    print(f'Making grain data-structures')
                    print(f'Beginning morphological characterization for tslice = {tslice}')
                    self.gs[tslice].char_morph_2d()
                    self.tslices_with_prop.append(tslice)
                    print(f'....... tslice = {tslice}: |-| COMPLETE |-|')
                else:
                    print(f"tslice = {tslice} not in `PXGS.m`. |-| Ignored |-|")
        else:
            print('Invalid tslice. |-| Ignored |-|')

    def hist(self,
             tslices=None,
             PROP_NAMES=None,
             bins=None,
             kdes=None,
             kdes_bw=None,
             stats=None,
             peaks=False,
             height=0,
             prominance=0.2,
             auto_xbounds=True,
             auto_ybounds=True,
             xbounds=[0, 50],
             ybounds=[0, 0.2]
             ):

        # Set null-hypothesis flag to exit hist computation
        exit_hist = False

        # validate user input for tslices
        if not tslices:
            tslices_prop_available = []
            for gs in self:
                if gs.are_properties_available:
                    tslices_prop_available.append(gs.m)
            if not tslices_prop_available:
                exit_hist=True
            else:
                tslices = tslices_prop_available
        else:
            if type(tslices) == int:
                tslices = [tslices]
            elif type(tslices) in dth.dt.ITERABLES:
                # Nothing to do here, now
                pass
            elif type(tslices) != int and type(tslices) not in dth.dt.ITERABLES:
                print("Invalid tslice datatype. Skipped")
                exit_hist=True

        # If tslice validation passed, then proceed
        if not exit_hist:

            # Make PROP_NAMES datatype valid
            if not PROP_NAMES:
                if tslices:
                    #PROP_NAMES = ['npixels' for tslice in tslices]
                    PROP_NAMES = ['npixels']
            elif type(PROP_NAMES) == str:
                PROP_NAMES = [PROP_NAMES for tslice in tslices]
            elif type(PROP_NAMES) != str and type(PROP_NAMES) not in dth.dt.ITERABLES:
                print("Invalid PROP_NAME. Considering npixels by default")

            # Make bins datatype valid
            if not bins:
                if tslices:
                    bins = [self.vizstyles['bins'] for tslice in tslices]
            if type(bins) in dth.dt.NUMBERS:
                bins = [bins for tslice in tslices]
            elif type(bins) in dth.dt.ITERABLES:
                if len(bins) == len(tslices):
                    if all([type(_) in dth.dt.NUMBERS for _ in bins]):
                        # Nothing more to do here
                        pass
                    else:
                        for i, nbin in enumerate(bins):
                            if type(nbin) not in dth.dt.NUMBERS:
                                bins[i] = self.vizstyles['bins']
                            else:
                                bins[i] = nbin

            # Make kdes datatype valid
            if not kdes:
                kdes = [False for _ in tslices]
                kdes_bw = [None for _ in tslices]
            elif type(kdes) == bool:
                kdes = [kdes for tslice in tslices]
                if type(kdes_bw) not in dth.dt.NUMBERS and type(kdes_bw) not in dth.dt.ITERABLES:
                    kdes_bw = [None for _ in tslices]
                elif type(kdes_bw) in dth.dt.NUMBERS:
                    kdes_bw = [kdes_bw for _ in tslices]
                elif type(kdes_bw) in dth.dt.ITERABLES:
                    if len(kdes_bw) == len(kdes):
                        _ = [type(_) for _ in kdes_bw]
                        if all(_):
                            pass
                        else:
                            for i, _ in enumerate(kdes_bw):
                                if type(_) not in dth.dt.NUMBERS:
                                    kdes_bw[i] = None
                    else:
                        print("len(kdes_bw) must be same as len(kdes)")
                        print("Current set of kdes_bw will results in an error!")
                        print("Please enter valid data.")


            # Make stats datatype valid
            if not stats:
                stats = ['density' for tslice in tslices]
            if type(stats) == str:
                stats = [stats for tslice in tslices]
            if type(stats) != str and type(stats) not in dth.dt.ITERABLES:
                print("Invalid stats datatype Considering density by default")
            hist_count=0
            for tslice in tslices:
                if tslice in self.tslices:
                    if self.gs[tslice].are_properties_available:
                        for PROP_NAME in PROP_NAMES:
                            if PROP_NAME in self.gs[tslice].prop.columns:
                                self.gs[tslice].hist(PROP_NAME=PROP_NAME,
                                                     bins=bins[hist_count],
                                                     kde=kdes[hist_count],
                                                     stat=stats[hist_count],
                                                     color=self.vizstyles['hist_colors_fill'],
                                                     edgecolor=self.vizstyles['hist_colors_edge'],
                                                     alpha=self.vizstyles['hist_colors_fill_alpha'],
                                                     bw_adjust=kdes_bw[hist_count],
                                                     line_kws={'color': self.vizstyles['kde_color'],
                                                                'lw': self.vizstyles['kde_thickness'],
                                                                'ls': '-',
                                                                },
                                                     peaks=peaks,
                                                     height=height,
                                                     auto_xbounds=auto_xbounds,
                                                     auto_ybounds=auto_ybounds,
                                                     xbounds=xbounds,
                                                     ybounds=ybounds,
                                                     prominance=prominance,
                                                     __stack_call__=True,
                                                     __tslice__=tslice
                                                     )
                            else:
                                print(f"PROP_NAME: {PROP_NAME} does not exist at tslice: {tslice}. Skipped")
                    else:
                        print(f"Properties have not been calculated for tslice: {tslice}. Skipped")
                else:
                    print(f"Invalid tslice: {tslice}. Skipped")
                hist_count += 1
        else:
            print("Invalid inputs. No histogram computed. Skipped")


    def find_grains_scilab_ndimage_3d(self,
                                      m
                                      ):
        """
        Detect grains using the library 'scikit-image'.
        m is a single temporal slice number.

        Parameters
        ----------
        m : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        from scipy.ndimage import label as spndimg_label
        _S_ = self.gs[m].s
        for i, _s_ in enumerate(np.unique(_S_)):
            # Mark the presence of this state
            self.gs[m].spart_flag[_s_] = True
            # Recognize the grains belonging to this state
            bin_img = (_S_ == _s_).astype(np.uint8)
            labels, num_labels = spndimg_label(bin_img)

            if i == 0:
                self.gs[m].lgi = labels
            else:
                labels[labels > 0] += self.gs[m].lgi.max()
                self.gs[m].lgi = self.gs[m].lgi + labels
            self.gs[m].s_gid[_s_] = tuple(np.delete(np.unique(labels), 0))
            self.gs[m].s_n[_s_-1] = len(self.gs[m].s_gid[_s_])
        # Get the total number of grains
        self.gs[m].n = np.unique(self.gs[m].lgi).size
        # Generate and store the gid-s mapping
        self.gs[m].gid = list(range(1, self.gs[m].n+1))
        _gid_s_ = []
        for _gs_, _gid_ in zip(self.gs[m].s_gid.keys(),
                               self.gs[m].s_gid.values()):
            if _gid_:
                for __gid__ in _gid_:
                    _gid_s_.append(_gs_)
            else:
                _gid_s_.append(0)
        self.gs[m].gid_s = _gid_s_
        # Make the output string to print on promnt
        optput_string_01 = f'Temporal slice number = {m}.'
        optput_string_02 = f' |||| No. of grains detected = {self.gs[m].n}'
        print(optput_string_01 + optput_string_02)

    def plotgs(self,
               M=[0, 4, 8, 12, 16],
               cmap='jet',
               figsize=(5,5),
               cbtick_incr=2,
               mbar=True,
               mbar_length=10,
               mbar_loc='bot_left'
               ):
        """


        Parameters
        ----------
        m : list, optional
            DESCRIPTION. List of temporal slices to plot.The default is [0]
        cmap : str, optional
            DESCRIPTION. The desired colour mapThe default is 'jet'.

        Returns
        -------
        None.

        """
        if type(M) == int:
            M = [M]
        if type(M) in dth.dt.ITERABLES and type(M[0]) == int:
            _S_max = self.uisim.S
            if mbar:
                xstart, ystart = self.uigrid.xmin, self.uigrid.ymin
                xsize = self.uigrid.xmax - self.uigrid.xmin
                ysize = self.uigrid.ymax - self.uigrid.ymin
                if mbar_loc == 'bot_left':
                    mbar_xstart = xstart + 0.05*min(xsize, ysize)
                    mbar_ystart = ystart + 0.95*min(xsize, ysize)
                elif mbar_loc == 'top_left':
                    mbar_xstart = xstart + 0.05*min(xsize, ysize)
                    mbar_ystart = ystart + 0.05*min(xsize, ysize)
                elif mbar_loc == 'bot_right':
                    mbar_xstart = xstart + 0.95*min(xsize, ysize)
                    mbar_ystart = ystart + 0.95*min(xsize, ysize)
                elif mbar_loc == 'top_right':
                    mbar_xstart = xstart + 0.95*min(xsize, ysize)
                    mbar_ystart = ystart + 0.05*min(xsize, ysize)
                mbar_xend = mbar_xstart+mbar_length
                mbar_ends = [[mbar_xstart, mbar_xend], [mbar_ystart, mbar_ystart]]

                xtext = 0.2*sum(mbar_ends[0])
                ytext = 0.5*sum(mbar_ends[1])-0.2*mbar_length

            for m in M:
                if m in self.tslices:
                    plt.figure(figsize=figsize)
                    cmap= cm.get_cmap('nipy_spectral', _S_max)
                    plt.imshow(self.gs[m].s, cmap=cmap)
                    plt.title(f"tslice: {m}")
                    plt.xlabel(r"X-axis, $\mu m$", fontsize=12)
                    plt.ylabel(r"Y-axis, $\mu m$", fontsize=12)
                    plt.colorbar(ticks=range(0,_S_max, cbtick_incr))
                    if mbar:
                        plt.plot(mbar_ends[0],
                                 mbar_ends[1],
                                 c = 'k',
                                 linewidth=8)
                        plt.text(xtext,
                                 ytext,
                                 f'{mbar_length} $\mu m$',
                                 fontsize = 12,
                                 bbox = {'color': 'white',
                                         'alpha': 0.75}
                                 )
                else:
                    print(f"m: {m} is invalid. Skipped")
        else:
            print("Please enter valid M")

    def finer(self,
              Grid_Data,
              ParentStateMatrix,
              Factor,
              InterpMethod):
        # Use to increase resolution
        # Unpack parent grid parameters
        xmin, xmax, xinc = Grid_Data['xmin'], Grid_Data['xmax'], Grid_Data['xinc']
        ymin, ymax, yinc = Grid_Data['ymin'], Grid_Data['ymax'], Grid_Data['yinc']

        # Reconstruct the original parent co-ordinate grid
        xvec_OG = np.arange(xmin, xmax+1, float(xinc))  # Parent grid axes
        yvec_OG = np.arange(ymin, ymax+1, float(yinc))  # Parent grid axes
        cogrid_OG = np.meshgrid(xvec_OG, yvec_OG, copy=True, sparse=False, indexing='xy')  # grid

        # Construct the new co-ordinate grid
        xvec_NG = np.arange(xmin, xmax+1, float(xinc/Factor))  # NM: 'of' New grid
        yvec_NG = np.arange(ymin, ymax+1, float(yinc/Factor))
        cogrid_NG = np.meshgrid(xvec_NG, yvec_NG, copy=True, sparse=False, indexing='xy')

        # Construct the new orientation state matrix
        OSM_NG = np.round(griddata((np.concatenate(cogrid_OG[0]),
                                    np.concatenate(cogrid_OG[1])),
                                   np.concatenate(ParentStateMatrix),
                                   (np.concatenate(cogrid_NG[0]),
                                    np.concatenate(cogrid_NG[1])),
                                   method=InterpMethod)
                          .reshape((xvec_NG.shape[0], yvec_NG.shape[0])))
        SMIN = np.min(ParentStateMatrix)
        SMAX = np.max(ParentStateMatrix)
        for i in range(np.shape(OSM_NG)[0]):
            for j in range(np.shape(OSM_NG)[1]):
                if OSM_NG[i, j] < SMIN:
                    OSM_NG[i, j] = SMIN
                elif OSM_NG[i, j] > SMAX:
                    OSM_NG[i, j] = SMAX
                elif np.isnan(OSM_NG[i, j]):
                    OSM_NG[i, j] = 1
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        return cogrid_NG, OSM_NG

    def coarser(self,
                Grid_Data,
                ParentStateMatrix,
                Factor,
                InterpMethod):
        # Use to decrease resolution
        # Unpack parent grid parameters
        xmin, xmax, xinc = Grid_Data['xmin'], Grid_Data['xmax'], Grid_Data['xinc']
        ymin, ymax, yinc = Grid_Data['ymin'], Grid_Data['ymax'], Grid_Data['yinc']

        # Reconstruct the parent co-ordinate grid
        xvec_OG = np.arange(xmin, xmax+1, float(xinc))  # Parent grid axes
        yvec_OG = np.arange(ymin, ymax+1, float(yinc))  # Parent grid axes
        cogrid_OG = np.meshgrid(xvec_OG, yvec_OG, copy=True, sparse=False, indexing='xy')  # grid

        # Construct the new co-ordinate grid
        xvec_NG = np.arange(xmin, xmax+1, float(xinc*Factor))  # NG: 'of' New grid
        yvec_NG = np.arange(ymin, ymax+1, float(yinc*Factor))
        cogrid_NG = np.meshgrid(xvec_NG, yvec_NG, copy=True, sparse=False, indexing='xy')

        # Construct the new orientation state matrix
        OSM_NG = np.round(griddata((np.concatenate(cogrid_OG[0]),
                                    np.concatenate(cogrid_OG[1])),
                                   np.concatenate(ParentStateMatrix),
                                   (np.concatenate(cogrid_NG[0]),
                                    np.concatenate(cogrid_NG[1])),
                                   method=InterpMethod)
                          .reshape((xvec_NG.shape[0], yvec_NG.shape[0])))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # print(abc.shape)
        return cogrid_NG, OSM_NG

    def _info_():
        """


        Returns
        -------
        table : TYPE
            DESCRIPTION.

        """

        doc1, doc2, doc3, doc4, doc5 = {}, {}, {}, {}, {}
        doc1['mcgs'] = """"UPXO. Generate, analyse, match, export and mesh,
        Monte-Carlo Simulation based grain structures."""
        doc1['capability A'] = """ Host of modifications to Q - Pott's
        Monte-Carlo, Markov-chain model """
        doc1['capability A'] = """ Rejection based state sampling possible,
        saving hours of computation time """
        doc1['capability A'] = """ State value based Transition probability.
        This enables state value wise preferential grain growth """
        doc1['capability A'] = """ 2D and 3D """
        doc1['capability A'] = """ Generate spatially gradient grain
        structures """
        doc1['capability '] = """ Pickle-save the grain structure instance """
        doc1['capability '] = """ Orientation sampling from EBSD data """
        doc1['capability '] = """ Assign crystallographic orientations """
        doc1['capability '] = """ Use algorithms with C-calls for highly
        enhanced performanceormance """
        doc1['capability '] = """ Control grain boundary roughness, state
        wise"""
        doc1['capability '] = """ Fast (very) grain recognition """
        doc1['capability '] = """ Fast (very) grain boundary extraction """
        doc1['capability '] = """ Fast (very) grain recognition """
        doc1['capability '] = """ 2 level neighbour grain identification """
        doc1['capability '] = """ Create grain boundary zones with specificed
        thicknesses """

        doc2['capability '] = """ Coarsen or fine your grain structure """
        doc2['capability '] = """ Realistically perturb grain structures """
        doc2['capability '] = """ Import and make-realistic, UPXO Voronoi
        tesellation grain structure """
        doc2['capability '] = """ Re-start grain growth in differant temporal
        branches at any temporal step of choice or triggered by an event"""
        doc2['capability '] = """ Uses DefDAP to load EBSD map and clean, and
        then digitally evolve these EBSD grain structures """
        doc2['capability '] = """ Clear, easy to use data structure """

        doc3['capability '] = """ Detailed grain structure characterisation """
        doc3['capability '] = """ Detailed grain structure anlalysis """
        doc3['capability '] = """ Select temporal slices best matching the
        target grain structure morphological statistics """

        doc4['capability '] = """ Bare-minimal user scripting """
        doc4['capability '] = """ Direct export any temporal or spatial slice
        of choice to .CTF file, ABAQUS .inp files """

        doc5['core developer - 1'] = """ Dr. Sunil Anandatheertha:
            conceptualization, design, programming """

        # -------------------------------------------------------------
        table = PrettyTable()
        table.field_names = ["item", "description"]
        table.align["item"] = "l"
        table.align["description"] = "l"
        for r, c in zip(doc1.keys(), doc1.values()):
            table.add_row([r, c])
        # -------------------------------------------------------------
        return table

    @property
    def pxtal_length(self):
        return self.uigrid.xmax-self.uigrid.xmin

    @property
    def pxtal_height(self):
        return self.uigrid.ymax-self.uigrid.ymin

    @property
    def pxtal_area(self):
        return self.pxtal_length*self.pxtal_height
# ---------------------------------------------------------------------


class monte_carlo_grain_structure(grid):

    def __init__(self,
                 study='independent',
                 info_message_display_level='detailed'
                 ):
        super().__init__(study=study)

    def __str__(self):
        """


        Returns
        -------
        string_1 : TYPE
            DESCRIPTION.

        """

        str_1 = f"x:({self.uigrid.xmin},{self.uigrid.xmax},{self.uigrid.xinc}), "
        str_2 = f"y:({self.uigrid.ymin},{self.uigrid.ymax},{self.uigrid.yinc}), "
        str_3 = f"z:({self.uigrid.zmin},{self.uigrid.zmax},{self.uigrid.zinc})."
        return str_1+str_2+str_3

        # string_1 = 'm, gs, (x,y,z)gr, ui(grid,sim,int,gsa,gsprop,georep)'
        # return string_1

    def __att__(self):
        return gops.att(self)

    def simulate(self):
        self.algo_hop = False
        # Initiate the grain-structure data-structure
        self.add_gs_data_structure_template(m=0,
                                            dim=self.uigrid.dim,
                                            study=self.study
                                            )
        # START THE MONTE-CARLO SIMULATIONS
        if self.uigrid.dim == 2 and len(self.uisim.algo_hops) == 1:
            print('I AM IN HERE -- 1')
            self.algo_hop = False
            self.start_algo2d_without_hops()
        elif self.uigrid.dim == 2 and len(self.uisim.algo_hops) > 1:
            if self.algo_hop:
                self.start_algo2d_with_hops()
            else:
                self.start_algo2d_without_hops()
        elif self.uigrid.dim == 3 and len(self.uisim.algo_hops) == 1:
            self.algo_hop = False
            self.start_algo3d_without_hops()
        elif self.uigrid.dim == 3 and len(self.uisim.algo_hops) > 1:
            if self.algo_hop:
                self.start_algo3d_with_hops()
            else:
                self.start_algo3d_without_hops()

    def start_algo2d_without_hops(self):
        _a, _b, _c = self.build_NLM()  # Unpack 3 rows of NLM
        if self.uisim.mcalg == '200':
            print('I AM IN HERE -- 2')
            print("Using ALG-200: SA's SL NL-1 TP1 C2 unweighted Q-Pott's model:")
            import upxo.algorithms.alg200 as alg200
            gs, fully_annealed = alg200.run(self.uisim, self.uiint,
                                            self.uidata_all, self.uigrid,
                                            self.xgr, self.ygr, self.zgr,
                                            self.px_size, _a, _b, _c,
                                            self.S, self.AIA0, self.AIA1,
                                            self.display_messages)
        elif self.uisim.mcalg == '201':
            print("Using ALG-200: SA's NL-1 weighted Q-Pott's model:")
            import upxo.algorithms.alg201 as alg201
            gs, fully_annealed = alg201.run(self.uisim, self.uiint,
                                            self.uidata_all, self.uigrid,
                                            self.xgr, self.ygr, self.zgr,
                                            self.px_size, _a, _b, _c,
                                            self.S, self.AIA0, self.AIA1,
                                            self.display_messages)
        elif self.uisim.mcalg == '202':
            print("Using SA's L0 modified Q-state Pott's model: ")
            print("    weighted (: ALG-200)")
            import upxo.algorithms.alg201 as alg201
            gs, fully_annealed = alg201.run(self.uisim, self.uiint,
                                            self.uidata_all, self.uigrid,
                                            self.xgr, self.ygr, self.zgr,
                                            self.px_size, _a, _b, _c,
                                            self.S, self.AIA0, self.AIA1,
                                            self.display_messages)

    def start_algo2d_with_hops(self):
        pass

    def start_algo3d_without_hops(self):
        print('I AM IN HERE -- 2')
        if self.uisim.mcalg == '300':
            print("Using ALG-300")
            print('////////////////////////////////')
            self.mc_iterations_3d_alg300()
        elif self.uisim.mcalg == '310':
            print("Using ALG-310")
            print('////////////////////////////////')
            self.mc_iterations_3d_alg310()

    def build_NLM(self):
        """


        Returns
        -------
        NLM : TYPE
            DESCRIPTION.

        """

        if self.uisim.NL == 1:
            NLM_00 = self.NLM_nd[0, 0]
            NLM_01 = self.NLM_nd[0, 1]
            NLM_02 = self.NLM_nd[0, 2]

            NLM_10 = self.NLM_nd[1, 0]
            NLM_11 = self.NLM_nd[1, 1]
            NLM_12 = self.NLM_nd[1, 2]

            NLM_20 = self.NLM_nd[2, 0]
            NLM_21 = self.NLM_nd[2, 1]
            NLM_22 = self.NLM_nd[2, 2]

            NLM = np.array([[NLM_00, NLM_01, NLM_02],
                            [NLM_10, NLM_11, NLM_12],
                            [NLM_20, NLM_21, NLM_22]])
        elif self.uisim.NL == 2:
            NLM_00 = self.NLM_nd[0, 0]
            NLM_01 = self.NLM_nd[0, 1]
            NLM_02 = self.NLM_nd[0, 2]
            NLM_03 = self.NLM_nd[0, 3]
            NLM_04 = self.NLM_nd[0, 4]

            NLM_10 = self.NLM_nd[1, 0]
            NLM_11 = self.NLM_nd[1, 1]
            NLM_12 = self.NLM_nd[1, 2]
            NLM_13 = self.NLM_nd[1, 3]
            NLM_14 = self.NLM_nd[1, 4]

            NLM_20 = self.NLM_nd[2, 0]
            NLM_21 = self.NLM_nd[2, 1]
            NLM_22 = self.NLM_nd[2, 2]
            NLM_23 = self.NLM_nd[2, 3]
            NLM_24 = self.NLM_nd[2, 4]

            NLM_30 = self.NLM_nd[3, 0]
            NLM_31 = self.NLM_nd[3, 1]
            NLM_32 = self.NLM_nd[3, 2]
            NLM_33 = self.NLM_nd[3, 3]
            NLM_34 = self.NLM_nd[3, 4]

            NLM_40 = self.NLM_nd[4, 0]
            NLM_41 = self.NLM_nd[4, 1]
            NLM_42 = self.NLM_nd[4, 2]
            NLM_43 = self.NLM_nd[4, 3]
            NLM_44 = self.NLM_nd[4, 4]

            NLM = np.array([[NLM_00, NLM_01, NLM_02, NLM_03, NLM_04],
                            [NLM_10, NLM_11, NLM_12, NLM_13, NLM_14],
                            [NLM_20, NLM_21, NLM_22, NLM_23, NLM_24],
                            [NLM_30, NLM_31, NLM_32, NLM_33, NLM_34],
                            [NLM_40, NLM_41, NLM_42, NLM_43, NLM_44]])
        return NLM

    @staticmethod
    def info_mcalg(alg):
        """


        Parameters
        ----------
        alg : TYPE
            DESCRIPTION.

        Returns
        -------
        table : TYPE
            DESCRIPTION.

        """

        doc = {}
        if alg == 200:
            doc['one_liner'] = "ALG-200: SL NL-1 TP1 unweighted Q-Pott's model"
            doc['generalization'] = 'high'
            doc['performanceormance'] = 'slow'
            doc['c_call'] = 'not yet available'
            doc['dim'] = 2
            doc['parentalg'] = "Modified Q-state Pott's model"
            doc['publication'] = "to be published"
            doc['lattice'] = "square"
            doc['non_locality'] = 1
            doc['connectivity'] = 2
            doc['tr_prob'] = "state partitioned"
            doc['state_partitioning_control'] = "possible"
            doc['developer'] = 'Dr. Sunil Anandatheertha'
            doc['coded_by'] = 'Dr. Sunil Anandatheertha'
        elif alg == 201:
            doc['one_liner'] = None
            doc['generalization'] = None
            doc['performanceormance'] = None
            doc['c_call'] = None
            doc['dim'] = None
            doc['parentalg'] = None
            doc['publication'] = None
            doc['lattice'] = None
            doc['non_locality'] = None
            doc['connectivity'] = None
            doc['tr_prob'] = None
            doc['state_partitioning_control'] = None
            doc['developer'] = None
            doc['coded_by'] = None
        # -------------------------------------------------------------
        table = PrettyTable()
        table.field_names = ["item", "description"]
        table.align["item"] = "l"
        table.align["description"] = "l"
        for r, c in zip(doc.keys(), doc.values()):
            table.add_row([r, c])
        # -------------------------------------------------------------
        return table



    def NLM_elements(self):
        # Build the Non-Locality Matrix
        _a, _b, _c = self.build_NLM()  # Unpack 3 rows of NLM
        NLM_00, NLM_01, NLM_02 = _a  # Unpack 3 colms of 1st row
        NLM_10, NLM_11, NLM_12 = _b  # Unpack 3 colms of 2nd row
        NLM_20, NLM_21, NLM_22 = _c  # Unpack 3 colms of 3rd row
        return NLM_00, NLM_01, NLM_02, NLM_10, NLM_11, NLM_12, NLM_20, NLM_21, NLM_22
