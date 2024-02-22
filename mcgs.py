"""s
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
import numpy.random as rand
# import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# import matplotlib.colors as colors
# import matplotlib.ticker as ticker
from random import sample as sample_rand
import gops
import os
# import re
import math
import numpy as np
from scipy.interpolate import griddata
import datatype_handlers as dth
# import scipy.stats as stats
import cv2
from skimage.measure import label as skim_label
# from point2d import point2d
# from mulpoint2d import mulpoint2d
import xlrd
import pandas as pd
from termcolor import colored

from typing import Union
# REFER: https://stackoverflow.com/questions/66055067/how-to-allow-multiple-types-of-arguments-in-function
# Union could be used in type specifying arguments in case where multiple
# types are allowed for an argument.


from copy import deepcopy
from prettytable import PrettyTable
import seaborn as sns
# import warnings, logging
from scipy.signal import argrelextrema
from scipy.signal import find_peaks
from mulpoint2d import mulpoint2d
__name__ = "UPXO-mcgs"
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



class parameter_sweep():
    """
    This is a core UPXO class. Use to generate and work with multiple mcgs
    using a combination of various generating and controlling parameters.

    Targetted at:
    -------------
        * research: understanding algorithm sensitivity
        * research: understanding parameter sensitivity
        * research: understanding grain growth evolution
        * development: UPXO

    Data structures:
    ----------------
        * Each PXTAl object is stored in a DICT under the instance number key.
        * Most properties are stored as Pandas dataframes.
        * The user input parameters are preserved under self.uiAAAAA,
        where, AAAAA is the additional name string of the corresponsing
        parameter set.
        * Mesh instances are stored seperately as a DICT under the instance
        number key, followed by a nested DICT under the mcstep key.

    Data attributes:
    ----------------
    domain: Size of the domain
    dim: dimensionality
    gmp: global morphological parameters
    qmp: Q-partitioned morphological parameters
    purge_previous: Purge previous study objects
    save_sims: BOOL: flag to pickle raw databases of simulations
    __GENLOCK__: str: flag to lock simulation capability of UPXO

    __GENLOCK__:
    ------------
        If 'locked', no simulations will be performed and grain structures
        will not be developed. Otherwise, if 'open'. Status set to 'locked'
        whenever following cases:
            * when user input validation fails.
            * when computations branch towards any lock imposed by developer
            in UPXO internals.
    """
    __slots__ = ('N', 'nstates', 'domain', 'dim',
                 'algo_hop', 'algo_hops',
                 'purge_previous', '_save_sims_',
                 'gsi',
                 'gmp', 'qmp',
                 'mesh_instances',
                 )
    __paramater_gsi_mapping_behaviour__ = 'one-many'
    __default_mcalg__ = '200'
    # ------------------------------------------------------------------
    '''
    NOTE: @ all locks :: Locked if True, Open if False

    __GS_GEN_LOCK__: GS generation lock.

    __GR_IDNT_LOCK__: Grain identification lock
    __GSMORPH_CHAR_LOCK__: Morphology characterisation lock
    __GS_ANALYSIS_LOCK__: Grain structure analysis lock
    __TEX_SAM_LOCK__: Texture sampling lock
    __TEX_MAP_LOCK__: Texture mapping lock
    __FE_GEN_LOCK__: Finite element mesh generation lock
    __FE_EXP_LOCK__: Finite element mesh export lock
    __REP_QUAL_LOCK__: Representativeness qualification lock
    __FERES_MAP_LOCK__: Finite element result map lock
    __FERES_ANALYSIS_LOCK__: Finite element result analysis lock
    '''
    __GS_GEN_LOCK__, __GR_IDNT_LOCK__ = [True, True], [True, True]
    __GSMORPH_CHAR_LOCK__, __GS_ANALYSIS_LOCK__ = [True, True], [True, True]
    __TEX_SAM_LOCK__, __TEX_MAP_LOCK__ = [True, True], [True, True]
    __FE_GEN_LOCK__, __FE_EXP_LOCK__ = [True, True], [True, True]
    __REP_QUAL_LOCK__ = [True, True]
    __FERES_MAP_LOCK__, __FERES_ANALYSIS_LOCK__ = [True, True], [True, True]
    # ------------------------------------------------------------------
    __GS_GEN_LOCK__ = [{'uigrid_type': True, 'uigrid_range': True,
                        'uisim_type': True, 'uisim_range': True,
                        'uigsc_type': True, 'uigsc_range': True,
                        'uimesh_type': True, 'uimesh_range': True,
                        'uigeomrepr_type': True, 'uigeomrepr_range': True, },
                       {'uigrid_type': True, 'uigrid_range': True,
                        'uisim_type': True, 'uisim_range': True,
                        'uigsc_type': True, 'uigsc_range': True,
                        'uimesh_type': True, 'uimesh_range': True,
                        'uigeomrepr_type': True, 'uigeomrepr_range': True, },
                       ]
    # ------------------------------------------------------------------
    def __init__(self,
                 use_default_values=False,
                 study='gs_analysis'):
        """
        Instantiates parameter sweep data-structure and perform user
        requested tasks

        Parameters
        ----------
        use_default_values : BOOL, optional
            If True, data-structure will be instantiated with default values
            for parameters of grid, simulation, grain structure analysis,
            meshing, geometric representations, etc. The default is False.
            If False, empty data-structure will be created. In which case,
            the user will have to seperately use the set_param_{ABCD} methods.

        study : srt, optional
            - If 'gs_analysis': All functionalities necessary for
            gs_analysis shall remain unlocked by default. All higher
            functionalities remain locked by default.
            - If 'gs_mesh': All functionalities necessary for grain
            structure meshing including the mesh export remain uncloked by
            default.
            - If 'gs_growth': All functionalities necessary for grain growth
            study remain unlocked by default.
            - If 'cpfe_data_analysis': All functionalities necessary for
            gs_analysis remain unlocked by default.
            - If '_development_': All functionalities necessary for
            gs_analysis remain unlocked by default.
            NOTE-1:
                User input validity may change the corrsponding sub-lock
            states.
            NOTE-2:
                Available options are contained in dth.opt.ps_studies

        Returns
        -------
        None.

        """
        # Initiate gsi=None to enable __repr__ when not assign_default_values.
        __gsi_index__ = None
        if use_default_values:
            self.initialize(N=2)
            # ----------------------------------------------------------------
            self.set_param_grid(domain_size=((0, 100),
                                             (0, 100),
                                             (0, 0),
                                             1),
                                read_from_file=False,
                                filename=None)
            for i, _ in enumerate(self.N):
                a = self.gsi[i].uisim.__uigrid_type_lock__
                b = self.gsi[i].uisim.__uigrid_range_lock__
                self.__GS_GEN_LOCK__[i]['uigrid_type'] = a
                self.__GS_GEN_LOCK__[i]['uigrid_range'] = b
            # ----------------------------------------------------------------
            self.set_param_sim(mcsteps=20,
                               nstates=32,
                               solver='python',
                               tgrad=None,
                               algo_hop=False,
                               algo_hops=[(200, 10),
                                          (201, 40),
                                          (202, 100)],
                               default_mcalg=self.default_mcalg,
                               save_at_mcsteps=np.linspace(0, 20, 5),
                               purge_previous=False,
                               read_from_file=False,
                               filename=None)
            self.set_param_gsc(char_grains=True,
                               char_stage='postsim',
                               library='scikit-image',
                               parallel=True,
                               find_gbseg=True,
                               g_area=True,
                               gb_length=True,
                               ofton=True,
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
                               filename=None)
            self.set_param_geomrepr(make_mp_grain_centoids=True,
                                    make_mp_grain_points=True,
                                    make_ring_grain_boundaries=True,
                                    make_xtal_grain=True,
                                    make_chull_grain=True,
                                    create_gbz=[False, True],
                                    gbz_thickness=[0.1, 0.2],
                                    read_from_file=False,
                                    filename=None)
            self.set_param_mesh(generate_mesh=False,
                                target_fe_software='abaqus',
                                par_treatment='global',
                                mesher='upxo',
                                gb_conformities=('conformal',
                                                 'non_conformal',
                                                 ),
                                global_elsizes=(0.5,
                                                1.0,
                                                ),
                                mesh_algos=(4,
                                            6,
                                            ),
                                grain_internal_el_gradient=('constant',
                                                            'constant',
                                                            ),
                                grain_internal_el_gradient_par=(('automin',
                                                                 'automax'),
                                                                ('automin',
                                                                 'automax'),
                                                                ),
                                target_eltypes=('CSP4',
                                                'CSP8',
                                                ),
                                elsets=('grains',
                                        'grains',
                                        ),
                                nsets=('x-', 'x+', 'y-', 'y+', ),
                                optimize=(False,
                                          False,
                                          ),
                                opt_par=('min_angle',
                                         [45, 60],
                                         'jacobian',
                                         [0.45, 0.6],
                                         ),
                                read_from_file=False,
                                filename=None)


    def __repr__(self):
        print('/'*60)
        print("+ + + + + UPXO MCGS PARAMETER SWEEP + + + + +")
        print('/'*60)
        print(f"Number of parameter datasets: {len(self.N)}")
        print('/'*60)
        for n in self.N:
            self.info_attributes(n)
            if n != self.N[-1]:
                print('= '*30)
        print('/'*60)
        return ''

    @property
    def info_message_display_level(self):
        return [self.gsi[n].info_message_display_level for n in self.N]

    @info_message_display_level.setter
    def info_message_display_level(self, level):
        if level == 'simple':
            self.set_gsi_info_message_display_level_simple
        elif level == 'detailed':
            self.set_gsi_info_message_display_level_detailed

    @property
    def info_message_display_level_simple(self):
        for n in self.N:
            self.gsi[n].info_message_display_level_simple

    @property
    def info_message_display_level_detailed(self):
        for n in self.N:
            self.gsi[n].info_message_display_level_detailed

    def __iter__(self):
        self.__gsi_index__ = 1
        return self

    def __next__(self):
        if self.__gsi_index__ <= len(self.N):
            _gsi_ = self.gsi[self.__gsi_index__]
            self.__gsi_index__ += 1
            return _gsi_
        else:
            raise StopIteration

    def limit_check(self):
        pass

    def generate_mcgs2d(self):
        pass

    def characterize_gs(self):
        for n in self.N:
            if self.gsi[n].uigsc.parallel:
                pass

    def initialize(self, N=2):
        if type(N) == int and N != 0 and N < 25:
            self.N = [n+1 for n in range(N)]
            from mcgs import monte_carlo_grain_structure as mcgs
            self.gsi = {}
            for n in range(N):
                self.gsi[n+1] = mcgs(study='para_sweep')
        else:
            print(f'Invalid N: {N}. None initialized. Skipped')

    def run(self,
            char_post_sim=True,
            parallel_char=False,
            ):
        '''
        Sweeps across pipelines to achieve as per user request, across
        all allowed and possible combination of individual and sets of
        various parameters. Will only be consideref if defaults is True.
        The default is False.

        char_post_sim:
            Characterize during sims or post sims
        parallel_char:
            Parallelise characterization.
            Only if char_post_sim=True
            If parchar=False, characterization GS will not be characterized,
            in which case, use the characterize method explicitly.
        '''
        pass

    def assemble_locks(self):
        for n in self.N:
            self.__GENLOCK__['uisim'] = self.gsi[n].uisim.__uisim_lock__

    def set_param_grid(self,
                       domain_size=((0, 100), (0, 100), (0, 0), 1),
                       read_from_file=False,
                       filename=None
                       ):
        """
        Set up the parameters pertaining to grid. Targetted at
        ps.gsi[:].uigrid

        Parameters
        ----------
        domain_size : list/tuple/deque/np.ndarray, optional
            ((x bounds), (y bounds), (z bounds), pixel size or increment).
            A bound is specified by two values, minimum and maximum. First
            value MUST be the numerically lower number. As no checks are made
            to validate this user entry, user must take care to enter these
            values.
            NOTE: pixel size will be uniform across x, y and z axes.
            The default is ((0, 100), (0, 100), (0, 0), 1).

        Returns
        -------
        None.

        """
        for n in self.N:
            self.gsi[n].set_uigrid(domain_size,
                                   read_from_file=False,
                                   filename=None
                                   )

    def set_param_sim(self,
                      mcsteps=20,
                      nstates=32,
                      solver='python',
                      tgrad=None,
                      algo_hop=False,
                      default_mcalg='200',
                      algo_hops=[(200, 10), (201, 40), (202, 100)],
                      save_at_mcsteps=np.linspace(0, 20, 5),
                      state_sampling_scheme='rejection',
                      consider_boltzmann_probability=False,
                      s_boltz_prob='q_related',
                      boltzmann_temp_factor_max=0.1,
                      boundary_condition_type='wrapped',
                      NL=1,
                      kineticity='static',
                      purge_previous=False,
                      read_from_file=False,
                      filename=None
                      ):
        """
        Explanation
        -----------
        This is part of parameter setting methods for
        parameter sweep studies. Helps set grain structure simulation
        parameters like. The followig parameters are set by set_param_sim:
            * mcsteps
            * nstates
            * solver
            * tgrad
            * algo_hop
            * algo_hops
            * save_at_mcsteps
            * purge_previous

        Usage
        -----
        UPXO internal and user. Will be used if the user prefers quick
        parameter sweep with default values. If user, user wishes to have
        specirfic values, which would be most often the case.

        Parameters
        ----------
        mcsteps : int, optional
            Number of Monte-Carlo steps. To restrict comparisons over
            different temporal scales, all grain structure instances will
            be simulated upto equal mcsteps.
            NOTE: However, if in a case, the grain structure temporally
            saturates during simulation before the total mcsteps for that
            specific gsi is reaced, then the total number of mcsteps actyually
            covered would be smaller than that specified by the user. This is
            becvause, all core-solver algorithms in UPXO are designed to
            STOP when the grain structure reaces temporal saturation. This
            saturation happens when there is a single grain in the GS, that
            is all state values in the gs lattice become same.
            The default is 20.
        nstates : int, optional
            The total number of unique state values in the simulation. A value
            of 2 would basically generate an Ising type lattice.
            The default is 32.
        solver : str, optional
            Specifies whether the solver is to be from python or C. The choice
            depens on the following parameters:
                * Size of the spatial domain
                * Total number of mcsteps
                * Computational cost of the algorithm
            The decision between 'python' and 'c' should happen based on
            simulation domain largness, which depend on the largeness of the
            spatial domain, largeness of the temporal dimension and finally
            largeness of the computational cost.
            Input is case-independent.
            Choosing 'C' will force UPXO to use
            the C-executatable for core solver. This option is only ava8ilable
            fo0r some algorithms. To know supported algorithms, please refer
            to info on algorithms.
            Choosing 'python' will make UPXO to decide between 'python' and
            'c', based on practicality of using 'python' for large simulation
            domains.
            The default option is 'python'
        tgrad : np.ndarray, optional
            Temperature gradient field of size. Size same as that of grid of
            lattice or that of the state value matrix. Each lattice point must
            accompanied by a temperature value.
            The default is None.
        algo_hop : bool/str, optional
            This helps decide whether to use a single algorithm for the
            entire temporal domain or whether you would need UPXO
            to hop across algorithms.
            If False: disallow algorithm hopping
            If True: allow algorithm hopping
            If str: only allowed value as of now is 'auto'.
            If 'auto': UPXO will decide which algorithms to use upon need of
            a algorithm hopping.
            The default is False.
        algo_hops : list/str/int, optional
            1. If options pertaining to algorithm hopping has
            been provided by the user, then the first available
            option pertaining to algorithm ID will be used to
            set the algorithm. For example, if algo_hops is
            [(200, 10), (201, 40), (202, 100)], then mcalg will be
            set to '200'.
            2. If a numerical entry has been made (in a case where the
            user has done through direct access through set_param_sim),
            then if it is valid, then str(value) will be set for mcalg.
            If invalid, mcalg will default to '200' for each hop.
            3. If a string entry has been made (in a case where the
            user has done through direct access through set_param_sim),
            then if it is valid, then it will be set for mcalg.
            If invalid, mcalg will default to '200'

            The default is [(200, 10), (201, 40), (202, 100)], meaning:
                algo200, upto 10% sim time
                algo201, upto 40% sim time
                algo202, upto 100% sim time
        save_at_mcsteps : ITERABLE, optional
            DESCRIPTION. The default is np.linspace(0, 20, 5).

        Returns
        -------
        None.

        """
        for n in self.N:
            sim_parameters = {'mcsteps': mcsteps,
                              'nstates': nstates,
                              'solver': solver,
                              'tgrad': tgrad,
                              'default_mcalg': default_mcalg,
                              'algo_hop': algo_hop,
                              'algo_hops':  algo_hops,
                              'save_at_mcsteps': save_at_mcsteps,
                              'state_sampling_scheme': state_sampling_scheme,
                              'consider_boltzmann_probability': consider_boltzmann_probability,
                              's_boltz_prob': s_boltz_prob,
                              'boltzmann_temp_factor_max': boltzmann_temp_factor_max,
                              'boundary_condition_type': boundary_condition_type,
                              'NL': NL,
                              'kineticity': kineticity,
                              'purge_previous': purge_previous,
                              }
            self.gsi[n].set_uisim(n=n,
                                  sim_parameters=sim_parameters,
                                  read_from_file=False,
                                  filename=None
                                  )

    def set_param_gsc(self,
                      char_grains=True, char_stage='postsim',
                      library='scikit-image', parallel=True,
                      find_gbseg=True, g_area=True, gb_length=True,
                      gb_length_crofton=True, gb_njp_order=True,
                      g_eq_dia=True, g_feq_dia=True, g_solidity=True,
                      g_circularity=True, g_mjaxis=True, g_mnaxis=True,
                      g_morph_ori=True, g_el=True, g_ecc=True,
                      read_from_file=False, filename=None
                      ):
        """
        Set flags for grain structure characterisaiton and analysis.

        Parameters
        ----------
        char_grains : bool, optional
            Flag to charatcterize grains. If True, grains will be
            characterised and not if False. If True, grain boundaries will also
            be characterised for basic properties. Once the grains have been
            identifies, the characterisation will be done using scikit-image
            by default, at this version of UPXO.
            The default is False.
        char_stage : str, optional
            Choose when to characterize the grains. Options include:
                * 'postsim'
                * 'insim'
            If 'postsim', grain structure will be characterised after all
            temporal slices have been extracted i.e. after all monte-carlo
            iterations have been completed. If 'insim', grain structure
            will be charactersed at the end of each monte-carlo iteration.
            The default is 'postsim'.
        library : str, optional
            Choose which library to identifying the grains. Options include:
                * scikit-image: 2D and 3D
                * opencv: 2D only
                * upxo: 2d only (deprecated)
            The default is 'scikit-image'.
        parallel : bool, optional
            Decides whether grain structure characterisation should be done
            using parallel execution. Following combinations of options are
            permitted:
                * If True and char_stage is 'post_sim', then grain structyure
                characterisation will be done with parallel computation.
                * If True and char_stage is 'in-sim', then the combination is
                invalid. The grain characterisaion will be done at the end of
                each mc iteration.
                * If False and char_stage is 'post_sim', then grain structure
                characterisation will done after all mc iterations are
                completed, but one temporal slice after the other. However,
                the calculation of individual morphological parameters will
                be done using pooling when possible. When this option is not
                possible, behaviour will be similar to that of the combination
                False and 'in-sim'.
                *  If False and char_stage is 'in-sim', then grain structure
                characterisation will be carried out at the end of each
                mc iteration. No part of the process will be threaded or
                pooled or  executed in parallel.
            The default is False.
        find_gbseg : bool, optional
            Flag to identify the grain boundary segments. If True, the grain
            boundary segments will be identified and not if False. GB segments
            will be identified by UPXO and no other oprtion is needed.
            The following behaviours should be kept in mind:
                * Will only work if char_grains is True. Assuming it is True,
                the following further points hold.
                * If find_gbseg is False, but gb_njp_order is True, then, the
                grain boundary segments will still be identified to allow the
                calculation of njp order.
            The default is False.
        g_area : bool, optional
            Flag to calculate grain area. The calculaion takes into
            consideration, the pixel area of the underlying grid. The default
            is False.
        gb_length : bool, optional
            Falg to calculate the grain boundary length. The calculation takjes
            into considertation, side length of the pixel of the underlying
            grid. The following behaviour should be noted:
                * If char_grains is False, and gb_length is True, then
                grain boundary lengths will not be calculated.
                * if grain boundary segments have been identified and gb_length
                is True, then along with calculating grain boundary lengths,
                the lengths of gerain boundaryu segments will also be
                calculated.
            The default is False.
        gb_length_crofton : bool, optional
            Flag to calculate the Crofton perimeter of the grain boundary.
            For more information, please refer to: https://scikit-image.org/
            docs/stable/auto_examples/segmentation/plot_perimeters.html
            The default is False.
        gb_njp_order : bool, optional
            Flag to calculate the 'n' of junction points, that is the value of
            grain boundary junction point order. Its value is the number of
            grains a grain boundary junction point is being shared with. If 3,
            we have a triple point junction, if 4, we have a quadruple point
            junction and so on.
            The following behaviours must be noted:
                * Will only be calculated if char_grains is True and grain
                boundary segments have been identified.
            The default is True.
        g_eq_dia : bool, optional
            Flag to calculate the equivalent diameter of the grain. If True,
            equiavelnt diamater of the grains will be caclculated, not if
            False. The following behaviours must be noted:
                * If grain_area is False, and g_eq_dia is True, then the
                grain area will still be calculated to allow claculation
                of  grain equivalent diameter. However, only the grain
                equivalent diameter will be saved as an attribute and not the
                grain area, which was not requested.
                * Equiavalent diameter caluclation will consider the area
                of the pixel in the grid. Infact, it gets carried from the
                grain_area claculation.
            The default is True.
        g_feq_dia : bool, optional
            Flag to calculate the Feret equivalent diameter. If True, the
            Feret equivalent will be calculated, not if False. Behaviours
            are similar to that of g_eq_dia.
            The default is True.
        g_solidity : bool, optional
            Flag to calculate the solidity of grain. The default is True.
        g_circularity : bool, optional
            Flag to calculate grain circularity. The default is True.
        g_mjaxis : bool, optional
            Flag to calculate the major axis of the grain. The default is True.
        g_mnaxis : bool, optional
            Flag to calculate the ninor axis of the grain. The default is True.
        g_morph_ori : bool, optional
            Flag to calculate the morphological orientation of the grains.
            Bounded in [-90, 90] degrees. The default is True.
        g_el : TYPE, optional
            DESCRIPTION. The default is True.
        g_ecc : bool, optional
            Flag to calculate the eccentricity of the grains. The default is
            True.

        Returns
        -------
        None.

        """
        for n in self.N:
            self.gsi[n].set_uigsc(char_grains=char_grains,
                                  char_stage=char_stage,
                                  library=library, parallel=parallel,
                                  find_gbseg=find_gbseg,
                                  g_area=g_area, gb_length=gb_length,
                                  gb_length_crofton=gb_length_crofton,
                                  gb_njp_order=gb_njp_order,
                                  g_eq_dia=g_eq_dia, g_feq_dia=g_feq_dia,
                                  g_solidity=g_solidity,
                                  g_circularity=g_circularity,
                                  g_mjaxis=g_mjaxis, g_mnaxis=g_mnaxis,
                                  g_morph_ori=g_morph_ori, g_el=g_el,
                                  g_ecc=g_ecc, read_from_file=read_from_file,
                                  filename=filename
                                  )

    def set_param_geomrepr(self,
                           make_mp_grain_centoids=True,
                           make_mp_grain_points=True,
                           make_ring_grain_boundaries=True,
                           make_xtal_grain=True, make_chull_grain=True,
                           create_gbz=True, gbz_thickness = 0.1,
                           read_from_file=False, filename=None
                           ):
        """
        Set parametwers needed to generate geometrical representations of the
        Monte-Carlo Grain Structure.

        Parameters
        ----------
        make_mp_grain_centoids : bool, optional
            Make UPXO multi-point object grom the grain centroids
            The default is True.
        make_mp_grain_points : bool, optional
            Make multi-point objects of all pixel cenrtoids in grains.
            NOTE: Not recommended for large domains.
            The default is False.
        make_ring_grain_boundaries : bool, optional
            Make UPXO multi-point object from all points on the grain boundary
            of a grains. Number of objects made will equal to the number
            of grains. The default is True.
        make_xtal_grain : bool, optional
            Make UPXO XTAL object for the grain. The default is True.
        make_chull_grain : bool, optional
            Flag to create convex hull object of the grain.
            The default is True.
        create_gbz : bool, optional
            Flag to create grain boundary zone. This operation will also make
            the grain core zone. Both of these will be available to be
            turned into element sets for FE mesh export.
            The default is True.
        gbz_thickness : float/int, optional
            Control the thickness of the grain boundary zone. Value must be
            between 0 and 1 and is the fraction of actual grain boundary
            thickness in grid units to minor axis length of the grain.
            NOTE: For grains, where grain boundary zones cannot be created
            due to morphological restrictions, data for the speciric grain
            will be kept at None. Default value is 0.1.

        Returns
        -------
        None.

        """
        for n in self.N:
            self.gsi[n].set_uigeomrepr(make_mp_grain_centoids=make_mp_grain_centoids,
                                       make_mp_grain_points=make_mp_grain_points,
                                       make_ring_grain_boundaries=make_ring_grain_boundaries,
                                       make_xtal_grain=make_xtal_grain,
                                       make_chull_grain=make_chull_grain,
                                       create_gbz=create_gbz,
                                       gbz_thickness=gbz_thickness,
                                       read_from_file=read_from_file,
                                       filename=filename)

    def set_param_mesh(self, generate_mesh=False, target_fe_software='abaqus',
                       par_treatment='global', mesher='upxo',
                       gb_conformities=('conformal', 'non_conformal'),
                       global_elsizes=(0.5, 1.0), mesh_algos=(4, 6),
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
                       read_from_file=False, filename=None
                       ):
        """
        Set the meshing parameters for parameter sweep studies

        Parameters
        ----------
        generate_mesh : BOOL, optional
            Flag to mesh the grain structure.
            The default is False.
        target_fe_software : STR, optional
            The FE software for which the mesh is targetted at.
            Current options include 'abaqus'.
            Future options shall be 'moose', 'damask'
            The default is 'abaqus'.
        par_treatment : STR, optional
            Specifies whether some (see below list) are to apply for
            all instances in the parameter sweep dataset, OR, whether,
            a unique parameter is to be used for a unique instance. This
            applies for the following user input parameters:
                * gb_conformities
                * global_elsizes
                * mesh_algos
                * grain_internal_el_gradient
                * target_eltypes
                * optimize
            If 'local', then 'n' values for each of the above parameters
            must be provided. 'n' is the number of parameter sweeps, which is
            len(ps.N).
            The default is 'global'.
        mesher : STR, optional
            Specify the mesher. Options are 'upxo', 'pygmsh', 'gmsh', 'abaqus'
            -'upxo': applies only to pizellated mesh (non-conformal) of the 2D,
            3D MCGS.
            -'pygmsh', 'gmsh': Applies to geometrised 2D MCGS, 3D MCGS, 2D VTGS
            and 3D VTGS
            -'abaqus': applies to 2D VTGS and geometrised 2D MCGS
            This will write data to disk. UPXO-ABAQUS python scripts are then
            to be used to construct and mesh the grain structure in ABAQUS
            The default is 'upxo'.
        gb_conformities : MIXED: STR/ITERABLE, optional
            Individual value options: 'conformal', 'non_conformal'
            If STR and 'conformal', then all instances will conformally meshed.
            if STR and 'non_conformal', then all instances will non-conformally
            meshed.
            if ITERABLE and of the right size, then each instance will be
            meshed as per the value in the location in gb_conformities
            corresponding to the instance.
            if ITERABLE and of the wrong size, parameter sweep study stops.
            The default is ('conformal', 'non_conformal').
        global_elsizes : MIXED: FLOAT/ITERABLE, optional
            If FLOAT, then it will be mapped to all instances
            If ITERABLE and of the right size, then each instance will be
            meshed with the corresponding element size.
            If ITERABLE and of the wrong size, then parameter sweep study
            stops.
            The default is (0.5, 1.0).
        mesh_algos : MIXED: INT/ITERABLE, optional
            If INT, then it will be mapped to all instances
            If ITERABLE and of the right size, then each instance will be
            meshed with the corresponding specified algirithm
            If ITERABLE and of the wrong size, then parameter sweep
            study will stop.
            The default is (4, 6).
        grain_internal_el_gradient : MIXED: STR/ITERABLE, optional
            If STR, then all instances will be meshed
            with the same element gradient specification
            If ITERABLE and of the right size, then all instances will be
            meshed using correpsoning values of element gradients
            If ITERABLE and of the wrong size, then parameter sweep study
            stops.
            Options are 'constant', 'linear_gb_to_centroid',
            'linear_centroid_to_gb', 'linear_gb_to_core', 'linear_core_to_gb'
            - For value other than 'constant', then global_elsizes will not
            be used. Instead values provided by grain_internal_el_gradient_par
            will be used.
            - For value 'linear_gb_to_centroid', min size will be near gb and
            max size will be at centroid. Variation will be linear.
            - For value 'linear_centroid_to_gb', max size will be near gb and
            min size will be at centroid. Variation will be linear.
            - For value 'linear_gb_to_core', min size will be near gb and
            size increases linearly towards the max size along vectors
            normal to the local gb edge. Vector will be directed towards
            inner region of the grain.
            - For value 'linear_core_to_gb', max size will be near gb and
            size decreases linearly towards the min size along vectors
            normal to the local gb edge. Vector will be directed towards
            inner region of the grain
            The default is ('constant', 'constant').
        grain_internal_el_gradient_par : MIXED: ITERABLE(STR/FLOAT)/ITERABLE,
        optional.
            If STR/FLOAT, same action will be mapped onto all instances.
            If (STR, STR), only allowed non-interchangeable values is 'automin'
            and 'automax'. If ('automin', 'automax'), then element sizes
            will be calculated using a combination of grain boundary
            properties, maximum intercept along the curve normal, grain
            shape factor, etc. The procedure is described in theoretical
            manual.
            If (FLOAT, FLOAT), then values will be chosen accordingly and
            maps accordingly to all instances.
            NOTE @ dev: RETAIN THIS TO BE ('automin', 'automax') and not
            replace with just 'auto', for reason of conformity to a standard
            user data specification format.
            The default is (('automin', 'automax'),).
        target_eltypes : MIXED: STR/ITERABLE, optional
            If STR, value is correct and allowed, then same element types
            will be mapped to all instances.
            If ITERABLE, all values are STR, correct and allowed, then values
            get mapped to each instance seperately and accordingly.
            The default is ('CSP4', 'CSP8').
        elsets : MIXED: STR/ITERABLE, optional
            If STR, valid and allowed, the resuested elment set will be
            mapped to all instances.
            If ITERABLE, values are STR, valid and allowed, then values will
            be mapped to corresponding instances.
            The default is ('grains', 'grains').
        nsets : ITERABLE, optional
            Nodal sets to make. Used to impose boundary conditions.
            Options: 'x-', 'x+', 'y-', 'y+', 'gb', 'rp_random_none_10'
            - Option 'gb': grain boundary nodes. A 'gn' nodal set will be
            created for each grain. Naming will be based on parent grain name.
            - Option 'rp_random_10': representative points, 10 in number.
            These points are points fully inside the grain. None of these
            points would lie on the grain boundary of the grain. 'random'
            denotes random positioning of representative points. Following
            'none' indicates completely randomised. If in place of 'none', we
            have a number (INT/FLOAT), then this number specifies the minimum
            distance of seperation between all representative points inside
            the grain. The following number 10, necessiates that there should
            be 10 coordinate positions (as ITERABLES), if 'random' locationing.
            If these input data-format rules are not conformed to, then
            parameter sweep study stops.
            The default is ('x-', 'x+', 'y-', 'y+', ).
        optimize : MIXED: BOOL/ITERABLE, optional
            If BOOL, then this optimization flag will be mapped to all
            instances.
            If ITERABLE and of right size, then each optimization flag will be
            mapped to each instance accordingly.
            If ITERABLE and of wrong size, then parameter sweep study will
            stop.
            Options: True, False
            The default is (False, False).
        opt_par : MIXED: STR/ITERABLE, optional
            Specifies the element quality metric to optimize the mesh for.
            The default is ('min_angle', [40, 60], 'jacobian', [0.45, 0.6])
            'min_angle' is the minimum angle in the distribution of
            minimum angles of all finite elements. [40, 60] denotes the
            bounds of acceptance. Note that if objectives are not met,
            UPXO will enable recursive mesh refinement near places
            where these minimum angle falls outside the specified bounds.
            'jacobian': similar explanations apply.

        Returns
        -------
        None.


        """
        for n in self.N:
            self.gsi[n].set_uimesh(generate_mesh=generate_mesh,
                                   target_fe_software=target_fe_software,
                                   par_treatment=par_treatment,
                                   mesher=mesher,
                                   gb_conformities=gb_conformities,
                                   global_elsizes=global_elsizes,
                                   mesh_algos=mesh_algos,
                                   grain_internal_el_gradient=grain_internal_el_gradient,
                                   grain_internal_el_gradient_par=grain_internal_el_gradient_par,
                                   target_eltypes=target_eltypes,
                                   elsets=elsets, nsets=nsets,
                                   optimize=optimize, opt_par=opt_par,
                                   read_from_file=read_from_file,
                                   filename=filename
                                   )

    def save(self):
        '''
        Pickle the dataset
        '''
        pass

    def update_gmp(self):
        pass

    def update_qmp(self):
        pass

    def plot(self,
             defaults=False,
             docformat='pdf',
             xax='time',
             yax='area',
             zax='sim',
             xaxpar='',
             yaxpar='',
             zaxpar='',
             plot_type='best',
             ):
        '''
        if defaults=True:
            It generates a set of plots to enable getting a quick
            overview of the data. Plots will be exported to a
            PDF [ref 1].

            Following plots are made.
                * Grain structure plots of the final MC step of all sims.
                * Grain size evolution for all sims.

            [ref 1]: Currently, only PDF is supported. It is planned to enable
            writing data to MS Word, MS PPT, Google Doc, Google presentation,
            MS Excel and Google Spreadsheet.
        --------------------------------------------------
        xax: x-axis
            Options: time, alg, temperature
        yax: y-axis
            Options: time, {morph. parameter}
        zax: z-axis
            Options:
        xaxpar: Parameter for the x-axis
        yaxpar: Parameter for the y-axis
        zaxpar: Parameter for the z-axis
        NOTE:
            if all in (xaxpar, yaxpar, zaxpar) is provided,
            UPXO will override xax, yax, zax
        --------------------------------------------------
        plot_type: Type of visualization
        --------------------------------------------------
        '''
        pass

    @property
    def uigrid(self):
        """
        This is an imitation method. It imitates the instantiated class
        which makes uigrid attribute. Imitation is of the 1st grain
        structure instance. The reason for making an imitation is:
            * All grain structure instances WILL have grids having the
            same dimensionality, bounds and increments.
            * Hence, it is not needed to make a seperate grid availabale
            to the paramater sweep object.
            * Instead, when this property method is called, it just
            returns the uigrid of the first grain structure isntance, if
            it exists.
            * No big deal here.
        NOTE: This documentation applies also to uisim, uigsc, uimesh,
        uigeomrepr
        """
        _ = None
        if hasattr(self, 'gsi'):
            if self.gsi and type(self.gsi) == dict and len(self.gsi) >= 1:
                if hasattr(self.gsi[1], 'uigrid'):
                    print('ps.uigrid:: taken from: ps.gsi[1].uigrid')
                    _ = self.gsi[1].uigrid
                else:
                    print('ps.gsi[1].uigrid has not been set. Skipped.')
            else:
                print('Invalid ps.gsi. Skipped.')
        else:
            print('ps.gsi has not been set. Skipped.')
        return _

    @property
    def uisim(self):
        """
        This is an imitation method. Refer to ps.uigrid for more documentation.
        """
        _ = None
        if hasattr(self, 'gsi'):
            if self.gsi and type(self.gsi) == dict and len(self.gsi) >= 1:
                if hasattr(self.gsi[1], 'uisim'):
                    print('ps.uisim:: taken from: ps.gsi[1].uisim')
                    _ = self.gsi[1].uisim
                else:
                    print('ps.gsi[1].uisim has not been set. Skipped.')
            else:
                print('Invalid ps.gsi. Skipped.')
        else:
            print('ps.gsi has not been set. Skipped.')
        return _

    @property
    def uigsc(self):
        """
        This is an imitation method. Refer to ps.uigrid for more documentation.
        """
        _ = None
        if hasattr(self, 'gsi'):
            if self.gsi and type(self.gsi) == dict and len(self.gsi) >= 1:
                if hasattr(self.gsi[1], 'uigsc'):
                    print('ps.uigsc:: taken from: ps.gsi[1].uigsc')
                    _ = self.gsi[1].uigsc
                else:
                    print('ps.gsi[1].uigsc has not been set. Skipped.')
            else:
                print('Invalid ps.gsi. Skipped.')
        else:
            print('ps.gsi has not been set. Skipped.')
        return _

    @property
    def uimesh(self):
        """
        This is an imitation method. Refer to ps.uigrid for more documentation.
        """
        _ = None
        if hasattr(self, 'gsi'):
            if self.gsi and type(self.gsi) == dict and len(self.gsi) >= 1:
                if hasattr(self.gsi[1], 'uimesh'):
                    print('ps.uimesh:: taken from: ps.gsi[1].uimesh')
                    _ = self.gsi[1].uimesh
                else:
                    print('ps.gsi[1].uimesh has not been set. Skipped.')
            else:
                print('Invalid ps.gsi. Skipped.')
        else:
            print('ps.gsi has not been set. Skipped.')
        return _

    @property
    def uigeomrepr(self):
        """
        This is an imitation method. Refer to ps.uigrid for more documentation.
        """
        _ = None
        if hasattr(self, 'gsi'):
            if self.gsi and type(self.gsi) == dict and len(self.gsi) >= 1:
                if hasattr(self.gsi[1], 'uigeomrepr'):
                    print('ps.uigeomrepr:: taken from: ps.gsi[1].uigeomrepr')
                    _ = self.gsi[1].uigeomrepr
                else:
                    print('ps.gsi[1].uigeomrepr has not been set. Skipped.')
            else:
                print('Invalid ps.gsi. Skipped.')
        else:
            print('ps.gsi has not been set. Skipped.')
        return _

    def info_attributes(self, n, throw=False):
        if n not in self.N:
            print('N not set. Skipped')
        else:
            str1 = f"~~~ Parameter sweep dataset: {n} ~~~"
            print(str1 + '\n')
            # ----------------------------------------------------------------
            str2 = "(A. GRID):: mc simulation grid: "
            _ = ' '*12
            if hasattr(self.gsi[n], 'uigrid'):
                str2 += f"{self.gsi[n].uigrid.dim}D."
                str2 += f"{ self.gsi[n].uigrid.type[:2]}."
                strxa = f" x:({self.gsi[n].uigrid.xmin}"
                strxb = f",{self.gsi[n].uigrid.xmax}"
                strxc = f",{self.gsi[n].uigrid.xinc})"
                strx = strxa + strxb + strxc
                strya = f" y:({self.gsi[n].uigrid.ymin}"
                stryb = f",{self.gsi[n].uigrid.ymax}"
                stryc = f",{self.gsi[n].uigrid.yinc})"
                stry = strya + stryb + stryc
                if self.gsi[n].uigrid.dim == 2:
                    str2 += strx + stry
                elif self.gsi[n].uigrid.dim == 3:
                    strza = f" z:({self.gsi[n].uigrid.zmin}"
                    strzb = f",{self.gsi[n].uigrid.zmax}"
                    strzc = f",{self.gsi[n].uigrid.zinc})"
                    strz = strza + strzb + strzc
                    str2 += strx + stry + strz
            else:
                str2 += '\n' + _
                str2 += "Grid parameters not set.\n"
                str2 += _ + "Use set_param_grid(..)"
            print(str2)
            # ----------------------------------------------------------------
            str3 = "(B: SIMPAR):: mc simulation: "
            _ = ' '*14
            if hasattr(self.gsi[n], 'uisim'):
                str3 += f"{self.gsi[n].uisim.mcsteps}"
                str3 += f"  SOLVER: {self.gsi[n].uisim.solver}"
                if len(self.gsi[n].uisim.save_at_mcsteps) > 2:
                    _0 = self.gsi[n].uisim.save_at_mcsteps[0]
                    _1 = self.gsi[n].uisim.save_at_mcsteps[1]
                    str3 += '  IN-SIM SAVES: every ' + str(int(_1 - _0))
                    str3 += ' mcsteps'
                else:
                    str3 += '  IN-SIM SAVES at 0 mcstep'
            else:
                str3 += '\n' + _
                str3 += "Simulation parameters not set.\n"
                str3 += _ + "Use set_param_sim(..)"
            print(str3)
            # ----------------------------------------------------------------
            str4 = "(C: GSCPAR):: gs characterisation: "
            _ = ' '*14
            if hasattr(self.gsi[n], 'uigsc'):
                str4 += f"{self.gsi[n].uigsc.char_grains} \n"
                if self.gsi[n].uigsc.char_grains:
                    str4 += _ + "CHARACTERIZATION STAGE: "
                    str4 += f"{self.gsi[n].uigsc.char_stage}\n"
                    str4 += _ + "PARALLEL CHARACTERIZATION: "
                    str4 += f"{self.gsi[n].uigsc.parallel}"
            else:
                str4 += '\n' + _
                str4 += 'Grain str. characterisation parameters not set.\n'
                str4 += ' '*14 + "Use set_param_uigsc(..)"
            print(str4)
            # ----------------------------------------------------------------
            str5 = "(D: MESHPAR):: fe mesh: "
            _ = ' '*15
            if hasattr(self.gsi[n], 'uimesh'):
                str5 += f"{self.gsi[n].uimesh.generate_mesh}"
                if self.gsi[n].uimesh.generate_mesh:
                    str5 += "\n" + _
                    str5 += "TARGET SOFTWARE: "
                    str5 += f"{self.gsi[n].uimesh.target_fe_software}\n"
                    str5 += _ + "PARAMETER TREATMENT: "
                    str5 += f"{self.gsi[n].uimesh.par_treatment}\n"
                    str5 += _ + "MESHER: {self.gsi[n].uimesh.mesher}\n"
                    str5 += _ + "GB CONFORMITIES: Use: "
                    str5 += f"{self.gsi[n].uimesh.gb_conformities}\n"
                    str5 += _ + "GLOBAL ELEMENT SIZES: Use: "
                    str5 += "ps.gsi[n].uimesh.global_elsizes\n"
                    str5 += _ + "MESH ALGORITHMS: Use: "
                    str5 += "ps.gsi[n].uimesh.mesh_algos\n"
                    str5 += _ + "GRAIN INTERNAL ELEMENT GRADIENT "
                    str5 += "SPECIFICATION: Use:"
                    str5 += "ps.gsi[n].uimesh.grain_internal_el_gradient\n"
                    str5 += _ + "GRAIN INTERNAL ELEMENT GRADIENT VALUES: "
                    str5 += f"{self.gsi[n].uimesh.grain_internal_el_gradient_par}\n"
                    str5 += _ + "ELMENT TARGET TYPES: Use: "
                    str5 += f"{self.gsi[n].uimesh.target_eltypes}\n"
                    str5 += _ + "ELEMENT SETS: Use: "
                    str5 += f"{self.gsi[n].uimesh.elsets}\n"
                    str5 += _ + "NODAL SETS: "
                    str5 += f"Use: {self.gsi[n].uimesh.nsets}\n"
                    str5 += _ + "OPTIMIZE MESH: Use: "
                    str5 += f"{self.gsi[n].uimesh.optimize}\n"
                    str5 += _ + "MESH OPTIMIZATION PARAMETERS: Use: "
                    str5 += f"{self.gsi[n].uimesh.opt_par}\n"
            else:
                str5 += '\n' + _
                str5 += 'FE Mesh parameters not set.\n'
                str5 += _ + "Use set_param_uimesh(..)"
            print(str5)
            # ----------------------------------------------------------------
            str6 = "(E: GEOMREPR):: PXTAL geometric repr(s).: "
            _ = ' '*16
            if hasattr(self.gsi[n], 'uigeomrepr'):
                str6 += "MAKE MULPOINT OF GRAIN CENTROIDS: "
                str6 += f"{self.gsi[n].uigeomrepr.make_mp_grain_centoids}\n"
                str6 += _ + "MAKE MULPOINT OF GRAIN POINTS: "
                str6 += f"{self.gsi[n].uigeomrepr.make_mp_grain_points}\n"
                str6 += _ + "MAKE RING OBJECTS OF GRAIN BOUNDARIES: "
                str6 += f"{self.gsi[n].uigeomrepr.make_ring_grain_boundaries}\n"
                str6 += _ + "MAKE XTAL OBJECTS OF GRAINS: "
                str6 += f"{self.gsi[n].uigeomrepr.make_xtal_grain}\n"
                str6 += _ + "MAKE CONVEX HULL OBJECTS OF GRAINS: "
                str6 += f"{self.gsi[n].uigeomrepr.make_chull_grain}\n"
                str6 += _ + "MAKE GRAIN BOUNDARY ZONE: "
                str6 += f"{self.gsi[n].uigeomrepr.create_gbz}\n"
                str6 += _ + "MAKE 2D.VTGS PXTAL FROM GRAIN CENTROIDS: "
                str6 += "Use: PS.gsi[n].uigeomrepr.make_2dvtgs"
            else:
                str6 += 'Geoemtric repr. parameters not set.\n'
                str6 += _ + "Use set_param_uigeomrepr(..)"
            print(str6)
            # ----------------------------------------------------------------

    def generate_report(self,
                        docformat='pdf'):
        """


        Parameters
        ----------
        docformat : TYPE, optional
            DESCRIPTION. The default is 'pdf'.

        Returns
        -------
        None.

        """
        pass

    def to_excel(self):
        pass

    def model(self):
        pass

    @property
    def default_mcalg(self):
        return self.__default_mcalg__


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
                 mcsteps_incr=10,
                 input_dashboard='input_dashboard.xls',
                 ):
        self.study = study
        self.__info_message_display_level__ = 'detailed'
        if study == 'independent':
            self.load_uidata(input_dashboard)
            self.initiate()
        elif study in ('para_sweep'):
            # Parameters to be manually set
            pass

    def simulate_for_parameter_sweep(self):
        print(self)
        if hasattr(self, 'uigrid') and hasattr(self, 'uigsim'):
            self.initiate()


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

    def load_uidata(self, input_dashboard):
        # Load user input data
        from mcgs import _load_user_input_data_
        self.__ui = _load_user_input_data_(xl_fname=input_dashboard)
        # Extract gridding parameters
        from mcgs import _uidata_mcgs_gridding_definitions_
        self.uigrid = _uidata_mcgs_gridding_definitions_(self.__ui)
        # Exrtact simulation parametrs
        from mcgs import _uidata_mcgs_simpar_
        self.uisim = _uidata_mcgs_simpar_(self.__ui)
        # Extract parameters for grain structure analysis
        from mcgs import _uidata_mcgs_grain_structure_characterisation_
        self.uigsc = _uidata_mcgs_grain_structure_characterisation_(self.__ui)
        # Extract interval counts which trigger speciric operations
        from mcgs import _uidata_mcgs_intervals_
        self.uiint = _uidata_mcgs_intervals_(self.__ui)
        # Extract grain structrue property calculation parameters (bools)
        from mcgs import _uidata_mcgs_property_calc_
        self.uigsprop = _uidata_mcgs_property_calc_(self.__ui)
        # Extract grain geometric representation flags
        from mcgs import _uidata_mcgs_generate_geom_reprs_
        self.uigeorep = _uidata_mcgs_generate_geom_reprs_(self.__ui)
        # Extract the user input data on meshing
        from mcgs import _uidata_mcgs_mesh_
        self.uimesh = _uidata_mcgs_mesh_(self.__ui)

    def set_uigrid(self,
                   domain_size=None,
                   read_from_file=False,
                   filename=None
                   ):
        from mcgs import _manual_uidata_mcgs_gridding_definitions_
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
        from mcgs import _manual_uidata_mcgs_simpar_ as _muisimpar_
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
        from mcgs import _manual_uidata_mcgs_gsc_par_
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
        from mcgs import _manual_uidata_mcgs_generate_geom_reprs_ as gr
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
        from mcgs import _manual_uidata_mesh_
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
        from mcgs import artwork
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
                    elif self.mcpar.NL == 2:
                        NLM_bw = ones+arts*np.array([[+1., +1., +1., +1., +1.],
                                                     [+1., +1., +1., +1., +1.],
                                                     [+1., +1., +1., +1., +1.],
                                                     [+1., +1., +1., +1., +1.],
                                                     [+1., +1., +1., +1., +1.]
                                                     ])
                    else:
                        None
            elif self.mcpar.kineticity == "kinetic":
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
        from mcgs import grain_structure
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
            _a_ = np.random.random(size=self.mcpar_core.S)
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
        # Initiate the grain-structure data-structure
        self.add_gs_data_structure_template(m=0,
                                            dim=self.uigrid.dim,
                                            study=self.study
                                            )
        # START THE MONTE-CARLO SIMULATIONS
        if self.uigrid.dim == 2 and len(self.uisim.algo_hops) == 1:
            self.algo_hop = False
            self.start_algo2d_without_hops()
        elif self.uigrid.dim == 2 and len(self.uisim.algo_hops) > 1:
            if self.algo_hop:
                self.start_algo2d_with_hops()
            else:
                self.start_algo2d_without_hops()
        elif self.uigrid.dim == 3 and len(self.uisim.algo_hops) == 1:
            self.algo_hop = False
            print('I AM IN HERE -- 1')
            self.start_algo3d_without_hops()
        elif self.uigrid.dim == 3 and len(self.uisim.algo_hops) > 1:
            if self.algo_hop:
                self.start_algo3d_with_hops()
            else:
                self.start_algo3d_without_hops()

    def start_algo2d_without_hops(self):
        if self.uisim.mcalg == '200':
            print("Using ALG-200: SA's SL NL-1 TP1 C2 unweighted Q-Pott's model:")
            print('////////////////////////////////')
            self.mc_iterations_2d_alg200()
        elif self.uisim.mcalg == '201':
            print("Using ALG-200: SA's NL-1 weighted Q-Pott's model:")
            print('////////////////////////////////')
            self.mc_iterations_2d_alg201()
        elif self.uisim.mcalg == '201':
            print("Using SA's L0 modified Q-state Pott's model: ")
            print("    weighted (: ALG-200)")
            print('////////////////////////////////')
            self.mc_iterations_2d_alg202()

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

    def mc_iterations_2d_alg200(self):
        # Build the Non-Locality Matrix
        _a, _b, _c = self.build_NLM()  # Unpack 3 rows of NLM
        NLM_00, NLM_01, NLM_02 = _a  # Unpack 3 colms of 1st row
        NLM_10, NLM_11, NLM_12 = _b  # Unpack 3 colms of 2nd row
        NLM_20, NLM_21, NLM_22 = _c  # Unpack 3 colms of 3rd row
        # ---------------------------------------------
        # Begin modified Markov-Chain annealing iterations
        fully_annealed = False
        for m in range(self.uisim.mcsteps):
            if self.S.min() == self.S.max():
                print(f'Single crystal achieved at iteration {m}.')
                print('Stopping the simulation.')
                fully_annealed = True
                break
            else:
                for s0 in list(range(self.S.shape[0])):  # along axis 0
                    s00, s01, s02 = s0+0, s0+1, s0+2
                    for s1 in list(range(self.S.shape[1])):  # along axis 1
                        s10, s11, s12 = s1+0, s1+1, s1+2
                        ssub_00 = self.S[self.AIA0[s00, s10],
                                         self.AIA1[s00, s10]]
                        ssub_01 = self.S[self.AIA0[s01, s10],
                                         self.AIA1[s01, s10]]
                        ssub_02 = self.S[self.AIA0[s02, s10],
                                         self.AIA1[s02, s10]]
                        ssub_10 = self.S[self.AIA0[s00, s11],
                                         self.AIA1[s00, s11]]
                        ssub_11 = self.S[self.AIA0[s01, s11],
                                         self.AIA1[s01, s11]]
                        ssub_12 = self.S[self.AIA0[s02, s11],
                                         self.AIA1[s02, s11]]
                        ssub_20 = self.S[self.AIA0[s00, s12],
                                         self.AIA1[s00, s12]]
                        ssub_21 = self.S[self.AIA0[s01, s12],
                                         self.AIA1[s01, s12]]
                        ssub_22 = self.S[self.AIA0[s02, s12],
                                         self.AIA1[s02, s12]]
                        Neigh = [ssub_00, ssub_01, ssub_02,
                                 ssub_10, ssub_11, ssub_12,
                                 ssub_20, ssub_21, ssub_22]
                        if min(Neigh) != max(Neigh):
                            DelH1 = NLM_00*int(ssub_11 == ssub_00) + \
                                    NLM_01*int(ssub_11 == ssub_01) + \
                                    NLM_02*int(ssub_11 == ssub_02) + \
                                    NLM_10*int(ssub_11 == ssub_10) + \
                                    NLM_12*int(ssub_11 == ssub_12) + \
                                    NLM_20*int(ssub_11 == ssub_20) + \
                                    NLM_21*int(ssub_11 == ssub_21) + \
                                    NLM_22*int(ssub_11 == ssub_22)
                            # ---------------------------------------------
                            # If the sampling is to be selected without
                            # weightage to dominant neighbour state, then:
                            Neigh = set([x for x in Neigh if x != ssub_11])
                            # ---------------------------------------------
                            ssub_11_b = sample_rand(Neigh, 1)[0]
                            DelH2 = NLM_00*int(ssub_11_b == ssub_00) + \
                                    NLM_01*int(ssub_11_b == ssub_01) + \
                                    NLM_02*int(ssub_11_b == ssub_02) + \
                                    NLM_10*int(ssub_11_b == ssub_10) + \
                                    NLM_12*int(ssub_11_b == ssub_12) + \
                                    NLM_20*int(ssub_11_b == ssub_20) + \
                                    NLM_21*int(ssub_11_b == ssub_21) + \
                                    NLM_22*int(ssub_11_b == ssub_22)
                            if DelH2 >= DelH1:
                                self.S[s0, s1] = ssub_11_b
                            elif self.uisim.consider_boltzmann_probability:
                                if self.uisim.s_boltz_prob[int(ssub_11_b-1)] < rand.random():
                                    self.S[s0, s1] = ssub_11_b
            if self.display_messages:
                if m % self.uiint.mcint_promt_display == 0:
                    print("Temporal step no.", m)
            cond_1 = m % self.uiint.mcint_save_at_mcstep_interval == 0.0
            if cond_1 or fully_annealed:
                #self.tslices.append(m)
                # Create and add grain structure data structure template
                self.add_gs_data_structure_template(m=m,
                                                    dim=self.uigrid.dim)
                self.gs[m].s = deepcopy(self.S)
                if self.display_messages:
                    print(f'State Updated @ mc step {m}')
        return fully_annealed

    def NLM_elements(self):
        # Build the Non-Locality Matrix
        _a, _b, _c = self.build_NLM()  # Unpack 3 rows of NLM
        NLM_00, NLM_01, NLM_02 = _a  # Unpack 3 colms of 1st row
        NLM_10, NLM_11, NLM_12 = _b  # Unpack 3 colms of 2nd row
        NLM_20, NLM_21, NLM_22 = _c  # Unpack 3 colms of 3rd row
        return NLM_00, NLM_01, NLM_02, NLM_10, NLM_11, NLM_12, NLM_20, NLM_21, NLM_22

    def mc_iterations_2d_alg200_general(self,
                                        mrange,
                                        NLM_00, NLM_01, NLM_02,
                                        NLM_10, NLM_11, NLM_12,
                                        NLM_20, NLM_21, NLM_22
                                        ):
        # ---------------------------------------------
        # Begin modified Markov-Chain annealing iterations
        fully_annealed = False
        for m in range(self.uisim.mcsteps):
            if self.S.min() == self.S.max():
                print(f'Single crystal achieved at iteration {m}.')
                print('Stopping the simulation.')
                fully_annealed = True
                break
            else:
                for s0 in list(range(self.S.shape[0])):  # along axis 0
                    s00, s01, s02 = s0+0, s0+1, s0+2
                    for s1 in list(range(self.S.shape[1])):  # along axis 1
                        s10, s11, s12 = s1+0, s1+1, s1+2
                        ssub_00 = self.S[self.AIA0[s00, s10],
                                         self.AIA1[s00, s10]]
                        ssub_01 = self.S[self.AIA0[s01, s10],
                                         self.AIA1[s01, s10]]
                        ssub_02 = self.S[self.AIA0[s02, s10],
                                         self.AIA1[s02, s10]]
                        ssub_10 = self.S[self.AIA0[s00, s11],
                                         self.AIA1[s00, s11]]
                        ssub_11 = self.S[self.AIA0[s01, s11],
                                         self.AIA1[s01, s11]]
                        ssub_12 = self.S[self.AIA0[s02, s11],
                                         self.AIA1[s02, s11]]
                        ssub_20 = self.S[self.AIA0[s00, s12],
                                         self.AIA1[s00, s12]]
                        ssub_21 = self.S[self.AIA0[s01, s12],
                                         self.AIA1[s01, s12]]
                        ssub_22 = self.S[self.AIA0[s02, s12],
                                         self.AIA1[s02, s12]]
                        Neigh = [ssub_00, ssub_01, ssub_02,
                                 ssub_10, ssub_11, ssub_12,
                                 ssub_20, ssub_21, ssub_22]
                        if min(Neigh) != max(Neigh):
                            DelH1 = NLM_00*int(ssub_11 == ssub_00) + \
                                    NLM_01*int(ssub_11 == ssub_01) + \
                                    NLM_02*int(ssub_11 == ssub_02) + \
                                    NLM_10*int(ssub_11 == ssub_10) + \
                                    NLM_12*int(ssub_11 == ssub_12) + \
                                    NLM_20*int(ssub_11 == ssub_20) + \
                                    NLM_21*int(ssub_11 == ssub_21) + \
                                    NLM_22*int(ssub_11 == ssub_22)
                            # ---------------------------------------------
                            # If the sampling is to be selected without
                            # weightage to dominant neighbour state, then:
                            Neigh = set([x for x in Neigh if x != ssub_11])
                            # ---------------------------------------------
                            ssub_11_b = sample_rand(Neigh, 1)[0]
                            DelH2 = NLM_00*int(ssub_11_b == ssub_00) + \
                                    NLM_01*int(ssub_11_b == ssub_01) + \
                                    NLM_02*int(ssub_11_b == ssub_02) + \
                                    NLM_10*int(ssub_11_b == ssub_10) + \
                                    NLM_12*int(ssub_11_b == ssub_12) + \
                                    NLM_20*int(ssub_11_b == ssub_20) + \
                                    NLM_21*int(ssub_11_b == ssub_21) + \
                                    NLM_22*int(ssub_11_b == ssub_22)
                            if DelH2 >= DelH1:
                                self.S[s0, s1] = ssub_11_b
                            elif self.uisim.consider_boltzmann_probability:
                                if self.uisim.s_boltz_prob[int(ssub_11_b-1)] < rand.random():
                                    self.S[s0, s1] = ssub_11_b
            if self.display_messages:
                if m % self.uiint.mcint_promt_display == 0:
                    print("Temporal step no.", m)
            cond_1 = m % self.uiint.mcint_save_at_mcstep_interval == 0.0
            if cond_1 or fully_annealed:
                #self.tslices.append(m)
                # Create and add grain structure data structure template
                self.add_gs_data_structure_template(m=m,
                                                    dim=self.uigrid.dim)
                self.gs[m].s = deepcopy(self.S)
                if self.display_messages:
                    print(f'State Updated @ mc step {m}')
        return fully_annealed


    def mc_iterations_2d_alg201(self):
        """


        Returns
        -------
        None.

        """

        # Build the Non-Locality Matrix
        _a, _b, _c = self.build_NLM()  # Unpack 3 rows of NLM
        NLM_00, NLM_01, NLM_02 = _a  # Unpack 3 colms of 1st row
        NLM_10, NLM_11, NLM_12 = _b  # Unpack 3 colms of 2nd row
        NLM_20, NLM_21, NLM_22 = _c  # Unpack 3 colms of 3rd row
        # ---------------------------------------------
        # Begin modified Markov-Chain annealing iterations
        fully_annealed = False
        for m in range(self.uisim.mcsteps):
            if self.S.min() == self.S.max():
                print(f'Single crystal achieved at iteration {m}.')
                print('Stopping the simulation.')
                fully_annealed = True
                break
            else:
                for s0 in list(range(self.S.shape[0])):  # along axis 0
                    s00, s01, s02 = s0+0, s0+1, s0+2
                    for s1 in list(range(self.S.shape[1])):  # along axis 1
                        s10, s11, s12 = s1+0, s1+1, s1+2
                        ssub_00 = self.S[self.AIA0[s00, s10],
                                         self.AIA1[s00, s10]]
                        ssub_01 = self.S[self.AIA0[s01, s10],
                                         self.AIA1[s01, s10]]
                        ssub_02 = self.S[self.AIA0[s02, s10],
                                         self.AIA1[s02, s10]]
                        ssub_10 = self.S[self.AIA0[s00, s11],
                                         self.AIA1[s00, s11]]
                        ssub_11 = self.S[self.AIA0[s01, s11],
                                         self.AIA1[s01, s11]]
                        ssub_12 = self.S[self.AIA0[s02, s11],
                                         self.AIA1[s02, s11]]
                        ssub_20 = self.S[self.AIA0[s00, s12],
                                         self.AIA1[s00, s12]]
                        ssub_21 = self.S[self.AIA0[s01, s12],
                                         self.AIA1[s01, s12]]
                        ssub_22 = self.S[self.AIA0[s02, s12],
                                         self.AIA1[s02, s12]]
                        Neigh = [ssub_00, ssub_01, ssub_02,
                                 ssub_10, ssub_11, ssub_12,
                                 ssub_20, ssub_21, ssub_22]
                        if min(Neigh) != max(Neigh):
                            DelH1 = NLM_00*int(ssub_11 == ssub_00) + \
                                    NLM_01*int(ssub_11 == ssub_01) + \
                                    NLM_02*int(ssub_11 == ssub_02) + \
                                    NLM_10*int(ssub_11 == ssub_10) + \
                                    NLM_12*int(ssub_11 == ssub_12) + \
                                    NLM_20*int(ssub_11 == ssub_20) + \
                                    NLM_21*int(ssub_11 == ssub_21) + \
                                    NLM_22*int(ssub_11 == ssub_22)
                            # ---------------------------------------------
                            # If the sampling is to be selected with
                            # weightage to dominant neighbour state, then:
                            Neigh = [x for x in Neigh if x != ssub_11]
                            # ---------------------------------------------
                            ssub_11_b = sample_rand(Neigh, 1)[0]
                            DelH2 = NLM_00*int(ssub_11_b == ssub_00) + \
                                    NLM_01*int(ssub_11_b == ssub_01) + \
                                    NLM_02*int(ssub_11_b == ssub_02) + \
                                    NLM_10*int(ssub_11_b == ssub_10) + \
                                    NLM_12*int(ssub_11_b == ssub_12) + \
                                    NLM_20*int(ssub_11_b == ssub_20) + \
                                    NLM_21*int(ssub_11_b == ssub_21) + \
                                    NLM_22*int(ssub_11_b == ssub_22)
                            if DelH2 >= DelH1:
                                self.S[s0, s1] = ssub_11_b
                            elif self.uisim.consider_boltzmann_probability:
                                if self.uisim.s_boltz_prob[int(ssub_11_b-1)] < rand.random():
                                    self.S[s0, s1] = ssub_11_b
            if self.display_messages:
                if m % self.uiint.mcint_promt_display == 0:
                    print("Temporal step no.", m)
            cond_1 = m % self.uiint.mcint_save_at_mcstep_interval == 0.0
            if cond_1 or fully_annealed:
                # self.tslices.append(m)
                # Create and add grain structure data structure template
                self.add_gs_data_structure_template(m=m,
                                                    dim=self.uigrid.dim)
                self.gs[m].s = deepcopy(self.S)
                if self.display_messages:
                    print(f'State updated @ mc step {m}')

    def mc_iterations_2d_alg202(self):
        """


        Returns
        -------
        None.

        """

        # Begin modified Markov-Chain annealing iterations
        fully_annealed = False
        for m in range(self.uisim.mcsteps):
            if self.S.min() == self.S.max():
                print(f'Single crystal achieved at iteration {m}.')
                print('Stopping the simulation.')
                fully_annealed = True
            else:
                for s0 in list(range(self.S.shape[0])):  # along axis 0
                    s00, s01, s02 = s0+0, s0+1, s0+2
                    for s1 in list(range(self.S.shape[1])):  # along axis 1
                        s10, s11, s12 = s1+0, s1+1, s1+2
                        ssub_00 = self.S[self.AIA0[s00, s10],
                                         self.AIA1[s00, s10]]
                        ssub_01 = self.S[self.AIA0[s01, s10],
                                         self.AIA1[s01, s10]]
                        ssub_02 = self.S[self.AIA0[s02, s10],
                                         self.AIA1[s02, s10]]
                        ssub_10 = self.S[self.AIA0[s00, s11],
                                         self.AIA1[s00, s11]]
                        ssub_11 = self.S[self.AIA0[s01, s11],
                                         self.AIA1[s01, s11]]
                        ssub_12 = self.S[self.AIA0[s02, s11],
                                         self.AIA1[s02, s11]]
                        ssub_20 = self.S[self.AIA0[s00, s12],
                                         self.AIA1[s00, s12]]
                        ssub_21 = self.S[self.AIA0[s01, s12],
                                         self.AIA1[s01, s12]]
                        ssub_22 = self.S[self.AIA0[s02, s12],
                                         self.AIA1[s02, s12]]
                        Neigh = [ssub_00, ssub_01, ssub_02,
                                 ssub_10, ssub_11, ssub_12,
                                 ssub_20, ssub_21, ssub_22]
                        if min(Neigh) != max(Neigh):
                            DelH1 = int(ssub_11 == ssub_00) + \
                                    int(ssub_11 == ssub_01) + \
                                    int(ssub_11 == ssub_02) + \
                                    int(ssub_11 == ssub_10) + \
                                    int(ssub_11 == ssub_12) + \
                                    int(ssub_11 == ssub_20) + \
                                    int(ssub_11 == ssub_21) + \
                                    int(ssub_11 == ssub_22)
                            # ---------------------------------------------
                            # If the sampling is to be selected without
                            # weightage to dominant neighbour state, then:
                            Neigh = set([x for x in Neigh if x != ssub_11])
                            # If the sampling is to be selected with
                            # weightage to dominant neighbour state, then:
                            # Neigh = [x for x in Neigh if x != ssub_11]
                            # ---------------------------------------------
                            ssub_11_b = sample_rand(Neigh, 1)[0]
                            DelH2 = int(ssub_11_b == ssub_00) + \
                                    int(ssub_11_b == ssub_01) + \
                                    int(ssub_11_b == ssub_02) + \
                                    int(ssub_11_b == ssub_10) + \
                                    int(ssub_11_b == ssub_12) + \
                                    int(ssub_11_b == ssub_20) + \
                                    int(ssub_11_b == ssub_21) + \
                                    int(ssub_11_b == ssub_22)
                            if DelH2 >= DelH1:
                                self.S[s0, s1] = ssub_11_b
                            elif self.uisim.consider_boltzmann_probability:
                                if self.uisim.s_boltz_prob[int(ssub_11_b-1)] < rand.random():
                                    self.S[s0, s1] = ssub_11_b
            if self.display_messages:
                if m % self.uiint.mcint_promt_display == 0:
                    print("Temporal step no.", m)
            cond_1 = m % self.uiint.mcint_save_at_mcstep_interval == 0.0
            if cond_1 or fully_annealed:
                # self.tslices.append(m)
                # Create and add grain structure data structure template
                self.add_gs_data_structure_template(m=m,
                                                    dim=self.uigrid.dim)
                self.gs[m].s = deepcopy(self.S)
                if self.display_messages:
                    print(f'State updated @ mc step {m}')
                print('__________________________')

    def mc_iterations_3d_alg220(self):
        """
        Each of the initial set of iterations is to contain the following

        STEP 1: Do the regular iteration using any of the 200 series of
        algorithms

        STEP 2: Identify grains and their neighbours

        STEP 3: Identify single pixel grains

        STEP 4: Merge them with neighbours
        """
        pass

    def mc_iterations_3d_alg221(self):
        """
        Each of the initial set of iterations is to contain the following

        STEP 1: Do the regular iteration using any of the 200 series of
        algorithms

        STEP 2: Identify grains and their neighbours

        STEP 3: Identify single pixel grains and straight line grains

        STEP 4: Merge them with neighbours
        """
        pass

    def mc_iterations_3d_alg222(self):
        """
        Each of the initial set of iterations is to contain the following

        STEP 1: Do the regular iteration using any of the 200 series of
        algorithms

        STEP 2: Identify grains and their neighbours

        STEP 3: Identify small grains with areas less than 5% of mean

        STEP 4: Merge them with neighbours
        """
        pass

    def mc_iterations_3d_alg223(self):
        """
        Each of the initial set of iterations is to contain the following

        STEP 1: Do the regular iteration using any of the 200 series of
        algorithms

        STEP 2: Identify grains and their neighbours

        STEP 3: Identify small grains with areas less than 5% of maximum

        STEP 4: Merge them with neighbours
        """
        pass

    def mc_iterations_3d_alg224(self):
        """
        DESIGNED TO ACHIEVE: Bi-modal grain size distribution

        Each of the initial set of iterations is to contain the following

        STEP 1: Do the regular iteration using any of the 200 series of
        algorithms

        STEP 2: Identify grains and their neighbours

        STEP 3: Calculate state partitioned grain area distribution

        STEP 4: Identify small grains with areas less than P % of mean area
        for each state

        STEP 5: Select the state with the largest mean area: S_large

        STEP 6: Select the state with the smallest mean area: S_small

        STEP 7: Prepare a merger list comprising of two columns. First column
        is to have the global grain IDs of certain grains belonging to S_small.
        The second column is to have a list of global grain IDs of neighbouring
        grains belonging to S_large. If for a grain of S_small, no neighbouring
        grains of S_large exit, then cancel the merger operation for the
        current S_small grain. Iterate through all the remaining grains.

        STEP 8: Calculate the grain area distribution. Calculate the modality
        Calculate the shift in peaks.

        STEP 9: If the peak shift is in the direction of target peak, then
        accept the present iteration using a iteration transition probability.
        """
        pass

    def mc_iterations_3d_alg230(self):
        """
        230 SERIES OF ALGORITHMS

        This series belongs to the Cluster Monte-Carlo Algorithms. Some of them
        may be well known existing ones, some of them, developed by the lead
        developer, Dr. Sunil Anandatheertha.
        ------------------------------------------------------------
        ALGORITHM 230
        ------------------------------------------------------------
        DESIGNED TO ACHIEVE: Multi-modal grain structure
        DEVELOPED BY: Dr. Sunil Anandatheertha
        ------------------------------------------------------------
        Each of the initial set of iterations is to contain the following

        STEP 1: Do the regular iteration using any of the 200 series of
        algorithms

        STEP 2: Identify grains and their neighbours

        STEP 3: Identify a state at random: S1

        STEP 4: Identify all grains of S1. Build a single list having
        IDs of all grains which neighbour the grains of S1.

        STEP 5: Identify the most frequent state amongst these grains, which
        would be S1_neigh_mostfrequent

        STEP 6: Flip the states of all S1 grains to S1_neigh_mostfrequent

        STEP 7: Characterise the grain structure.
        """
        pass

    def mc_iterations_3d_alg300(self):
        """


        Returns
        -------
        None.

        """

        # _a, _b, _c, _d, _e = self.build_NLM()
        # NLM_00, NLM_01, NLM_02, NLM_03, NLM_04 = _a
        # NLM_10, NLM_11, NLM_12, NLM_13, NLM_14 = _b
        # NLM_20, NLM_21, NLM_22, NLM_23, NLM_24 = _c
        # NLM_30, NLM_31, NLM_32, NLM_33, NLM_34 = _d
        # NLM_40, NLM_41, NLM_42, NLM_43, NLM_44 = _e

        S_sz0, S_sz1, S_sz2 = self.S.shape[0], self.S.shape[1], self.S.shape[2]
        S_sz0_list, S_sz1_list = list(range(S_sz0)), list(range(S_sz1))
        S_sz2_list = list(range(S_sz2))
        # --------------------------------------
        if self.uisim.NL == 1:
            NLM_000 = self.NLM_nd[0, 0, 0]
            NLM_001 = self.NLM_nd[0, 0, 1]
            NLM_002 = self.NLM_nd[0, 0, 2]
            NLM_010 = self.NLM_nd[0, 1, 0]
            NLM_011 = self.NLM_nd[0, 1, 1]
            NLM_012 = self.NLM_nd[0, 1, 2]
            NLM_020 = self.NLM_nd[0, 2, 0]
            NLM_021 = self.NLM_nd[0, 2, 1]
            NLM_022 = self.NLM_nd[0, 2, 2]

            NLM_100 = self.NLM_nd[1, 0, 0]
            NLM_101 = self.NLM_nd[1, 0, 1]
            NLM_102 = self.NLM_nd[1, 0, 2]
            NLM_110 = self.NLM_nd[1, 1, 0]
            # NLM_111 = self.NLM_nd[1, 1, 1]
            NLM_112 = self.NLM_nd[1, 1, 2]
            NLM_120 = self.NLM_nd[1, 2, 0]
            NLM_121 = self.NLM_nd[1, 2, 1]
            NLM_122 = self.NLM_nd[1, 2, 2]

            NLM_200 = self.NLM_nd[2, 0, 0]
            NLM_201 = self.NLM_nd[2, 0, 1]
            NLM_202 = self.NLM_nd[2, 0, 2]
            NLM_210 = self.NLM_nd[2, 1, 0]
            NLM_211 = self.NLM_nd[2, 1, 1]
            NLM_212 = self.NLM_nd[2, 1, 2]
            NLM_220 = self.NLM_nd[2, 2, 0]
            NLM_221 = self.NLM_nd[2, 2, 1]
            NLM_222 = self.NLM_nd[2, 2, 2]
        # --------------------------------------
        xinda = self.xinda
        yinda = self.yinda
        zinda = self.zinda
        # sa = self.sa
        # --------------------------------------
        fully_annealed = False
        # --------------------------------------
        # Begin modified Markov-Chain iterations
        for m in range(self.uisim.mcsteps):
            if self.S.min() == self.S.max():
                print('Domain fully annealed at iteration', m)
                break
            else:
                for P in S_sz2_list:  # along axis 2, along plane
                    for R in S_sz0_list:  # along axis 1, along row
                        for C in S_sz1_list:  # along axis 0, along column
                            ssub_000 = self.S[zinda[P+0, R+0, C+0],
                                              yinda[P+0, R+0, C+0],
                                              xinda[P+0, R+0, C+0]]
                            ssub_001 = self.S[zinda[P+0, R+0, C+1],
                                              yinda[P+0, R+0, C+1],
                                              xinda[P+0, R+0, C+1]]
                            ssub_002 = self.S[zinda[P+0, R+0, C+2],
                                              yinda[P+0, R+0, C+2],
                                              xinda[P+0, R+0, C+2]]
                            ssub_010 = self.S[zinda[P+0, R+1, C+0],
                                              yinda[P+0, R+1, C+0],
                                              xinda[P+0, R+1, C+0]]
                            ssub_011 = self.S[zinda[P+0, R+1, C+1],
                                              yinda[P+0, R+1, C+1],
                                              xinda[P+0, R+1, C+1]]
                            ssub_012 = self.S[zinda[P+0, R+1, C+2],
                                              yinda[P+0, R+1, C+2],
                                              xinda[P+0, R+1, C+2]]
                            ssub_020 = self.S[zinda[P+0, R+2, C+0],
                                              yinda[P+0, R+2, C+0],
                                              xinda[P+0, R+2, C+0]]
                            ssub_021 = self.S[zinda[P+0, R+2, C+1],
                                              yinda[P+0, R+2, C+1],
                                              xinda[P+0, R+2, C+1]]
                            ssub_022 = self.S[zinda[P+0, R+2, C+2],
                                              yinda[P+0, R+2, C+2],
                                              xinda[P+0, R+2, C+2]]

                            ssub_100 = self.S[zinda[P+1, R+0, C+0],
                                              yinda[P+1, R+0, C+0],
                                              xinda[P+1, R+0, C+0]]
                            ssub_101 = self.S[zinda[P+1, R+0, C+1],
                                              yinda[P+1, R+0, C+1],
                                              xinda[P+1, R+0, C+1]]
                            ssub_102 = self.S[zinda[P+1, R+0, C+2],
                                              yinda[P+1, R+0, C+2],
                                              xinda[P+1, R+0, C+2]]
                            ssub_110 = self.S[zinda[P+1, R+1, C+0],
                                              yinda[P+1, R+1, C+0],
                                              xinda[P+1, R+1, C+0]]
                            ssub_111_a = self.S[zinda[P+1, R+1, C+1],
                                                yinda[P+1, R+1, C+1],
                                                xinda[P+1, R+1, C+1]]
                            ssub_112 = self.S[zinda[P+1, R+1, C+2],
                                              yinda[P+1, R+1, C+2],
                                              xinda[P+1, R+1, C+2]]
                            ssub_120 = self.S[zinda[P+1, R+2, C+0],
                                              yinda[P+1, R+2, C+0],
                                              xinda[P+1, R+2, C+0]]
                            ssub_121 = self.S[zinda[P+1, R+2, C+1],
                                              yinda[P+1, R+2, C+1],
                                              xinda[P+1, R+2, C+1]]
                            ssub_122 = self.S[zinda[P+1, R+2, C+2],
                                              yinda[P+1, R+2, C+2],
                                              xinda[P+1, R+2, C+2]]

                            ssub_200 = self.S[zinda[P+2, R+0, C+0],
                                              yinda[P+2, R+0, C+0],
                                              xinda[P+2, R+0, C+0]]
                            ssub_201 = self.S[zinda[P+2, R+0, C+1],
                                              yinda[P+2, R+0, C+1],
                                              xinda[P+2, R+0, C+1]]
                            ssub_202 = self.S[zinda[P+2, R+0, C+2],
                                              yinda[P+2, R+0, C+2],
                                              xinda[P+2, R+0, C+2]]
                            ssub_210 = self.S[zinda[P+2, R+1, C+0],
                                              yinda[P+2, R+1, C+0],
                                              xinda[P+2, R+1, C+0]]
                            ssub_211 = self.S[zinda[P+2, R+1, C+1],
                                              yinda[P+2, R+1, C+1],
                                              xinda[P+2, R+1, C+1]]
                            ssub_212 = self.S[zinda[P+2, R+1, C+2],
                                              yinda[P+2, R+1, C+2],
                                              xinda[P+2, R+1, C+2]]
                            ssub_220 = self.S[zinda[P+2, R+2, C+0],
                                              yinda[P+2, R+2, C+0],
                                              xinda[P+2, R+2, C+0]]
                            ssub_221 = self.S[zinda[P+2, R+2, C+1],
                                              yinda[P+2, R+2, C+1],
                                              xinda[P+2, R+2, C+1]]
                            ssub_222 = self.S[zinda[P+2, R+2, C+2],
                                              yinda[P+2, R+2, C+2],
                                              xinda[P+2, R+2, C+2]]

                            Neigh = [ssub_000, ssub_001, ssub_002,
                                     ssub_010, ssub_011, ssub_012,
                                     ssub_020, ssub_021, ssub_022,
                                     ssub_100, ssub_101, ssub_102,
                                     ssub_110, ssub_111_a, ssub_112,
                                     ssub_120, ssub_121, ssub_122,
                                     ssub_200, ssub_201, ssub_202,
                                     ssub_210, ssub_211, ssub_212,
                                     ssub_220, ssub_221, ssub_222]
                            if min(Neigh) != max(Neigh):
                                DelH1 = NLM_000 * float(ssub_111_a == ssub_000) + \
    								NLM_001 * float(ssub_111_a == ssub_001) + \
    								NLM_002 * float(ssub_111_a == ssub_002) + \
    								NLM_010 * float(ssub_111_a == ssub_010) + \
    								NLM_011 * float(ssub_111_a == ssub_011) + \
    								NLM_012 * float(ssub_111_a == ssub_012) + \
    								NLM_020 * float(ssub_111_a == ssub_020) + \
    								NLM_021 * float(ssub_111_a == ssub_021) + \
    								NLM_022 * float(ssub_111_a == ssub_022) + \
    								NLM_100 * float(ssub_111_a == ssub_100) + \
    								NLM_101 * float(ssub_111_a == ssub_101) + \
    								NLM_102 * float(ssub_111_a == ssub_102) + \
    								NLM_110 * float(ssub_111_a == ssub_110) + \
    								NLM_112 * float(ssub_111_a == ssub_112) + \
    								NLM_120 * float(ssub_111_a == ssub_120) + \
    								NLM_121 * float(ssub_111_a == ssub_121) + \
    								NLM_122 * float(ssub_111_a == ssub_122) + \
    								NLM_200 * float(ssub_111_a == ssub_200) + \
    								NLM_201 * float(ssub_111_a == ssub_201) + \
    								NLM_202 * float(ssub_111_a == ssub_202) + \
    								NLM_210 * float(ssub_111_a == ssub_210) + \
    								NLM_211 * float(ssub_111_a == ssub_211) + \
    								NLM_212 * float(ssub_111_a == ssub_212) + \
    								NLM_220 * float(ssub_111_a == ssub_220) + \
    								NLM_221 * float(ssub_111_a == ssub_221) + \
    								NLM_222 * float(ssub_111_a == ssub_222)
    							# ---------------------------------------------
    							# If the sampling is to be selected without weightage to dominant neighbour state, then:
                                Neigh = list(set([x for x in Neigh if x != ssub_111_a]))
    							# If the sampling is to be selected with weightage to dominant neighbour state, then:
    							# Neigh = [x for x in Neigh if x != ssub_111_a]
    							# ---------------------------------------------
                                ssub_111_b = sample_rand(Neigh, 1)[0]
                                DelH2 = NLM_000 * float(ssub_111_b == ssub_000) + \
    								NLM_001 * float(ssub_111_b == ssub_001) + \
    								NLM_002 * float(ssub_111_b == ssub_002) + \
    								NLM_010 * float(ssub_111_b == ssub_010) + \
    								NLM_011 * float(ssub_111_b == ssub_011) + \
    								NLM_012 * float(ssub_111_b == ssub_012) + \
    								NLM_020 * float(ssub_111_b == ssub_020) + \
    								NLM_021 * float(ssub_111_b == ssub_021) + \
    								NLM_022 * float(ssub_111_b == ssub_022) + \
    								NLM_100 * float(ssub_111_b == ssub_100) + \
    								NLM_101 * float(ssub_111_b == ssub_101) + \
    								NLM_102 * float(ssub_111_b == ssub_102) + \
    								NLM_110 * float(ssub_111_b == ssub_110) + \
    								NLM_112 * float(ssub_111_b == ssub_112) + \
    								NLM_120 * float(ssub_111_b == ssub_120) + \
    								NLM_121 * float(ssub_111_b == ssub_121) + \
    								NLM_122 * float(ssub_111_b == ssub_122) + \
    								NLM_200 * float(ssub_111_b == ssub_200) + \
    								NLM_201 * float(ssub_111_b == ssub_201) + \
    								NLM_202 * float(ssub_111_b == ssub_202) + \
    								NLM_210 * float(ssub_111_b == ssub_210) + \
    								NLM_211 * float(ssub_111_b == ssub_211) + \
    								NLM_212 * float(ssub_111_b == ssub_212) + \
    								NLM_220 * float(ssub_111_b == ssub_220) + \
    								NLM_221 * float(ssub_111_b == ssub_221) + \
    								NLM_222 * float(ssub_111_b == ssub_222)
                                if DelH2 >= DelH1:
    								# S[P, R, C] = ssub_111_b
                                    _p = zinda[P+1, R+1, C+1]
                                    _r = yinda[P+1, R+1, C+1]
                                    _c = xinda[P+1, R+1, C+1]
                                    self.S[_p, _r, _c] = ssub_111_b
                                elif self.uisim.consider_boltzmann_probability:
                                    if self.uisim.s_boltz_prob[int(ssub_111_b-1)] < rand.random():
                                        self.S[P, R, C] = ssub_111_b
            if self.display_messages:
                if m % self.uiint.mcint_promt_display == 0:
                    print("Annealing step no.", m)
            cond_1 = m % self.uiint.mcint_save_at_mcstep_interval == 0.0
            if cond_1 or fully_annealed:
                # self.tslices.append(m)
                # Create and add grain structure data structure template
                self.add_gs_data_structure_template(m=m,
                                                    dim=self.uigrid.dim)
                self.gs[m].s = deepcopy(self.S)
                if self.display_messages:
                    print(f'State updated @ mc step {m}')
                print('__________________________')

    def mc_iterations_3d_alg310(self):
        """


        Returns
        -------
        None.

        """

        # _a, _b, _c, _d, _e = self.build_NLM()
        # NLM_00, NLM_01, NLM_02, NLM_03, NLM_04 = _a
        # NLM_10, NLM_11, NLM_12, NLM_13, NLM_14 = _b
        # NLM_20, NLM_21, NLM_22, NLM_23, NLM_24 = _c
        # NLM_30, NLM_31, NLM_32, NLM_33, NLM_34 = _d
        # NLM_40, NLM_41, NLM_42, NLM_43, NLM_44 = _e

        S_sz0, S_sz1, S_sz2 = self.S.shape[0], self.S.shape[1], self.S.shape[2]
        S_sz0_list, S_sz1_list = list(range(S_sz0)), list(range(S_sz1))
        S_sz2_list = list(range(S_sz2))
        # --------------------------------------
        xinda = self.xinda
        yinda = self.yinda
        zinda = self.zinda
        # sa = self.sa
        # --------------------------------------
        fully_annealed = False
        # --------------------------------------
        # Begin modified Markov-Chain iterations
        for m in range(self.uisim.mcsteps):
            if self.S.min() == self.S.max():
                print('Domain fully annealed at iteration', m)
                break
            else:
                for P in S_sz2_list:  # along axis 2, along plane
                    for R in S_sz0_list:  # along axis 1, along row
                        for C in S_sz1_list:  # along axis 0, along column
                            ssub_000 = self.S[zinda[P+0, R+0, C+0],
                                              yinda[P+0, R+0, C+0],
                                              xinda[P+0, R+0, C+0]]
                            ssub_001 = self.S[zinda[P+0, R+0, C+1],
                                              yinda[P+0, R+0, C+1],
                                              xinda[P+0, R+0, C+1]]
                            ssub_002 = self.S[zinda[P+0, R+0, C+2],
                                              yinda[P+0, R+0, C+2],
                                              xinda[P+0, R+0, C+2]]
                            ssub_010 = self.S[zinda[P+0, R+1, C+0],
                                              yinda[P+0, R+1, C+0],
                                              xinda[P+0, R+1, C+0]]
                            ssub_011 = self.S[zinda[P+0, R+1, C+1],
                                              yinda[P+0, R+1, C+1],
                                              xinda[P+0, R+1, C+1]]
                            ssub_012 = self.S[zinda[P+0, R+1, C+2],
                                              yinda[P+0, R+1, C+2],
                                              xinda[P+0, R+1, C+2]]
                            ssub_020 = self.S[zinda[P+0, R+2, C+0],
                                              yinda[P+0, R+2, C+0],
                                              xinda[P+0, R+2, C+0]]
                            ssub_021 = self.S[zinda[P+0, R+2, C+1],
                                              yinda[P+0, R+2, C+1],
                                              xinda[P+0, R+2, C+1]]
                            ssub_022 = self.S[zinda[P+0, R+2, C+2],
                                              yinda[P+0, R+2, C+2],
                                              xinda[P+0, R+2, C+2]]

                            ssub_100 = self.S[zinda[P+1, R+0, C+0],
                                              yinda[P+1, R+0, C+0],
                                              xinda[P+1, R+0, C+0]]
                            ssub_101 = self.S[zinda[P+1, R+0, C+1],
                                              yinda[P+1, R+0, C+1],
                                              xinda[P+1, R+0, C+1]]
                            ssub_102 = self.S[zinda[P+1, R+0, C+2],
                                              yinda[P+1, R+0, C+2],
                                              xinda[P+1, R+0, C+2]]
                            ssub_110 = self.S[zinda[P+1, R+1, C+0],
                                              yinda[P+1, R+1, C+0],
                                              xinda[P+1, R+1, C+0]]
                            ssub_111_a = self.S[zinda[P+1, R+1, C+1],
                                                yinda[P+1, R+1, C+1],
                                                xinda[P+1, R+1, C+1]]
                            ssub_112 = self.S[zinda[P+1, R+1, C+2],
                                              yinda[P+1, R+1, C+2],
                                              xinda[P+1, R+1, C+2]]
                            ssub_120 = self.S[zinda[P+1, R+2, C+0],
                                              yinda[P+1, R+2, C+0],
                                              xinda[P+1, R+2, C+0]]
                            ssub_121 = self.S[zinda[P+1, R+2, C+1],
                                              yinda[P+1, R+2, C+1],
                                              xinda[P+1, R+2, C+1]]
                            ssub_122 = self.S[zinda[P+1, R+2, C+2],
                                              yinda[P+1, R+2, C+2],
                                              xinda[P+1, R+2, C+2]]

                            ssub_200 = self.S[zinda[P+2, R+0, C+0],
                                              yinda[P+2, R+0, C+0],
                                              xinda[P+2, R+0, C+0]]
                            ssub_201 = self.S[zinda[P+2, R+0, C+1],
                                              yinda[P+2, R+0, C+1],
                                              xinda[P+2, R+0, C+1]]
                            ssub_202 = self.S[zinda[P+2, R+0, C+2],
                                              yinda[P+2, R+0, C+2],
                                              xinda[P+2, R+0, C+2]]
                            ssub_210 = self.S[zinda[P+2, R+1, C+0],
                                              yinda[P+2, R+1, C+0],
                                              xinda[P+2, R+1, C+0]]
                            ssub_211 = self.S[zinda[P+2, R+1, C+1],
                                              yinda[P+2, R+1, C+1],
                                              xinda[P+2, R+1, C+1]]
                            ssub_212 = self.S[zinda[P+2, R+1, C+2],
                                              yinda[P+2, R+1, C+2],
                                              xinda[P+2, R+1, C+2]]
                            ssub_220 = self.S[zinda[P+2, R+2, C+0],
                                              yinda[P+2, R+2, C+0],
                                              xinda[P+2, R+2, C+0]]
                            ssub_221 = self.S[zinda[P+2, R+2, C+1],
                                              yinda[P+2, R+2, C+1],
                                              xinda[P+2, R+2, C+1]]
                            ssub_222 = self.S[zinda[P+2, R+2, C+2],
                                              yinda[P+2, R+2, C+2],
                                              xinda[P+2, R+2, C+2]]

                            Neigh = [ssub_000, ssub_001, ssub_002,
                                     ssub_010, ssub_011, ssub_012,
                                     ssub_020, ssub_021, ssub_022,
                                     ssub_100, ssub_101, ssub_102,
                                     ssub_110, ssub_111_a, ssub_112,
                                     ssub_120, ssub_121, ssub_122,
                                     ssub_200, ssub_201, ssub_202,
                                     ssub_210, ssub_211, ssub_212,
                                     ssub_220, ssub_221, ssub_222]
                            if min(Neigh) != max(Neigh):
                                DelH1 = float(ssub_111_a == ssub_000) + \
    								float(ssub_111_a == ssub_001) + \
    								float(ssub_111_a == ssub_002) + \
    								float(ssub_111_a == ssub_010) + \
    								float(ssub_111_a == ssub_011) + \
    								float(ssub_111_a == ssub_012) + \
    								float(ssub_111_a == ssub_020) + \
    								float(ssub_111_a == ssub_021) + \
    								float(ssub_111_a == ssub_022) + \
    								float(ssub_111_a == ssub_100) + \
    								float(ssub_111_a == ssub_101) + \
    								float(ssub_111_a == ssub_102) + \
    								float(ssub_111_a == ssub_110) + \
    								float(ssub_111_a == ssub_112) + \
    								float(ssub_111_a == ssub_120) + \
    								float(ssub_111_a == ssub_121) + \
    								float(ssub_111_a == ssub_122) + \
    								float(ssub_111_a == ssub_200) + \
    								float(ssub_111_a == ssub_201) + \
    								float(ssub_111_a == ssub_202) + \
    								float(ssub_111_a == ssub_210) + \
    								float(ssub_111_a == ssub_211) + \
    								float(ssub_111_a == ssub_212) + \
    								float(ssub_111_a == ssub_220) + \
    								float(ssub_111_a == ssub_221) + \
    								float(ssub_111_a == ssub_222)
    							# ---------------------------------------------
    							# If the sampling is to be selected without weightage to dominant neighbour state, then:
                                Neigh = list(set([x for x in Neigh if x != ssub_111_a]))
    							# If the sampling is to be selected with weightage to dominant neighbour state, then:
    							# Neigh = [x for x in Neigh if x != ssub_111_a]
    							# ---------------------------------------------
                                ssub_111_b = sample_rand(Neigh, 1)[0]
                                DelH2 = float(ssub_111_b == ssub_000) + \
    								float(ssub_111_b == ssub_001) + \
    								float(ssub_111_b == ssub_002) + \
    								float(ssub_111_b == ssub_010) + \
    								float(ssub_111_b == ssub_011) + \
    								float(ssub_111_b == ssub_012) + \
    								float(ssub_111_b == ssub_020) + \
    								float(ssub_111_b == ssub_021) + \
    								float(ssub_111_b == ssub_022) + \
    								float(ssub_111_b == ssub_100) + \
    								float(ssub_111_b == ssub_101) + \
    								float(ssub_111_b == ssub_102) + \
    								float(ssub_111_b == ssub_110) + \
    								float(ssub_111_b == ssub_112) + \
    								float(ssub_111_b == ssub_120) + \
    								float(ssub_111_b == ssub_121) + \
    								float(ssub_111_b == ssub_122) + \
    								float(ssub_111_b == ssub_200) + \
    								float(ssub_111_b == ssub_201) + \
    								float(ssub_111_b == ssub_202) + \
    								float(ssub_111_b == ssub_210) + \
    								float(ssub_111_b == ssub_211) + \
    								float(ssub_111_b == ssub_212) + \
    								float(ssub_111_b == ssub_220) + \
    								float(ssub_111_b == ssub_221) + \
    								float(ssub_111_b == ssub_222)
                                if DelH2 >= DelH1:
    								# S[P, R, C] = ssub_111_b
                                    _p = zinda[P+1, R+1, C+1]
                                    _r = yinda[P+1, R+1, C+1]
                                    _c = xinda[P+1, R+1, C+1]
                                    self.S[_p, _r, _c] = ssub_111_b
                                elif self.uisim.consider_boltzmann_probability:
                                    if self.uisim.s_boltz_prob[int(ssub_111_b-1)] < rand.random():
                                        self.S[P, R, C] = ssub_111_b
                    if self.display_messages:
                        if m % self.uiint.mcint_promt_display == 0:
                            print("Annealing step no.", m, "Kernel core in slice. ", P, "/", S_sz2)
            cond_1 = m % self.uiint.mcint_save_at_mcstep_interval == 0.0
            if cond_1 or fully_annealed:

                # self.tslices.append(m)
                # Create and add grain structure data structure template
                self.add_gs_data_structure_template(m=m,
                                                    dim=self.uigrid.dim)
                self.gs[m].s = deepcopy(self.S)
                self.gs[m]
                if self.display_messages:
                    print(f'State updated @ mc step {m}')
                print('__________________________')

class grain_structure():
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
                 'study'
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
    EPS = 0.000000000001
    __maxGridSizeToIgnoreStoringGrids = 25**3

    def __init__(self,
                 study='idependent',
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
        self.study = study
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
        from mcgs import grain2d
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

    def export_vtk3d(self, filePath, fileName, addSuffix=True):
        """
        Export data to .vtk format.

        Parameters
        ----------
        filePath :str
            DESCRIPTION.
        fileName : str
            DESCRIPTION.
        addSuffix : bool, optional
            If True, the suffix '_upxo' will be added at end. This is advised
            to enable distinguising any .vtk files you may create using
            applications such as Dream3D etc. The default is True.

        Returns
        -------
        None.

        """
        grid = pv.StructuredGrid(xgr, ygr, zgr)
        grid['values'] = state_matrix.flatten(order='F')  # Flatten in Fortran order to match the VTK's indexing
        grid['gid_values'] = gid_matrix.flatten(order='F')  # Flatten in Fortran order to match the VTK's indexing
        # Save the grid to a VTK file
        grid.save('yellowSubmarine2b.vtk')

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

# --------------------------------------------------------------------------
from abc import ABCMeta, abstractstaticmethod

class partition2d(metaclass=ABCMeta):
    pass


# --------------------------------------------------------------------------
class grain2d():
    """
    brec: bounding rectangle of the grain
    bbox: bounding box
    bbox_ex: bbox expanded (by unit pixel along permitted direction)

    gid: This grain ID. Same as the lgi number of this grain
    gbid: Grain boundary ID
    gind: Indices of grain pixel in the parent state matrix
    gbind: Indices of grain boundary pixel in the parent state matrix
    gbsegs_pre: Partially processed grain boundary segments

    s: State value of the grain
    sn: Number (n) of this grain in the list of 's' stated grains

    neigh: list of neighbouring grain ids --> User calculated

    grep_cen: Geometric repr: centroid point --> User calculated
    grep_mp: Geometric repr: UPXO multi-point object of grain boundary --> User calculated
    grep_me: Geometric repr: UPXO multi-edge object of grain boundary --> User calculated
    grep_ring: Geometric representation: UPXO ring object of grain boundary --> User calculated

    feat_bbox: Feature: Bounding box <<-- STORED UPON CREATION
    feat_islands: Feature: Island grains
    feat_gc: Feature: Grain core
    feat_gbz: Feature: Grain boundary zone
    feat_ell: Feature: Ellipse --> User calculated
    feat_rec: Feature: Rectangle --> User calculated

    x_xsys: Crystal system: 'fcc', 'bcc', 'hcp': USER INPUT / EXTRACTED
    x_phase: Phase ID: USER INPUT / EXTRACTED
    x_mean: Mean crystallographic orientation: Bunge's Euler angle: USER INPUT / EXTRACTED

    gbvert: Grain boundary vertices <<-- STORED UPON CREATION
    gbseg: Grain boundary segments --> User calculated

    _xgr_min_: Minimum value of the xgr of the parent grain structure
    _xgr_max_: Maximum value of the xgr of the parent grain structure
    _xgr_incr_: Increment value of the xgr of the parent grain structure

    _ygr_min_: Minimum value of the ygr of the parent grain structure
    _ygr_max_: Maximum value of the ygr of the parent grain structure
    _xgr_incr_: Increment value of the ygr of the parent grain structure
    """
    __slots__ = ('loc',
                 'position',
                 'coords',
                 'gbloc',
                 'brec',
                 'bbox_bounds',
                 'bbox_ex_bounds',
                 'bbox',
                 'bbox_ex',
                 'px_area',
                 'skprop',
                 'gid',
                 'gbid',
                 'gind',
                 'gbind',
                 'gbsegs_pre',
                 's',
                 'sn',
                 'neigh',
                 'grep_cen',
                 'grep_mp',
                 'grep_me',
                 'grep_ring',
                 'feat_bbox',
                 'feat_islands',
                 'feat_gc',
                 'feat_gbz',
                 'feat_ell',
                 'feat_rec',
                 'x_xsys',
                 'x_phase',
                 'x_mean',
                 'gbvert',
                 'gbseg',
                 'xmin',
                 'xmax',
                 'ymin',
                 'ymax'
                 )

    def __init__(self):
        self.loc = None
        self.gbloc = None
        self.brec = None
        self.bbox_bounds = None
        self.bbox_ex_bounds = None
        self.bbox = None
        self.bbox_ex = None
        self.px_area = None
        self.skprop = None
        self.gid = None
        self.gind = None
        self.s = None
        self.sn = None
        self.position = None
        self.coords = None
        self.xmin, self.xmax = None, None
        self.ymin, self.ymax = None, None

    def __str__(self):
        return f's{self.s}, sn{self.sn}'

    def __len__(self):
        return len(self.loc)

    def __mul__(self, k):
        self.px_area *= k

    def __att__(self):
        return gops.att(self)

    def make_prop(self, generator, skprop=True):
        if skprop:
            self.skprop = generator(self.bbox_ex, cache=False)[0]

    @property
    def centroid(self):
        coords = self.coords.T
        return (coords[0].mean(), coords[1].mean())

    def plot(self, hold_on=False):
        if not hold_on:
            plt.figure()
        plt.imshow(self.bbox_ex)
        plt.title(f"Grain plot \n Grain: {self.gid}. Area: {round(self.skprop.area*100)/100} $\mu m^2$")
        if not hold_on:
            plt.xlabel(r"X-axis, $\mu m$", fontsize=12)
            plt.ylabel(r"Y-axis, $\mu m$", fontsize=12)
            plt.show()

    def plotgb(self, hold_on=False):
        z = np.zeros_like(self.bbox_ex)
        rmin = self.bbox_ex_bounds[0]
        cmin = self.bbox_ex_bounds[2]
        for rc in self.gbloc:
            z[rc[0]-rmin, rc[1]-cmin] = 1
        # ------------------------------------
        if not hold_on:
            plt.figure()
        plt.imshow(z)
        if not hold_on:
            plt.title(f"Grain boundary plot \n Grain: {self.gid}. Area: {round(self.skprop.area*100)/100} $\mu m^2$. Perimeter: {round(self.skprop.perimeter*10000)/10000} $\mu m$")
            plt.xlabel(r"X-axis, $\mu m$", fontsize=12)
            plt.ylabel(r"Y-axis, $\mu m$", fontsize=12)
            plt.show()

    def plotgbseg(self, hold_on=False):
        if not hold_on:
            plt.figure()
        plt.imshow(self.gbsegs_pre)
        if not hold_on:
            plt.title(f"Grain boundary segment plot \n Grain: {self.gid}. Area: {round(self.skprop.area*100)/100} $\mu m^2$. Perimeter: {round(self.skprop.perimeter*10000)/10000} $\mu m$")
            plt.xlabel(r"X-axis, $\mu m$", fontsize=12)
            plt.ylabel(r"Y-axis, $\mu m$", fontsize=12)
            plt.show()

class mesh():
    """
    This is a core mcgs.meshing class
    Intended only for Monte-Carlo simulation grain structures
    Works only on square lattice.
    For non-square lattices, interpolate to a square lattice and use

    meshing_package:
        # UPXO, GMSH, PYGMSH
    gb_conformity:
        # Conformal or Non-conformal
    element_type:
        # Type of finite element
    target_fe_software:
        # Abaqus or Moose
    reduced_integration:
        # BOOL: Flag for reduced integration
    abaqus_prop:
        # Abaqus mesh properties
    m:
        # Temporal slice
    coords_element_nodes:
        # Coordinates of FE nodes
    nodal_connectivity:
        DEscription
    n_nodes:
        # Number of nodes
    n_elements:
        # Number of elements
    ABQ_NODES:
        # Nodes contained in ABAQUS friendly format
    ABQ_ELEMENTS:
        # Elements contained in ABAQUS friendly format
    __uimesh:
        # Private copy of pxtal.uimesh
    __uigrid:
        # Private copy of pxtal.uigrid
    xgrid:
        DEscription
    ygrid:
        DEscription
    lgi:
        DEscription
    dim:
        DEscription
    elsets_grains:
        Element sets for grains
    elsets_gbz:
        Element sets for grain boundary zones
    elsets_gcr:
        Element sets for grain core
    ndsets_xl:
        Nodal set for left extreme edge boundary condition
    ndsets_xer:
        Nodal set for right extreme edge boundary condition
    ndsets_yl:
        Nodal set for bottom extreme edge boundary condition
    ndsets_yr:
        Nodal set for top extreme edge boundary condition
    """
    __slots__ = ('meshing_package',
                 'gb_conformity',
                 'element_type',
                 'target_fe_software',
                 'reduced_integration',
                 'abaqus_prop',
                 'm',
                 'coords_element_nodes',
                 'nodal_connectivity',
                 'n_nodes',
                 'n_elements',
                 'ABQ_NODES',
                 'ABQ_ELEMENTS',
                 '__uimesh',
                 '__uigrid',
                 'xgrid',
                 'ygrid',
                 'lgi',
                 'dim',
                 'elsets_grains',
                 'elsets_gbz',
                 'elsets_gcr',
                 'ndsets_xl',
                 'ndsets_xr',
                 'ndsets_yl',
                 'ndsets_yr',
                 )

    def __init__(self, uimesh, uigrid, dim, m, lgi):
        self.target_fe_software = uimesh.mesh_target_fe_software
        self.meshing_package = uimesh.mesh_meshing_package
        self.gb_conformity = uimesh.mesh_gb_conformity
        self.element_type = uimesh.mesh_element_type
        self.reduced_integration = uimesh.mesh_reduced_integration
        self.__uimesh = uimesh
        self.__uigrid = uigrid
        self.m = m
        self.lgi = lgi
        self.dim = dim
        self.mesher_brancher()

    def __att__(self):
        return gops.att(self)

    def mesher_brancher(self):
        if self.target_fe_software == 'Abaqus':
            self.abaqus_mesher_brancher()
        elif self.target_fe_software == 'Moose':
            self.moose_mesher_brancher()
        elif self.target_fe_software == 'DAMASK':
            self.damask_mesher_brancher()

    def abaqus_mesher_brancher(self):
        if self.meshing_package == 'UPXO' and self.gb_conformity == 'Non-Conformal':
            self.upxo_abaqus_nonconformal_mesher()
        elif self.meshing_package in ('UPXO',
                                      'GMSH') and self.gb_conformity == 'Conformal':
            self.gmsh_abaqus_conformal_mesher()  # <-------- MAKE NEW DEF
        elif self.meshing_package == 'PyGMSH' and self.gb_conformity == 'Conformal':
            self.pygmsh_abaqus_conformal_mesher()  # <-------- MAKE NEW DEF
        elif self.meshing_package == 'Abaqus' and self.gb_conformity == 'Conformal':
            self.abaqus_abaqus_conformal_mesher()  # <-------- MAKE NEW DEF
        pass

    def moose_mesher_brancher(self):
        pass

    def damask_mesher_brancher(self):
        pass

    def upxo_abaqus_nonconformal_mesher(self):
        if self.dim == 2 and self.element_type == 'quad4':
            self.mesh_abaqus_upxo_nonconformal_quad4()
        elif self.dim == 2 and self.element_type == 'quad8':
            self.mesh_abaqus_upxo_nonconformal_quad8()  # <-------- MAKE NEW DEF
        elif self.dim == 3 and self.element_type == 'hex8':
            self.mesh_abaqus_upxo_nonconformal_hex8()  # <-------- MAKE NEW DEF
        elif self.dim == 3 and self.element_type == 'hex20':
            self.mesh_abaqus_upxo_nonconformal_hex20()  # <-------- MAKE NEW DEF

    def upxo_abaqus_conformal_mesher(self):
        if self.dim == 2 and self.element_type == 'quad4':
            self.mesh_abaqus_upxo_conformal_quad4()  # <-------- MAKE NEW DEF
        elif self.dim == 2 and self.element_type == 'quad8':
            self.mesh_abaqus_upxo_conformal_quad8()  # <-------- MAKE NEW DEF
        elif self.dim == 3 and self.element_type == 'hex8':
            self.mesh_abaqus_upxo_conformal_tri3()  # <-------- MAKE NEW DEF
        elif self.dim == 3 and self.element_type == 'hex20':
            self.mesh_abaqus_upxo_conformal_tri6()  # <-------- MAKE NEW DEF

    def mesh_abaqus_upxo_nonconformal_quad4(self):
        """
        Generates ABAQUS compatible mesh data-structure inside UPXO
        targetted at non-conformal mesh with 4 noded quadrilateral elements.
        """
        # ------------------------------------------------------
        Xlat, Ylat = np.meshgrid(np.arange(self.__uigrid.xmin,
                                           self.__uigrid.xmax+1,
                                           self.__uigrid.xinc),
                                 np.arange(self.__uigrid.ymin,
                                           self.__uigrid.ymax+1,
                                           self.__uigrid.yinc),
                                 indexing='xy')
        self.xgrid, self.ygrid = deepcopy(Xlat), deepcopy(Ylat)
        Xlat, Ylat = np.concatenate(Xlat), np.concatenate(Ylat)
        # ------------------------------------------------------
        X, Y = np.meshgrid(np.arange(self.__uigrid.xmin-0.5*self.__uigrid.xinc,
                                     self.__uigrid.xmax+1.0*self.__uigrid.xinc,
                                     self.__uigrid.xinc),
                           np.arange(self.__uigrid.ymin-0.5*self.__uigrid.yinc,
                                     self.__uigrid.ymax+1.0*self.__uigrid.yinc,
                                     self.__uigrid.yinc),
                           indexing='xy')
        # ------------------------------------------------------
        dx, dy = 1.1*0.5*self.__uigrid.xinc, 1.1*0.5*self.__uigrid.xinc
        # ------------------------------------------------------
        xbl, ybl = list(np.concatenate(X)), list(np.concatenate(Y))
        xtl, ytl = list(np.concatenate(X)), list(np.concatenate(Y))
        xtr, ytr = list(np.concatenate(X)), list(np.concatenate(Y))
        xbr, ybr = list(np.concatenate(X)), list(np.concatenate(Y))
        _x_, _y_ = [], []
        for _ in xbl:
            _x_.append(_)
        for _ in xtl:
            _x_.append(_)
        for _ in xtr:
            _x_.append(_)
        for _ in xbr:
            _x_.append(_)
        for _ in ybl:
            _y_.append(_)
        for _ in ytl:
            _y_.append(_)
        for _ in ytr:
            _y_.append(_)
        for _ in ybr:
            _y_.append(_)
        _x_, _y_ = np.array(_x_), np.array(_y_)
        # ------------------------------------------------------
        _xy_ = np.vstack((_x_, _y_)).T
        u, indices = np.unique(_xy_, axis=0, return_index=True)
        X_vec, Y_vec = np.array([_xy_[i] for i in sorted(indices)]).T
        # ------------------------------------------------------
        self.coords_element_nodes, self.nodal_connectivity = [], []
        # ------------------------------------------------------
        for xl in np.arange(self.__uigrid.xmin,
                            self.__uigrid.xmax+1,
                            self.__uigrid.xinc):
            for yl in np.arange(self.__uigrid.ymin,
                                self.__uigrid.ymax+1,
                                self.__uigrid.yinc):
                # Calculate the expanded nodal bounds of the current elemewnt
                xmin = xl-dx
                xmax = xl+dx
                ymin = yl-dy
                ymax = yl+dy
                # Find nodes within these bounds
                locs_x_possibilities = np.logical_and(X_vec >= xmin,
                                                      X_vec <= xmax)
                locs_y_possibilities = np.logical_and(Y_vec >= ymin,
                                                      Y_vec <= ymax)
                nodes_locs = np.logical_and(locs_x_possibilities,
                                            locs_y_possibilities)
                # Get the numbers  nodes inside this bound and
                # take it nodal connectivity array
                el_nodes = list(np.where(nodes_locs)[0])
                el_nodes = [el_nodes[0]+1,
                            el_nodes[1]+1,
                            el_nodes[3]+1,
                            el_nodes[2]+1]
                self.nodal_connectivity.append(el_nodes)
                # Get the x and y coordinates of the bounded nodes
                a, b, c, d = X_vec[nodes_locs]
                nodal_coords_x = [a, b, d, c]
                a, b, c, d = Y_vec[nodes_locs]
                nodal_coords_y = [a, b, d, c]
                # Get coords of all nodes in the presente element
                self.coords_element_nodes.append(np.vstack((nodal_coords_x,
                                                            nodal_coords_y)))
        # Get nodes and the corresponding coordinates
        nodes_and_coords = []
        for ncon, ncoord in zip(self.nodal_connectivity,
                                self.coords_element_nodes):
            for _nc_, _ncoord_x_, _ncoord_y_ in zip(ncon,
                                                    ncoord[0],
                                                    ncoord[1]):
                nodes_and_coords.append([_nc_, _ncoord_x_, _ncoord_y_])
        # Build nodes data suitable for ABAQUS input file
        self.ABQ_NODES = np.unique(np.array(nodes_and_coords), axis=0)
        ## Add the 0's for the z-dimension
        #self.ABQ_NODES = np.hstack((self.ABQ_NODES,
        #                            np.zeros((self.ABQ_NODES.shape[0], 1))))
        # Build elements data suitable for ABAQUS input file
        # i.e. nodal connectivity table
        self.ABQ_ELEMENTS = np.array([[i+1]+self.nodal_connectivity[i]
                                      for i in range(len(self.nodal_connectivity))],
                                     dtype=int)
        # Store the number of nodes
        self.n_nodes = len(self.ABQ_NODES)
        # Store the number of elements
        self.n_elements = len(self.ABQ_ELEMENTS)

    def map_elements_grainids(self):
        '''
        This makes a map of grain ids and the pixel ids. Map is stored
        in a dictionary. grain id forms the keys and pixel ids for the
        values

        This is akin to the element-set defined in ABAQUS
        '''
        # Make global element ids array
        eids = np.reshape(self.ABQ_ELEMENTS.T[0], self.xgrid.shape)
        # Find locations in lgi, where values equal different grain id numbers
        a = [(np.where(self.lgi == gid)) for gid in range(1, self.lgi.max()+1)]
        a = [[list(_a[0]), list(_a[1])] for _a in a]  # Reformat data structure
        # Construct element sets. There will be as many elmewnts in
        # grain_element_sets, as the number of grains in the domain.
        grain_element_sets = {}
        for gid in range(0, self.lgi.max()):
            grain_element_sets[gid] = []
            for r, c in zip(a[gid][0], a[gid][1]):
                grain_element_sets[gid].append(eids[r, c])
        return grain_element_sets

    def abaqus_make_element_sets(self):
        return self.map_elements_grainids()

    def info(self):
        print(f"Number of elements, nodes: {self.n_elements}, {self.n_nodes}")
        print(f"Element type: {self.element_type}")
        print(f"Grain boundary conformity: {self.gb_conformity}")
        print(f"Temporal slice number of the mcgs: {self.m}")
        print(f"Target FE software: {self.target_fe_software}")

    def export_abaqus_inp_file(self,
                               folder="pxtal_mesh",
                               file="pxtal_mesh.inp",
                               ):
        os.makedirs(folder, exist_ok=True)
        file = file
        file_path = os.path.join(folder, file)
        with open(file_path, 'w') as f:
            f.write('*Heading\n')
            f.write('** Job name: UPXO-ABAQUS Model name: Model-1\n')
            f.write('** Generated by: Abaqus/CAE 6.14-1\n')
            # f.write('*Preprint, echo=NO, model=NO, history=NO, contact=NO\n')
            f.write('**\n')
            f.write('** PARTS\n')
            f.write('**\n')
            f.write('*Part, name=Part-1\n')
            f.write('*Node\n')

            np.savetxt(f,
                       self.ABQ_NODES,
                       fmt='    %d, %4.8f, %4.8f',
                       delimiter=', ',
                       header='',
                       comments='')

            f.write('*Element, type=CPS4\n')

            np.savetxt(f,
                       self.ABQ_ELEMENTS,
                       fmt='    %d, %d, %d, %d, %d',
                       delimiter=', ',
                       header='',
                       comments='')

            elsets = self.abaqus_make_element_sets().values()
            n_elsets = len(elsets)
            elset_names = []
            section_names = []
            material_names = []

            for esetnum, _ in enumerate(elsets):
                if esetnum < 10:
                    elset_names.append(f'Grain-000{esetnum}')
                    section_names.append(f'Sec-Grain-000{esetnum}')
                    material_names.append(f'Mat-Grain-000{esetnum}')
                elif esetnum >= 10 and 1 < 100:
                    elset_names.append(f'Grain-00{esetnum}')
                    section_names.append(f'Sec-Grain-00{esetnum}')
                    material_names.append(f'Mat-Grain-00{esetnum}')
                elif esetnum >= 100 and 1 < 1000:
                    elset_names.append(f'Grain-0{esetnum}')
                    section_names.append(f'Sec-Grain-0{esetnum}')
                    material_names.append(f'Mat-Grain-0{esetnum}')
                else:
                    elset_names.append(f'Grain-{esetnum}')
                    section_names.append(f'Sec-Grain-{esetnum}')
                    material_names.append(f'Mat-Grain-{esetnum}')

            # Write element sets
            for esetnum, eset in enumerate(elsets):
                f.write(f'*Elset, elset={elset_names[esetnum]}\n')
                line = ''
                for i, number in enumerate(eset):
                    line += str(number)
                    if i % 9 != 8 and i != len(eset) - 1:
                        line += ', '
                    elif i == len(eset) - 1:
                        f.write(line + '\n')
                    else:
                        f.write(line + ',\n')
                        line = ''

            # Wirte sections
            for section, elset, material in zip(section_names,
                                                elset_names,
                                                material_names):
                f.write(f"**Section: {section}\n")
                f.write(f"*Solid Section, elset={elset}, material={material}\n")
                f.write(",\n\n")

            f.write("*End Part\n")
            f.write("**\n")

            # Write materials
            for n_mat, mat in enumerate(material_names):
                f.write(f'*Material, name={mat}\n')
                f.write('*Depvar\n')
                f.write('12,\n')
                f.write('*User Material, constants=6\n')
                ea1 = np.random.randint(0, 359)
                ea2 = np.random.randint(0, 359)
                ea3 = np.random.randint(0, 180)
                f.write(f'{ea1},{ea2},{ea3},{n_mat},2,0\n')

            f.write('\n')
            f.write('**\n')
            f.write('** STEP: Loading\n')
            f.write('**\n')
            f.write('*Step, name=Loading, nlgeom=YES, inc=10000\n')
            f.write('*Static\n')
            f.write('0.01, 10., 1e-05, 1.\n')
            f.write('**\n')
            f.write('** OUTPUT REQUESTS\n')
            f.write('**\n')
            f.write('*Restart, write, frequency=0\n')
            f.write('**\n')
            f.write('** FIELD OUTPUT: F-Output-1\n')
            f.write('**\n')
            f.write('*Output, field, variable=PRESELECT\n')
            f.write('**\n')
            f.write('** FIELD OUTPUT: F-Output-2\n')
            f.write('**\n')
            f.write('*Element Output, directions=YES\n')
            f.write('SDV,\n')
            f.write('**\n')
            f.write('** HISTORY OUTPUT: H-Output-1\n')
            f.write('**\n')
            f.write('*Output, history, variable=PRESELECT\n')
            f.write('**\n')
            f.write('*End Step\n')
            print('\n')
            self.info()
            print('..............')
            print(f"Number of grain element sets: {len(elset_names)}")
            print(f"Numebr of materials: {len(material_names)}")
            print('..............')
            print('ABAQUS input file has been successfully written')
            print('\n')
            print('-------------------------------------------')
        f.close()

class artwork():
    """
    TO BE DEPRECATED
    """
    __slots__ = ('GrColArr',
                 )
    def __init__(self):
        pass

    def s_partitioned_tranition_probabilities(self,
                                              S,
                                              s_boltz_prob):
        fig = plt.figure(0,
                         figsize=(3.5, 3.5),
                         dpi=75,
                         )
        plt.scatter(np.arange(S), s_boltz_prob)
        plt.axis('auto')  # square, equal
        # plt.title("Q={}, m={}".format(Q, m+1), fontdict=font)
        plt.xlim([0., S])
        plt.xticks(np.linspace(0, S, 5))
        plt.ylim([0., 1.])
        plt.yticks(np.linspace(0, 1., 5))
        #plt.xlabel('Allowed no. of unique orientations', fontdict=font)
        #plt.ylabel('Probability of the unique orientation', fontdict=font)
        plt.grid(True)
        plt.show()

    def q_Col_Mat(self,
                  Q):
        """
        Summary line.

        State orientation based colour definitions: DESCRIPTION
        q_Col_Mat inputs
            1. Q        : No. of orientation states
        q_Col_Mat outputs
            1. GrColArr : Grain colour Array in RGB format. Q rows and 3 columns
        """
        if Q == 2:
            self.GrColArr = [[1, 0, 0],
                        [0, 0, 1]]
        elif Q == 3:
            self.GrColArr = [[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]]
        elif Q == 4:
            self.GrColArr = [[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]]
            self.GrColArr = np.vstack((self.GrColArr,
                                  [[1, 1, 0],
                                   [0, 1, 1],
                                   [1, 0, 1]][np.random.randint(3)]))
        elif Q > 4:
            gradient = 'random'
            if gradient == 'random':
                self.GrColArr = np.random.rand(Q, 3)
            elif gradient == 'GreyShades1':
                normFactor = Q-1
                red = np.arange(Q)/normFactor
                red = np.power(red, 2)
                green = np.arange(Q)/normFactor
                green = np.power(green, 2)
                blue = np.arange(Q)/normFactor
                blue = np.power(blue, 2)

                self.GrColArr = np.vstack((red, green, blue)).T
            elif gradient == 'GreyShades2':
                normFactor = Q-1
                red = np.arange(Q)/normFactor
                green = np.arange(Q)/normFactor
                blue = np.arange(Q)/normFactor
                self.GrColArr = np.vstack((red, green, blue)).T
            elif gradient == 'RedShades':
                normFactor = Q-1
                red = np.arange(Q)/normFactor
                green = 0.*np.arange(Q)
                blue = 0.*np.arange(Q)
                self.GrColArr = np.vstack((red, green, blue)).T
            elif gradient == 'BlueShades':
                normFactor = Q-1
                red = 0.*np.arange(Q)
                green = 0.*np.arange(Q)
                blue = np.arange(Q)/normFactor
                self.GrColArr = np.vstack((red, green, blue)).T
            elif gradient == 'GreenShades':
                normFactor = Q-1
                red = 0.*np.arange(Q)
                green = np.arange(Q)/normFactor
                blue = 0.*np.arange(Q)
                self.GrColArr = np.vstack((red, green, blue)).T
            elif gradient == 'RedGreenShades':
                normFactor = Q-1
                red = np.arange(Q)/normFactor
                green = np.arange(Q)/normFactor
                blue = 0.*np.arange(Q)
                self.GrColArr = np.vstack((red, green, blue)).T
            elif gradient == 'Custom_01':
                normFactor = Q-1
                red = np.arange(Q)/normFactor
                red[::-1].sort()
                red = np.power(red, 3)
                green = np.arange(Q)/normFactor
                green[::-1].sort()
                green = np.power(green, 3)
                blue = np.arange(Q)/normFactor
                # blue[::-1].sort()
                blue = np.power(blue, 3)
                self.GrColArr = np.vstack((red, green, blue)).T
            elif gradient == 'lemon':
                normFactor = Q-1
                red = np.arange(Q)/normFactor
                # red[::-1].sort()
                red = np.power(red, 3)
                green = np.arange(Q)/normFactor
                # green[::-1].sort()
                green = np.power(green, 0.2)
                blue = np.arange(Q)/normFactor
                # blue[::-1].sort()
                blue = np.power(blue, 4)
                self.GrColArr = np.vstack((red, green, blue)).T
            elif gradient == 'Custom_02':
                normFactor = Q-1
                red = np.arange(Q)/normFactor
                # red[::-1].sort()
                red = np.power(red, 0.2)  # Fixed
                green = np.arange(Q)/normFactor
                green[::-1].sort()
                green = np.power(green, 0.3)  # Fixed
                blue = np.arange(Q)/normFactor
                blue[::-1].sort()
                blue = np.power(blue, 6)  # Fixed
                self.GrColArr = np.vstack((red, green, blue)).T

def _load_user_input_data_(xl_fname='input_dashboard.xls'):
    """
    CALL:
        from mcgs import _load_user_input_data_
        uidata = _load_user_input_data_(xl_fname='input_dashboard.xls')
    """
    workbook = xlrd.open_workbook('input_dashboard.xls')
    _sheet_, uidata = workbook.sheet_by_index(0), {}
    for r in range(_sheet_.nrows):
        cellname = _sheet_.cell_value(r, 0)
        cellvalue = _sheet_.cell_value(r, 1)
        uidata[cellname] = cellvalue
    return uidata


class _uidata_mcgs_gridding_definitions_:
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
    transformation: str :: Geometric transformation operation for the grid

    CALL:
        from mcgs import _uidata_mcgs_gridding_definitions_
        uidata_gridpar = _uidata_mcgs_gridding_definitions_(uidata)
    """
    DEV = True
    __slots__ = ('type', 'dim', 'xmin', 'xmax', 'xinc',
                 'ymin', 'ymax', 'yinc', 'zmin', 'zmax', 'zinc', 'px_size',
                 'transformation', '__npixles_lock__', '__type_lock__')

    npixels_max = 500*500

    def __init__(self, uidata):
        self.type, self.dim = uidata['type'], int(uidata['dim'])
        self.xmin, self.xmax = uidata['xmin'], uidata['xmax']
        self.xinc = uidata['xinc']
        self.ymin, self.ymax = uidata['ymin'], uidata['ymax']
        self.yinc = uidata['yinc']
        self.zmin, self.zmax = uidata['zmin'], uidata['zmax']
        self.zinc = uidata['zinc']
        self.px_size = self.xinc*self.yinc*self.zinc
        self.transformation = uidata['transformation']
        _, __, npixels = self.grid
        self.__npixles_lock__ = False
        if npixels >= self.npixels_max:
            self.__npixles_lock__ = True

    def __repr__(self):
        _ = ' '*5
        retstr = 'Attribues of gridding definitions: \n'
        retstr += _ + f"{colored('TYPE', 'red', attrs=['bold'])}: {colored(self.type, 'cyan')}\n"
        retstr += _ + f"{colored('DIMENSIONALITY', 'red', attrs=['bold'])}: {colored(self.dim, 'cyan')}\n"
        retstr += _ + f"{colored('X', 'red', attrs=['bold'])}: ({colored(self.xmin, 'cyan')}, {colored(self.xmax, 'cyan')}, {colored(self.xinc, 'cyan')})\n"
        retstr += _ + f"{colored('Y', 'red', attrs=['bold'])}: ({colored(self.ymin, 'cyan')}, {colored(self.ymax, 'cyan')}, {colored(self.yinc, 'cyan')})\n"
        retstr += _ + f"{colored('Z', 'red', attrs=['bold'])}: ({colored(self.zmin, 'cyan')}, {colored(self.zmax, 'cyan')}, {colored(self.zinc, 'cyan')})\n"
        retstr += _ + f"{colored('PIXEL SIZE', 'red', attrs=['bold'])}: {colored(self.px_size, 'cyan')}\n"
        retstr += _ + f"{colored('TRANSFORMATION', 'red', attrs=['bold'])}: {colored(self.transformation, 'cyan')}"
        return retstr


    @property
    def xbound(self):
        return (self.xmin, self.xmax, self.xinc)

    @property
    def ybound(self):
        return (self.ymin, self.ymax, self.yinc)

    @property
    def zbound(self):
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
        pass

    @property
    def grid(self):
        if self.dim == 2:
            # make the grid from linear spaces of x and y
            x, y = np.meshgrid(self.xls, self.yls)
            npixels = x.size*y.size
        elif self.dim == 3:
            x, y, z = np.meshgrid(self.xls, self.yls, self.zls)
            npixels = x.size*y.size*z.size
        return x, y, npixels


class _uidata_mcgs_simpar_:
    """
    Class to port user inputs on simulation parameters into UPXO.
    --------------------------------------------------
    Following list of attributes are available:
        * PUBLIC:
            - S: int: Number of individual state values
            - mcsteps: int: Number of Monte-Carlo iterations
            - state_sampling_scheme: str: Sampling scheme to use
            - mcstep_hops: list[list]: List of mcstep ranges for each algo
            - consider_boltzmann_probability: bool: Flag to consider
            Boltzmann prob.
            - s_boltz_prob: str: Type of state dependent B.Probabilty
            - boltzmann_temp_factor_max: float: Multiplication factor
            - boundary_condition_type: str: Type of boundary condition to use
            - NL: int: times unit pixel dist of non-locality for energy
            calculation. Value supported in current UPXO version: 1
                @ value = 2: algorithm needs debugging in math and
                implementation
            - kineticity: str: mobility of temporally evolving state partitions
        * PRIVATE:
            - __lock__: dict
                lock on simulation parameters. Locked if any is True. Has the
                following sublocks:
                    * __lock__['mcstep_hops']:
                        > True for invalid mc-step ranges.
                    * __lock__['non_locality']:
                        > True for invalid Non-Locality value.
                    * __lock__['kineticity']:
                        > True for invalid kineticity value
                        > True for kineticity value incompatible with other
                        atttributes
    --------------------------------------------------
    CALL: INTERNAL
        from mcgs import _uidata_mcgs_simpar_
        _ = _uidata_mcgs_simpar_(uidata)
    --------------------------------------------------
    DEPRECATIONS @DEVELOPER:
        > mcsteps
    --------------------------------------------------
    TODO set: Behaviour overrides:
        * mcalg should now contain the right order of algorityhms to use fior
        each algo_hop. This is internally constructed.
        * mcalg should be made fully private. Impose source accerss only by
        overriding using __वृत्ति१__ uinstead of mcalg. This makes access only
        possible through algo_hops and nothing else.Removes source for
        ambiguities and simplifies user-code-interface. This also forces
        developer to adhere to source more often than making a chain of
        variables all pointing to ther same source.

    TODO set: Following to suit data-structure of algo_hops
        1. state_sampling_scheme
        2. consider_boltzmann_probability: auto based on values in algo_hops
        3. s_boltz_prob: auto based on values in algo_hops
        4. boltzmann_temp_factor_max: auto based on values in algo_hops
        5. NL: auto based on values in algo_hops
        6. kineticity: auto based on values in algo_hops
    """
    DEV = True
    __lock__ = {'mcstep_hops': False,
                'non_locality': False,
                '_': False
                }
    __slots__ = ('S', 'mcsteps',
                 'mcalg', 'algo_hop', 'algo_hops', 'mcstep_hops',
                 'state_sampling_scheme', 'consider_boltzmann_probability',
                 's_boltz_prob', 'boltzmann_temp_factor_max',
                 'boundary_condition_type',
                 'NL', 'kineticity',
                                 '__वृत्ति१__',
                 )

    def __init__(self, uidata=None):
        # ------------------------------------------------------
        self.set_algorithm_hopping(uidata)
        self.set_s(uidata)
        self.set_kbt(uidata)
        self.boundary_condition_type = uidata['boundary_condition_type']
        self.NL = int(uidata['NL'])
        self.kineticity = uidata['kineticity']
        if any(self.__lock__.values()):
            self.__lock__['_'] = True

    def __repr__(self):
        _ = ' '*5
        retstr = "Attributes of Simulation parameters:\n"
        retstr += _ + f"{colored('MCSTEPS', 'red', attrs=['bold'])}: {colored(self.mcsteps, 'cyan')}\n"
        retstr += _ + f"{colored('S', 'red', attrs=['bold'])}: {colored(self.S, 'cyan')} - will be deprecated.\n"
        retstr += _ + f"{colored('STATE SAMPLING SCHEME', 'red', attrs=['bold'])}: {colored(self.state_sampling_scheme, 'cyan')}\n"
        retstr += _ + f"{colored('CONSIDER BOLTZMANN PROBABILITY', 'red', attrs=['bold'])}: {colored(self.consider_boltzmann_probability, 'cyan')}\n"
        retstr += _ + f"{colored('S BOLTZAMNN PROBABILITY', 'red', attrs=['bold'])}: {colored(self.s_boltz_prob, 'cyan')}\n"
        retstr += _ + f"{colored('MAXIMUM BOLTZMANN TEMPERATURE FACTOR', 'red', attrs=['bold'])}: {colored(self.boltzmann_temp_factor_max, 'cyan')}\n"
        retstr += _ + f"{colored('BOUNDARY CONDITION TYPE', 'red', attrs=['bold'])}: {colored(self.boundary_condition_type, 'cyan')}\n"
        retstr += _ + f"{colored('NON LOCALITY', 'red', attrs=['bold'])}: {colored(self.NL, 'cyan')}\n"
        retstr += _ + f"{colored('KINETICITY', 'red', attrs=['bold'])}: {colored(self.kineticity, 'cyan')}\n"
        return retstr

    def set_algorithm_hopping(self, uidata):
        self.mcsteps = int(uidata['mcsteps'])
        self.mcstep_hops = []
        self.mcalg = str(int(uidata['mcalg']))
        self.algo_hop = True
        self.algo_hops = ((str(int(uidata['mcalg'])),
                           100
                           ),
                          )
        self.__वृत्ति१__ = tuple([_[0] for _ in self.algo_hops])
        self.validate_algo_hops(uidata)

    def set_s(self, uidata):
        self.S = int(uidata['S'])
        self.state_sampling_scheme = uidata['state_sampling_scheme']

    def set_kbt(self, uidata):
        self.consider_boltzmann_probability = bool(uidata['consider_boltzmann_probability'])
        self.s_boltz_prob = uidata['s_boltz_prob']
        self.boltzmann_temp_factor_max = uidata['boltzmann_temp_factor_max']

    def validate_algo_hops(self, uidata):
        # --------------------------------------------------
        # mcalg_hops = self.algo_hops[0][0]
        # mcstep_hops = [[0, self.algo_hops[0][1]]]
        # AAA = self.algo_hops
        # AAA = (('200', 20), ('200', 40), ('200', 41))
        # AAA = (('200', 20), )
        # --------------------------------------------------
        PRINT_algo_hops = lambda: print(self.algo_hops) if self.DEV else None
        PRINT_mcsteps_lock = lambda: print(mcsteps_lock) if self.DEV else None
        # --------------------------------------------------
        PRINT_algo_hops()
        t = [_[1] for _ in self.algo_hops]
        t[0], t[-1] = 0, 100
        mcsteps_lock = [False for _ in t]

        for i in range(len(t)):
            if i > 0:
                if t[i] < t[i-1]:
                    mcsteps_lock[i] = True

        PRINT_mcsteps_lock()

        if any(mcsteps_lock):
            # locked as True in it.
            self.__lock__['mcstep_hops'] = True
            print(f"{colored('Invalid algorithm hopping specification. LOCKED. ','red')}")
        else:
            # open as True not in it.
            mcstep_hops = []
            for i, ah in enumerate(self.algo_hops):
                if i == 0 and len(self.algo_hops) == 1:
                    mcstep_hops = [0, 100]
                elif i == 0 and len(self.algo_hops) > 1:
                    mcstep_hops.append([0, t[i]])
                elif i > 0:
                    mcstep_hops.append([t[i-1]+1, t[i]])
            self.mcstep_hops = mcstep_hops

    @property
    def lock_status(self):
        return self.__lock__['_']

    @property
    def locks(self):
        return self.__lock__


class _uidata_mcgs_simpar_1:

    DEV = True
    __lock__ = {'mcsteps': False,  # True for invalid mcsteps
                'nstates': False,  # True for invalid nstates
                'tgrad': False,  # True for invalid temperature gradient
                'algo_hop': False,
                'algo_hops': False,
                'mcstep_hops': False,
                'save_at_mcsteps': False,
                'algo_prop_compatability': False,
                '_': True
                }
    __slots__ = ('gsi', 'S', 'mcsteps', 'nstates', 'solver', 'tgrad',
                 'default_mcalg',
                 'algo_hop', 'algo_hops',
                 '__epoch_hops_mcsteps_pct__', 'epoch_hops_mcsteps',
                 'epoch_hops_algos',
                 '__mcstep_hop_locks__',
                 'mcalg', 'save_at_mcsteps',
                 'state_sampling_scheme', 'consider_boltzmann_probability',
                 's_boltz_prob', 'boltzmann_temp_factor_max',
                 'boundary_condition_type',
                 'NL', 'kineticity', '__वृत्ति१__',  '__sp__'
                 )

    def __init__(self,
                 n,
                 sim_parameters=None,
                 read_from_file=False,
                 filename=None
                 ):
        # Port all incoming simulation parameters to a convineient
        # private variable
        self.__sp__ = sim_parameters
        '''
        Set the grain structure index, which could be:
            * Index of the current grain structure in the parameter sweep
            study. This is the same as the key value of the present grain
            structure in ps.gsi dictionary.
            * Slightly Perturbed Grain Structure. FUTURE FEATURE.
            NOTE: Work needed in on
            this in: (1) concept development (2) procedure development
            (3) interface development (4) code development (5) implementation
        '''
        self.gsi = n
        self.set_s()
        self.set_algo_details(read_from_file)
        self.set_kbt()
        self.set_non_locality()
        self.boundary_condition_type = self.__sp__['boundary_condition_type']
        if any(self.__lock__.values()):
            self.__lock__['_'] = True
        self.__वृत्ति१__ = tuple([_[0] for _ in self.algo_hops])
        self.lock_update

    def __repr__(self):
        _ = ' '*5
        retstr = "Attributes of Simulation parameters:\n"
        retstr += _ + f"{colored('GSI', 'red', attrs=['bold'])}:  {colored(self.gsi, 'cyan')}\n"
        retstr += _ + "--/"*12 + "\n"
        retstr += _ + f"{colored('Num of MCSTEPS', 'red', attrs=['bold'])}:  {colored(self.mcsteps, 'cyan')}\n"
        retstr += _ + f"{colored('S', 'red', attrs=['bold'])}:  {colored(self.S, 'cyan')} - will be deprecated.\n"
        retstr += _ + "--/"*12 + "\n"
        retstr += _ + f"{colored('ALGO_HOP', 'red', attrs=['bold'])}:  {colored(self.algo_hop, 'cyan')}\n"
        retstr += _ + f"{colored('ALGO_HOPS', 'red', attrs=['bold'])}:  {colored(self.algo_hops, 'cyan')} - {colored('TO BE MADE PRIVATE', 'red')}\n"
        retstr += _ + f"{colored('EPOCH HOPS: MCSTEPS (%)', 'red', attrs=['bold'])}:  {colored(self.__epoch_hops_mcsteps_pct__, 'cyan')}\n"
        retstr += _ + f"{colored('EPOCH HOPS: MCSTEPS', 'red', attrs=['bold'])}:  {colored(self.epoch_hops_mcsteps, 'cyan')}\n"
        retstr += _ + f"{colored('EPOCH HOPS: ALGORITHMS', 'red', attrs=['bold'])}:  {colored(self.epoch_hops_algos, 'cyan')}\n"
        retstr += _ + f"{colored('SAVE_AT_MCSTEPS', 'red', attrs=['bold'])}:  {colored(self.save_at_mcsteps, 'cyan')}\n"
        retstr += _ + f"{colored('STATE_SAMPLING_SCHEME', 'red', attrs=['bold'])}:  {colored(self.state_sampling_scheme, 'cyan')}\n"
        retstr += _ + "--/"*12 + "\n"
        retstr += _ + f"{colored('CONSIDER BOLTZMANN PROBABILITY', 'red', attrs=['bold'])}:  {colored(self.consider_boltzmann_probability, 'cyan')}\n"
        retstr += _ + f"{colored('S BOLTZMANN PROBABILITY', 'red', attrs=['bold'])}:  {colored(self.s_boltz_prob, 'cyan')}\n"
        retstr += _ + f"{colored('BOLTZMANN TEMPERATURE FACTOR MAXIMUM', 'red', attrs=['bold'])}:  {colored(self.boltzmann_temp_factor_max, 'cyan')}\n"
        retstr += _ + f"{colored('BOUNDARY CONDITION TYPE', 'red', attrs=['bold'])}:  {colored(self.boundary_condition_type, 'cyan')}\n"
        retstr += _ + f"{colored('TGRAD', 'red', attrs=['bold'])}:  {colored(self.tgrad, 'cyan')}\n"
        retstr += _ + "--/"*12 + "\n"
        retstr += _ + f"{colored('NON LOCALITY', 'red', attrs=['bold'])}:  {colored(self.NL, 'cyan')}\n"
        retstr += _ + f"{colored('KINETICITY', 'red', attrs=['bold'])}:  {colored(self.kineticity, 'cyan')}\n"
        retstr += _ + "--/"*12 + "\n"
        retstr += _ + f"{colored('DEFAULT_MCALG', 'red', attrs=['bold'])}:  {colored(self.default_mcalg, 'cyan')}\n"
        retstr += _ + "--/"*12 + "\n"
        retstr += _ + f"{colored('SOLVER', 'red', attrs=['bold'])}:  {colored(self.solver, 'cyan')}\n"
        return retstr

    def set_algo_details(self, read_from_file):
        self.default_mcalg = self.__sp__['default_mcalg']
        self.kineticity = self.__sp__['kineticity']
        # --------------------------------------------
        self.set_algorithm_hopping(read_from_file)
        self.set_mcstep_hop_lock_and_epoch_hops_mcsteps_pct()
        self.set_epoch_hops_algos()

    def set_algorithm_hopping(self, read_from_file):

        if not read_from_file:
            if type(self.__sp__['mcsteps']) == int:
                self.mcsteps = self.__sp__['mcsteps']
            else:
                self.__lock__['mcsteps'] = True
            # ----------------------------------------
            # ----------------------------------------
            if type(self.__sp__['solver']) == str:
                self.solver = self.__sp__['solver']
            else:
                self.__lock__['solver'] = True
            # ----------------------------------------
            if type(self.__sp__['tgrad']) == np.ndarray:
                self.tgrad = self.__sp__['tgrad']
            else:
                self.tgrad = 'invalid'
                self.__lock__['tgrad'] = False
            # =================================================================
            self.algo_hop = self.__sp__['algo_hop']
            algo_hops = self.__sp__['algo_hops']
            # ----------------------------------------
            if not self.algo_hop:
                """If algorithm hopping is off (algo_hop=False), then, this
                branching helps set the algorithm (ps.gsi[:].uisim.mcalg) using
                the options provided in the algo_hops.
                """
                if type(algo_hops) in dth.dt.ITERABLES:
                    print(1)
                    """if options pertaining to algorithm hopping has
                    been provided by the user, then the first available
                    option pertaining to algorithm ID will be used to
                    set the algorithm. For example, if algo_hops is
                    [(200, 10), (201, 40), (202, 100)], then mcalg will be
                    set to '200'.
                    """
                    if algo_hops[0][0] in dth.valg.mc2d:
                        self.algo_hops = ((str(algo_hops[0][0]), 100), )
                    else:
                        self.algo_hops = ((self.default_mcalg, 100), )
                elif type(algo_hops) in dth.dt.NUMBERS or type(algo_hops) == str:
                    print(2)
                    """If a numerical entry has been made (in a case where the
                    user has done through direct access through set_param_sim),
                    then if it is valid, then str(value) will be set for mcalg.
                    If invalid, mcalg will default to '200'.
                    """
                    if algo_hops in dth.valg.mc2d:
                        self.algo_hops = ((str(int(algo_hops)), 100), )
                    else:
                        self.algo_hops = ((self.default_mcalg, 100), )
                else:
                    print(3)
                    """ This branch when user input could not be validated
                    or corrected.
                    """
                    print("MCALG could not be validated and/or corrected. Skipped")
                    self.__lock__['algo_hops'] = True
                    self.algo_hops = ((self.default_mcalg, 100), )
                # -------------------------------------
            else:
                """
                This involves two steps. First, validated mcalg array will be
                built. Then the mcsteps breakup will be validated based on
                values in algo_hops. Invalidities will be attempted to be
                corrected to enable simulation completion.
                """
                # STEP 1: Build validated mcalg array
                mcalg = ['invalid' for _ in self.N]
                self.algo_hops = [(None, None) for _ in self.N]
                if type(algo_hops) in dth.dt.ITERABLES:
                    print(5)
                    for n in self.N:
                        if algo_hops[n][0] in dth.valg.mc2d:
                            self.algo_hops[n][0] = str(algo_hops[n][0])
                        else:
                            self.algo_hops[n][0] = self.__default_mcalg__
                        mcalg[n-1] = self.algo_hops[n][0]
                elif type(algo_hops) in dth.dt.NUMBERS or type(algo_hops) == str:
                    print(6)
                    self.algo_hops = [('invalid', ) for _ in self.N]
                    for n in self.N:
                        if algo_hops in dth.valg.mc2d:
                            self.algo_hops[n][0] = str(int(algo_hops))
                        else:
                            self.algo_hops[n][0] = self.__default_mcalg__
                        mcalg[n-1] = self.algo_hops[n][0]
                else:
                    print(7)
                    mcalg = ['invalid' for _ in self.N]
                    self.__lock__['algo_hops'] = True
                    print('MCALG could not be validated and/or corrected. Skipped')
                self.gsi[n+1].uisim.mcalg = self.algo_hops[n][0]
                # STEP 2: Validate mcsteps breakup based on values in algo_hops
            # =================================================================
            # =================================================================
            self.save_at_mcsteps = [int(_) for _ in self.__sp__['save_at_mcsteps']]
        else:
            pass

    def set_mcstep_hop_lock_and_epoch_hops_mcsteps_pct(self):
        """
        Takes in the info from validated algo_hops and build up
        two secondary lists which are easier to handle in algorithm
        selection. In the process, it may decide to perform additional
        validations which may be required.
        """
        # --------------------------------------------------
        ''' STEP 1    GENERATE MONTE-CARLO EPOCHS
        Each pair of adjacent entries denotes a hop range, which is basically
        the range of monte-carlo iterations within which an algorithm works.
        As to which algorithm is it, which would be working, is isolated
        later on in this method space, but the data is already present in the
        self.algo_hops. Example: If epochs is [0, 20, 100], then there would be
        two epoch hops, which are [0, 20] and [20, 100]. As you can see, its
        actually a range of iteration numbers. Note; To be accurate, the epoch
        hops would actually be [0, 20] and [21, 100].
        '''
        epoch = [_[1] for _ in self.algo_hops]
        epoch[0], epoch[-1] = 0, 100
        # --------------------------------------------------
        '''STEP 2    GENERATE TEMPORARY LOCKS FOR EACH EPOCH HOP.
        If the epoch hops contains startings and endings, which are invalid,
        then the mcsteps_lock will be set to True for the specific epoch hop.
        Obviously, there will be as many elements in it as there are
        epoch hops.
        '''
        self.__mcstep_hop_locks__ = [False for _ in epoch]
        for i in range(len(epoch)):
            if i > 0:
                if epoch[i] < epoch[i-1]:
                    self.__mcstep_hop_locks__[i] = True
        print(f"__mcstep_hop_locks__: {self.__mcstep_hop_locks__}")
        # --------------------------------------------------
        '''STEP 3: generate __epoch_hops_mcsteps_pct__ list for each algorithm.
        The outer list contains lists of all epoxh hops. Each inner list
        contains ther start and end of the monte-carlo iteration number. At
        mc iterartion number of srtart, the corresponding algorithm will be
        switched to, from the previous algorithm, if there was one active.
        If any locks in __mcstep_hop_locks__ have been locked in STEP 2, then
        'mcstep_hops' sublock of the global uisim lock gets locked and
        everything stops.
        '''
        if any(self.__mcstep_hop_locks__):
            # Branch: Locked if True in it.
            self.__lock__['mcstep_hops'] = True
            self.__epoch_hops_mcsteps_pct__ = ['invalid' for _ in self.algo_hops]
            self.epoch_hops_mcsteps = ['invalid' for _ in self.algo_hops]
            print(f"{colored('Invalid algorithm hopping specification. LOCKED. ','red')}")
        else:
            # Branch: Open if True not in it.
            self.__epoch_hops_mcsteps_pct__ = []
            for i, ah in enumerate(self.algo_hops):
                if i == 0 and len(self.algo_hops) == 1:
                    self.__epoch_hops_mcsteps_pct__.append([0, 100])
                elif i == 0 and len(self.algo_hops) > 1:
                    self.__epoch_hops_mcsteps_pct__.append([0, epoch[i]])
                elif i > 0:
                    self.__epoch_hops_mcsteps_pct__.append([epoch[i-1]+1, epoch[i]])
            '''Set the actual mcstep start end values including validity check.
            Validity check includes:
                If self.mcsteps is too small to accomodate the algo_hops,
                then uisim parameters get LOCKED.
            '''
            self.epoch_hops_mcsteps = []
            for _mcsteps_ in self.__epoch_hops_mcsteps_pct__:
                starting_mcstep = int(_mcsteps_[0]*self.mcsteps/100)
                c = int(_mcsteps_[1]*self.mcsteps/100)
                self.epoch_hops_mcsteps.append([starting_mcstep,
                                                starting_mcstep])
            self.epoch_hops_mcsteps[-1][1] = self.mcsteps
        print(f"__epoch_hops_mcsteps_pct__: {self.__epoch_hops_mcsteps_pct__}")

    def set_epoch_hops_algos(self):
        if not self.__lock__['mcstep_hops']:
            self.epoch_hops_algos = [am[0] for am in self.algo_hops]
        else:
            self.epoch_hops_algos = ['invalid' for _ in self.algo_hops]
        print(f"epoch_hops_algos:  {self.epoch_hops_algos}")

    def set_kbt(self):
        self.consider_boltzmann_probability = bool(self.__sp__['consider_boltzmann_probability'])
        self.s_boltz_prob = self.__sp__['s_boltz_prob']
        self.boltzmann_temp_factor_max = self.__sp__['boltzmann_temp_factor_max']

    def set_non_locality(self):
        self.NL = int(self.__sp__['NL'])

    def set_s(self):
        if type(self.__sp__['nstates']) == int:
            self.S = self.__sp__['nstates']
            self.nstates = self.S
        else:
            self.__lock__['nstates'] = True
        self.state_sampling_scheme = self.__sp__['state_sampling_scheme']

    @property
    def lock_status(self):
        if self.__lock__['_']:
            print(f"{colored('Locked sub-locks exist', 'red', attrs=['bold'])}")
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


class _uidata_mcgs_grain_structure_characterisation_:
    """
    CALL:
        from mcgs import _uidata_mcgs_grain_structure_characterisation_
        uidata_grain_identification = _uidata_mcgs_grain_structure_characterisation_(uidata)
    """
    DEV = True
    __slots__ = ('grain_identification_library', '__uisim_lock__'
                 )

    def __init__(self, uidata):
        self.grain_identification_library = uidata['grain_identification_library']

    def __repr__(self):
        str0 = 'Attributes of grain structure analysis parameters: \n'
        str1 = '    grain_identification_library'
        return str0 + str1


class _uidata_mcgs_intervals_:
    """
    mcint_grain_size_par_estim: int  ::
    mcint_gb_par_estimation: int  ::
    mcint_grain_shape_par_estim: int  ::
    mcint_save_at_mcstep_interval: int  ::
    save_final_S_only: bool  ::
    mcint_promt_display: int  ::
    mcint_plot_gs: int  ::

    CALL:
        from mcgsa import _uidata_mcgs_intervals_
        uidata_intervals = _uidata_mcgs_intervals_(uidata)
    """
    DEV = True
    __slots__ = ('mcint_grain_size_par_estim',
                 'mcint_gb_par_estimation',
                 'mcint_grain_shape_par_estim',
                 'mcint_save_at_mcstep_interval',
                 'save_final_S_only',
                 'mcint_promt_display',
                 'mcint_plot_grain_structure', '__uiint_lock__'
                 )

    def __init__(self, uidata):
        self.mcint_grain_size_par_estim = uidata['mcint_grain_size_par_estim']
        self.mcint_gb_par_estimation = uidata['mcint_gb_par_estimation']
        self.mcint_grain_shape_par_estim = uidata['mcint_grain_shape_par_estim']
        self.mcint_save_at_mcstep_interval = uidata['mcint_save_at_mcstep_interval']
        self.save_final_S_only = bool(uidata['save_final_S_only'])
        self.mcint_promt_display = uidata['mcint_promt_display']
        self.mcint_plot_grain_structure = bool(uidata['mcint_plot_grain_structure'])

    def __str__(self):
        _ = ' '*5
        retstr = "Attributes of mcgs intervals related parameters: \n"
        retstr += f"{colored('MCINT_GRAIN_SIZE_PAR_ESTIM', 'red')}: {colored(self.mcint_grain_size_par_estim, 'green')}\n"
        retstr += f"{colored('MCINT_GB_PAR_ESTIMATION', 'red')}: {colored(self.mcint_gb_par_estimation, 'green')}\n"
        retstr += f"{colored('MCINT_GRAIN_SHAPE_PAR_ESTIM', 'red')}: {colored(self.mcint_grain_shape_par_estim, 'green')}\n"
        retstr += f"{colored('MCINT_SAVE_AT_MCSTEP_INTERVAL', 'red')}: {colored(self.mcint_save_at_mcstep_interval, 'green')}\n"
        retstr += f"{colored('SAVE_FINAL_S_ONLY', 'red')}: {colored(self.save_final_S_only, 'green')}\n"
        retstr += f"{colored('MCINT_PROMT_DISPLAY', 'red')}: {colored(self.mcint_promt_display, 'green')}\n"
        retstr += f"{colored('MCINT_PLOT_GRAIN_STRUCTURE', 'red')}: {colored(self.mcint_plot_grain_structure, 'green')}\n"
        return retstr


class _uidata_mcgs_property_calc_:
    """
    compute_grain_area_pol: bool :: Flag to compute polygonal grain area
    compute_grain_area_pix: bool :: Flag to compute pixelated grain area
    compute_gb_length_pol: bool :: Flag to compute grain boundayr length polygonal
    compute_gb_length_pxl: bool :: Flag to compute grain boundary length pixelated
    compute_grain_moments: bool :: Flag to compute grain moments
    grain_area_type_to_consider: str :: Flag to select type of area to calculate
    compute_grain_area_distr: bool :: Flag to compute
    compute_grain_area_distr_kde: bool :: Flag to compute
    compute_grain_area_distr_prop: bool :: Flag to select type of grain boundary length to calculate
    gb_length_type_to_consider: str :: Flag to compute
    compute_gb_length_distr: bool :: Flag to compute
    compute_gb_length_distr_kde: bool :: Flag to compute
    compute_gb_length_distr_prop: bool :: Flag to compute

    CALL:
        from mcgs import _uidata_mcgs_property_calc_
        uidata_propcalc = _uidata_mcgs_property_calc_(uidata)
    """
    DEV = True
    __slots__ = ('compute_grain_area_pol',
                 'compute_grain_area_pix',
                 'compute_gb_length_pol',
                 'compute_gb_length_pxl',
                 'compute_grain_moments',
                 'grain_area_type_to_consider',
                 'compute_grain_area_distr',
                 'compute_grain_area_distr_kde',
                 'compute_grain_area_distr_prop',
                 'gb_length_type_to_consider',
                 'compute_gb_length_distr',
                 'compute_gb_length_distr_kde',
                 'compute_gb_length_distr_prop',
                 '__uiprop_lock__'
                 )
    def __init__(self, uidata):
        self.compute_grain_area_pix = bool(uidata['compute_grain_area_pix'])
        if self.compute_grain_area_pix:
            self.compute_grain_area_pol = False
        else:
            self.compute_grain_area_pol = bool(uidata['compute_grain_area_pol'])
        self.compute_gb_length_pol = bool(uidata['compute_gb_length_pol'])
        self.compute_gb_length_pxl = bool(uidata['compute_gb_length_pxl'])
        self.compute_grain_moments = bool(uidata['compute_grain_moments'])
        self.grain_area_type_to_consider = bool(uidata['grain_area_type_to_consider'])
        self.compute_grain_area_distr = bool(uidata['compute_grain_area_distr'])
        self.compute_grain_area_distr_kde = bool(uidata['compute_grain_area_distr_kde'])
        self.compute_grain_area_distr_prop = bool(uidata['compute_grain_area_distr_prop'])
        self.gb_length_type_to_consider = bool(uidata['gb_length_type_to_consider'])
        self.compute_gb_length_distr = bool(uidata['compute_gb_length_distr'])
        self.compute_gb_length_distr_kde = bool(uidata['compute_gb_length_distr_kde'])
        self.compute_gb_length_distr_prop = bool(uidata['compute_gb_length_distr_prop'])

    def __repr__(self):
        _ = ' '*5
        retstr = "Attributes of grain structure property estimation: \n"
        retstr += f"{colored('COMPUTE_GRAIN_AREA_POL', 'red')}: {colored(self.compute_grain_area_pol, 'green')}\n"
        retstr += f"{colored('COMPUTE_GRAIN_AREA_PIX', 'red')}: {colored(self.compute_grain_area_pix, 'green')}\n"
        retstr += f"{colored('COMPUTE_GB_LENGTH_POL', 'red')}: {colored(self.compute_gb_length_pol, 'green')}\n"
        retstr += f"{colored('COMPUTE_GB_LENGTH_PXL', 'red')}: {colored(self.compute_gb_length_pxl, 'green')}\n"
        retstr += f"{colored('COMPUTE_GRAIN_MOMENTS', 'red')}: {colored(self.compute_grain_moments, 'green')}\n"
        retstr += f"{colored('COMPUTE_GRAIN_CENTROIDS', 'red')}: {colored(self.compute_grain_centroids, 'green')}\n"
        retstr += f"{colored('CREATE_GRAIN_BOUNDARY_ZONE', 'red')}: {colored(self.create_grain_boundary_zone, 'green')}\n"
        return retstr


class _uidata_mcgs_generate_geom_reprs_():
    """
    make_mp_grain_centoids: bool :: Make MP of grain_centroids
    make_mp_grain_points: bool :: Grains as multi-point
    make_ring_grain_boundaries: bool :: GB as UPXO ring
    make_xtal_grain: bool :: Grains as UPXO XTAL object
    make_chull_grain: bool :: Make convex hull for each grain
    create_gbz: bool :: create_grain_boundary_zone

    CALL:
        from mcgs import _uidata_mcgs_generate_geom_reprs_
        uidata_georepr = _uidata_mcgs_generate_geom_reprs_(uidata)
    """
    DEV = True
    __slots__ = ('make_mp_grain_centoids',
                 'make_mp_grain_points',
                 'make_ring_grain_boundaries',
                 'make_xtal_grain',
                 'make_chull_grain',
                 'create_gbz', '__uigeomrepr_lock__'
                 )

    def __init__(self, uidata):
        self.make_mp_grain_centoids = bool(uidata['make_mp_grain_centoids'])
        self.make_mp_grain_points = bool(uidata['make_mp_grain_points'])
        self.make_ring_grain_boundaries = bool(uidata['make_ring_grain_boundaries'])
        self.make_xtal_grain = bool(uidata['make_xtal_grain'])
        self.make_chull_grain = bool(uidata['make_chull_grain'])
        self.create_gbz = bool(uidata['create_gbz'])

    def __repr__(self):
        _ = ' '*5
        retstr = "Attributes of geometric representation: \n"
        retstr += f"{colored('MAKE_MP_GRAIN_CENTOIDS', 'red')}: {colored(self.make_mp_grain_centoids, 'green')}\n"
        retstr += f"{colored('MAKE_MP_GRAIN_POINTS', 'red')}: {colored(self.make_mp_grain_points, 'green')}\n"
        retstr += f"{colored('MAKE_RING_GRAIN_BOUNDARIES', 'red')}: {colored(self.make_ring_grain_boundaries, 'green')}\n"
        retstr += f"{colored('MAKE_XTAL_GRAIN', 'red')}: {colored(self.make_xtal_grain, 'green')}\n"
        retstr += f"{colored('MAKE_CHULL_GRAIN', 'red')}: {colored(self.make_chull_grain, 'green')}\n"
        retstr += f"{colored('CREATE_GBZ', 'red')}: {colored(self.create_gbz, 'green')}\n"
        return retstr

class _uidata_mcgs_mesh_():
    """
    mesh_gb_conformity
    mesh_target_fe_software
    mesh_meshing_package
    mesh_reduced_integration
    mesh_element_type
    """
    __slots__ = ('mesh_gb_conformity',
                 'mesh_target_fe_software',
                 'mesh_meshing_package',
                 'mesh_reduced_integration',
                 'mesh_element_type',
                 '__uimesh_lock__'
                 )
    DEV = True
    def __init__(self, uidata):
        self.mesh_gb_conformity = uidata['mesh_gb_conformity']
        self.mesh_target_fe_software = uidata['mesh_target_fe_software']
        self.mesh_meshing_package = uidata['mesh_meshing_package']
        self.mesh_reduced_integration = bool(uidata['mesh_reduced_integration'])
        self.mesh_element_type = uidata['mesh_element_type']

    def __repr__(self):
        _ = ' '*5
        retstr = "Attriutes of meshing parameters: \n"
        retstr += _ + f"{colored('MESH_GB_CONFORMITY', 'red')}: {colored(self.mesh_gb_conformity, 'green')}\n"
        retstr += _ + f"{colored('MESH_TARGET_FE_SOFTWARE', 'red')}: {colored(self.mesh_target_fe_software, 'green')}\n"
        retstr += _ + f"{colored('MESH_MESHING_PACKAGE', 'red')}: {colored(self.mesh_meshing_package, 'green')}\n"
        retstr += _ + f"{colored('MESH_REDUCED_INTEGRATION', 'red')}: {colored(self.mesh_reduced_integration, 'green')}\n"
        retstr += _ + f"{colored('MESH_ELEMENT_TYPE', 'red')}: {colored(self.mesh_element_type, 'green')}\n"
        return retstr


class _manual_uidata_mcgs_gridding_definitions_:
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
    DEV = True
    __slots__ = ('type', 'dim',
                 'xmin', 'xmax', 'xinc',
                 'ymin', 'ymax', 'yinc',
                 'zmin', 'zmax', 'zinc',
                 'px_size',
                 'transformation')
    __npixx__ = 500  # Number of pixels along x-axis
    __npixy__ = 500  # Number of pixels along y-axis
    __npixels_max__ = __npixx__*__npixy__
    __lock__ = {'type': False,  # True if invalid type
                'npixx': False,  # True if npixx > self.npixx
                'npixy': False,  # True if npixy > self.npixy
                'npix': False,  # True if npix > self.npixx*self.npixy
                '_': False,  # True if any of above is True
                }

    def __init__(self,
                 domain_size=None,
                 read_from_file=False, filename=None
                 ):
        self.__lock__['_'] = False
        # ----------------------------------
        if not read_from_file:
            self.type = 'square'
            self.xmin, self.xmax = domain_size[0][0], domain_size[0][1]
            self.xinc = domain_size[3]
            self.ymin, self.ymax = domain_size[1][0], domain_size[1][1]
            self.yinc = domain_size[3]
            self.zmin, self.zmax = domain_size[2][0], domain_size[2][1]
            self.zinc = domain_size[3]
            self.px_size = self.xinc*self.yinc*self.zinc
            # ----------------------------------
            if 2 in (len(domain_size[0]),
                     len(domain_size[1]),
                     len(domain_size[2])):
                self.dim = 2
            elif len(domain_size[0])==3 and len(domain_size[1])==3 and len(domain_size[2])==3:
                self.dim = 3
            else:
                print('Invalid grid specification')
                self.__lock__ = 'locked'
            # ----------------------------------
            self.transformation = None
            # ----------------------------------
            _, __, npixels = self.grid
            if npixels >= self.__npixels_max__:
                self.__lock__['_'] = True
        else:
            pass
        self.lock_update

    def __repr__(self):
        _ = ' '*5
        retstr = 'Attribues of gridding definitions: \n'
        retstr += _ + f"{colored('TYPE', 'red', attrs=['bold'])}: {colored(self.type, 'cyan')}\n"
        retstr += _ + f"{colored('DIMENSIONALITY', 'red', attrs=['bold'])}: {colored(self.dim, 'cyan')}\n"
        retstr += _ + f"{colored('X', 'red', attrs=['bold'])}: ({colored(self.xmin, 'cyan')}, {colored(self.xmax, 'cyan')}, {colored(self.xinc, 'cyan')})\n"
        retstr += _ + f"{colored('Y', 'red', attrs=['bold'])}: ({colored(self.ymin, 'cyan')}, {colored(self.ymax, 'cyan')}, {colored(self.yinc, 'cyan')})\n"
        retstr += _ + f"{colored('Z', 'red', attrs=['bold'])}: ({colored(self.zmin, 'cyan')}, {colored(self.zmax, 'cyan')}, {colored(self.zinc, 'cyan')})\n"
        retstr += _ + f"{colored('PIXEL SIZE', 'red', attrs=['bold'])}: {colored(self.px_size, 'cyan')}\n"
        retstr += _ + f"{colored('TRANSFORMATION', 'red', attrs=['bold'])}: {colored(self.transformation, 'cyan')}"
        return retstr

    @property
    def xbound(self):
        return (self.xmin, self.xmax, self.xinc)

    @property
    def ybound(self):
        return (self.ymin, self.ymax, self.yinc)

    @property
    def zbound(self):
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
        pass

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


class _manual_uidata_mcgs_simpar_:
    """
    * gsi: Grain structure index
    * S: number of states - TO BE DEPRECATED
    * mcsteps: Number of mcsteps
    * nstates: Number of states
    * solver: Python / C
    * tgrad: temperature gradient grid
    * default_mcalg: The default fall-back monte-carlo algorithm
    * algo_hop: Bool flag whether to consider algorithm hopping
    * algo_hops: User provided algorithm hopping specification
    * __epoch_hops_mcsteps_pct__: percentage values of mcsteps of all epoch
        hops
    * epoch_hops_mcsteps: mcstep values of all epock hops
    * epoch_hops_algos: algorithms for each of the epoch hops
    * __mcstep_hop_locks__: lock for each of the epoch hops on validity of
        mcstep values of corresponding epoch hops
    * mcalg: monte-carlo algorithm - TO BE DEPRECATED by phasing out usage
    * save_at_mcsteps - mcstep intervals where temporal grain structure
        instancesto be stored
    * state_sampling_scheme: sampling scheme for use in state flipping.
        NOTE: TO BE GENERALIZED for suite of applicable algorithms
    * consider_boltzmann_probability: Bool flag indicvating whether an
        algorithm should use Boltzmann (i.e. transition) probability
    * s_boltz_prob: Provides choice to select state dependent kbT or
        state independent kbT
        NOTE: An accompanying variable is to be made to provide the
        exact dependency behaviour in case state dependent kbT is needed
    * boltzmann_temp_factor_max: Provides the maximum kbT factor. refer
        to theory manual for more information. In simple terms, use this
        to control grain boundary roughness.
    * boundary_condition_type: Specify type of boundary condition. Currently
        only valid entry is 'wrapped'.
        Option 'closed' is TO BE IMPLEMENTED to consider boundary effects.
        Some of the practical cases where this will be needed include
        the following of grain structures, where gradients in grain structure
        are natural due to boundary effects:
            1. chilled cast grain structure
            2. welded grain strcutures
            3. etc.
    * NL: Non-locality in lattice energy calculation. RESTRICTED to 1.
    * kineticity: str flag to allow UPXO to select algorithms which result in
        an spatially unbalanced skewed energetics in the pixel to pixel
        Hamiltonian estimation.
    * __sp__: A private copy of all user input simulation parameter values

    CALL:
        from mcgs import _uidata_mcgs_simpar_
        uisim = _uidata_mcgs_simpar_(sim_parameters)
    """
    DEV = True
    __lock__ = {'mcsteps': False,  # True for invalid mcsteps
                'nstates': False,  # True for invalid nstates
                'tgrad': False,  # True for invalid temperature gradient
                'algo_hop': False,
                'algo_hops': False,
                'mcstep_hops': False,
                'save_at_mcsteps': False,
                'algo_prop_compatability': False,
                '_': True
                }
    __slots__ = ('gsi', 'S', 'mcsteps', 'nstates', 'solver', 'tgrad',
                 'default_mcalg',
                 'algo_hop', 'algo_hops',
                 '__epoch_hops_mcsteps_pct__', 'epoch_hops_mcsteps',
                 'epoch_hops_algos',
                 '__mcstep_hop_locks__',
                 'mcalg', 'save_at_mcsteps',
                 'state_sampling_scheme', 'consider_boltzmann_probability',
                 's_boltz_prob', 'boltzmann_temp_factor_max',
                 'boundary_condition_type',
                 'NL', 'kineticity', '__वृत्ति१__',  '__sp__'
                 )

    def __init__(self,
                 n,
                 sim_parameters=None,
                 read_from_file=False,
                 filename=None
                 ):
        # Port all incoming simulation parameters to a convineient
        # private variable
        self.__sp__ = sim_parameters
        '''
        Set the grain structure index, which could be:
            * Index of the current grain structure in the parameter sweep
            study. This is the same as the key value of the present grain
            structure in ps.gsi dictionary.
            * Slightly Perturbed Grain Structure. FUTURE FEATURE.
            NOTE: Work needed in on
            this in: (1) concept development (2) procedure development
            (3) interface development (4) code development (5) implementation
        '''
        self.gsi = n
        self.set_s()
        self.set_algo_details(read_from_file)
        self.set_kbt()
        self.set_non_locality()
        self.boundary_condition_type = self.__sp__['boundary_condition_type']
        if any(self.__lock__.values()):
            self.__lock__['_'] = True
        self.__वृत्ति१__ = tuple([_[0] for _ in self.algo_hops])
        self.lock_update

    def __repr__(self):
        _ = ' '*5
        retstr = "Attributes of Simulation parameters:\n"
        retstr += _ + f"{colored('GSI', 'red', attrs=['bold'])}:  {colored(self.gsi, 'cyan')}\n"
        retstr += _ + "--/"*12 + "\n"
        retstr += _ + f"{colored('Num of MCSTEPS', 'red', attrs=['bold'])}:  {colored(self.mcsteps, 'cyan')}\n"
        retstr += _ + f"{colored('S', 'red', attrs=['bold'])}:  {colored(self.S, 'cyan')} - will be deprecated.\n"
        retstr += _ + "--/"*12 + "\n"
        retstr += _ + f"{colored('ALGO_HOP', 'red', attrs=['bold'])}:  {colored(self.algo_hop, 'cyan')}\n"
        retstr += _ + f"{colored('ALGO_HOPS', 'red', attrs=['bold'])}:  {colored(self.algo_hops, 'cyan')} - {colored('TO BE MADE PRIVATE', 'red')}\n"
        retstr += _ + f"{colored('EPOCH HOPS: MCSTEPS (%)', 'red', attrs=['bold'])}:  {colored(self.__epoch_hops_mcsteps_pct__, 'cyan')}\n"
        retstr += _ + f"{colored('EPOCH HOPS: MCSTEPS', 'red', attrs=['bold'])}:  {colored(self.epoch_hops_mcsteps, 'cyan')}\n"
        retstr += _ + f"{colored('EPOCH HOPS: ALGORITHMS', 'red', attrs=['bold'])}:  {colored(self.epoch_hops_algos, 'cyan')}\n"
        retstr += _ + f"{colored('SAVE_AT_MCSTEPS', 'red', attrs=['bold'])}:  {colored(self.save_at_mcsteps, 'cyan')}\n"
        retstr += _ + f"{colored('STATE_SAMPLING_SCHEME', 'red', attrs=['bold'])}:  {colored(self.state_sampling_scheme, 'cyan')}\n"
        retstr += _ + "--/"*12 + "\n"
        retstr += _ + f"{colored('CONSIDER BOLTZMANN PROBABILITY', 'red', attrs=['bold'])}:  {colored(self.consider_boltzmann_probability, 'cyan')}\n"
        retstr += _ + f"{colored('S BOLTZMANN PROBABILITY', 'red', attrs=['bold'])}:  {colored(self.s_boltz_prob, 'cyan')}\n"
        retstr += _ + f"{colored('BOLTZMANN TEMPERATURE FACTOR MAXIMUM', 'red', attrs=['bold'])}:  {colored(self.boltzmann_temp_factor_max, 'cyan')}\n"
        retstr += _ + f"{colored('BOUNDARY CONDITION TYPE', 'red', attrs=['bold'])}:  {colored(self.boundary_condition_type, 'cyan')}\n"
        retstr += _ + f"{colored('TGRAD', 'red', attrs=['bold'])}:  {colored(self.tgrad, 'cyan')}\n"
        retstr += _ + "--/"*12 + "\n"
        retstr += _ + f"{colored('NON LOCALITY', 'red', attrs=['bold'])}:  {colored(self.NL, 'cyan')}\n"
        retstr += _ + f"{colored('KINETICITY', 'red', attrs=['bold'])}:  {colored(self.kineticity, 'cyan')}\n"
        retstr += _ + "--/"*12 + "\n"
        retstr += _ + f"{colored('DEFAULT_MCALG', 'red', attrs=['bold'])}:  {colored(self.default_mcalg, 'cyan')}\n"
        retstr += _ + "--/"*12 + "\n"
        retstr += _ + f"{colored('SOLVER', 'red', attrs=['bold'])}:  {colored(self.solver, 'cyan')}\n"
        return retstr

    def set_algo_details(self, read_from_file):
        self.default_mcalg = self.__sp__['default_mcalg']
        self.kineticity = self.__sp__['kineticity']
        # --------------------------------------------
        self.set_algorithm_hopping(read_from_file)
        self.set_mcstep_hop_lock_and_epoch_hops_mcsteps_pct()
        self.set_epoch_hops_algos()

    def set_algorithm_hopping(self, read_from_file):

        if not read_from_file:
            if type(self.__sp__['mcsteps']) == int:
                self.mcsteps = self.__sp__['mcsteps']
            else:
                self.__lock__['mcsteps'] = True
            # ----------------------------------------
            # ----------------------------------------
            if type(self.__sp__['solver']) == str:
                self.solver = self.__sp__['solver']
            else:
                self.__lock__['solver'] = True
            # ----------------------------------------
            if type(self.__sp__['tgrad']) == np.ndarray:
                self.tgrad = self.__sp__['tgrad']
            else:
                self.tgrad = 'invalid'
                self.__lock__['tgrad'] = False
            # =================================================================
            self.algo_hop = self.__sp__['algo_hop']
            algo_hops = self.__sp__['algo_hops']
            # ----------------------------------------
            if not self.algo_hop:
                """If algorithm hopping is off (algo_hop=False), then, this
                branching helps set the algorithm (ps.gsi[:].uisim.mcalg) using
                the options provided in the algo_hops.
                """
                if type(algo_hops) in dth.dt.ITERABLES:
                    print(1)
                    """if options pertaining to algorithm hopping has
                    been provided by the user, then the first available
                    option pertaining to algorithm ID will be used to
                    set the algorithm. For example, if algo_hops is
                    [(200, 10), (201, 40), (202, 100)], then mcalg will be
                    set to '200'.
                    """
                    if algo_hops[0][0] in dth.valg.mc2d:
                        self.algo_hops = ((str(algo_hops[0][0]), 100), )
                    else:
                        self.algo_hops = ((self.default_mcalg, 100), )
                elif type(algo_hops) in dth.dt.NUMBERS or type(algo_hops) == str:
                    print(2)
                    """If a numerical entry has been made (in a case where the
                    user has done through direct access through set_param_sim),
                    then if it is valid, then str(value) will be set for mcalg.
                    If invalid, mcalg will default to '200'.
                    """
                    if algo_hops in dth.valg.mc2d:
                        self.algo_hops = ((str(int(algo_hops)), 100), )
                    else:
                        self.algo_hops = ((self.default_mcalg, 100), )
                else:
                    print(3)
                    """ This branch when user input could not be validated
                    or corrected.
                    """
                    print("MCALG could not be validated and/or corrected. Skipped")
                    self.__lock__['algo_hops'] = True
                    self.algo_hops = ((self.default_mcalg, 100), )
                # -------------------------------------
            else:
                """
                This involves two steps. First, validated mcalg array will be
                built. Then the mcsteps breakup will be validated based on
                values in algo_hops. Invalidities will be attempted to be
                corrected to enable simulation completion.
                """
                # STEP 1: Build validated mcalg array
                mcalg = ['invalid' for _ in self.N]
                self.algo_hops = [(None, None) for _ in self.N]
                if type(algo_hops) in dth.dt.ITERABLES:
                    print(5)
                    for n in self.N:
                        if algo_hops[n][0] in dth.valg.mc2d:
                            self.algo_hops[n][0] = str(algo_hops[n][0])
                        else:
                            self.algo_hops[n][0] = self.__default_mcalg__
                        mcalg[n-1] = self.algo_hops[n][0]
                elif type(algo_hops) in dth.dt.NUMBERS or type(algo_hops) == str:
                    print(6)
                    self.algo_hops = [('invalid', ) for _ in self.N]
                    for n in self.N:
                        if algo_hops in dth.valg.mc2d:
                            self.algo_hops[n][0] = str(int(algo_hops))
                        else:
                            self.algo_hops[n][0] = self.__default_mcalg__
                        mcalg[n-1] = self.algo_hops[n][0]
                else:
                    print(7)
                    mcalg = ['invalid' for _ in self.N]
                    self.__lock__['algo_hops'] = True
                    print('MCALG could not be validated and/or corrected. Skipped')
                self.gsi[n+1].uisim.mcalg = self.algo_hops[n][0]
                # STEP 2: Validate mcsteps breakup based on values in algo_hops
            # =================================================================
            # =================================================================
            self.save_at_mcsteps = [int(_) for _ in self.__sp__['save_at_mcsteps']]
        else:
            pass

    def set_mcstep_hop_lock_and_epoch_hops_mcsteps_pct(self):
        """
        Takes in the info from validated algo_hops and build up
        two secondary lists which are easier to handle in algorithm
        selection. In the process, it may decide to perform additional
        validations which may be required.
        """
        # --------------------------------------------------
        ''' STEP 1    GENERATE MONTE-CARLO EPOCHS
        Each pair of adjacent entries denotes a hop range, which is basically
        the range of monte-carlo iterations within which an algorithm works.
        As to which algorithm is it, which would be working, is isolated
        later on in this method space, but the data is already present in the
        self.algo_hops. Example: If epochs is [0, 20, 100], then there would be
        two epoch hops, which are [0, 20] and [20, 100]. As you can see, its
        actually a range of iteration numbers. Note; To be accurate, the epoch
        hops would actually be [0, 20] and [21, 100].
        '''
        epoch = [_[1] for _ in self.algo_hops]
        epoch[0], epoch[-1] = 0, 100
        # --------------------------------------------------
        '''STEP 2    GENERATE TEMPORARY LOCKS FOR EACH EPOCH HOP.
        If the epoch hops contains startings and endings, which are invalid,
        then the mcsteps_lock will be set to True for the specific epoch hop.
        Obviously, there will be as many elements in it as there are
        epoch hops.
        '''
        self.__mcstep_hop_locks__ = [False for _ in epoch]
        for i in range(len(epoch)):
            if i > 0:
                if epoch[i] < epoch[i-1]:
                    self.__mcstep_hop_locks__[i] = True
        print(f"__mcstep_hop_locks__: {self.__mcstep_hop_locks__}")
        # --------------------------------------------------
        '''STEP 3: generate __epoch_hops_mcsteps_pct__ list for each algorithm.
        The outer list contains lists of all epoxh hops. Each inner list
        contains ther start and end of the monte-carlo iteration number. At
        mc iterartion number of srtart, the corresponding algorithm will be
        switched to, from the previous algorithm, if there was one active.
        If any locks in __mcstep_hop_locks__ have been locked in STEP 2, then
        'mcstep_hops' sublock of the global uisim lock gets locked and
        everything stops.
        '''
        if any(self.__mcstep_hop_locks__):
            # Branch: Locked if True in it.
            self.__lock__['mcstep_hops'] = True
            self.__epoch_hops_mcsteps_pct__ = ['invalid' for _ in self.algo_hops]
            self.epoch_hops_mcsteps = ['invalid' for _ in self.algo_hops]
            print(f"{colored('Invalid algorithm hopping specification. LOCKED. ','red')}")
        else:
            # Branch: Open if True not in it.
            self.__epoch_hops_mcsteps_pct__ = []
            for i, ah in enumerate(self.algo_hops):
                if i == 0 and len(self.algo_hops) == 1:
                    self.__epoch_hops_mcsteps_pct__.append([0, 100])
                elif i == 0 and len(self.algo_hops) > 1:
                    self.__epoch_hops_mcsteps_pct__.append([0, epoch[i]])
                elif i > 0:
                    self.__epoch_hops_mcsteps_pct__.append([epoch[i-1]+1, epoch[i]])
            '''Set the actual mcstep start end values including validity check.
            Validity check includes:
                If self.mcsteps is too small to accomodate the algo_hops,
                then uisim parameters get LOCKED.
            '''
            self.epoch_hops_mcsteps = []
            for _mcsteps_ in self.__epoch_hops_mcsteps_pct__:
                starting_mcstep = int(_mcsteps_[0]*self.mcsteps/100)
                c = int(_mcsteps_[1]*self.mcsteps/100)
                self.epoch_hops_mcsteps.append([starting_mcstep,
                                                starting_mcstep])
            self.epoch_hops_mcsteps[-1][1] = self.mcsteps
        print(f"__epoch_hops_mcsteps_pct__: {self.__epoch_hops_mcsteps_pct__}")

    def set_epoch_hops_algos(self):
        if not self.__lock__['mcstep_hops']:
            self.epoch_hops_algos = [am[0] for am in self.algo_hops]
        else:
            self.epoch_hops_algos = ['invalid' for _ in self.algo_hops]
        print(f"epoch_hops_algos:  {self.epoch_hops_algos}")

    def set_kbt(self):
        self.consider_boltzmann_probability = bool(self.__sp__['consider_boltzmann_probability'])
        self.s_boltz_prob = self.__sp__['s_boltz_prob']
        self.boltzmann_temp_factor_max = self.__sp__['boltzmann_temp_factor_max']

    def set_non_locality(self):
        self.NL = int(self.__sp__['NL'])

    def set_s(self):
        if type(self.__sp__['nstates']) == int:
            self.S = self.__sp__['nstates']
            self.nstates = self.S
        else:
            self.__lock__['nstates'] = True
        self.state_sampling_scheme = self.__sp__['state_sampling_scheme']

    @property
    def lock_status(self):
        if self.__lock__['_']:
            print(f"{colored('Locked sub-locks exist', 'red', attrs=['bold'])}")
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


class _manual_uidata_mcgs_gsc_par_:
    """
    Parameters for grain strcuture characterisation
    """
    DEV = True
    __lock__ = {'char_grains': False,
                'char_stage': False,
                'library': False,
                'parallel': False,
                'find_gbseg': False,
                '_': True
                }
    __slots__ = ('char_grains', 'char_stage', 'library', 'parallel',
                 'g_area', 'gb_length', 'find_gbseg',
                 'gb_length_crofton', 'gb_njp_order', 'g_eq_dia',
                 'g_feq_dia', 'g_solidity', 'g_circularity',
                 'g_mjaxis', 'g_mnaxis', 'g_morph_ori',
                 'g_el', 'g_ecc',
                 )

    def __init__(self,
                 char_grains=True, char_stage='postsim',
                 library='scikit-image', parallel=True,
                 find_gbseg=True, g_area=True, gb_length=True,
                 gb_length_crofton=True, gb_njp_order=True,
                 g_eq_dia=True, g_feq_dia=True, g_solidity=True,
                 g_circularity=True, g_mjaxis=True, g_mnaxis=True,
                 g_morph_ori=True, g_el=True, g_ecc=True,
                 read_from_file=False, filename=None
                 ):
        if not read_from_file:
            self.char_grains, self.char_stage = char_grains, char_stage
            self.library, self.parallel = library, parallel
            self.find_gbseg = find_gbseg
            self.g_area, self.gb_length = g_area, gb_length
            self.gb_length_crofton = gb_length_crofton
            self.gb_njp_order = gb_njp_order
            self.g_eq_dia, self.g_feq_dia = g_eq_dia, g_feq_dia
            self.g_solidity, self.g_circularity = g_solidity, g_circularity
            self.g_mjaxis, self.g_mnaxis = g_mjaxis, g_mnaxis
            self.g_morph_ori = g_morph_ori
            self.g_el, self.g_ecc = g_el, g_ecc
        else:
            pass
        self.lock_update

    def __repr__(self):
        _ = ' '*5
        retstr = "Grain structure characterisation parameters: \n"
        retstr += _ + f"{colored('CHAR_GRAINS', 'red', attrs=['bold'])}: {colored(self.char_grains, 'cyan')}\n"
        retstr += _ + f"{colored('CHAR_STAGE', 'red', attrs=['bold'])}: {colored(self.char_stage, 'cyan')}\n"
        retstr += _ + f"{colored('LIBRARY', 'red', attrs=['bold'])}: {colored(self.library, 'cyan')}\n"
        retstr += _ + f"{colored('PARALLEL', 'red', attrs=['bold'])}: {colored(self.parallel, 'cyan')}\n"
        retstr += _ + f"{colored('G_AREA', 'red', attrs=['bold'])}: {colored(self.g_area, 'cyan')}\n"
        retstr += _ + f"{colored('GB_LENGTH', 'red', attrs=['bold'])}: {colored(self.gb_length, 'cyan')}\n"
        retstr += _ + f"{colored('FIND_GBSEG', 'red', attrs=['bold'])}: {colored(self.find_gbseg, 'cyan')}\n"
        retstr += _ + f"{colored('GB_LENGTH_CROFTON', 'red', attrs=['bold'])}: {colored(self.gb_length_crofton, 'cyan')}\n"
        retstr += _ + f"{colored('GB_NJP_ORDER', 'red', attrs=['bold'])}: {colored(self.gb_njp_order, 'cyan')}\n"
        retstr += _ + f"{colored('G_EQ_DIA', 'red', attrs=['bold'])}: {colored(self.g_eq_dia, 'cyan')}\n"
        retstr += _ + f"{colored('G_FEQ_DIA', 'red', attrs=['bold'])}: {colored(self.g_feq_dia, 'cyan')}\n"
        retstr += _ + f"{colored('G_SOLIDITY', 'red', attrs=['bold'])}: {colored(self.g_solidity, 'cyan')}\n"
        retstr += _ + f"{colored('G_CIRCULARITY', 'red', attrs=['bold'])}: {colored(self.g_circularity, 'cyan')}\n"
        retstr += _ + f"{colored('G_MJAXIS', 'red', attrs=['bold'])}: {colored(self.g_mjaxis, 'cyan')}\n"
        retstr += _ + f"{colored('G_MNAXIS', 'red', attrs=['bold'])}: {colored(self.g_mnaxis, 'cyan')}\n"
        retstr += _ + f"{colored('G_MORPH_ORI', 'red', attrs=['bold'])}: {colored(self.g_morph_ori, 'cyan')}\n"
        retstr += _ + f"{colored('G_EL', 'red', attrs=['bold'])}: {colored(self.g_el, 'cyan')}\n"
        retstr += _ + f"{colored('G_ECC', 'red', attrs=['bold'])}: {colored(self.g_ecc, 'cyan')}\n"
        return retstr

    @property
    def lock_status(self):
        if self.__lock__['_']:
            print(f"{colored('Locked sub-locks exist', 'red', attrs=['bold'])}")
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


class _manual_uidata_mcgs_generate_geom_reprs_():
    """
    make_mp_grain_centoids: bool :: Make MP of grain_centroids
    make_mp_grain_points: bool :: Grains as multi-point
    make_ring_grain_boundaries: bool :: GB as UPXO ring
    make_xtal_grain: bool :: Grains as UPXO XTAL object
    make_chull_grain: bool :: Make convex hull for each grain
    create_gbz: bool :: create_grain_boundary_zone

    CALL:
        from mcgs import _uidata_mcgs_generate_geom_reprs_
        uidata_georepr = _uidata_mcgs_generate_geom_reprs_(uidata)
    """
    DEV = True
    __lock__ = {'mp': False,
                'ring': False,
                'xtal': False,
                'chull': False,
                'gbz': False,
                '_': True
                }
    __slots__ = ('make_mp_grain_centoids', 'make_mp_grain_points',
                 'make_ring_grain_boundaries', 'make_xtal_grain',
                 'make_chull_grain', 'create_gbz', 'gbz_thickness',
                 )

    def __init__(self,
                 make_mp_grain_centoids=True,
                 make_mp_grain_points=True,
                 make_ring_grain_boundaries=True,
                 make_xtal_grain=True,
                 make_chull_grain=True,
                 create_gbz=True,
                 gbz_thickness=0.1,
                 read_from_file=False, filename=None
                 ):
        self.make_mp_grain_centoids = make_mp_grain_centoids
        self.make_mp_grain_points = make_mp_grain_points
        self.make_ring_grain_boundaries = make_ring_grain_boundaries
        self.make_xtal_grain = make_xtal_grain
        self.make_chull_grain = make_chull_grain
        self.create_gbz = create_gbz
        self.gbz_thickness = gbz_thickness
        self.lock_update

    def __repr__(self):
        _ = ' '*5
        retstr = "Geometric representation parameters: \n"
        retstr += _ + f"{colored('MAKE_MP_GRAIN_CENTOIDS', 'red', attrs=['bold'])}: {colored(self.make_mp_grain_centoids, 'cyan')}\n"
        retstr += _ + f"{colored('MAKE_MP_GRAIN_POINTS', 'red', attrs=['bold'])}: {colored(self.make_mp_grain_points, 'cyan')}\n"
        retstr += _ + f"{colored('MAKE_RING_GRAIN_BOUNDARIES', 'red', attrs=['bold'])}: {colored(self.make_ring_grain_boundaries, 'cyan')}\n"
        retstr += _ + f"{colored('MAKE_XTAL_GRAIN', 'red', attrs=['bold'])}: {colored(self.make_xtal_grain, 'cyan')}\n"
        retstr += _ + f"{colored('MAKE_CHULL_GRAIN', 'red', attrs=['bold'])}: {colored(self.make_chull_grain, 'cyan')}\n"
        retstr += _ + f"{colored('CREATE_GBZ', 'red', attrs=['bold'])}: {colored(self.create_gbz, 'cyan')}\n"
        return retstr

    @property
    def lock_status(self):
        if self.__lock__['_']:
            print(f"{colored('Locked sub-locks exist', 'red', attrs=['bold'])}")
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


class _manual_uidata_mesh_():
    DEV = True
    __lock__ = {'mesh': False,
                'conformal': False,
                'elgradient': False,
                'optimize': False,
                'cps3': True,
                'cps4': False,
                'cps6': True,
                'cps8': True,
                'c3d4': True,
                'c3d6': True,
                'c3d8': True,
                '_': True
                }
    __slots__ = ('generate_mesh',
                 'target_fe_software',
                 'par_treatment',
                 'mesher',
                 'gb_conformities',
                 'global_elsizes',
                 'mesh_algos',
                 'grain_internal_el_gradient',
                 'grain_internal_el_gradient_par',
                 'target_eltypes',
                 'elsets',
                 'nsets',
                 'optimize',
                 'opt_par',
                 )

    def __init__(self,
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
                 read_from_file=False, filename=None
                 ):
        """
        Please refer documentation of parameter_sweep.set_param_mesh()
        """
        if not read_from_file:
            self.generate_mesh = generate_mesh
            self.target_fe_software = target_fe_software
            self.par_treatment = par_treatment
            self.mesher = mesher
            self.gb_conformities = gb_conformities
            self.global_elsizes = global_elsizes
            self.mesh_algos = mesh_algos
            self.grain_internal_el_gradient = grain_internal_el_gradient
            self.grain_internal_el_gradient_par = grain_internal_el_gradient_par
            self.target_eltypes = target_eltypes
            self.elsets = elsets
            self.nsets = nsets
            self.optimize = optimize
            self.opt_par = opt_par
        else:
            pass
        self.lock_update

    def __repr__(self):
        _ = ' '*5
        retstr = "Attributes of meshing: \n"
        retstr += _ + f"{colored('GENERATE_MESH', 'red', attrs=['bold'])}: {colored(self.generate_mesh, 'cyan')}\n"
        retstr += _ + f"{colored('TARGET_FE_SOFTWARE', 'red', attrs=['bold'])}: {colored(self.target_fe_software, 'cyan')}\n"
        retstr += _ + f"{colored('PAR_TREATMENT', 'red', attrs=['bold'])}: {colored(self.par_treatment, 'cyan')}\n"
        retstr += _ + f"{colored('MESHER', 'red', attrs=['bold'])}: {colored(self.mesher, 'cyan')}\n"
        retstr += _ + f"{colored('GB_CONFORMITIES', 'red', attrs=['bold'])}: {colored(self.gb_conformities, 'cyan')}\n"
        retstr += _ + f"{colored('GLOBAL_ELSIZES', 'red', attrs=['bold'])}: {colored(self.global_elsizes, 'cyan')}\n"
        retstr += _ + f"{colored('MESH_ALGOS', 'red', attrs=['bold'])}: {colored(self.mesh_algos, 'cyan')}\n"
        retstr += _ + f"{colored('GRAIN_INTERNAL_EL_GRADIENT', 'red', attrs=['bold'])}: {colored(self.grain_internal_el_gradient, 'cyan')}\n"
        retstr += _ + f"{colored('GRAIN_INTERNAL_EL_GRADIENT_PAR', 'red', attrs=['bold'])}: {colored(self.grain_internal_el_gradient_par, 'cyan')}\n"
        retstr += _ + f"{colored('TARGET_ELTYPES', 'red', attrs=['bold'])}: {colored(self.target_eltypes, 'cyan')}\n"
        retstr += _ + f"{colored('ELSETS', 'red', attrs=['bold'])}: {colored(self.elsets, 'cyan')}\n"
        retstr += _ + f"{colored('NSETS', 'red', attrs=['bold'])}: {colored(self.nsets, 'cyan')}\n"
        retstr += _ + f"{colored('OPTIMIZE', 'red', attrs=['bold'])}: {colored(self.optimize, 'cyan')}\n"
        retstr += _ + f"{colored('OPT_PAR', 'red', attrs=['bold'])}: {colored(self.opt_par, 'cyan')}\n"
        return retstr

    @property
    def lock_status(self):
        if self.__lock__['_']:
            print(f"{colored('Locked sub-locks exist', 'red', attrs=['bold'])}")
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


class mcrepr():
    """
    Representativeness qualificartion
    ----------------------------------
    target_type: str
        Source of targer data. Options:
            1. ebsd0 - un-processed 2D EBSD map: DefDAP object.
            2. ebsd1 - processed DefDAP data. Remapped with avg. ori.
            3. umc2 - UPXO Monte-Carlo Grain structure 2D.
            4. umc3 - UPXO Monte-Carlo Grain structure 3D.
            5. uvt2 - UPXO Voronoi-Tessellation Grain Structure 2D.
            6. stats - Data samples across grain morphology par. Needs xori.
                       Could be in the form of dictionary or panadas dataframe.
                       If dict or pandas dataframe, key or column name
                       respectively, must be name of the parameter.
                       Examples of parameter names include:
                           1. area, perimeter
                           2. aspecrt ratio, morphologhical orientation
    ----------------------------------
    target: object
        Target grain structure data. Details:
            1. `MCGS.gs[tslice]` for umc2 and umc3
            2. `VTGS` for uvt2
            3. ddap_ebsd - for un-processed or processed DefDAP data
    ----------------------------------
    samples: dict
        Samples to match against the target.
        Keys should be sample_names
        Values should contain either:
            grain structure objects, or
            flag-string, 'make'
        If a value is a grain strucutre object, then it will be used as
        samples. It can be of types (a) umc2, (b) umc3 and (c) uvt2
        If a value is 'make', then the following will be performanceormed:
            1. read the excel file for grain structure generation parameters
            2. simulate the grain structure evolution
            3. Pull out specified slices at specified temporal slice intervals
            4. Characterize the temporal slices
    ----------------------------------
    par_bounds: dict
        DESCRIPTION:
            For each parameter in the key, value must be a list of:
                [match bounds for peak locations in percentage,
                 match bounds for peak location density in percentage,
                 J-S test bounds
                 ]
        KEYS:
            area, perimeter, aspect ratio
        VALUES:
            bounds: [ [5, 5], [5, 5], [0.1, 0.1]]
    ----------------------------------
    metrics: list
        DESCRIPTION:
            List of metrics to use to enable representativeness qualification
            Examples include:
                1. modes_n
                2. modes_loc
                3. modes_width
                4. distr_type
                5. skewness
                6. kurtosis
    ----------------------------------
    kde_options: dict
        DESCRIPTION:
            key: bw_method
            value: choose from 'scott', 'silverman' or a scalar value
    ----------------------------------
    """
    __slots__ = ('target_type',
                 'target',
                 'samples',
                 'par_bounds',
                 'metrics',
                 'kde_options',
                 'stat_tests',
                 'test_threshold',
                 'stest',
                 'test_metrics',
                 'parameters',
                 'distr_type',
                 'performance'  # Performance
                 )

    def __init__(self,
                 target_type=None,
                 target=None,
                 samples=None,
                 par_bounds=None,
                 metrics=None,
                 kde_options=None,
                 stest={'tests': ['correlation',
                                  'kldiv',
                                  'ks',
                                  'jsdiv',
                                  'mannwhitneyu',
                                  'kruskalwallis',
                                  ],
                        'mw_p_threshold': 0.90,
                        'kw_p_threshold': 0.90,
                        'ks_p_threshold': 0.90,
                        },
                 test_metrics=['mode0_location',
                               'mode0_count',
                               'mode1_location',
                               'mode1_count',
                               'mean',
                               ],
                 parameters=['area',
                             ],
                 ):
        """
        This is a core UPXO class and has the following functions:

            * Caclulate type of statistical distribution of the specified
              morphological properties of the target grain structure
              and sample grain structures.

            * Estimate statistical similarity between the target grain
              structure and each of the "samples" grain structures

            * Provide an acceptance flag for each samples grain structures
        """
        self.target_type = target_type
        self.target = target
        self.samples = samples
        self.par_bounds = par_bounds
        self.metrics = metrics
        self.kde_options = kde_options
        self.stest = stest
        self.test_metrics = test_metrics
        self.parameters = parameters
        self.performance = {}
        # from scipy.stats import gaussian_kde

    def load_target(self,
                    target=None,
                    target_type=None):
        self.target = target
        self.target_type = target_type

    def load_samples(self,
                     samples=None):
        if type(samples) in dth.dt.ITERABLES:
            self.samples = samples
        else:
            print('samples must be of the type list.')

    def add_sample(self,
                   sample=None):
        if sample:
            self.samples.append(sample)

    def set_stests(self,
                   tests):
        self.stest['tests'] = tests

    def set_cor_thresh(self,
                       cor_threshold):
        while cor_threshold < 0 or cor_threshold > 1:
            self.stest['cor_threshold'] = float(input("cor_threshold [0, 1]: "))

    def set_kldiv_thresh(self,
                         kldiv_thresh):
        while kldiv_thresh < 0 or kldiv_thresh > 1:
            self.stest['kldiv_thresh'] = float(input("kldiv_thresh [0, 1]: "))

    def set_ks_thresh(self,
                      ks_thresh_D,
                      ks_thresh_P):
        while ks_thresh_D < 0 or ks_thresh_D > 1:
            self.stest['ks_thresh_D'] = float(input("ks_thresh_D [0, 1]: "))
        while ks_thresh_P < 0 or ks_thresh_P > 1:
            self.stest['ks_thresh_P'] = float(input("ks_thresh_P [0, 1]: "))

    def set_jsdiv_thresh(self,
                         jsdiv_thresh):
        while jsdiv_thresh < 0 or jsdiv_thresh > 1:
            self.stest['jsdiv_thresh'] = float(input("jsdiv_thresh [0, 1]: "))

    def prop_to_excel(self,
                      filename="pxtal_properties",
                      ):
        with pd.ExcelWriter(f"{filename}.xlsx") as writer:
            self.target.prop.to_excel(writer,
                                      sheet_name='target',
                                      index=False)
            for i, sample in enumerate(self.samples.values(), start=1):
                sample.prop.to_excel(writer,
                                     sheet_name=f"sample{i}",
                                     index=False
                                     )

    def build_distribution_dataset(self):
        self.distr_type = {'target': {}}
        for sample_name in self.samples.keys():
            self.distr_type[sample_name] = {}
        for key in self.distr_type.keys():
            for parameter in self.parameters:
                self.distr_type[key][parameter] = {'right_skewed': None,
                                                   'left_skewed': None,
                                                   'leptokurtic': None,
                                                   'platykurtic': None,
                                                   'normal': None,
                                                   'kurtosis': None,
                                                   'skewness': None
                                                   }

    def determine_distr_type(self):
        self.build_distribution_dataset()
        from scipy.stats import skew
        from scipy.stats import kurtosis
        from scipy.stats import shapiro

        for parameter_name in self.parameters:
            target_skewness = skew(self.target.prop[parameter_name])
            target_kurt = kurtosis(self.target.prop[parameter_name])
            shapiro_stat, shapiro_p = shapiro(self.target.prop[parameter_name])
            self.distr_type['target'][parameter_name]['skewness'] = target_skewness
            self.distr_type['target'][parameter_name]['kurtosis'] = target_kurt
            if target_skewness > 0:
                self.distr_type['target'][parameter_name]['right_skewed'] = True
                if target_kurt > 0:
                    self.distr_type['target'][parameter_name]['leptokurtic'] = True
                else:
                    self.distr_type['target'][parameter_name]['platykurtic'] = True
            else:
                self.distr_type['target'][parameter_name]['left_skewed'] = True
                if target_kurt > 0:
                    self.distr_type['target'][parameter_name]['leptokurtic'] = True
                else:
                    self.distr_type['target'][parameter_name]['platykurtic'] = True
            if abs(target_skewness) < 0.5 and abs(target_kurt) < 1 and shapiro_p > 0.05:
                self.distr_type['target'][parameter_name]['normal'] = True
            else:
                self.distr_type['target'][parameter_name]['normal'] = False

        for sample_name, sample in self.samples.items():
            for parameter_name in self.parameters:
                sample_skewness = skew(sample.prop[parameter_name])
                sample_kurt = kurtosis(sample.prop[parameter_name])
                stat, p = shapiro(sample.prop[parameter_name])
                self.distr_type[sample_name][parameter_name]['skewness'] = target_skewness
                self.distr_type[sample_name][parameter_name]['kurtosis'] = target_kurt
                if sample_skewness > 0:
                    self.distr_type[sample_name][parameter_name]['right_skewed'] = True
                    if sample_kurt > 0:
                        self.distr_type[sample_name][parameter_name]['leptokurtic'] = True
                    else:
                        self.distr_type[sample_name][parameter_name]['platykurtic'] = True
                else:
                    self.distr_type[sample_name][parameter_name]['left_skewed'] = True
                    if sample_kurt > 0:
                        self.distr_type[sample_name][parameter_name]['leptokurtic'] = True
                    else:
                        self.distr_type[sample_name][parameter_name]['platykurtic'] = True
                if abs(sample_skewness) < 0.5 and abs(sample_kurt) < 1 and shapiro_p > 0.05:
                        self.distr_type[sample_name][parameter_name]['normal'] = True
                else:
                    self.distr_type[sample_name][parameter_name]['normal'] = False

    def test(self):
        """
        TEST 1: correlation: For two datasets, it is a measure of the linear
        relationship between them. If correlation is close to 1 then, the
        distributions are very similar.

        TEST 2: kldiv:

        TEST 3: ks: Kolmogorov-Smirnov test: Determines of the two distribution
        samples differ significantly. It uses cumulative distributions of the
        two datasets. Retyurns D-statistic and P-value.
            * D-statistic: maximum absolute difference of the cumulative
            distributions (absolute max distance (supremum) b/w the CDFs
            of the two samples). A smaller D-static value is indicative of
            similar distributions.
            * P-value: probability that thwe tywo distributions are similar. If
            p-value is low (<= 0.05), distributions are different. If p-value
            is high (> 0.05), we cannot reject the null-hypothesis that the
            two distributions are the same.
            * Note: if P <= 0.05: the null hypothesis that the two samples are
            drawn from tyhe sample sample can be rejected, indicating that the
            samples are not representative of the target

        TEST 4: jsdiv: P value will allways be between 0 and 1.
        @ 0: Distributions are identical. @ 1: Distributions are completely
        different

        TEST 5: mannwhitneyu: Mann-Whitney test: Used to determine if two '
        distribution samples are drawn from a population having the same
        population. If P-value is less than or equal to 0.05, then different
        distributiopns. If P-value is > 0.05, then the two disrtirbutions
        are similar.

        TEST 6: kruskalwallis: Kruskal-wallis test. Used to determine if there
        are statistically significant differences between two distributions.
        """
        if 'kldiv' in self.stest['tests']:
            from scipy.stats import entropy
        if 'jsdiv' in self.stest['tests']:
            from scipy.spatial.distance import jensenshannon
        if 'ks' in self.stest['tests']:
            from scipy.stats import ks_2samp
        if 'mannwhitneyu' in self.stest['tests']:
            from scipy.stats import mannwhitneyu
        if 'kruskalwallis' in self.stest['tests']:
            from scipy.stats import kruskal
        if self.stest['tests']:
            # Iterate through each of the sample object
            for sample_name, sample in self.samples.items():
                print('-----------sample-----------')
                self.performance[sample_name] = {}
                for ipar, par in enumerate(self.parameters, start=1):
                    self.performance[sample_name][par] = {}
                    for test in self.stest['tests']:
                        self.performance[sample_name][par][test] = None
                        if test == 'correlation':
                            correlation = self.target.prop[par].corr(sample.prop[par])
                            self.performance[sample_name][par][test] = correlation
                        # -------------------------------------
                        if test == 'kldiv':
                            print('kldiv test not available')
                        # -------------------------------------
                        if test == 'ks':
                            ks_D, ks_P = ks_2samp(self.target.prop[par],
                                                  sample.prop[par])
                            self.performance[sample_name][par][test] = (ks_D,
                                                                         ks_P)
                        # -------------------------------------
                        if test == 'jsdiv':
                            # TODO: DEBUG the length mismatch
                            # SOLn: Make KDE and resample data iteratively
                            # based on user satisfaction of number of bins in
                            # histogram and bandwidth in KDE calculation
                            pass
                            #js_P = jensenshannon(self.target.prop[par],
                            #                     sample.prop[par])
                            #self.performance[sample_name][par][test] = js_P
                        # -------------------------------------
                        if test == 'mannwhitneyu':
                            mwu_D, mwu_P = mannwhitneyu(self.target.prop[par].dropna(),
                                                        sample.prop[par].dropna())
                            self.performance[sample_name][par][test] = (mwu_D,
                                                                        mwu_P)
                        # -------------------------------------
                        if test == 'kruskalwallis':
                            kw_D, kw_P = kruskal(self.target.prop[par].dropna(),
                                                 sample.prop[par].dropna())
                            self.performance[sample_name][par][test] = (kw_D,
                                                                        kw_P)
                        # -------------------------------------

class ebsd_data():
    __slots__ = ('map_raw',
                 'map',
                 'gid',
                 'ea_avg',
                 'prop',
                 'n',
                 'quat_avg',
                 'filename'
                 )
    def __init__(self,
                 filename=None,
                 ):
        from defdap.quat import Quat
        self.filename = filename  # Cugrid_after 2nd_15kv_2kx_2
        self.load_ctf()

    def load_ctf(self):
        import defdap.ebsd as DEFDAP_EBSD
        self.map_raw = DEFDAP_EBSD.Map(self.filename,
                                       dataType="OxfordText")
        self.map = deepcopy(self.map_raw)

    def build_quatarray(self):
        self.map.buildQuatArray()

    def detect_grains(self, size_min=10):
        self.map.findGrains(minGrainSize=size_min)

    def calc_grain_ori_avg(self):
        self.map.calcGrainMisOri()

"""
algorithm वृत्ति
iterative refinements कुट्टक
"""
