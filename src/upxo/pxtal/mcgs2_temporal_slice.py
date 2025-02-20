import os
import math
import numpy as np
import cv2
import random
import seaborn as sns
from pathlib import Path
from copy import deepcopy
from typing import Iterable
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from skimage.measure import regionprops
from scipy.ndimage import generic_filter
from scipy.interpolate import RegularGridInterpolator
# from skimage.measure import label as skim_label
# from upxo.geoEntities.point2d import point2d
# from scipy.ndimage import label, generate_binary_structure
from upxo.meshing.mesher_2d import mesh_mcgs2d
from upxo.dclasses.features import twingen
# from upxo.geoEntities.mulpoint2d import mulpoint2d
from upxo.xtal.mcgrain2d_definitions import grain2d
from upxo._sup.gops import att
from upxo._sup.data_ops import find_intersection, find_union_with_counts
from upxo._sup.data_ops import increase_grid_resolution, decrease_grid_resolution
from upxo._sup import dataTypeHandlers as dth
from upxo._sup.validation_values import _validation
from upxo._sup.data_templates import pd_templates
from upxo._sup.data_templates import dict_templates
from upxo._sup.console_formats import print_incrementally
from upxo.geoEntities.sline2d import Sline2d as sl2d

class mcgs2_grain_structure():
    """
    SLOTS
    -----
    dim: Dimensionality of the grain structure
    uigrid: Copy of uigrid datastructure
    uimesh: Copy of uimesh datastructure
    xgr: min, incr, max of x-axis
    ygr: min, incr, max of y-axis
    zgr: min, incr, max of z-axis
    m: MC temporal step to which this GS belongs to.
    s: State array
    S: Total number of states
    binaryStructure2D: 2D Binary Structure to identify grains
    binaryStructure3D: 3D Binary Structure to identify grains
    n: Number of grains
    lgi: Lattice of Grains Ids
    spart_flag: State wise partitioning
    gid: Grain numbers used as grain IDs
    s_gid: DICT: {s: overall grain id i.e grain number}
    gid_s: LIST: [a, b, c, ...] see explanation below.
    s_n: DICT: State partitioned number of grains
    g: DICT: grains
    gb: DICT: grains
    positions: DICT: gids as per spatial location string
    mp: DICT: UPXO mul-point objects
    vtgs: DICT: VTGS instances
    mesh: OBJECT: mesh data structure
    px_size: FLOAT: pixel area if dim=2 else volume of dim=3
    dim: INT: DImensionaality
    prop_flag: DICT: flags indicating variables to compute
    prop: PANDAS TABLE of properties
    are_properties_available: True if properties have been caculated
    prop_stat: PANDAS TABLE of property statistics
    __gi__: Grain index used for __iter__
    uinputs: Stores original user inp used by grid() instance
    display_messages',
    info',
    print_interval_bool',
    EAPGLB', # EA Primary Global
    EASGLB', # EA Secondary Global



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

    """
    __slots__ = ('dim', 'uigrid', 'uimesh', 'xgr', 'ygr', 'zgr', 'm', 's', 'S',
                 'binaryStructure2D', 'binaryStructure3D', 'n', 'lgi',
                 'spart_flag', 'gid', 's_gid', 'gid_s', 's_n', 'g', 'gb',
                 'positions', 'mp', 'vtgs', 'mesh', 'px_size', 'dim',
                 'prop_flag', 'prop', 'are_properties_available', 'prop_stat',
                 '__gi__', 'uinputs', 'display_messages', 'info',
                 'print_interval_bool', 'EAPGLB', 'EASGLB',
                 '__ori_assign_status_stack__', '__ori_assign_status_slice__',
                 'scaled', 'scaled_gs', '__resolution_state__', 'gbjp',
                 'xomap', 'val', 'neigh_gid', 'valid_mprops', 'features',
                 'twingen', 'pxtal', '_gid_bf_merger_'
                 )

    EPS = 1e-12
    __maxGridSizeToIgnoreStoringGrids = 250**2

    def __init__(self, dim=2, m=None, uidata=None, S_total=None, px_size=None,
                 xgr=None, ygr=None, zgr=None, uigrid=None, uimesh=None,
                 EAPGLB=None, assign_ori_stack=False, assign_ori_slice=True,
                 oripert_tc=True, oripert_gr=True):
        self.uinputs = uidata
        self.val = _validation()
        self.dim, self.m, self.S, self.px_size = 2, m, S_total, px_size
        self.uigrid, self.uimesh = uigrid, uimesh
        self.xgr, self.ygr = xgr, ygr
        self.set__spart_flag(S_total)
        self.set__s_gid(S_total)
        self.set__gid_s()
        self.set__s_n(S_total)
        self.g, self.gb, self.info = {}, {}, {}
        self.EAPGLB = {}
        self.EAPGLB['statewise'] = EAPGLB
        self.EASGLB = self.EAPGLB
        # Above EASGLB needs to be updated in the orinetation mapping stage
        self.mp = dict_templates.mulpnt_gs2d
        self.scaled = {'xmin': None, 'xmax': None, 'xinc': None, 'xgr': None,
                       'ymin': None, 'ymax': None, 'yinc': None, 'ygr': None,
                       's': None, 'grains': None, 'prop': None}
        self.scaled_gs = None
        self.are_properties_available, self.display_messages = False, False
        self.__setup__positions__()
        self.xomap = None
        self.neigh_gid = None

        if assign_ori_stack:
            self.__ori_assign_status_stack__ = {'status': False,
                                                'info': 'to be developed'}
        if assign_ori_slice:
            __info = "..-t:u-s:u-..-ru-..-d:c-..-ea:s-.."
            self.__ori_assign_status_slice__ = {'status': True,
                                                'info': __info}
            # DOCUMENTATION:ad
            '''
            # TODO: REPLACE THIS LONG NAME BY SIMPLER ALPHA-NUMERIC CODES
            status: True or False
            info options:
                "..-t:u-s:u-..-ru-..-d:c-..-ea:s-.."
                        Temporally Spatially Untracked, Random perturbation,
                        Cartesian, Euler angle Sepoerate.
                    -t:u-s:u-
                        No temporal tracking, as noi relationshiop between gid
                        and spatial location of gids is tracked !
                    -ru-
                        Perturbations will be introduced randomly, respectivng
                        uniformity (i.e. uniform random numbers..)
                    -d:c-
                        Cartesian distance measure employed. No consideration
                        will be paid to the crystallographic misorientations
                        and symmetries. Rather, the perturbation would be the
                        Cartesian distance from the mean state wise assigned
                        orientation.
                    -ea:s-
                        Euler angle Seperate. EA1, EA2 and EA3 will have
                        different levels of perturbations.

                    EXPLANATION:
                        Orientations are assigned afresh for each temporal
                        slice, on top of the orientations in self.EAPGLB. This
                        is done by introducung perturbations. Perturbations
                        are defined two ways: state wise and euler angle wise,
                        which are expolained later. As a consequence, for
                        non-zero perturbation values, spatial ori distr in a
                        gs temporal slice have no relationship between previosu
                        and next temporal slices. Example: a smaller grain in a
                        previous temporal, which has now grow in sizew in the
                        current temporal slice may have a completely different
                        crys. ori. in the present temporal slice. The distance
                        of this difference, however, is defined by the max
                        perturbation values, used by the user.
            if TRUE, the following will be considered
            oripert_tc:
                Template: {'bool': True/False,
                           glb_pert_min_ea1=0, glb_pert_max_ea1=7.5,
                           glb_pert_min_ea2=0, glb_pert_max_ea2=7.5,
                           glb_pert_min_ea3=0, glb_pert_max_ea3=7.5,
                           }
            oripert_gr
            '''
        self.valid_mprops = {'npixels': False,
                             'npixels_gb': False,
                             'area': True,
                             'eq_diameter': False,
                             'perimeter': False,
                             'perimeter_crofton': False,
                             'compactness': False,
                             'gb_length_px': False,
                             'aspect_ratio': False,
                             'solidity': False,
                             'morph_ori': False,
                             'circularity': False,
                             'eccentricity': False,
                             'feret_diameter': False,
                             'major_axis_length': False,
                             'minor_axis_length': False,
                             'euler_number': False,
                             'char_grain_positions': False}
        self.pxtal = {}

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
        return 'grains :: att : n, lgi, id, ind, spart'

    def __att__(self):
        return att(self)

    @property
    def get_px_size(self):
        '''Get size of the pixel.'''
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

    def set__s_gid(self, S_total):
        """
        Set up dict of s as keys with None values.

        Parameters
        ----------
        S_total : int
            Specify the total number of states.

        Returns
        -------
        None
        """
        self.s_gid = {s: None for s in range(1, S_total+1)}

    def set__gid_s(self):
        '''Set up empty list. This would contain list of s values for every gid.'''
        self.gid_s = []

    def set__spart_flag(self, S_total):
        self.spart_flag = {_s_: False for _s_ in range(1, S_total+1)}

    def _check_lgi_dtype_uint8(self, lgi):
        """Validates and modifies (if needed) lgi user input data-type."""
        if type(lgi) == np.ndarray and np.size(lgi) > 0 and np.ndim(lgi) == 2:
            if self.lgi.dtype.name != 'uint8':
                self.lgi = lgi.astype(np.uint8)
            else:
                self.lgi = lgi
        else:
            self.lgi = 'invalid mcgs 4685'

    def calc_num_grains(self, throw=False):
        """Calculate the total number of grains in this grain structure."""
        if self.lgi:
            self.n = self.lgi.max()
            if throw:
                return self.n

    def find_neigh(self):
        '''Find the gids of neighbours for every gid.'''
        for idx, _gid_ in enumerate(self.gid):
            if (idx+1) % 100 == 0:
                print(f'Extracting neigh list for grain: {_gid_}')
            self.find_neigh_gid(_gid_)

    def find_neigh_gid(self, gid, throw=False):
        # Find the gids of neighbours of a given gid.
        bounds = self.g[gid]['grain'].bbox_ex_bounds
        probable_grains_locs = self.lgi[bounds[0]:bounds[1]+1,
                                        bounds[2]:bounds[3]+1]
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
        """ if values in probable_grains_locs not equal to -1, then replace
        them with 0 """
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
        """ Store the neighbnour_ids inside this GS instance """
        self.neigh_gid = {_gid_: self.g[_gid_]['grain'].neigh
                          for _gid_ in self.gid}
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

    def find_extended_bounding_box(self, gid):
        """
        Find the extended bounded box of a given gid.

        Example
        -------
        from upxo.ggrowth.mcgs import mcgs
        pxtal = mcgs(study='independent',
                     input_dashboard='input_dashboard_for_testing_50x50_alg202.xls')
        pxtal.simulate()
        pxtal.detect_grains()
        pxtal.gs[16].find_extended_bounding_box(10)
        """
        loc = np.where(self.lgi == gid)
        xi, xj, yi, yj = loc[0].min(), loc[0].max(), loc[1].min(), loc[1].max()
        xmax, ymax = self.lgi.shape
        if xi == 0 and xi == xmax:
            if yi == 0 and yi == ymax:
                grain_lgi_ex = self.lgi[xi:xj, yi:yj]
            elif yi == 0 and yi < ymax:
                grain_lgi_ex = self.lgi[xi:xj, yi:yj+2]
            elif yi > 0 and yi == ymax:
                grain_lgi_ex = self.lgi[xi:xj, yi-2:yj]
            elif yi > 0 and yi < ymax:
                grain_lgi_ex = self.lgi[xi:xj, yi-2:yj+2]
        elif xi == 0 and xi < xmax:
            if yi == 0 and yi == ymax:
                grain_lgi_ex = self.lgi[xi:xj+2, yi:yj]
            elif yi == 0 and yi < ymax:
                grain_lgi_ex = self.lgi[xi:xj+2, yi:yj+2]
            elif yi > 0 and yi == ymax:
                grain_lgi_ex = self.lgi[xi:xj+2, yi-1:yj]
            elif yi > 0 and yi < ymax:
                grain_lgi_ex = self.lgi[xi:xj+2, yi-1:yj+2]
        elif xi > 0 and xi == xmax:
            if yi == 0 and yi == ymax:
                grain_lgi_ex = self.lgi[xi-1:xj, yi:yj]
            elif yi == 0 and yi < ymax:
                grain_lgi_ex = self.lgi[xi-1:xj, yi:yj+2]
            elif yi > 0 and yi == ymax:
                grain_lgi_ex = self.lgi[xi-1:xj, yi-1:yj]
            elif yi > 0 and yi < ymax:
                grain_lgi_ex = self.lgi[xi-1:xj, yi-1:yj+2]
        elif xi > 0 and xi < xmax:
            if yi == 0 and yi == ymax:
                grain_lgi_ex = self.lgi[xi-1:xj+2, yi:yj]
            elif yi == 0 and yi < ymax:
                grain_lgi_ex = self.lgi[xi-1:xj+2, yi:yj+2]
            elif yi > 0 and yi == ymax:
                grain_lgi_ex = self.lgi[xi-1:xj+2, yi-1:yj]
            elif yi > 0 and yi < ymax:
                grain_lgi_ex = self.lgi[xi-1:xj+2, yi-1:yj+2]
        return grain_lgi_ex

    def find_extended_bounding_box_all_grains(self):
        """
        Find the extended bounded box of every gid.

        Example
        -------
        from upxo.ggrowth.mcgs import mcgs
        pxtal = mcgs(study='independent',
                     input_dashboard='input_dashboard_for_testing_50x50_alg202.xls')
        pxtal.simulate()
        pxtal.detect_grains()
        pxtal.gs[16].find_extended_bounding_box_all_grains()
        """
        grain_lgi_ex_all = {gid: None for gid in self.gid}
        for gid in self.gid:
            grain_lgi_ex_all[gid] = self.find_extended_bounding_box(gid)
        return grain_lgi_ex_all

    def find_neigh_gid_fast(self, gid, include_parent=False):
        """
        Find neighbouring grains of a given gid.

        Example
        -------
        from upxo.ggrowth.mcgs import mcgs
        pxtal = mcgs(study='independent',
                     input_dashboard='input_dashboard_for_testing_50x50_alg202.xls')
        pxtal.simulate()
        pxtal.detect_grains()
        np.unique(pxtal.gs[16].find_extended_bounding_box(10))
        pxtal.gs[16].find_neigh_gid_fast(10)
        """
        neighbours = list(np.unique(self.find_extended_bounding_box(gid)))
        if not include_parent:
            neighbours.remove(gid)
        return tuple(neighbours)

    def find_neigh_gid_fast_all_grains(self, include_parent=False,
                                       saa=True, throw=False):
        """
        Example
        -------
        from upxo.ggrowth.mcgs import mcgs
        pxtal = mcgs(study='independent',
                     input_dashboard='input_dashboard_for_testing_50x50_alg202.xls')
        pxtal.simulate()
        pxtal.detect_grains()
        np.unique(pxtal.gs[16].find_extended_bounding_box(10))
        pxtal.gs[16].find_neigh_gid_fast_all_grains(include_parent=False)
        pxtal.gs[16].neigh_gid
        """
        neigh_gid = {gid: self.find_neigh_gid_fast(gid, include_parent=include_parent)
                     for gid in self.gid}
        if saa:
            self.neigh_gid = neigh_gid
        if throw:
            return neigh_gid

    def get_upto_nth_order_neighbors(self, grain_id, neigh_order,
                                     recalculate=False, include_parent=True,
                                     output_type='list'):
        """
        Calculates the nth order neighbours for a given gid.

        Args:
            cell_id: The ID of the cell for which to find neighbors.
            n: The order of neighbors to calculate (1st order, 2nd order, etc.).

        Returns:
            A set containing the nth order neighbours.

        Example
        -------
        from upxo.ggrowth.mcgs import mcgs
        pxtal = mcgs(study='independent',
                     input_dashboard='input_dashboard_for_testing_50x50_alg202.xls')
        pxtal.simulate()
        pxtal.detect_grains()
        gid = 10
        np.unique(pxtal.gs[16].find_extended_bounding_box(gid))
        pxtal.gs[16].find_neigh_gid_fast_all_grains(include_parent=False)
        neigh_order = 3
        pxtal.gs[16].get_upto_nth_order_neighbors(gid,
                                                  neigh_order,
                                                  recalculate=False,
                                                  include_parent=True,
                                                  output_type='list')
        """
        if neigh_order == 0:
            return grain_id
        if recalculate or not self.neigh_gid:
            self.find_neigh_gid_fast_all_grains(include_parent=False)
        # Start with 1st-order neighbors
        neighbors = set(self.neigh_gid.get(grain_id, []))

        for _ in range(neigh_order - 1):
            new_neighbors = set()
            for neighbor in neighbors:
                new_neighbors.update(self.neigh_gid.get(neighbor, []))
            neighbors.update(new_neighbors)

        if not include_parent:
            neighbors.discard(grain_id)
        if output_type == 'list':
            return list(neighbors)
        if output_type == 'nparray':
            return np.array(list(neighbors))
        elif output_type == 'set':
            return neighbors

    def get_nth_order_neighbors(self, grain_id, neigh_order,
                                recalculate=False, include_parent=True):
        """
        Calculates the 1st till nth order neighbours for a given gid.

        Example
        -------
        from upxo.ggrowth.mcgs import mcgs
        pxtal = mcgs(study='independent',
                     input_dashboard='input_dashboard_for_testing_50x50_alg202.xls')
        pxtal.simulate()
        pxtal.detect_grains()
        gid = 10
        np.unique(pxtal.gs[16].find_extended_bounding_box(gid))
        pxtal.gs[16].find_neigh_gid_fast_all_grains(include_parent=False)
        neigh_order = 2
        pxtal.gs[16].get_nth_order_neighbors(gid,
                                             neigh_order,
                                             recalculate=False,
                                             include_parent=True)
        """
        neigh_upto_n_minus_1 = self.get_upto_nth_order_neighbors(grain_id,
                                                                 neigh_order-1,
                                                                 include_parent=include_parent,
                                                                 output_type='set')
        if type(neigh_upto_n_minus_1) in dth.dt.NUMBERS:
            neigh_upto_n_minus_1 = set([neigh_upto_n_minus_1])

        neigh_upto_n = self.get_upto_nth_order_neighbors(grain_id, neigh_order,
                                                         include_parent=include_parent,
                                                         output_type='set')
        if type(neigh_upto_n) in dth.dt.NUMBERS:
            neigh_upto_n = set([neigh_upto_n])
        return list(neigh_upto_n.difference(neigh_upto_n_minus_1))

    def get_upto_nth_order_neighbors_all_grains(self, neigh_order,
                                                recalculate=False,
                                                include_parent=True,
                                                output_type='list'):
        """
        Calculates 1st to nth order neighbors of all gids.

        Args:
            cell_id: The ID of the cell for which to find neighbors.
            n: The order of neighbors to calculate (1st order, 2nd order, etc.).

        Returns:
            A set containing the nth order neighbors.

        Example
        -------
        from upxo.ggrowth.mcgs import mcgs
        pxtal = mcgs(study='independent',
                     input_dashboard='input_dashboard_for_testing_50x50_alg202.xls')
        pxtal.simulate()
        pxtal.detect_grains()
        neigh_order = 3
        pxtal.gs[16].get_upto_nth_order_neighbors_all_grains(neigh_order,
                                                             recalculate=False,
                                                             include_parent=True,
                                                             output_type='list')
        """
        neighs_upto_nth_order = {gid: self.get_upto_nth_order_neighbors(gid,
                                                                        neigh_order,
                                                                        recalculate=recalculate,
                                                                        include_parent=include_parent,
                                                                        output_type='list')
                                 for gid in self.gid}
        return neighs_upto_nth_order

    def get_nth_order_neighbors_all_grains(self, neigh_order,
                                           recalculate=False,
                                           include_parent=True):
        """
        Calculates the nth order neighbours of all gids.

        Example
        -------
        from upxo.ggrowth.mcgs import mcgs
        pxtal = mcgs(study='independent',
                     input_dashboard='input_dashboard_for_testing_50x50_alg202.xls')
        pxtal.simulate()
        pxtal.detect_grains()
        neigh_order = 2
        A = pxtal.gs[16].get_upto_nth_order_neighbors_all_grains(neigh_order,
                                                             recalculate=False,
                                                             include_parent=True,
                                                             output_type='list')
        B = pxtal.gs[16].get_nth_order_neighbors_all_grains(neigh_order,
                                                        recalculate=False,
                                                        include_parent=True)

        """
        neighs_nth_order = {gid: self.get_nth_order_neighbors(gid,
                                                              neigh_order,
                                                              recalculate=recalculate,
                                                              include_parent=include_parent)
                            for gid in self.gid}
        return neighs_nth_order

    def get_upto_nth_order_neighbors_all_grains_prob(self, neigh_order,
                                                     recalculate=False,
                                                     include_parent=False,
                                                     print_msg=False):
        """
        from upxo.ggrowth.mcgs import mcgs
        pxt = mcgs()
        pxt.simulate()
        pxt.detect_grains()
        tslice = 10
        def_neigh = pxt.gs[tslice].get_upto_nth_order_neighbors_all_grains_prob

        neigh0 = def_neigh(1, recalculate=False, include_parent=True)
        neigh1 = def_neigh(1.06, recalculate=False, include_parent=True)
        neigh2 = def_neigh(1.5, recalculate=False, include_parent=True)
        neigh0[22]
        neigh1[2][22]
        neigh2[2][22]
        """
        # @dev:
            # no: neighbour order in these definitions.
        no = neigh_order
        on_neigh_all_grains_upto = self.get_upto_nth_order_neighbors_all_grains
        on_neigh_all_grains_at = self.get_nth_order_neighbors_all_grains
        if isinstance(no, (int, np.int32)):
            if print_msg:
                print('neigh_order is of type int. Adopting the usual method.')
            neigh_on = on_neigh_all_grains_upto(no, recalculate=recalculate,
                                           include_parent=include_parent)
            return neigh_on
        elif isinstance(no, (float, np.float64)):
            if abs(no-round(no)) < 0.05:
                if print_msg:
                    print('neigh_order is close to being int. Adopting usual method.')
                neigh_on = on_neigh_all_grains_upto(math.floor(no),
                                                    recalculate=recalculate,
                                                    include_parent=include_parent)
                return neigh_on
            else:
                if print_msg:
                    # Nothing to print
                    pass
                no_low, no_high = math.floor(no), math.ceil(no)
                neigh_upto_low = on_neigh_all_grains_upto(no_low,
                                                          recalculate=recalculate,
                                                          include_parent=include_parent)
                neigh_at_high = on_neigh_all_grains_at(no_low + 1,
                                                       recalculate=recalculate,
                                                       include_parent=False)
                delno = np.round(abs(neigh_order-math.floor(neigh_order)), 4)
                neighbours = {}
                for gid in self.gid:
                    nselect = math.ceil(delno * len(neigh_at_high[gid]))
                    if len(neigh_at_high[gid]) > 1:
                        neighbours[gid] = neigh_upto_low[gid] + random.sample(neigh_at_high[gid],
                                                                              nselect)
                return neighbours
        else:
            raise ValueError('Invalid neigh_order')

    def char_morph_2d(self, bbox=True, bbox_ex=True, npixels=False,
                      npixels_gb=False, area=False, eq_diameter=False,
                      perimeter=False, perimeter_crofton=False,
                      compactness=False, gb_length_px=False, aspect_ratio=False,
                      solidity=False, morph_ori=False, circularity=False,
                      eccentricity=False, feret_diameter=False,
                      major_axis_length=False, minor_axis_length=False,
                      euler_number=False, append=False, saa=True, throw=False,
                      char_grain_positions=False, find_neigh=False,
                      char_gb=False, make_skim_prop=False,
                      get_grain_coords=True):
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
        __ = pd_templates()
        __a, __b, __c = __.make_prop2d_df(bbox=bbox,
                                          bbox_ex=bbox_ex,
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
                                          append=append, )
        # ---------------------------------------------
        self.prop_flag, self.prop, self.prop_stat = __a, __b, __c
        # ---------------------------------------------
        Rlab, Clab = self.lgi.shape[0], self.lgi.shape[1]
        # ---------------------------------------------
        print('Extracting requested GS props across all available states')
        for s in self.s_gid.keys():
            if s % 5 == 0:
                print(f"--------State value: {s} of {self.S}")
            s_gid_keys_npy = [skey for skey in self.s_gid.keys()
                              if self.s_gid[skey]]
            # ---------------------------------------------
            sn = 1
            for state in s_gid_keys_npy:
                grains = self.s_gid[state]
                # Iterate through each grain of this state value
                _ngrains_ = len(grains)
                for i, gn in enumerate(grains, start=1):
                    if _ngrains_%100 == 0:
                        print(f'....grain no. {i}/{_ngrains_}')
                    _, L = cv2.connectedComponents(np.array(self.lgi == gn,
                                                            dtype=np.uint8))
                    self.g[gn] = {'s': state,
                                  'grain': grain2d()}
                    self.g[gn]['grain'].gid = gn
                    locations = np.argwhere(L == 1)
                    self.g[gn]['grain'].loc = locations
                    self.g[gn]['grain'].npixels = locations.size
                    _ = locations.T
                    self.g[gn]['grain'].xmin = _[0].min()
                    self.g[gn]['grain'].xmax = _[0].max()
                    self.g[gn]['grain'].ymin = _[1].min()
                    self.g[gn]['grain'].ymax = _[1].max()
                    self.g[gn]['grain'].s = state
                    self.g[gn]['grain'].sn = sn
                    self.g[gn]['grain']._px_area = self.px_size
                    sn += 1
                    # ---------------------------------------------
                    # Extract grain boundary indices
                    if char_gb:
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
                    rmin = np.where(L == 1)[0].min()
                    rmax = np.where(L == 1)[0].max()+1
                    cmin = np.where(L == 1)[1].min()
                    cmax = np.where(L == 1)[1].max()+1
                    # ---------------------------------------------
                    if bbox_ex:
                        # Extract bounding rectangle
                        Rlab = L.shape[0]
                        Clab = L.shape[1]

                        rmin_ex = rmin - int(rmin != 0)
                        rmax_ex = rmax + int(rmin != Rlab)
                        cmin_ex = cmin - int(cmin != 0)
                        cmax_ex = cmax + int(cmax != Clab)
                    # ---------------------------------------------
                    if bbox:
                        # Store the bounds of the bounding box
                        self.g[gn]['grain'].bbox_bounds = [rmin, rmax,
                                                           cmin, cmax]
                    if bbox:
                        # Store bounding box
                        self.g[gn]['grain'].bbox = np.array(L[rmin:rmax,
                                                              cmin:cmax],
                                                            dtype=np.uint8)
                    if bbox_ex:
                        # Store the bounds of the extended bounding box
                        self.g[gn]['grain'].bbox_ex_bounds = [rmin_ex, rmax_ex,
                                                              cmin_ex, cmax_ex]
                    if bbox_ex:
                        # Store the extended bounding box
                        self.g[gn]['grain'].bbox_ex = np.array(L[rmin_ex:rmax_ex,
                                                                 cmin_ex:cmax_ex],
                                                               dtype=np.uint8)
                    if make_skim_prop:
                        # Store the scikit-image regionproperties generator
                        self.g[gn]['grain'].make_prop(regionprops, skprop=True)
                    if get_grain_coords:
                        # Make coordinates
                        _coords_ = np.array([[self.xgr[ij[0], ij[1]],
                                              self.ygr[ij[0], ij[1]]]
                                             for ij in self.g[gn]['grain'].loc])
                        self.g[gn]['grain'].coords = deepcopy(_coords_)
        print(40*'-')
        self.build_prop()
        self.are_properties_available = True
        if char_grain_positions:
            self.char_grain_positions_2d()
        if find_neigh:
            print('Identifying grain neighbours.')
            self.find_neigh()

    def find_grain_boundary_junction_points(self, xorimap=False, IN=None):

        def __find_junctions(pixel_values):
            """
            Function to be applied on each pixel. It checks if the central
            pixel is a junction point. pixel_values: A flattened array of the
            central pixel and its neighbors. Returns 1 if the central pixel
            is a junction, else 0.
            """
            unique_grains = np.unique(pixel_values)
            '''Count the unique grain IDs excluding the background or border
            if needed'''
            count = np.sum(unique_grains > 0)
            return 1 if count >= 3 else 0
        __footprint__ = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        # Apply the filter to identify junctions
        if not xorimap:
            self.gbjp = generic_filter(self.lgi, __find_junctions,
                                       footprint=__footprint__, mode='nearest',
                                       cval=0)
        else:
            if IN in self.pxtal.keys():
                self.pxtal[IN].gbjp = generic_filter(self.pxtal[IN].lgi,
                                                     __find_junctions,
                                                     footprint=__footprint__,
                                                     mode='nearest',
                                                     cval=0)
            else:
                print(f'Invalid Instance number, IN: {IN}')

    def do_single_pixel_grains_exist(self):
        # MAKE THIS A DECORATOR
        pass

    def do_straightline_grains_exist(self):
        # MAKE THIS A DECORATOR
        pass


    def remove_single_pixel_grains(self, acceptable_percentage_fraction=0):
        if acceptable_percentage_fraction > 0.1:
            # VALIDATE IF AT ALL THERE ARE SINGLE PIXEL GRAINS
            acceptable_fraction = acceptable_percentage_fraction/100
            # Calculate fraction to remove in the first iteration:
            # MAY-BE CHOOSE 5% TO START
            frac_iter = [0.05]
            fraction_remaining = 1.00
            count = 0
            while fraction_remaining > acceptable_fraction:
                # REMOVE THE SINGLE PIXEL GRAINS
                fraction_to_remove = frac_iter[count]
        else:
            # JUST REMOVE ALL SINGLE PIXEL GRAINS
            pass

    def remove_straight_line_grains(self,
                                    acceptable_percentage_fraction=0):
        '''
        By default, removes only those straight lines of unit pixel width.
        '''
        if acceptable_percentage_fraction > 0.1:
            # VALIDATE IF AT ALL THERE ARE SINGLE PIXEL GRAINS
            acceptable_fraction = acceptable_percentage_fraction/100
            # Calculate fraction to remove in the first iteration:
            # MAY-BE CHOOSE 5% TO START
            frac_iter = [0.05]
            fraction_remaining = 1.00
            count = 0
            while fraction_remaining > acceptable_fraction:
                # REMOVE THE SINGLE PIXEL GRAINS
                fraction_to_remove = frac_iter[count]
        else:
            # JUST REMOVE ALL SINGLE PIXEL GRAINS
            pass

    def check_for_neigh(self, parent_gid, other_gid):
        """
        Check if other_gid is indeed a O(1) neighbour of parent_gid.

        Parameters
        ----------
        parent_gid:
            Grain ID of the parent.
        other_gid:
            Grain ID of the other grain being checked for O(1) neighbourhood
            with parent_gid.

        Returns
        -------
        True if other_gid is a valid O(1) neighbour of parent_gid, else False.
        """
        return True if other_gid in self.neigh_gid[parent_gid] else False

    def get_two_rand_o1_neighs(self):
        """
        Calculate at random, two neighbouring O(1) grains.

        Example
        -------
        from upxo.ggrowth.mcgs import mcgs
        mcgs = mcgs(study='independent', input_dashboard='input_dashboard.xls')
        mcgs.simulate()
        mcgs.detect_grains()
        mcgs.gs[35].char_morph_2d()
        mcgs.gs[35].find_neigh()
        mcgs.gs[35].neigh_gid
        mcgs.gs[35].get_two_rand_o1_neighs()
        mcgs.gs[35].plot_two_rand_neighs(return_gids=True)
        """
        if self.neigh_gid:
            rand_gid = random.sample(self.gid, 1)[0]
            rand_neigh_rand_grain = random.sample(self.neigh_gid[rand_gid],
                                                  1)[0]
            return [rand_gid, rand_neigh_rand_grain]
        else:
            print('Please build neigh_gid data before using this function.')
            return [None, None]

    def plot_two_rand_neighs(self, return_gids=True):
        """
        Plot two random neighbouring grains.

        Parameters
        ----------
        return_gids: bool
            Flag to return the random neigh gid numbers. Defaults to True.

        Return
        ------
        rand_neigh_gids: list
            random neigh gid numbers. Will be gids if return_gids is True.
            Else, will be [None, None].

        Example
        -------
        Please refer to use in the example provided for the definition,
        get_two_rand_o1_neighs()
        """
        rand_neigh_gids = self.get_two_rand_o1_neighs()
        self.plot_grains_gids(rand_neigh_gids, cmap_name='viridis')
        if return_gids:
            return rand_neigh_gids
        else:
            return [None, None]

    def _merge_two_grains_(self, parent_gid, other_gid, print_msg=False):
        """Low level merge operartion. No checks done. Just merging.

        Parameters
        ----------
        parent_gid: int
            Parent grain ID number.
        other_gid: int
            Otrher grain ID number.
        print_msg: bool
            Defgaults to False.

        Returns
        -------
        None

        Usage
        -----
        Internal use only.
        """
        self.lgi[self.lgi == other_gid] = parent_gid
        if print_msg:
            print(f"Grain {other_gid} merged with grain {parent_gid}.")

    def merge_two_neigh_grains(self, parent_gid, other_gid,
                               check_for_neigh=True, simple_merge=True):
        """
        Merge other_gid grain to the parent_gid grain.

        Paramters
        ---------
        parent_gid:
            Grain ID of the parent.
        other_gid:
            Grain ID of the other grain being merged into the parent.
        check_for_neigh: bool.
            If True, other_gid will be checked if it can be merged to the
            parent grain. Defaults to True.

        Returns
        -------
        merge_success: bool
            True, if successfully merged, else False.
        """
        def MergeGrains():
            if simple_merge:
                self._merge_two_grains_(parent_gid, other_gid, print_msg=False)
                merge_success = True
            else:
                print("Special merge process. To be developed.")
                merge_success = False  # As of now, this willd efault to False.
            return merge_success
        # ---------------------------------------
        if not check_for_neigh:
            merge_success = MergeGrains()
        else:
            if check_for_neigh and not self.check_for_neigh(parent_gid, other_gid):
                # print('Check for neigh failed. Nothing merged.')
                merge_success = False
            # ---------------------------------------
            if any((check_for_neigh, self.check_for_neigh(parent_gid, other_gid))):
                merge_success = MergeGrains()
                # print(f"Grain {other_gid} merged with grain {parent_gid}.")
        return merge_success

    def perform_post_grain_merge_ops(self, merge_success, merged_gid):
        self.renumber_gid_post_merge(merged_gid)
        self.recalculate_ngrains_post_merge()
        # Update lgi
        # Update neigh_gid
        pass

    def renumber_gid_post_grain_merge(self, merged_gid):
        # self._gid_bf_merger_ = deepcopy(self.gid) # May nor be needed
        GID_left = self.gid[0:merged_gid-1]
        GID_right = [gid-1 for gid in self.gid[merged_gid:]]
        self.gid = GID_left + GID_right

    def recalculate_ngrains_post_grain_merge(self):
        # gid must have been recalculated for tjhis as a pre-requisite.
        self.n = len(self.gid)

    def renumber_lgi_post_grain_merge(self, merged_gid):
        LGI_left = self.lgi[self.lgi < merged_gid]
        self.lgi[self.lgi > merged_gid] -= 1

    def validate_propnames(self, mpnames, return_type='dict'):
        """
        Validate an iterable containing propnames. Mostly for internal use.

        Parameters
        ----------
        mpnames: dth.dt.ITERABLES
            Property names to be validated.
        return_type: str
            Type of function return. Valid choices: dict (default), list,
            tuple.

        Returns
        -------
        validation: dict (default) / tuple
            If return_type is other than dictionary and either list or
            tuple, or numpy array, only tuple will be returned. If return_type
            is dict, then dict with mpnames keys and their individual
            validations will be the values. The values will all be bool.
            If a property is a valid property, then True, else False.

        Example
        -------
        self.validate_propnames(['area', 'perimeter', 'solidity'])
        """
        _ = {pn: pn in self.valid_mprops.keys() for pn in mpnames}
        if return_type == 'dict':
            return _
        elif return_type in ('list', 'tuple'):
            return tuple(_.values())
        else:
            raise ValueError('Invalid return_type specification.')

    def check_mpnamevals_exists(self, mpnames, return_type='dict'):
        if return_type == 'dict':
            return {mpn: mpn in self.prop.columns for mpn in mpnames}
        elif return_type in ('list', 'tuple'):
            return [mpn in self.prop.columns for mpn in mpnames]

    def set_mprops(self, mpnames, char_grain_positions=True,
                   char_gb=False, set_grain_coords=True,
                   saa=True, throw=False):
        """
        Targetted use of char_morph_2d.

        Parameters
        ----------
        mpnames: dth.dt.ITERABLES
            List of user specified names of morphological properties
        char_grain_positions: bool
            If True, grain positions will also be characterized. Defaults to
            True.
        char_gb:
            If True, grain boundary will be characterized/re-characterized.
            Degfaults to False.
        set_grain_coords:
            If True, self.g[gn]['grain'].coords will be updated, else not, for
            all gn in self.gid.

        Example
        -------
        self.set_mprops(mpnames, recharacterize=True)
        """
        VALMPROPS = deepcopy(self.valid_mprops)
        # ----------------------------
        if not all(self.validate_propnames(mpnames, return_type='tuple')):
            raise ValueError('Invalid propnames.')
        # ----------------------------
        for mpn in mpnames:
            # Check if each user input morph0ological propetrty name and
            # corresponding values exist in self.prop pd dataFrame.
            VALMPROPS[mpn] = True
        if char_grain_positions:
            VALMPROPS['char_grain_positions'] = True
        # ----------------------------
        self.char_morph_2d(bbox=True, bbox_ex=True, append=False, saa=saa,
                           throw=False, char_gb=char_gb, make_skim_prop=True,
                           get_grain_coords=set_grain_coords, **VALMPROPS)
        # ----------------------------
        if throw:
            mprop_values = {mpn: self.prop[mpn].to_numpy() for mpn in mpnames}
        else:
            mprop_values = {mpn: None for mpn in mpnames}

        return mprop_values

    def get_mprops(self, mpnames, set_missing_mprop=False):
        """
        Get values of mpnames.

        Example
        -------
        from upxo.ggrowth.mcgs import mcgs
        mcgs = mcgs(study='independent', input_dashboard='input_dashboard.xls')
        mcgs.simulate()
        mcgs.detect_grains()
        mcgs.gs[mcgs.m[-1]].char_morph_2d(bbox=True, bbox_ex=True,
                                     area=True,aspect_ratio=True,
                                     make_skim_prop=True,)

        mpnames=['area', 'aspect_ratio', 'perimeter', 'solidity']
        mcgs.gs[mcgs.m[-1]].prop
        mprop_values = mcgs.gs[mcgs.m[-1]].get_mprops(mpnames,
                                                      set_missing_mprop=True)
        mprop_values
        """
        if not all(self.validate_propnames(mpnames, return_type='list')):
            raise ValueError('Invalid mpname values.')
        val_exists = self.check_mpnamevals_exists(mpnames, return_type='dict')
        # ----------------------------
        if not set_missing_mprop:
            mprop_values = {}
            for mpn in mpnames:
                if val_exists[mpn]:
                    mprop_values[mpn] = self.prop[mpn].to_numpy()
                else:
                    mprop_values[mpn] = None
        # ----------------------------
        if set_missing_mprop:
            set_propnames = [mpn for mpn in mpnames if not val_exists[mpn]]
            self.set_mprops(mpnames, char_grain_positions=False,
                            char_gb=False, set_grain_coords=False)
            mprop_values = self.get_mprops(mpnames, set_missing_mprop=False)

        return mprop_values

    def validata_gids(self, gids):
        """
        Validate the gid values.

        Parameters
        ----------
        gids: Iterable of ints.

        Returns
        -------
        True if all gids are in self.gid else False
        """
        return all([gid in self.gid for gid in gids])

    def get_gids_in_params_bounds(self,
                                  search_gid_source='all',
                                  search_gids=None,
                                  mpnames=['area', 'aspect_ratio',
                                           'perimeter', 'solidity'],
                                  fx_stats=[np.mean, np.mean, np.mean, np.mean],
                                  pdslh=[[50, 50], [50, 50], [50, 50], [50, 50]],
                                  param_priority=[1, 2, 3, 2],
                                  plot_mprop=True
                                  ):
        """
        pdslh: Percentages of distance from stat to minimum and stat to maximum.

        Example
        -------
        """
        # Validations
        # ---------------------------
        pname_val = self.validate_propnames(mpnames, return_type='dict')
        # pname_val = mcgs.gs[35].validate_propnames(mpnames, return_type='dict')
        mprop_values = self.get_mprops(mpnames, set_missing_mprop=True)
        # mcgs.gs[35].prop
        # mprop_values = mcgs.gs[35].get_mprops(mpnames, set_missing_mprop=True)
        # ---------------------------
        '''Sub-select gids as per user request.'''
        if search_gid_source == 'user' and dth.IS_ITER(search_gids):
            if self.validata_gids(search_gids):
                search_gids = np.sort(search_gids)
                for mpn in mpnames:
                    mprop_values[mpn] = mprop_values[mpn][search_gids]
        # ---------------------------
        '''Data processing and extract indices of parameters for parameter
        values valid to the user provided bound.'''
        mprop_KEYS = list(mprop_values.keys())
        mprop_VALS = list(mprop_values.values())
        mpinds = {mpn: None for mpn in mprop_KEYS}
        mp_stats = {mpn: None for mpn in mprop_KEYS}
        mp_bounds = {mpn: None for mpn in mprop_KEYS}
        for i, (KEY, VAL) in enumerate(zip(mprop_KEYS, mprop_VALS)):
            masked_VAL = np.ma.masked_invalid(VAL)
            # Compute the stat value of the morpho prop
            mp_stat = fx_stats[i](masked_VAL)
            mp_stats[KEY] = mp_stat
            # COmpute min and max of the mprop array
            mp_gmin, mp_gmax = np.min(masked_VAL), np.max(masked_VAL)
            # Compute distance from stat to low and stat to high
            mp_dlow, mp_dhigh = abs(mp_stat-mp_gmin), abs(mp_stat-mp_gmax)
            # Compute bounds of arrays using varper
            dfsmin = pdslh[i][0]/100  # Distance factor from stat to prop min.
            dfsmax = pdslh[i][1]/100  # Distance factor from stat to prop max.
            # Compute lower bound and upper boubnd
            boundlow = mp_stat - dfsmin*mp_dlow
            boundhigh = mp_stat + dfsmax*mp_dhigh
            mp_bounds[KEY] = [boundlow, boundhigh]
            # Mask the mprop array and get indices
            mpinds[KEY] = np.where((VAL >= boundlow) & (VAL <= boundhigh))[0]
            # ---------------------------

        # Find the intersection
        intersection = find_intersection(mpinds.values())
        # Find the union with counts
        union, counts = find_union_with_counts(mpinds.values())
        # Copnvert array indices to gid notation.
        intersection = [i+1 for i in intersection]
        union = [u+1 for u in union]
        counts = {c+1: v for c, v in counts.items()}
        mpinds_gids = {}
        for mpn in mpinds:
            mpinds_gids[mpn] = [i+1 for i in mpinds[mpn]]
        # Collate the GID related results
        GIDs = {'intersection': intersection,
                'union': union,
                'presence': counts,
                'mpmapped': mpinds_gids}
        # Collate the Values and Indices related results
        VALIND = {'stat': mp_stats,
                  'statmap': fx_stats,
                  'bounds': mp_bounds,
                  'indices': mpinds,
                  }

        if plot_mprop:
            fig, ax = plt.subplots(nrows=1, ncols=len(GIDs['mpmapped'].keys()),
                                   figsize=(5, 5), dpi=120, sharey=True)
            for i, mpn in enumerate(GIDs['mpmapped'].keys(), start=0):
                LGI = deepcopy(self.lgi)
                if len(GIDs['mpmapped'][mpn]) > 0:
                    for gid in self.gid:
                        if gid in GIDs['mpmapped'][mpn]:
                            pass
                        else:
                            LGI[LGI == gid] = -10
                ax[i].imshow(LGI, cmap='nipy_spectral')
                bounds = ", ".join(f"{b:.2f}" for b in VALIND['bounds'][mpn])
                ax[i].set_title(f"{mpn}: bounds: [{bounds}]", fontsize=10)

        return GIDs, VALIND

    def get_gid_mprop_map(self, mpropname, querry_gids):
        """
        Provide gid mapped values of a valid mprop for valid querry_gids.

        Parameters
        ----------
        mpropname: str
        querry_gids: Iterable

        Returns
        -------
        gid_mprop_map: dict
            Dictionary with querry_gids keys and corresponding mprop values.
        """
        # Validations
        self.validate_propnames([mpropname])
        self.validata_gids(querry_gids)
        # ------------------------------------
        if mpropname in self.prop.columns:
            mpvalues = self.prop.loc[[gid-1 for gid in list(querry_gids)],
                                     'aspect_ratio'].to_numpy()
            gid_mprop_map = {gid: mpv
                             for gid, mpv in zip(querry_gids, mpvalues)}
            return gid_mprop_map
        else:
            raise ValueError(f'mpropname must be in {list(self.prop.columns())}.')

    def map_scalar_to_lgi(self, scalars_dict, default_scalar=-1,
                          plot=True, throw_axis=True, plot_centroid=True,
                          plot_gid_number=True,
                          title='title',
                          centroid_kwargs={'marker': 'o',
                                           'mfc': 'yellow',
                                           'mec': 'black',
                                           'ms': 2.5},
                          gid_text_kwargs={'fontsize': 10},
                          title_kwargs={'fontsize': 10},
                          label_kwargs={'fontsize': 10}):
        """
        Map to LGI, the gid keyed values in scalars_dict.

        Parameters
        ----------
        scalars_dict: dict
            Dictionary with keys being a subset of self.gid. Each key must have
            a single numeric or bool value.
        default_scalar: int
            Defauts to -1.

        Example
        -------
        from upxo.ggrowth.mcgs import mcgs
        pxt = mcgs()
        pxt.simulate()
        pxt.detect_grains()
        tslice = 10
        def_neigh = pxt.gs[tslice].get_upto_nth_order_neighbors_all_grains_prob

        neigh1 = def_neigh(1.38, recalculate=False, include_parent=True)

        sf_no = pxt.gs[tslice]



        from upxo.ggrowth.mcgs import mcgs
        mcgs = mcgs(study='independent', input_dashboard='input_dashboard.xls')
        mcgs.simulate()
        mcgs.detect_grains()
        mcgs.gs[35].char_morph_2d(bbox=True, bbox_ex=True, area=True,
                                  aspect_ratio=True,
                                  make_skim_prop=True,)
        GIDs, VALIND = mcgs.gs[35].get_gids_in_params_bounds(mpnames=['aspect_ratio', 'area'],
                                              fx_stats=[np.mean, np.mean],
                                              pdslh=[[50, 30], [50, 30]], plot_mprop=False
                                              )
        mcgs.gs[35].map_scalar_to_lgi(GIDs['presence'], default_scalar=-1,
                              plot=True, throw_axis=True)

        gid_mprop_map = mcgs.gs[35].get_gid_mprop_map('aspect_ratio',
                                                      GIDs['mpmapped']['aspect_ratio'])
        MPLGIAX = mcgs.gs[35].map_scalar_to_lgi(gid_mprop_map, default_scalar=-1,
                              plot=True, throw_axis=True)
        """
        # Validations
        self.validata_gids(scalars_dict.keys())
        # -------------------
        LGI = deepcopy(self.lgi).astype(float)
        for gid in self.gid:
            if gid in scalars_dict.keys():
                LGI[LGI == gid] = scalars_dict[gid]
            else:
                LGI[LGI == gid] = default_scalar
        # -------------------
        if plot:
            # VMIN, VMAX = min(scalars_dict.values()), max(scalars_dict.values())
            plt.figure(figsize=(5, 5), dpi=120)
            plt.imshow(LGI, cmap='viridis')
            if plot_centroid or plot_gid_number:
                centroid_x, centroid_y = [], []
                for gid in scalars_dict.keys():
                    centroid_x.append(self.xgr[self.lgi == gid].mean())
                    centroid_y.append(self.ygr[self.lgi == gid].mean())
            if plot_centroid:
                plt.plot(centroid_x, centroid_y, linestyle='None',
                         marker=centroid_kwargs['marker'],
                         mfc=centroid_kwargs['mfc'],
                         mec=centroid_kwargs['mec'],
                         ms=centroid_kwargs['ms'])
            if plot_gid_number:
                for i, (cenx, ceny) in enumerate(zip(centroid_x,
                                                     centroid_y), start=1):
                    plt.text(cenx, ceny, str(i),
                             fontsize=gid_text_kwargs['fontsize'])
            ax = plt.gca()
            ax.set_title('Title', fontsize=10)
            ax.set_xlabel(r"X-axis, $\mu m$", fontsize=10)
            ax.set_ylabel(r"Y-axis, $\mu m$", fontsize=10)
            plt.colorbar()
        # -------------------
        if plot and throw_axis:
            return {'lgi': LGI, 'ax': ax}
        else:
            return {'lgi': LGI, 'ax': None}

    def merge_two_neigh_grains_simple(self,
                                      method_id='1',
                                      method_params_parent_sel=['area'],
                                      method_params_other_sel=['area'],
                                      method_params_merging=['area'],
                                      parent_gid=[],
                                      return_gids=True,
                                      plot_gs_bf=True,
                                      plot_gs_af=True,
                                      plot_area_kde_diff=True,
                                      bandwidth=1.0):
        """
        Find two random neighbouring grains and merge them.

        Parameters
        ----------
        method_id: int
            0: parenmt_gid will be selected at random and other_gid will also
                be selected at random.
            1: parent_gid should be provided by user and othet_gid should also
                be provided by user.
            2: parent_gid sahould be provide by user and other_gid will be
                selected at random.
            NOTE: other_grain will allways be O(1) neighbour of parent_grain.
        method_params_parent_sel: str
            Morphological parameter of choice for parent grain selection.
        method_params_other_sel: str
            Morphological parameter of choice for other grain selection.
        method_params_merging: str
            Morphological parameter of choice for merging decision makjing.
        plot_bf: bool
            Plot grain structure before merging. Defaults to True.
        plot_af: bool
            Plot grain structure after merging. Defaults to True.

        Returns
        -------
        gids: list
            [parent_gid, other_gid]. Other_gid merged into parent_gid.
        """
        def plot_kde_difference(area1, area2, bandwidth=1):
            """
            Calculates KDEs for two arrays of data and plots their difference.

            Parameters
            ----------
            area1: np.ndarray
                The first array of data.
            area2: np.ndarray
                The second array of data.
            bandwidth: float, optional
                The bandwidth (smoothing parameter) for KDEs (default: 0.2).
            """
            plt.figure(figsize=(5,5), dpi=120)
            kde1 = sns.kdeplot(area1, bw_adjust=bandwidth, fill=True,
                               label='Area 1', color='blue')
            kde2 = sns.kdeplot(area2, bw_adjust=bandwidth, fill=True,
                               label='Area 2', color='orange')
            # Get the KDE curve data
            x1, y1 = kde1.get_lines()[0].get_data()
            x2, y2 = kde2.get_lines()[0].get_data()
            # Interpolate if x values don't exactly match
            # (to ensure we can subtract)
            y2_interp = np.interp(x1, x2, y2)
            # Calculate and plot the difference
            y_diff = y1 - y2_interp
            plt.fill_between(x1, y_diff, 0, color='green',
                             alpha=0.5, label='Difference (Area 1 - Area 2)')
            # Label axes and add a title
            plt.xlabel('Area')
            plt.ylabel('Density')
            plt.title('KDEs of area distributions and their difference.')
            plt.legend()
            plt.show()
        # ============================================================
        if method_id == '1':
            '''parenmt_gid will be selected at random and other_gid will also
            be selected at random. NOTE: other_grain will allways be O(1)
            neighbour of parent_grain.'''
            parent_gid, other_gid = self.get_two_rand_o1_neighs()
        if method_id == '2':
            '''parent_gid should be provided by user and othet_gid should also
            be provided by user.'''
            parent_gid = parent_gid
            other_gid = self.get_o1_neigh(parent_gid)
            parent_gid, other_gid = self.get_two_rand_o1_neighs()
        if method_id == '3':
            '''parent_gid sahould be provide by user and other_gid will be
            selected at random. NOTE: other_grain will allways be O(1)
            neighbour of parent_grain.'''
            pass
        if method_id == '1-stat($varper$)':
            '''method1 + more. Below provides deytails.
            stat: statistic. Valids: mean, median.
            varper: percentage variation in the stat defining target area
            bound for parent_gid.
            '''
            pass
        # -------------------------------------
        if plot_gs_bf:
            self.plotgs(plot_centroid=True, plot_gid_number=True,
                        plot_cbar=False,
                        title=f'Before merging {other_gid} into {parent_gid}.')
        # -------------------------------------
        if plot_area_kde_diff:
            # Get area before merging
            area_bf = self.prop['area'].to_numpy()
        # -------------------------------------
        self.merge_two_neigh_grains(parent_gid, other_gid,
                                    check_for_neigh=False,
                                    simple_merge=True,)
        # -------------------------------------
        if plot_gs_af:
            self.plotgs(plot_centroid=True,
                        plot_gid_number=True,
                        plot_cbar=False,
                        title=f'After merging {other_gid} into {parent_gid}')
        # -------------------------------------
        if plot_area_kde_diff:
            area_af = deepcopy(area_bf)
            area_af[parent_gid-1] += area_af[other_gid-1]
            area_af = np.delete(area_af, other_gid-1)
            # Get area after merging
            plot_kde_difference(area_bf, area_af, bandwidth=bandwidth)
        # -------------------------------------
        if return_gids:
            return parent_gid, other_gid

    def merge_neigh_grains(self, gid_pairs,
                           check_for_neigh=True, simple_merge=True):
        hit = 0
        for parent_gid, other_gid in gid_pairs:
            if self.check_for_neigh(parent_gid, other_gid):
                self.merge_two_neigh_grains(parent_gid, other_gid,
                                            check_for_neigh=check_for_neigh,
                                            simple_merge=simple_merge)

    def set_twingen(self, vf=0.2, tspec='absolute', trel='minil',
                    tdis='user', t=[0.2, 0.5, 0.6, 0.7], tw=[1, 1, 1, 1],
                    tmin=0.2, tmean=0.5, tmax=1.0,
                    nmax_pg=1, placement='centroid', factor_min=0.0,
                    factor_max=1.0,
                    ):
        self.twingen = twingen(vf=0.2, tmin=0.2, tmean=0.5, tmax=1.0,
                     tdis='user', tvalues=[0.2, 0.5, 1.0, 0.75],
                     allow_partial=True, partial_prob=0.2)

    def introduce_single_twins(self, GIDs=[1], full_twin=True,
                               throw_lgi=True,
                               LFAL_kwargs={'factor': 0.5,
                                            'angle_min': 0,
                                            'angle_max': 360,
                                            'length': 50},
                               twdis_kwargs={'max_count_per_grain': 1,
                                             'min_thickness': 4.5,
                                             'mean_thickness': 4.5,
                                             'max_thickness': 4.5,
                                             'distribution': 'normal',
                                             'variance': 1.0,
                                             },
                               plotgs_original=True,
                               plotgs_twinned=True,
                               save_to_features=True,
                               ):
        """
        Introduce twinned grain features into self.lgi.

        Parameters
        ----------
        GIDs: Iterable
            Iterable of grain ID numbers which would host the twin regions.
        throw_lgi: bool
            Return the new LGI array.
        plot: bool
            Plot if Trure else, not.

        LFAL_kwargs:
            location, factor, angle and length kwargs for UPXO.sline2d.

        twdis_kwargs: Twin Distribution kwargs

        Example
        -------
        from upxo.ggrowth.mcgs import mcgs
        mcgs = mcgs(study='independent', input_dashboard='input_dashboard.xls')
        mcgs.simulate()
        mcgs.detect_grains()
        mcgs.gs[35].char_morph_2d(bbox=True, bbox_ex=True, area=True,
                                  aspect_ratio=True, perimeter=True, solidity=True,
                                  make_skim_prop=True,)
        mcgs.gs[35].prop.columns
        mcgs.gs[35].find_neigh()
        mcgs.gs[35].g[12]['grain'].coords
        mcgs.gs[35].g[12]['grain'].centroid
        GIDs, VALIND = mcgs.gs[35].get_gids_in_params_bounds(mpnames=['area'],
                                              fx_stats=[np.mean],
                                              pdslh=[[50, 50]],
                                              plot_mprop=False
                                              )
        gids = GIDs['mpmapped']['area']
        mcgs.gs[35].introduce_single_twins(GIDs=gids, full_twin=True,
                                   throw_lgi=True, plotgs_original=False,
                                   plotgs_twinned=True)
        """
        # Validations
        # -----------------------------------------
        if plotgs_original:
            self.plotgs(figsize=(6, 6), dpi=120,
                        cmap='coolwarm', plot_centroid=True,
                        centroid_kwargs={'marker': 'o', 'mfc': 'yellow',
                                         'mec': 'black', 'ms': 2.5},
                        plot_gid_number=True)
        # -----------------------------------------
        gscoords = (self.xgr.ravel(), self.ygr.ravel())
        LGI_1 = deepcopy(self.lgi)
        # -----------------------------------------
        for gid in GIDs:
            LGI = deepcopy(self.lgi).ravel()
            xc = self.xgr[self.lgi == gid].mean()
            yc = self.ygr[self.lgi == gid].mean()
            remaining_indices = list(range(gscoords[0].size))
            # -----------------------------------------
            lines = [sl2d.by_LFAL(location=[xc, yc], **LFAL_kwargs)]
            # -----------------------------------------
            twin_indices = []
            for line in lines:
                _fx_ = line.find_neigh_point_by_perp_distance
                _twin_indices_ = _fx_(gscoords, 4.5, use_bounding_rec=True)
                if _twin_indices_:
                    twin_indices.append(_twin_indices_)
                    remaining_indices = list(set(remaining_indices) - set(_twin_indices_))
            # -----------------------------------------
            LGI[twin_indices[0]] = -1
            LGI = np.reshape(LGI, self.lgi.shape)
            LGI_1[(LGI == -1) & (LGI_1 == gid)] = -1
        # -----------------------------------------
        self.plotgs(figsize=(6, 6), dpi=120,
                    cmap='coolwarm', custom_lgi=LGI_1,
                    plot_centroid=True,
                    centroid_kwargs={'marker': 'o', 'mfc': 'yellow',
                                     'mec': 'black', 'ms': 2.5},
                    plot_gid_number=True)

    def add_pxtal(self):
        from upxo.pxtal.pxtal_ori_map_2d import polyxtal2d as PXTAL
        if len(self.pxtal.keys()) == 0:
            self.pxtal[1] = PXTAL()
        else:
            self.pxtal[max(list(self.pxtal.keys()))+1] = PXTAL()

    def set_pxtal(self, instance_no=1,
                  path_filename_noext=None,
                  map_type='ebsd',
                  apply_kuwahara=False, kuwahara_misori=5, gb_misori=10,
                  min_grain_size=1,
                  print_closs=True,
                  ):
        """
        Crystal Orientation Map. EBSD dataswt is one which can be loadsed.

        Example
        -------
        from upxo.ggrowth.mcgs import mcgs
        pxt = mcgs()
        pxt.simulate()
        pxt.detect_grains()
        tslice = 20  # Temporal slice number
        pxt.char_morph_2d(tslice)
        pxt.gs[tslice].export_ctf(r'D:\export_folder', 'sunil')
        path_filename_noext = r'D:\export_folder\sunil'
        pxt.gs[tslice].set_pxtal(path_filename_noext=path_filename_noext)
        pxt.gs[tslice].pxtal.map
        """
        IN, _fn_ = instance_no, path_filename_noext
        _khfflag_, _khfmo_ = apply_kuwahara, kuwahara_misori
        # -------------------------------------------
        self.add_pxtal()
        print(_fn_)
        self.pxtal[IN].setup(map_type='ebsd',
                             path_filename_noext=_fn_,
                             apply_kuwahara=_khfflag_,
                             kuwahara_misori=_khfmo_,)
        self.pxtal[IN].find_grains_gb(gb_misori=gb_misori,
                                      min_grain_size=min_grain_size,
                                      print_msg=True,)
        self.pxtal[IN].port_essentials(print_msg=True)
        # self.pxtal[IN].char_grain_positions_2d()
        self.pxtal[IN].set_conversion_loss(refn=np.unique(self.lgi).size)
        self.find_grain_boundary_junction_points(xorimap=True, IN=IN)
        self.pxtal[IN].set_bjp()
        self.pxtal[IN].find_neigh(update_gid=True, reset_lgi=False)
        self.pxtal[IN].find_gbseg1()

    def __make_linear_grid(self, sf=1):
        # Validate for maximum sf
        # -------------------------------------------------
        # Make the base space
        xinc, yinc = self.uigrid.xinc*sf, self.uigrid.yinc*sf
        xmin, xmax, xinc = self.uigrid.xmin, self.uigrid.xmax, self.uigrid.xinc
        ymin, ymax, yinc = self.uigrid.ymin, self.uigrid.ymax, self.uigrid.yinc
        x = np.arange(xmin, xmax+xinc, xinc)
        y = np.arange(ymin, ymax+yinc, yinc)
        return x, y, xinc, yinc

    def scale(self, sf):
        """
        Apply a scale factor to the current grain structure temporal slice.
        """
        # -------------------------------------------------
        # VALIDATE input, f
        # Make the base linear space
        x, y, xinc, yinc = self.__make_linear_grid(sf=1)
        # Construct the inerpolator
        intmeth = 'nearest'
        interpolator = RegularGridInterpolator((x, y), self.s, method=intmeth)
        # Make the updated linear space
        _x_, _y_, _xinc_, _yinc_ = self.__make_linear_grid(sf=sf)
        # Make the new grid
        _xgr_, _ygr_ = np.meshgrid(_x_, _y_)
        # Interpolate state values
        s = interpolator(np.array([_xgr_.flatten(), _ygr_.flatten()]).T)
        s = s.reshape(len(_x_), len(_y_)).T
        # ---------------------------------------------
        # TODO: VALIDATE IF CREATED S DIMENTSIONS ARE CONSISTENT
        # ---------------------------------------------
        # Create a new grain structure database
        self.scaled['sf'] = sf
        self.scaled['xmin'], self.scaled['xmax'] = _x_.min(), _x_.max()
        self.scaled['ymin'], self.scaled['ymax'] = _y_.min(), _y_.max()
        self.scaled['xinc'], self.scaled['yinc'] = _xinc_, _yinc_
        self.scaled.xgr, self.scaled.ygr, self.s = _xgr_, _ygr_, s
        self.scaled.char_morph_2d()
        if sf != 1:
            self.__resolution_state__ = f'finer_sf={sf}'


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

    def __setup__positions__(self):
        self.positions = {'top_left': [], 'bottom_left': [],
                          'bottom_right': [], 'top_right': [],
                          'pure_right': [], 'pure_bottom': [],
                          'pure_left': [], 'pure_top': [],
                          'left': [], 'bottom': [], 'right': [], 'top': [],
                          'boundary': [], 'corner': [], 'internal': [], }

    def char_grain_positions_2d(self):
        row_max, col_max = self.lgi.shape[0]-1, self.lgi.shape[1]-1
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

    def find_border_internal_grains_fast(self):
        """
        Identify border and internal grains.

        Parameters
        ----------
        None

        Return
        ------
        border_gids: ids of border grains
        internal_gids: ids of internal grains
        lgi_border: lgi of only border grains
        lgi_internal: lgi of only internal grains

        Use
        ---
        border_gids, internal_gids, lgi_border, lgi_internal = find_border_internal_grains_fast()

        plt.figure()
        plt.imshow(lgi_border)

        plt.figure()
        plt.imshow(lgi_internal)

        """
        lgi = self.lgi
        lgi_border = deepcopy(lgi)
        lgi_border[1:-1, 1:-1] = 0
        border_gids = np.unique(lgi_border[lgi_border != 0])
        internal_gids = np.array(list(set(self.gid) - set(border_gids)))

        lgi_border = deepcopy(lgi)

        lgi_internal = deepcopy(lgi)

        for bgid in border_gids:
            lgi_internal[lgi_internal == bgid] = 0

        for nbgid in internal_gids:
            lgi_border[lgi_border == nbgid] = 0

        return border_gids, internal_gids, lgi_border, lgi_internal

    def find_grain_size_fast(self, metric='npixels'):
        """
        Quickly find the grain sizes without doing anything else.

        Explanations
        ------------
        Order of grain_sizes is that of pxtal.gs[m].gid

        Parameters
        ----------
        metric: Specify which ares metric is needed. Optoins include:
            * 'npixels': Number of pixels
            * 'pxarea': Pixel wise calculated area
            * 'eq_dia': Equivalent diameter

        Return
        ------
        grain_sizes: Numpy array of grain areas.

        Example
        -------
        from upxo.ggrowth.mcgs import mcgs
        pxtal = mcgs(study='independent', input_dashboard='input_dashboard.xls')
        pxtal.simulate()
        pxtal.detect_grains()
        grain_areas_all_grains = pxtal.gs[2].find_grain_size_fast(metric='npixels')
        """
        grain_sizes = []
        if metric in ('npixels'):
            for gid in self.gid:
                grain_sizes.append(np.where(self.lgi == gid)[0].size)

        return np.array(grain_sizes)

    def find_npixels_border_grains_fast(self, metric='npixels'):
        """
        Find the number of pixels in each of the border grains.

        Parameters
        ----------
        metric: Specify which ares metric is needed. Optoins include:
            * 'npixels': Number of pixels
            * 'pxarea': Pixel wise calculated area
            * 'eq_dia': Equivalent diameter

        Return
        ------
        border_grain_npixels: Numpy array of number of pixels in each border
            grain.

        Example
        -------
        from upxo.ggrowth.mcgs import mcgs
        pxtal = mcgs(study='independent',input_dashboard='input_dashboard.xls')
        pxtal.simulate()
        pxtal.detect_grains()
        grain_areas_border_grains = pxtal.gs[2].find_npixels_border_grains_fast(metric='npixels')
        """
        border_gids, _, __, ___ = self.find_border_internal_grains_fast()

        border_grain_npixels = []

        if metric in ('npixels'):
            for bg in border_gids:
                border_grain_npixels.append(np.where(self.lgi == bg)[0].size)

        return np.array(border_grain_npixels)

    def find_npixels_internal_grains_fast(self, metric='npixels'):
        """
        Find the number of pixels in each of the internal grains.

        Parameters
        ----------
        metric: Specify which ares metric is needed. Optoins include:
            * 'npixels': Number of pixels
            * 'pxarea': Pixel wise calculated area
            * 'eq_dia': Equivalent diameter

        Reyturn
        -------
        internal_grain_npixels: Numpy array of number of pixels in each
            internal grain.

        Example
        -------
        from upxo.ggrowth.mcgs import mcgs
        pxtal = mcgs(study='independent',input_dashboard='input_dashboard.xls')
        pxtal.simulate()
        pxtal.detect_grains()
        grain_areas_internal_grains = pxtal.gs[2].find_npixels_internal_grains_fast(metric='npixels')
        """
        _, internal_grains, __, ___ = self.find_border_internal_grains_fast()

        internal_grain_npixels = []

        if metric in ('npixels'):
            for ig in internal_grains:
                internal_grain_npixels.append(np.where(self.lgi == ig)[0].size)

        return np.array(internal_grain_npixels)

    def find_ags(self, grains_to_include='all', gids=None, method='npixels'):
        if grains_to_include == 'all':
            pass
        elif grains_to_include == 'border':
            pass
        elif grains_to_include == 'internal':
            pass
        elif grains_to_include in ('gid', 'gids'):
            pass
        return ags

    def find_prop_npixels(self):
        # Get grain NUMBER OF PIXELS into pandas dataframe
        # if self.prop_flag['npixels']:
        npixels = []
        for g in self.g.values():
            npixels.append(len(g['grain'].loc))
        self.prop['npixels'] = npixels
        if self.display_messages:
            print('    Number of Pixels making the grains: DONE')

    def find_prop_npixels_gb(self):
        # Get grain GRAIN BOUNDARY LENGTH (NO. PIXELS) into pandas dataframe
        # if self.prop_flag['npixels_gb']:
        npixels_gb = []
        for g in self.g.values():
            npixels_gb.append(len(g['grain'].gbloc))
        self.prop['npixels_gb'] = npixels_gb

    def find_prop_gb_length_px(self):
        # Get grain GRAIN BOUNDARY LENGTH (NO. PIXELS) into pandas dataframe
        # if self.prop_flag['gb_length_px']:
        gb_length_px = []
        for g in self.g.values():
            gb_length_px.append(len(g['grain'].gbloc))
        self.prop['gb_length_px'] = gb_length_px

    def find_prop_area(self):
        # Get grain AREA into pandas dataframe
        # if self.prop_flag['area']:
        area = []
        for g in self.g.values():
            area.append(g['grain'].skprop.area)
        self.prop['area'] = area

    def find_prop_eq_diameter(self):
        # Get grain EQUIVALENT DIAMETER into pandas dataframe
        # if self.prop_flag['eq_diameter']:
        eq_diameter = []
        for g in self.g.values():
            eq_diameter.append(g['grain'].skprop.equivalent_diameter_area)
        self.prop['eq_diameter'] = eq_diameter

    def find_prop_perimeter(self):
        # Get grain PERIMETER into pandas dataframe
        # if self.prop_flag['perimeter']:
        perimeter = []
        for g in self.g.values():
            perimeter.append(g['grain'].skprop.perimeter)
        self.prop['perimeter'] = perimeter

    def find_prop_perimeter_crofton(self):
        # Get grain CROFTON PERIMETER into pandas dataframe
        # if self.prop_flag['perimeter_crofton']:
        perimeter_crofton = []
        for g in self.g.values():
            perimeter_crofton.append(g['grain'].skprop.perimeter_crofton)
        self.prop['perimeter_crofton'] = perimeter_crofton

    def find_prop_compactness(self):
        # Get grain COMPACTNESS into pandas dataframe
        # if self.prop_flag['compactness']:
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

    def find_prop_aspect_ratio(self):
        # Get grain ASPECT RATIO into pandas dataframe
        # if self.prop_flag['aspect_ratio']:
        aspect_ratio = []
        for g in self.g.values():
            maj_axis = g['grain'].skprop.major_axis_length
            min_axis = g['grain'].skprop.minor_axis_length
            if min_axis <= self.EPS:
                aspect_ratio.append(np.inf)
            else:
                aspect_ratio.append(maj_axis/min_axis)
        self.prop['aspect_ratio'] = aspect_ratio

    def find_prop_solidity(self):
        # Get grain SOLIDITY into pandas dataframe
        # if self.prop_flag['solidity']:
        solidity = []
        for g in self.g.values():
            solidity.append(g['grain'].skprop.solidity)
        self.prop['solidity'] = solidity

    def find_prop_circularity(self):
        # Get grain CIRCULARITY into pandas dataframe
        # if self.prop_flag['circularity']:
        circularity = []

    def find_prop_major_axis_length(self):
        # Get grain MAJOR AXIS LENGTH into pandas dataframe
        # if self.prop_flag['major_axis_length']:
        major_axis_length = []
        for g in self.g.values():
            major_axis_length.append(g['grain'].skprop.axis_major_length)
        self.prop['major_axis_length'] = major_axis_length

    def find_prop_minor_axis_length(self):
        # Get grain MINOR AXIS LENGTH into pandas dataframe
        # if self.prop_flag['minor_axis_length']:
        minor_axis_length = []
        for g in self.g.values():
            minor_axis_length.append(g['grain'].skprop.axis_minor_length)
        self.prop['minor_axis_length'] = minor_axis_length

    def find_prop_morph_ori(self):
        # Get grain MORPHOLOGICAL ORIENTATION into pandas dataframe
        # if self.prop_flag['morph_ori']:
        morph_ori = []
        for g in self.g.values():
            morph_ori.append(g['grain'].skprop.orientation)
        self.prop['morph_ori'] = [mo*180/np.pi for mo in morph_ori]

    def find_prop_feret_diameter(self):
        # Get grain FERET DIAMETER into pandas dataframe
        # if self.prop_flag['feret_diameter']:
        feret_diameter = []
        for g in self.g.values():
            feret_diameter.append(g['grain'].skprop.feret_diameter_max)
        self.prop['feret_diameter'] = feret_diameter

    def find_prop_euler_number(self):
        # Get grain EULER NUMBER into pandas dataframe
        # if self.prop_flag['euler_number']:
        euler_number = []
        for g in self.g.values():
            euler_number.append(g['grain'].skprop.euler_number)
        self.prop['euler_number'] = euler_number

    def find_prop_eccentricity(self):
        # Get grain ECCENTRICITY into pandas dataframe
        # if self.prop_flag['eccentricity']:
        eccentricity = []
        for g in self.g.values():
            eccentricity.append(g['grain'].skprop.eccentricity)
        self.prop['eccentricity'] = eccentricity

    def build_prop(self):
        if self.prop_flag['npixels']:
            self.find_prop_npixels()
        if self.prop_flag['npixels_gb']:
            self.find_prop_npixels_gb()
        if self.prop_flag['gb_length_px']:
            self.find_prop_gb_length_px()
        if self.prop_flag['area']:
            self.find_prop_area()
        if self.prop_flag['eq_diameter']:
            self.find_prop_eq_diameter()
        if self.prop_flag['perimeter']:
            self.find_prop_perimeter()
        if self.prop_flag['perimeter_crofton']:
            self.find_prop_perimeter_crofton()
        if self.prop_flag['compactness']:
            self.find_prop_compactness()
        if self.prop_flag['aspect_ratio']:
            self.find_prop_aspect_ratio()
        if self.prop_flag['solidity']:
            self.find_prop_solidity()
        if self.prop_flag['circularity']:
            self.find_prop_circularity()
        if self.prop_flag['major_axis_length']:
            self.find_prop_major_axis_length()
        if self.prop_flag['minor_axis_length']:
            self.find_prop_minor_axis_length()
        if self.prop_flag['morph_ori']:
            self.find_prop_morph_ori()
        if self.prop_flag['feret_diameter']:
            self.find_prop_feret_diameter()
        if self.prop_flag['euler_number']:
            self.find_prop_euler_number()
        if self.prop_flag['eccentricity']:
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
                print("Storing all requested grain structure properties "
                      "to pandas dataframe")
            else:
                print("No properties calulated as none were requested."
                      " Skipped")

    def docu(self):
        print("ACCESS-1:")
        print("---------")
        print("You can access all properties across all states as: ")
        print("    >> PXGS.gs[M].prop['PROP_NAME']")
        print("ACCESS-2:")
        print("---------")
        print("You can access all state-partitioned properties as:")
        print("    >> PXGS.gs[M].s_prop(s, PROP_NAME)")
        print('    Here, M: requested requested nth temporal slice of grain'
              ' structure\n')
        print("          s: Desired state value\n")

        print('BASIC STATS:')
        print('------------')
        print("You can readily extract some basic statstics as:")
        print("    >> PXGS.gs[M].prop['area'].describe()[STAT_PARAMETER_NAME]")
        print('    Here, M: requested requested nth temporal slice of grain '
              'structure\n')
        print("    Permitted STAT_PARAMETER_NAME are:")
        print("    'count', 'mean', 'std', 'min', '25%', '50%', '75%',"
              " 'max'\n")

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
        print("You can extract further grain properties as permitted by: "
              "''skimage.measure.regionprops'', as:")
        print("    >> PXGS.gs[M].g[Ng]['grain'].PROP_NAME")
        print("    Here, M: temporal slice")
        print("          Ng: nth grain")
        print("          PROP_NAME: as permitted by sckit-image")
        print("    REF: https://github.com/scikit-image/scikit-image/blob/"
              "v0.21.0/skimage/measure/_regionprops.py#L1046-L1329")

    def get_stat(self, PROP_NAME, saa=True, throw=False, ):
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

    def make_valid_prop(self, PROP_NAME='aspect_ratio',
                        rem_nan=True, rem_inf=True, PROP_df_column=None, ):
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
                    print(f"Property {PROP_NAME} has not been calculated in"
                          " temporal slice {self.m}")
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
                    # __ = self.make_valid_prop(rem_nan=True,
                    #                           rem_inf=True,
                    #                           PROP_df_column=self.prop[PROP_NAME], )
                    # PROP_VALUES_VALID = __
                    subset = self.prop[PROP_NAME].iloc[[i-1 for i in self.s_gid[s]]]
                else:
                    subset = None
                    print(f"Temporal slice {self.m} has no grains in s:"
                          " {s}. Skipped")
            else:
                subset, ratio = None, None
                print(f"Property {PROP_NAME} has not been calculated in "
                      "temporal slice {self.m}")
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
                                                   value_range=[80, 100]
                                                   )
        Example-2
        ---------
        gid, value, df_loc = PXGS.gs[8].get_gid_prop_range(PROP_NAME='area',
                                                   range_type='percentage',
                                                   value_range=[80, 100]
                                                   )
        Example-3
        ---------
        gid, value, df_loc = PXGS.gs[8].get_gid_prop_range(PROP_NAME='aspect_ratio',
                                                   range_type='value',
                                                   value_range=[2, 2.5]
                                                   )
        '''
        gids, A_B_values, A_B_indices = [], [], []
        if PROP_NAME in self.prop.columns:
            PROPERTY = self.prop[PROP_NAME].replace([-np.inf, np.inf],
                                                    np.nan).dropna()
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
            elif range_type in ('value', 'by_value'):
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
            elif range_type in ('rank', 'by_rank'):
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
        if 'area' in self.prop.columns:
            gid = self.prop['area'].idxmax()+1
        else:
            areas = self.find_grain_size_fast(metric='npixels')
            gid = 1
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
        gids, _, _ = self.get_gid_prop_range(PROP_NAME='aspect_ratio',
                                             range_type='percentage',
                                             percentage_range=[100, 100],
                                             )
        # plt.imshow(self.g[gid[0]]['grain'].bbox_ex)
        self.plot_grains_gids(list(gids))
        #for _gid_ in gid:
        #    self.g[_gid_]['grain'].plot()

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
        print('========================================')
        print(gids)
        print('========================================')
        for gid in gids:
            if gid in self.gid:
                lgi_masked[lgi_masked == gid] = masker
            else:
                print(f"Invalid gid: {gid}. Skipped")
        # -----------------------------------------
        return lgi_masked, masker

    def mask_s_with_gids(self, gids, masker=-10, force_masker=False):
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

    def plotgs(self, figsize=(6, 6), dpi=120,
               custom_lgi=None,
               cmap='coolwarm', plot_cbar=True,
               title='Title',
               plot_centroid=False, plot_gid_number=False,
               centroid_kwargs={'marker': 'o',
                                'mfc': 'yellow',
                                'mec': 'black',
                                'ms': 2.5},
               gid_text_kwargs={'fontsize': 10},
               title_kwargs={'fontsize': 10},
               label_kwargs={'fontsize': 10}
               ):
        """
        from upxo.ggrowth.mcgs import mcgs
        mcgs = mcgs(study='independent', input_dashboard='input_dashboard.xls')
        mcgs.simulate()
        mcgs.detect_grains()
        mcgs.gs[35].plotgs(figsize=(6, 6), dpi=120, cmap='coolwarm',
                           plot_centroid=True,
                           centroid_kwargs={'marker':'o','mfc':'yellow',
                                            'mec':'black','ms':2.5},
                           plot_gid_number=True)
        """
        plt.figure(figsize=figsize, dpi=dpi)
        if custom_lgi is None:
            LGI = self.lgi
        else:
            LGI = custom_lgi
        plt.imshow(LGI, cmap=cmap)
        if plot_centroid or plot_gid_number:
            centroid_x, centroid_y = [], []
            for gid in self.gid:
                centroid_x.append(self.xgr[self.lgi == gid].mean())
                centroid_y.append(self.ygr[self.lgi == gid].mean())
        if plot_centroid:
            plt.plot(centroid_x, centroid_y, linestyle='None',
                     marker=centroid_kwargs['marker'],
                     mfc=centroid_kwargs['mfc'], mec=centroid_kwargs['mec'],
                     ms=centroid_kwargs['ms'])
        if plot_gid_number:
            for i, (cenx, ceny) in enumerate(zip(centroid_x, centroid_y), start=1):
                plt.text(cenx, ceny, str(i),
                         fontsize=gid_text_kwargs['fontsize'])
        plt.xlabel(r"X-axis, $\mu m$", fontsize=label_kwargs['fontsize'])
        plt.ylabel(r"Y-axis, $\mu m$", fontsize=label_kwargs['fontsize'])
        plt.title(f'tslice={self.m}. {title}')
        if plot_cbar:
            plt.colorbar()

    def plot_grains_gids(self, gids,
                         gclr='color',
                         title="user grains",
                         cmap_name='CMRmap_r', ):
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
        if not dth.IS_ITER(gids):
            gids = [gids]
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

    def plot_grains(self, gids):
        """
        self.plot_grains([1, 2, 3, 4])
        """
        if not isinstance(gids, Iterable):
            raise TypeError('gids should be an Iterable')

        lgi = {gid: None for gid in gids}
        for gid in gids:
            lgi[gid] = gid*(self.lgi == gid)
        lgi = list(lgi.values())
        plt.imshow(np.sum(lgi, axis=0))

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
            LGI[LGI == ba] = pseudo
        LGI[LGI > 0] = 0
        for i, pseudo in enumerate(pseudos):
            LGI[LGI == pseudo] = boundary_array[i]
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

    def hist(self, PROP_NAME=None, bins=20, kde=True, bw_adjust=None,
             stat='density', color='blue', edgecolor='black', alpha=1.0,
             line_kws={'color': 'k', 'lw': 2, 'ls': '-'},
             auto_xbounds=True, auto_ybounds=True,
             xbounds=[0, 50], ybounds=[0, 0.2], peaks=False, height=0,
             prominance=0.2, __stack_call__=False, __tslice__=None, ):
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

    def kde(self, PROP_NAMES, bw_adjust, ):
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

    def plot_histograms(self, props=['area', 'perimeter',
                                     'orientation', 'solidity', ],
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

    def femesh(self, saa=True, throw=False, ):
        '''
        Set up finite element mesh of the poly-xtal
        Use saa=True to update grain structure mesh atttribute
        Use saa=True and throw=True to update and return mesh
        Use saa=False and throw=True to only return mesh
        '''
        # from mcgs import _uidata_mcgs_gridding_definitions_
        # uigrid = _uidata_mcgs_gridding_definitions_(self.uinputs)
        # from mcgs import _uidata_mcgs_mesh_
        # uimesh = _uidata_mcgs_mesh_(self.uinputs)

        if saa:
            self.mesh = mesh_mcgs2d(self.uinputs['uimesh'],
                                    self.uigrid,
                                    self.dim,
                                    self.m,
                                    self.lgi)
            if throw:
                return self.mesh
        if not saa:
            if throw:
                return mesh_mcgs2d(self.uinputs['uimesh'],
                                   self.uigrid,
                                   self.dim,
                                   self.m,
                                   self.lgi)
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
        centroids = []
        for gid in self.gid:
            locs = self.lgi == gid
            centroids.append([self.xgr[locs].mean(), self.ygr[locs].mean()])
        return np.array(centroids)
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
        plt.xlabel('x-axis $\mu m$', fontdict={'fontsize': 12})
        plt.ylabel('y-axis $\mu m$', fontdict={'fontsize': 12})
        plt.title(f"MCGS tslice:{self.m}.\nUPXO.mulpoint2d of grain centroids",
                  fontdict={'fontsize': 12})
        plt.show()

    def vtgs2d(self, visualize=True):
        # from polyxtal import polyxtal2d as polyxtal
        from upxo.pxtal.polyxtal import vtpolyxtal2d as vtpxtal
        self.make_mulpoint2d_grain_centroids()
        self.vtgs = vtpxtal(gsgen_method='vt',
                            vt_base_tool='shapely',
                            point_method='mulpoints',
                            mulpoint_object=self.mp['gc'],
                            xbound=[self.uigrid.xmin,
                                    self.uigrid.xmax+self.uigrid.xinc],
                            ybound=[self.uigrid.ymin,
                                    self.uigrid.ymax+self.uigrid.yinc],
                            vis_vtgs=visualize
                            )
        if visualize:
            self.vtgs.plot(dpi=100,
                           default_par_faces={'clr': 'teal', 'alpha': 1.0, },
                           default_par_lines={'width': 1.5, 'clr': 'black', },
                           xtal_marker_vertex=True, xtal_marker_centroid=True)

    def ebsd_write_ctf(self, folder='upxo_ctf', file='ctf.ctf'):
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

    def export_vtk2d(self):
        pass

    def export_ctf(self, folder, fileName, factor=1, method='nearest'):
        """
        ctf.export_ctf('D:/export_folder', 'sunil')

        CODES before the modication:

            from upxo._sup.export_data import ctf
            ctf = ctf()
            ctf.load_header_file()
            ctf.make_header_from_lines()
            ctf.set_phase_name(phase_name='PHNAME')
            # ------------------------------------
            ctf.set_grid(self.xgr, self.ygr)
            ctf.set_state(self.S, self.s)
            # ------------------------------------
            '''UPDATE TO BE MADE ASAP.'''
            # ctf.set_ori(self.euler1, self.euler2, self.euler3)
            ctf.set_grid_data()
            # ndata = ctf.assemble_grid_data()
            # ndata = ctf.assemble_grid_data_orix()
            ctf.write_ctf_file_ORIX(folder, fileName)
        """
        if method not in ('nearest', 'decimate'):
            raise ValueError('Invalid method provided. Valid: nearest or decimate')
        from upxo._sup.export_data import ctf
        ctf = ctf()
        ctf.load_header_file()
        ctf.make_header_from_lines()
        ctf.set_phase_name(phase_name='PHNAME')
        # ------------------------------------
        if factor > 0.0 and factor < 1.0:
            XGRID, YGRID, SMATRIX = decrease_grid_resolution(self.xgr, self.ygr, self.s, factor)
        elif factor == 1.0:
            XGRID, YGRID, SMATRIX = self.xgr, self.ygr, self.s
        elif factor > 1.0:
            XGRID, YGRID, SMATRIX = increase_grid_resolution(self.xgr, self.ygr, self.s, factor)
        # ------------------------------------
        ctf.set_grid(XGRID, YGRID)
        ctf.set_state(self.S, SMATRIX)
        """UPDATE TO BE MADE ASAP."""
        # ctf.set_ori(self.euler1, self.euler2, self.euler3)
        ctf.set_grid_data()
        # ndata = ctf.assemble_grid_data()
        # ndata = ctf.assemble_grid_data_orix()
        ctf.write_ctf_file_ORIX(folder, fileName)

    def export_slices(self, xboundPer=None, yboundPer=None, zboundPer=None,
                      mlist=None, sliceStepSize=None, sliceNormal=None,
                      xoriConsideration=None, resolution_factor=None,
                      exportDir=None, fileFormats=None, overwrite=None, ):
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
        xsz = math.floor((self.uigrid.xmax-self.uigrid.xmin)/self.uigrid.xinc)
        ysz = math.floor((self.uigrid.ymax-self.uigrid.ymin)/self.uigrid.yinc)
        zsz = math.floor((self.uigrid.zmax-self.uigrid.zmin)/self.uigrid.zinc)
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
