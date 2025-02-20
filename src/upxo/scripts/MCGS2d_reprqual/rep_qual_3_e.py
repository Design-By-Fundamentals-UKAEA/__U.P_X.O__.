# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 16:21:16 2024

@author: Dr. Sunil Anandatheertha
"""
plot_while_testing = True

import numpy as np
from upxo.repqual.grain_network_repr_assesser import KREPR
import matplotlib.pyplot as plt
kr = KREPR.from_gsgen(gstype_tgt='mcgs2d', gstype_smp='mcgs2d',
                      is_smp_same_as_tgt = False,
                      characterize_tgt=True, characterize_smp=True,
					  tgt_dashboard='input_dashboard.xls',
					  smp_dashboard='input_dashboard.xls',
					  ordern=[1, 5],
                      tsid_source='user', ssid_source='user',
					  tid=np.arange(1, 10, 2), sid=np.arange(1, 10, 2),
                      _cim_='from_gsgen')
kr.set_rkf(js=True, wd=True, ksp=True, ed=True, nlsd=True,
           degcen=False, btwcen=False, clscen=False, egnvcen=False)
kr.calculate_rkf()

if plot_while_testing:
    kr.plot_rkf(neigh_orders=[1, 5], power=1, figsize=(7, 5), dpi=120,
                xtick_incr=5, ytick_incr=5, lfs=7, tfs=8,
                cmap='nipy_spectral', cbarticks=np.arange(0, 1.1, 0.1), cbfs=10,
                cbtitle='Measure of representativeness R(S|T)',
                cbfraction=0.046, cbpad=0.04, cbaspect=30, shrink=0.5,
                cborientation='vertical',
                flags={'rkf_js': False, 'rkf_wd': True,
                       'rkf_ksp': True, 'rkf_ed': True,
                       'rkf_nlsd': True, 'rkf_degcen': False,
                       'rkf_btwcen': False, 'rkf_clscen': False,
                       'rkf_egnvcen': False})

kr.set_ordern([1, 3, 5])
kr.find_neigh_order_n(saa=True, throw=False)
kr.create_tgt_smp_networks(saa=True, throw=False)
kr.set_rkf(js=True, wd=True, ksp=True, ed=True, nlsd=True, degcen=False,
           btwcen=False, clscen=False, egnvcen=False)
kr.calculate_rkf()

kr.plot_rkf(neigh_orders=[1, 3, 5], power=1, figsize=(7, 5), dpi=120,
            xtick_incr=5, ytick_incr=5, lfs=7, tfs=8,
            cmap='nipy_spectral', cbarticks=np.arange(0, 1.1, 0.1), cbfs=10,
            cbtitle='Measure of representativeness R(S|T)',
            cbfraction=0.046, cbpad=0.04, cbaspect=30, shrink=0.5,
            cborientation='vertical',
            flags={'rkf_js': False, 'rkf_wd': True,
                   'rkf_ksp': True, 'rkf_ed': True,
                   'rkf_nlsd': True, 'rkf_degcen': False,
                   'rkf_btwcen': False, 'rkf_clscen': False,
                   'rkf_egnvcen': False})


data_title = 'R-Field measure: Energy Distance'
n_bins = 5
AD, AX = kr.calculate_uncertainty_angdist(rkf_measure='ed',
                                          neigh_orders=[1, 3, 5], n_bins=n_bins,
                                          data_title=data_title,
                                          throw=True, plot_ad=False)

if plot_while_testing:
    kr.plot_ang_dist(AD, neigh_orders=[1, 3, 5], n_bins=n_bins,
                     figsize=(5, 5), dpi=150, data_title=data_title,
                     cmap='nipy_spectral')
