# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 09:41:00 2024

@author: Dr. Sunil Anandatheertha

NOTES: to enable quick running of this script, following parame6ters
are recommended.
    Domain size: 25 x 25
    mcsteps = 5, Q32
    algorithm: 202
"""
plot_while_testing = True

import numpy as np
from upxo.repqual.grain_network_repr_assesser import KREPR
import matplotlib.pyplot as plt
neigh_orders = np.arange(1, 3, 0.5)
kr = KREPR.from_gsgen(gstype_tgt='mcgs2d', gstype_smp='mcgs2d',
                      is_smp_same_as_tgt = False,
                      characterize_tgt=True, characterize_smp=True,
					  tgt_dashboard='input_dashboard.xls',
					  smp_dashboard='input_dashboard.xls',
					  ordern=neigh_orders,
                      tsid_source='from_gs', ssid_source='from_gs',
					  tid=None, sid=None, _cim_='from_gsgen')
kr.set_rkf(js=True, wd=True, ksp=True, ed=True, nlsd=True, degcen=False,
           btwcen=False, clscen=False, egnvcen=False)
kr.calculate_rkf()
kr.calculate_mprop2d()

on = list(kr.tnset.keys())[1]
gsid = 2
gid = 2
kr.tnset[2][gsid][gid]
kr.mprop2d['tgt']['area_pix'][on][gsid][gid]
len(kr.tnset[on][gsid][gid])
kr.mprop2d['tgt']['area_pix'][on][gsid][gid].size

if plot_while_testing:
    kr.plot_rkf(neigh_orders=neigh_orders, power=1, figsize=(7, 5), dpi=120,
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
AD, AX = kr.calculate_uncertainty_angdist(rkf_measure='ed',
                                          neigh_orders=neigh_orders, n_bins=20,
                                          data_title=data_title,
                                          throw=True, plot_ad=False)

if plot_while_testing:
    kr.plot_ang_dist(AD, neigh_orders=neigh_orders, n_bins=20,
                     figsize=(5, 5), dpi=150, data_title=data_title,
                     cmap='Set1')

kr.calculate_rkf_pairwise(1, 2, 4, printmsg=True)
kr.calculate_rkf_pairwise_generalized(1, 1, 3, 2, printmsg=True)

kr.calculate_rkf_js()
kr.calculate_rkf_js_on(neigh_order=2)

kr.calculate_rkf_js_pairwise(kr.tkset[1][4], kr.skset[1][2])
kr.calculate_rkf_js_pairwise(kr.tkset[1][4], kr.skset[2][3])

len(kr.tkset[1][1].nodes)
len(kr.tkset[1][2].nodes)
len(kr.tkset[2][1].nodes)
len(kr.tkset[2][3].nodes)

len(kr.skset[1][1].nodes)
len(kr.skset[1][3].nodes)

from upxo.netops.kmake import make_gid_net_from_neighlist
kr.tnset[1.5][3]
kr.tnset[2][2]

kr.tkset[1.5][3].degree(1)
kr.tkset[2][2].degree(1)

kr.skset[1][3].degree(1)
kr.skset[2][2].degree(1)

len(make_gid_net_from_neighlist(kr.tnset[1][2]).nodes)
len(make_gid_net_from_neighlist(kr.tnset[1][3]).nodes)

import upxo.netops.kcmp as kcmp

kr.tkset[1][2].degree(1)
kr.tkset[1][3].degree(1)

kr.tkset[1.5][3].degree(1)

kcmp.calculate_rkfield_js(kr.tkset[1][2], kr.tkset[1][3])

kr.rkf['js'][2]
if plot_while_testing:
    plt.figure, plt.imshow(kr.rkf['js'][2])
    plt.figure, plt.imshow(kr.rkf['wd'][2])
    plt.figure, plt.imshow(kr.rkf['ed'][2])
kr.rkf_flags
