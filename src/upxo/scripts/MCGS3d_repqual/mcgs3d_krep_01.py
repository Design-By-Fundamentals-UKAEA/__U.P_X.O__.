# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 11:18:13 2024

@author: rg5749
"""
import numpy as np
from upxo.repqual.grain_network_repr_assesser_3D import KREPR
import matplotlib.pyplot as plt
neigh_orders=[1.5, 2.0, 2.5]
# ----------------------------------------------------------
# tid=np.arange(10, 20, 1), sid=np.arange(10, 20, 1),
kr = KREPR.from_gsgen(gstype_tgt='mcgs3d', gstype_smp='mcgs3d',
                      is_smp_same_as_tgt = False,
                      char_tgt=True, char_smp=True,
                      set_mprops_tgt=False, set_mprops_smp=False,
					  tgt_dashboard='input_dashboard.xls',
					  smp_dashboard='input_dashboard.xls',
					  ordern=neigh_orders,
                      tsid_source='user', ssid_source='user',
					  tid=np.arange(40, 49, 1), sid=np.arange(40, 49, 1),
                      _cim_='from_gsgen',
                      label_str_order=1,
                      mpflags={'volnv': True,
                               'sanv': True,
                               'arbbox': True,
                               'rat_sanv_volnv': True
                               }
                      )

kr.set_rkf(js=True, wd=True, ksp=True, ed=True, nlsd=True,
           degcen=False, btwcen=False, clscen=False, egnvcen=False)

kr.calculate_rkf()

# ----------------------------------------------------------
kr.plot_rkf(neigh_orders=neigh_orders, power=1, figsize=(7, 5), dpi=120,
            xtick_incr=1, ytick_incr=1, lfs=7, tfs=8,
            cmap='nipy_spectral', cbarticks=np.arange(0, 1.1, 0.1), cbfs=10,
            cbtitle='Measure of representativeness R(S|T)',
            cbfraction=0.046, cbpad=0.04, cbaspect=30, shrink=0.5,
            cborientation='vertical',
            flags={'rkf_js': False, 'rkf_wd': True,
                   'rkf_ksp': True, 'rkf_ed': True,
                   'rkf_nlsd': True, 'rkf_degcen': False,
                   'rkf_btwcen': False, 'rkf_clscen': False,
                   'rkf_egnvcen': False})
# ----------------------------------------------------------
data_title = 'R-Field measure: Energy Distance'
AD, AX = kr.calculate_uncertainty_angdist(rkf_measure='ed', neigh_orders=neigh_orders,
                                          n_bins=25, data_title=data_title, throw=True, plot_ad=False)
kr.plot_ang_dist(AD, neigh_orders=neigh_orders, n_bins=20, figsize=(5, 5), dpi=150,
                 data_title=data_title, cmap='Set1')
# ----------------------------------------------------------
data_title = 'R-Field measure: ksp'
AD, AX = kr.calculate_uncertainty_angdist(rkf_measure='ksp', neigh_orders=neigh_orders,
                                          n_bins=50, data_title=data_title, throw=True, plot_ad=False)
kr.plot_ang_dist(AD, neigh_orders=neigh_orders, n_bins=50, figsize=(5, 5), dpi=150,
                 data_title=data_title, cmap='Set1')
# =========================================================================
'''kr.tgset[3].pvgrid.plot()
kr.tgset[3].skimrp[2].centroid
kr.tgset[3].neigh_gid[2]
kr.tgset[3].mprop.keys()
# =========================================================================
[kr.tgset[3].mprop['volnv'][i] for i in kr.tgset[3].neigh_gid[2]]
[kr.tgset[3].mprop['eqdia']['values'][i-1] for i in kr.tgset[3].neigh_gid[2]]
[kr.tgset[3].mprop['arbbox'][i] for i in kr.tgset[3].neigh_gid[2]]'''
# =========================================================================
# kr.rkf['js'][2]


kr.set_mprop3d_flags(volnv=True, volsr=False, volch=False,
                     sanv=True, savi=False, sasr=False,
                     psa=False,
                     pernv=False, pervl=False, pergl=False,
                     eqdia=False, feqdia=False,
                     kx=False, ky=False, kz=False, ksr=False,
                     arbbox=False, arellfit=False,
                     sol=False, ecc=False, com=False, sph=False,
                     fn=False, rnd=False, mi=False, fdim=False,
                     rat_sanv_volnv=True)



kr.calculate_mprop3d(print_msg_tors=True, print_msg_prnm=True,
                     print_msg_no=True, print_msg_gsid=False,
                     print_msg_gid=False)


kr.mprop3d['tgt']['rat_sanv_volnv'][no][tid][gid]

kr.rkf['ed'][3]
