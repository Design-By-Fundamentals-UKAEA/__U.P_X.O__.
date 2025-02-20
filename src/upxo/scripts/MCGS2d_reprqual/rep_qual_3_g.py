"""
Created on Tue Jun 18 14:53:39 2024

@author: Dr. Sunil Anandatheertha
"""
import numpy as np
from upxo.repqual.grain_network_repr_assesser import KREPR
import matplotlib.pyplot as plt
import seaborn as sns

ngrains_tgt_SIMSETS = {}
ngrains_smp_SIMSETS = {}
gareas_tgt_mean_SIMSETS = {}
gareas_tgt_std_SIMSETS = {}
neigh_stats_tgt_SIMSETS = {}

LON_values_tgt_SIMSETS = {}
LON_values_smp_SIMSETS = {}
gareas_smp_mean_SIMSETS = {}
gareas_smp_std_SIMSETS = {}
neigh_stats_smp_SIMSETS = {}

tgt_dashboards = {1: 'idash-rep_qual_3_g_smp1.xls',
                  2: 'idash-rep_qual_3_g_smp2.xls',
                  3: 'idash-rep_qual_3_g_smp3.xls',
                  4: 'idash-rep_qual_3_g_smp4.xls',
                  5: 'idash-rep_qual_3_g_smp5.xls'}

smp_dashboards = {1: 'idash-rep_qual_3_g_tgt1.xls',
                  2: 'idash-rep_qual_3_g_tgt2.xls',
                  3: 'idash-rep_qual_3_g_tgt3.xls',
                  4: 'idash-rep_qual_3_g_tgt4.xls',
                  5: 'idash-rep_qual_3_g_tgt5.xls'}
KR = {}
NEIGH_ORDERS = [1]
for nset in tgt_dashboards.keys():
    kr = KREPR.from_gsgen(gstype_tgt='mcgs2d', gstype_smp='mcgs2d',
                          is_smp_same_as_tgt = False,
                          characterize_tgt=True, characterize_smp=True,
    					  tgt_dashboard=tgt_dashboards[nset],
    					  smp_dashboard=smp_dashboards[nset],
    					  ordern=NEIGH_ORDERS,
                          tsid_source='from_gs', ssid_source='from_gs',
    					  tid=None, sid=None, _cim_='from_gsgen')
    KR[nset] = kr

    '''neigh_orders = np.hstack((np.arange(1, 2, 0.1),
                              np.arange(2, 5, 0.5),
                              np.arange(5, 10, 1),
                              np.arange(10, 20, 2)
                              ))'''

    neigh_count_vf_max=0.95
    LON_values_tgt, ngrains_tgt, LON_Ng_ratio_tgt = [], [], []
    gareas_tgt_mean, gareas_tgt_std = [], []
    neigh_stats_tgt = {gsid: None for gsid in kr.tgset.keys()}
    for gsid in kr.tgset.keys():
        print(40*'=', f'\n Target gsid: {gsid}')
        LON, neighn_stats, Ng = kr.estimate_upper_ordern_bycount(tors='tgt', gsid=gsid,
                                                                 on_start=5, on_max=30.0, on_incr=1.5,
                                                                 neigh_count_vf_max=neigh_count_vf_max,
                                                                 include_parent=True, kdeplot=False,
                                                                 statplot=False, gsplot=False,)
        LON_values_tgt.append(LON)
        ngrains_tgt.append(Ng)
        LON_Ng_ratio_tgt.append(LON/Ng)
        gareas_tgt_mean.append(kr.tgset[gsid].areas.mean())
        gareas_tgt_std.append(kr.tgset[gsid].areas.std())
        neigh_stats_tgt[gsid] = neighn_stats

    ngrains_tgt_SIMSETS['set'+str(nset)] = ngrains_tgt
    LON_values_tgt_SIMSETS['set'+str(nset)] = LON_values_tgt
    gareas_tgt_mean_SIMSETS['set'+str(nset)] = gareas_tgt_mean
    gareas_tgt_std_SIMSETS['set'+str(nset)] = gareas_tgt_std
    neigh_stats_tgt_SIMSETS['set'+str(nset)] = neigh_stats_tgt

    LON_values_smp, ngrains_smp, LON_Ng_ratio_smp = [], [], []
    gareas_smp_mean, gareas_smp_std = [], []
    neigh_stats_smp = {gsid: None for gsid in kr.sgset.keys()}
    for gsid in kr.sgset.keys():
        print(40*'=', f'\n Sample gsid: {gsid}')
        LON, neighn_stats, Ng = kr.estimate_upper_ordern_bycount(tors='smp', gsid=gsid,
                                                                 on_start=5, on_max=30.0, on_incr=1.5,
                                                                 neigh_count_vf_max=neigh_count_vf_max,
                                                                 include_parent=True, kdeplot=False,
                                                                 statplot=False, gsplot=False,)
        LON_values_smp.append(LON)
        ngrains_smp.append(Ng)
        LON_Ng_ratio_smp.append(LON/Ng)
        gareas_smp_mean.append(kr.sgset[gsid].areas.mean())
        gareas_smp_std.append(kr.sgset[gsid].areas.std())
        neigh_stats_smp[gsid] = neighn_stats

    ngrains_smp_SIMSETS['set'+str(nset)] = ngrains_smp
    LON_values_smp_SIMSETS['set'+str(nset)] = LON_values_smp
    gareas_smp_mean_SIMSETS['set'+str(nset)] = gareas_smp_mean
    gareas_smp_std_SIMSETS['set'+str(nset)] = gareas_smp_std
    neigh_stats_smp_SIMSETS['set'+str(nset)] = neigh_stats_smp

SETKEYS = list(ngrains_tgt_SIMSETS.keys())

GSIDs = kr.tgset.keys()
# ============================================================================
set_q_map = {'set1': 'Q: 8', 'set2': 'Q: 16', 'set3': 'Q: 32', 'set4': 'Q: 64',
             'set5': 'Q: 128', }
markers = {'set1': 'x', 'set2': '+', 'set3': '*', 'set4': 's', 'set5': 'o', }
# ============================================================================
''' IMSHOW plot All target grain structures.'''
for krkey in KR.keys():
    fig, axes = plt.subplots(nrows=5, ncols = 10, figsize=(5, 5), dpi=150, sharex=True, sharey=True)
    TIDs = np.reshape(list(kr.tgset.keys()), (5, 10))
    XTICKS, YTICKS = np.arange(0, 60, 10), np.arange(0, 60, 10)
    kr = KR[krkey]
    for r in range(TIDs.shape[0]):
        for c in range(TIDs.shape[1]):
            if c == 0:
                axes[r, c].set_title(f'gsid: {TIDs[r, c]}', fontsize=8)
            axes[r, c].imshow(kr.tgset[TIDs[r, c]].lgi, cmap='viridis')
            axes[r, c].set_xticks(XTICKS)
            axes[r, c].set_xticklabels(XTICKS, fontsize=8, rotation=0)
            axes[r, c].set_yticks(YTICKS)
            axes[r, c].set_yticklabels(YTICKS, fontsize=8, rotation=0)
    fig.suptitle(f"Monte-Carlo state length is {set_q_map['set'+str(krkey)]}", fontsize=10)
''' IMSHOW plot All sample grain structures.'''
for krkey in KR.keys():
    fig, axes = plt.subplots(nrows=5, ncols = 10, figsize=(5, 5), dpi=150, sharex=True, sharey=True)
    SIDs = np.reshape(list(kr.sgset.keys()), (5, 10))
    XTICKS, YTICKS = np.arange(0, 60, 10), np.arange(0, 60, 10)
    kr = KR[krkey]
    for r in range(SIDs.shape[0]):
        for c in range(SIDs.shape[1]):
            if c == 0:
                axes[r, c].set_title(f'gsid: {SIDs[r, c]}', fontsize=8)
            axes[r, c].imshow(kr.sgset[SIDs[r, c]].lgi, cmap='viridis')
            axes[r, c].set_xticks(XTICKS)
            axes[r, c].set_xticklabels(XTICKS, fontsize=8, rotation=0)
            axes[r, c].set_yticks(YTICKS)
            axes[r, c].set_yticklabels(YTICKS, fontsize=8, rotation=0)
    fig.suptitle(f"Monte-Carlo state length is {set_q_map['set'+str(krkey)]}", fontsize=10)
# ============================================================================
''' Histogram of target grain areas '''
for krkey in KR.keys():
    fig, axes = plt.subplots(nrows=5, ncols = 10, figsize=(5, 5), dpi=150, sharex=False, sharey=False)
    TIDs = np.reshape(list(kr.tgset.keys()), (5, 10))
    # XTICKS, YTICKS = np.arange(0, 60, 10), np.arange(0, 60, 10)
    kr = KR[krkey]
    for r in range(TIDs.shape[0]):
        for c in range(TIDs.shape[1]):
            if c == 0:
                axes[r, c].set_title(f'gsid: {TIDs[r, c]}', fontsize=8)
            sns.histplot(kr.tgset[TIDs[r, c]].areas, kde=True, ax=axes[r, c])
            # axes[r, c].imshow(kr.tgset[TIDs[r, c]].lgi, cmap='viridis')
            axes[r, c].tick_params(axis='x', labelsize=8)
            axes[r, c].tick_params(axis='y', labelsize=8)
            #axes[r, c].set_xticks(XTICKS)
            #axes[r, c].set_xticklabels(XTICKS, fontsize=8, rotation=0)
            #axes[r, c].set_yticks(YTICKS)
            #axes[r, c].set_yticklabels(YTICKS, fontsize=8, rotation=0)
    fig.suptitle(f"Monte-Carlo state length is {set_q_map['set'+str(krkey)]}", fontsize=10)
''' Histogram of sample grain areas '''
for krkey in KR.keys():
    fig, axes = plt.subplots(nrows=5, ncols = 10, figsize=(5, 5), dpi=150, sharex=False, sharey=False)
    SIDs = np.reshape(list(kr.sgset.keys()), (5, 10))
    # XTICKS, YTICKS = np.arange(0, 60, 10), np.arange(0, 60, 10)
    kr = KR[krkey]
    for r in range(SIDs.shape[0]):
        for c in range(SIDs.shape[1]):
            if c == 0:
                axes[r, c].set_title(f'gsid: {SIDs[r, c]}', fontsize=8)
            sns.histplot(kr.sgset[SIDs[r, c]].areas, kde=True, ax=axes[r, c])
            # axes[r, c].imshow(kr.tgset[SIDs[r, c]].lgi, cmap='viridis')
            axes[r, c].tick_params(axis='x', labelsize=8)
            axes[r, c].tick_params(axis='y', labelsize=8)
            #axes[r, c].set_xticks(XTICKS)
            #axes[r, c].set_xticklabels(XTICKS, fontsize=8, rotation=0)
            #axes[r, c].set_yticks(YTICKS)
            #axes[r, c].set_yticklabels(YTICKS, fontsize=8, rotation=0)
# ============================================================================
''' GSIDs vs. grain areas. '''
plt.figure(figsize=(5, 5), dpi=150)
msz_incr = 0.1
for setkey in SETKEYS:
    plt.plot(GSIDs, gareas_tgt_mean_SIMSETS[setkey],
             linestyle='-', color='black', linewidth=0.75,
             marker='o', markersize=2+msz_incr,
             markerfacecolor='maroon', markeredgecolor='black',
             markeredgewidth=0.5,
             label=f'Target gs @ {set_q_map[setkey]}')
    plt.plot(GSIDs, gareas_smp_mean_SIMSETS[setkey],
             linestyle='--', color='black', linewidth=0.75,
             marker='o', markersize=2+msz_incr,
             markerfacecolor='yellow', markeredgecolor='black',
             markeredgewidth=0.5,
             label=f'Sample gs @ {set_q_map[setkey]}')
    msz_incr += msz_incr
plt.xlabel('Grain strcture ID', fontsize=12)
plt.ylabel('Mean grain size, um^2', fontsize=12)
plt.legend(fontsize=8)
# ============================================================================
''' Number of grains vs. LON_values. '''
plt.figure(figsize=(5, 5), dpi=150)
msz_incr = 0.125
for setkey in SETKEYS:
    plt.plot(ngrains_tgt_SIMSETS[setkey], LON_values_tgt_SIMSETS[setkey],
             linestyle='-', color='black', linewidth=0.75,
             marker=markers[setkey], markersize=4+msz_incr,
             markerfacecolor='maroon', markeredgecolor='black',
             markeredgewidth=0.5,
             label=f'Target gs @ {set_q_map[setkey]}')
    plt.plot(ngrains_smp_SIMSETS[setkey], LON_values_smp_SIMSETS[setkey],
             linestyle='--', color='black', linewidth=0.75,
             marker=markers[setkey], markersize=4+msz_incr,
             markerfacecolor='green', markeredgecolor='black',
             markeredgewidth=0.5,
             label=f'Sample gs @ {set_q_map[setkey]}')
    msz_incr += msz_incr
plt.xlabel('Number of grains', fontsize=11)
plt.ylabel(f'Limiting O(n) values for max neigh count Vf of {neigh_count_vf_max}', fontsize=11)
plt.legend(fontsize=8, loc='best')
# ============================================================================
''' Grain areas vs. LON_values. '''
plt.figure(figsize=(5, 5), dpi=150)
msz_incr = 0.125
for setkey in SETKEYS:
    plt.plot(gareas_tgt_mean_SIMSETS[setkey], LON_values_tgt_SIMSETS[setkey],
             linestyle='-', color='black', linewidth=0.75,
             marker=markers[setkey], markersize=4+msz_incr,
             markerfacecolor='maroon', markeredgecolor='black',
             markeredgewidth=0.5,
             label=f'Target gs @ {set_q_map[setkey]}')
    plt.plot(gareas_smp_mean_SIMSETS[setkey], LON_values_smp_SIMSETS[setkey],
             linestyle='--', color='black', linewidth=0.75,
             marker=markers[setkey], markersize=4+msz_incr,
             markerfacecolor='green', markeredgecolor='black',
             markeredgewidth=0.5,
             label=f'Sample gs @ {set_q_map[setkey]}')
    msz_incr += msz_incr
plt.xlabel('Mean grain area', fontsize=11)
plt.ylabel(f'Limiting O(n) values for max neigh count Vf of {neigh_count_vf_max}', fontsize=11)
plt.legend(fontsize=8, loc='best')
# ============================================================================
for krkey in KR.keys():
    kr = KR[krkey]
    kr.set_rkf(js=True, wd=True, ksp=True, ed=True, nlsd=True, degcen=False,
               btwcen=False, clscen=False, egnvcen=False)
    kr.calculate_rkf()
    kr.calculate_mprop2d()
# ============================================================================
for krkey in KR.keys():
    kr = KR[krkey]
    kr.plot_rkf(neigh_orders=NEIGH_ORDERS, power=1, figsize=(7, 5), dpi=120,
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
# ============================================================================
for krkey in KR.keys():
    kr = KR[krkey]
    data_title = 'R-Field measure: Energy Distance'
    AD, AX = kr.calculate_uncertainty_angdist(rkf_measure='js',
                                              neigh_orders=NEIGH_ORDERS, n_bins=20,
                                              data_title=data_title,
                                              throw=True, plot_ad=False)
    kr.plot_ang_dist(AD, neigh_orders=NEIGH_ORDERS, n_bins=20,
                     figsize=(5, 5), dpi=150, data_title=data_title,
                     cmap='Set1')
# ============================================================================
KR[1].tgset[4].plotgs()
gstr_nneighgids_1 = KR[1].tgset[4].get_upto_nth_order_neighbors_all_grains(1, include_parent=True,
                                                                         output_type='list')
gstr_nneighgids_2 = KR[1].tgset[4].get_upto_nth_order_neighbors_all_grains(2, include_parent=True,
                                                                         output_type='list')
gstr_nneighgids_5 = KR[1].tgset[4].get_upto_nth_order_neighbors_all_grains(5, include_parent=True,
                                                                         output_type='list')
KR[1].tgset[4].plot_grains_gids(gstr_nneighgids_1[1])



NEIGH_ORDERS = [1, 2, 5, 8]
GSKR = KREPR.from_gsgen(gstype_tgt='mcgs2d', gstype_smp='mcgs2d',
                      is_smp_same_as_tgt = False,
                      characterize_tgt=True, characterize_smp=True,
					  tgt_dashboard='idash-rep_qual_3_g_tgt3.xls',
					  smp_dashboard='idash-rep_qual_3_g_tgt3.xls',
					  ordern=NEIGH_ORDERS,
                      tsid_source='from_gs', ssid_source='from_gs',
					  tid=None, sid=None, _cim_='from_gsgen')
GSKR.tgset[4].plotgs()
gstr_nneighgids_1 = GSKR.tgset[4].get_upto_nth_order_neighbors_all_grains(1, include_parent=True,
                                                                         output_type='list')
gstr_nneighgids_2 = GSKR.tgset[4].get_upto_nth_order_neighbors_all_grains(2, include_parent=True,
                                                                         output_type='list')
gstr_nneighgids_5 = GSKR.tgset[4].get_upto_nth_order_neighbors_all_grains(5, include_parent=True,
                                                                         output_type='list')
gstr_nneighgids_8 = GSKR.tgset[4].get_upto_nth_order_neighbors_all_grains(8, include_parent=True,
                                                                         output_type='list')
gstr_nneighgids_ON = {1: gstr_nneighgids_1,
                      2: gstr_nneighgids_2,
                      5: gstr_nneighgids_5,
                      8: gstr_nneighgids_8}

GSKR.tgset[4].plot_grains_gids(gstr_nneighgids_ON[8][1], cmap_name='viridis')

GSKR.set_rkf(js=True, wd=True, ksp=True, ed=True, nlsd=True, degcen=False,
           btwcen=False, clscen=False, egnvcen=False)
GSKR.calculate_rkf()
GSKR.calculate_mprop2d()
GSKR.plot_rkf(neigh_orders=NEIGH_ORDERS, power=1, figsize=(7, 5), dpi=120,
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
AD, AX = GSKR.calculate_uncertainty_angdist(rkf_measure='wd',
                                          neigh_orders=NEIGH_ORDERS, n_bins=20,
                                          data_title=data_title,
                                          throw=True, plot_ad=False)


GSKR.plot_ang_dist(AD, neigh_orders=NEIGH_ORDERS, n_bins=20,
                 figsize=(5, 5), dpi=150, data_title=data_title,
                 cmap='viridis')
