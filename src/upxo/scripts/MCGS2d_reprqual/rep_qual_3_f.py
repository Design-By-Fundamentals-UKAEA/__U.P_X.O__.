# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 14:24:45 2024

@author: rg5749
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
					  ordern=[1],
                      tsid_source='from_gs', ssid_source='from_gs',
					  tid=None, sid=None, _cim_='from_gsgen')
kr.set_rkf(js=True, wd=True, ksp=True, ed=True, nlsd=True, degcen=False,
           btwcen=False, clscen=False, egnvcen=False)
kr.calculate_rkf()
kr.calculate_mprop2d()

gsid, on_start = 1, 1.0

LON, neighn_stats, Ng = kr.estimate_upper_ordern_bycount(tors='tgt', gsid=gsid,
                                                         on_start=on_start,
                                                         on_max=30.0, on_incr=1,
                                                         neigh_count_vf_max=0.95,
                                                         include_parent=True, kdeplot=True,
                                                         kdeplot_kwargs={'figsize': (5, 5),
                                                                        'dpi': 120, 'fill': True,
                                                                        'cmap': 'viridis',
                                                                        'fs_xlabel': 12, 'fs_ylabel': 12,
                                                                        'fs_legend': 9, 'fs_xticks': 10,
                                                                        'fs_yticks': 10, 'legend_ncols': 3,
                                                                        'legend_loc': 'best'},
                                                         statplot=True, gsplot=True,)
