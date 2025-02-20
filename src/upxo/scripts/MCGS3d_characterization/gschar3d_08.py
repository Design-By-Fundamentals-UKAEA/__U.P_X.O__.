# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 11:42:59 2025

@author: rg5749
"""

from upxo.ggrowth.mcgs import mcgs
import numpy as np
import pyvista as pv
# # -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## -- ## --
pxt = mcgs()
pxt.simulate(rsfso=6, verbose=False)

tslice = 99
gstslice = pxt.gs[tslice]
gstslice.char_morphology_of_grains(label_str_order=1)
gstslice.set_mprop_volnv()
gstslice.mprop['volnv']
gstslice.make_pvgrid()
gstslice.add_scalar_field_to_pvgrid(sf_name="lgi", sf_value=None)
gstslice.plot_gs_pvvox(scalar="lgi", show_edges=False)
gstslice.n

GIDS = gstslice.get_s_gids(2)
print(len(GIDS))
gstslice.plot_grains(GIDS, scalar="lgi", cmap='viridis', style='surface',
                     show_edges=False, opacity=0.8)

# dir(gstslice)

#
RSFSO = [27, 20, 12, 8, 7, 6, 5, 4, 3, 2]
counts = 4
Ng = np.array([[0 for _ in range(counts)] for __ in RSFSO])
for rsfso_count in range(len(RSFSO)):
    print(40*'-')
    print(RSFSO[rsfso_count])
    print(40*'-')
    for count in range(counts):
        pxt = mcgs()
        pxt.simulate(rsfso=RSFSO[rsfso_count], verbose=False)
        tslice = 24
        gstslice = pxt.gs[tslice]
        gstslice.char_morphology_of_grains(label_str_order=1)
        Ng[rsfso_count][count] = gstslice.n

print(Ng)
