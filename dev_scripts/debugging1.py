import upxo as upxo
from upxo.pxtal.mcgs import monte_carlo_grain_structure as mcgs
from upxo.pxtalops import detect_grains_from_mcstates as get_grains

PXGS = mcgs()
PXGS.simulate()

print(PXGS.gs)
#PXGS.detect_grains(PXGS.m)
gs_dict, state_ng = get_grains.mcgs2d(library='scikit-image',
                                      gs_dict = PXGS.gs,
                                      msteps = PXGS.tslices,
                                      isograin_pxl_neigh_order=2,
                                      store_state_ng=True
                                      )
