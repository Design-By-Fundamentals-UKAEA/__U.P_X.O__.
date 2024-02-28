import upxo as upxo
from upxo.pxtal.mcgs import monte_carlo_grain_structure as mcgs

PXGS = mcgs()
PXGS.simulate()

print(PXGS.gs)
# //////////////////////////////////////////////////
#               GRAIN DETECTION PHASE

""" Method - 1 """
PXGS.detect_grains()
""" Method - 1: detailed """
PXGS.detect_grains(mcsteps=None,
                   kernelOrder=2,
                   store_state_ng=True,
                   library='scikit-image')
# Where, [1] mcsteps is the Monte-Carlo temporal slice numbers of interest
# mcsteps must be a subset of PXGS.gs.keys()
# [2] isograin_pxl_neigh_order is the pixel kernel structure used in grain
# detection. This is needed only if library is 'scikit-image'
""" Method - 2 """
# You would need to manually do stuff here
from upxo.pxtalops import detect_grains_from_mcstates as get_grains
gs_dict, state_ng = get_grains.mcgs2d(library='scikit-image',
                                      gs_dict = PXGS.gs,
                                      msteps = PXGS.tslices,
                                      isograin_pxl_neigh_order=2,
                                      store_state_ng=True)
# //////////////////////////////////////////////////
