import upxo as upxo
from upxo.pxtal.mcgs import monte_carlo_grain_structure as mcgs

PXGS = mcgs()
PXGS.simulate()

print(PXGS.gs)
# //////////////////////////////////////////////////
#               GRAIN DETECTION PHASE

""" Method-1 """
PXGS.detect_grains()

""" Method-1: detailed """
PXGS.detect_grains(mcsteps=None,
                   kernel_order=2,
                   store_state_ng=True,
                   library='scikit-image')

""" Method-2 """
# You would need to manually do stuff here
from upxo.pxtalops import detect_grains_from_mcstates as get_grains
gs_dict, state_ng = get_grains.mcgs2d(library='scikit-image',
                                      gs_dict = PXGS.gs,
                                      msteps = PXGS.tslices,
                                      kernel_order=2,
                                      store_state_ng=True)
# //////////////////////////////////////////////////
#               GRAIN STRUCTURE VISUALIZATION
""" View the grain streucture """
# Simple
#PXGS.gs[10].plot()



PXGS.gs[10].char_morph_2d()
