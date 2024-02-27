import upxo as upxo
from upxo.pxtal.mcgs import monte_carlo_grain_structure as mcgs

PXGS = mcgs()
PXGS.simulate()

print(PXGS.gs)
#PXGS.detect_grains(PXGS.m)
