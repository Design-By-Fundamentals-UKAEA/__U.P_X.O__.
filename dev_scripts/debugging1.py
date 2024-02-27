import upxo as upxo
from upxo.pxtal.mcgs import monte_carlo_grain_structure as mcgs

PXGS = mcgs()
dir(PXGS)

PXGS.uigeomrepr #
PXGS.uigrid #
PXGS.uigsc
PXGS.uigsprop
PXGS.uiint
PXGS.uimesh
PXGS.uisim


PXGS.simulate()
