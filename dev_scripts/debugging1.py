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
PXGS.gs[8].plot()
# //////////////////////////////////////////////////
#               GRAIN STRUCTURE CHARACTERISATION
mt = 8
PXGS.gs[mt].char_morph_2d()
# //////////////////////////////////////////////////
#               IDENTIFY GRAIN NEIGHBOURS
PXGS.gs[mt].neigh()
# //////////////////////////////////////////////////
#               GENERATE VTGS EQUIVALENT
PXGS.gs[mt].vtgs2d(visualize=True)
PXGS.gs[mt].vtgs.get_L0_ng()
# //////////////////////////////////////////////////
from upxo.meshing.pxtalmesh_01 import vtpxtalmesh
pxtal_mesh = vtpxtalmesh(pxtal = PXGS.gs[8].vtgs,
                         mesher = 'pygmsh',
                         elshape = 'tri',
                         elorder = 1,
                         algorithm = 6,
                         elsize_global = [2., 2.5, 2.5],
                         )

pxtal_mesh.get_pygmsh_mesh_feature_count()
