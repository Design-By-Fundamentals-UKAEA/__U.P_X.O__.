__name__ = "UPXO"
__author__ = "Dr. Sunil Anandatheertha"
__version__ = "1.26.1"
__doc__ = "This module initializes the UPXO and simplifies the pipeline."


import numpy as np
from upxo.geoEntities.mulpoint2d_3 import mulpoint2d


class gsets():
    """
    Manages grain-structure sets for materials science simulations.

    This class supports initializing and manipulating databases for
    different grain structure types and levels, with a focus on Monte
    Carlo simulations.

    Example:
        from upxo.initialize import gsets
        gsdb = gsets()
        gsdb.initialize(gstype='mc2d')
        print(gsdb.db['l0'])

    Attributes:
        type (str): Placeholder for future use, indicating the type of
                    grain structure.
        db (dict): Database for storing grain structure information.
        dim (int): Placeholder for dimensionality of the grain structure.
    """
    __slots__ = ('type',
                 'db',
                 'dim',
                 )

    def __init__(self):
        pass

    def initialize(self, gstype='mc2d', level='l0'):
        """
        Initializes the database with a specified grain structure type.

        Args:
            gstype (str): Type of grain structure simulation. Defaults to
            'mc2d'.
            level (str): The simulation level. Defaults to 'l0'.
        """
        self.db = {'l0': None,
                   'l1': None}
        if gstype in ('mc2d', 'monte-carlo'):
            if level in ('L0', 'l0'):
                self.setup_mcgsl0()

    def setup_mcgsl0(self):
        """
        Sets up Monte Carlo grain structure simulation for level 'l0'.

        It imports and utilizes the Monte Carlo grain structure simulation
        module, performs simulations, detects grains, characterizes
        morphologies, calculates neighboring grains, and generates Voronoi
        Tessellation grain structures based on the simulation results.
        """
        from upxo.ggrowth.mcgs import monte_carlo_grain_structure as mcgs
        self.db['l0'] = mcgs()
        if self.db['l0'].uigrid.dim == 2:
            self.db['l0'].simulate()
            self.db['l0'].detect_grains(mcsteps=None,
                                        kernel_order=2,
                                        store_state_ng=True,
                                        library='scikit-image')
            for mt in self.db['l0'].tslices:
                print('Characterising mcgs temporal slice: @{mt}')
                self.db['l0'].gs[mt].char_morph_2d()
            for mt in self.db['l0'].tslices:
                print('Calculating neighbouring grains @ temporal slice: {mt}')
                self.db['l0'].gs[mt].neigh()
            for mt in self.db['l0'].tslices:
                print('Generating Voronoi Tessellation gs from centroids')
                self.db['l0'].gs[mt].vtgs2d(visualize=False)

    def setup_vtgs(self):
        coordx, coordy = np.random.rand(10, 2).T
        grid = mulpoint2d(mulpoint_type='seed',
                          method='points',
                          gridding_technique='random',
                          sampling_technique='uniform',
                          nrndpnts=25,
                          randuni_calc='by_points',
                          char_length_mean=0.10,
                          char_length_min=0.05,
                          char_length_max=0.15,
                          n_trials=10,
                          n_iterations=10,
                          point_objects=[],
                          make_point_objects=True,
                          mulpoint_objects=None,
                          coordx=[],
                          coordy=[],
                          coordxy=[],
                          space='linear',
                          xbound=[0, 1],
                          ybound=[0, 1],
                          char_length=[0.25, 0.25],
                          n_char_lengths=[10, 10],
                          latvecs=[0.1, 0.1],
                          angles=[0, 60],
                          bridson_sampling_radius=0.01,
                          bridson_sampling_k=30,
                          perturb_flag=False,
                          perturb_type='local_uniform',
                          perturb_mag=[0.05, 0.05],
                          lean='ignore',
                          pdom='grain',
                          make_rid=True,
                          make_ckdtree=True,
                          vis=False,
                          print_summary=True,
                          )
        pass
# ---------------------------------------------------------------------------
