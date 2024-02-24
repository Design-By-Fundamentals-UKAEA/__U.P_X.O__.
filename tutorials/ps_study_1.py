
import numpy as np
from mcgs import parameter_sweep
ps = parameter_sweep()
ps.initialize(N = 2)

ps.set_param_grid(domain_size=((0, 100), (0, 100), (0, 0), 1),
                  read_from_file=False, filename=None)

ps.set_param_sim(mcsteps=20, nstates=32, solver='python',
				 tgrad=None, algo_hop=False, algo_hops=[(200, 10), (201, 40), (202, 100)],
                 default_mcalg=ps.default_mcalg,
				 save_at_mcsteps=np.linspace(0, 20, 5),
                 state_sampling_scheme='rejection',
                 consider_boltzmann_probability=False,
                 s_boltz_prob='q_related', boltzmann_temp_factor_max=0.1,
                 boundary_condition_type='wrapped', NL=1,
                 kineticity='static', purge_previous=False, read_from_file=False, filename=None)

ps.set_param_gsc(char_grains=True, char_stage='postsim', library='scikit-image', parallel=True,
				 find_gbseg=True, g_area=True, gb_length=True,
				 gb_length_crofton=True, gb_njp_order=True,
				 g_eq_dia=True, g_feq_dia=True, g_solidity=True,
				 g_circularity=True, g_mjaxis=True, g_mnaxis=True,
				 g_morph_ori=True, g_el=True, g_ecc=True,read_from_file=False, filename=None)

ps.set_param_geomrepr(make_mp_grain_centoids=True, make_mp_grain_points=True,
					  make_ring_grain_boundaries=True, make_xtal_grain=True, make_chull_grain=True,
					  create_gbz=True, gbz_thickness = 0.1, read_from_file=False, filename=None)

ps.set_param_mesh(generate_mesh=False, target_fe_software='abaqus', par_treatment='global',
				  mesher='upxo', gb_conformities=('conformal', 'non_conformal'),
				  global_elsizes=(0.5, 1.0), mesh_algos=(4, 6),
				  grain_internal_el_gradient=('constant', 'constant'),
				  grain_internal_el_gradient_par=(('automin', 'automax'), ('automin', 'automax'),),
				  target_eltypes=('CSP4', 'CSP8'), elsets=('grains', 'grains'),
				  nsets=('x-', 'x+', 'y-', 'y+', ), optimize=(False, False),
				  opt_par=('min_angle', [45, 60], 'jacobian', [0.45, 0.6]),
				  read_from_file=False, filename=None)
# --------------------------------------------------------------------------
ps.uigrid
ps.uisim
ps.uigsc
ps.uimesh
ps.uigeomrepr

ps.gsi[1].uigrid
ps.gsi[1].uisim
ps.gsi[1].uigsc
ps.gsi[1].uimesh
ps.gsi[1].uigeomrepr

ps.gsi[1].uigrid.locks
ps.gsi[1].uisim.locks
ps.gsi[1].uigsc.locks
ps.gsi[1].uimesh.locks
ps.gsi[1].uigeomrepr.locks

ps.gsi[1].uigrid.lock_status
ps.gsi[1].uisim.lock_status
ps.gsi[1].uigsc.lock_status
ps.gsi[1].uimesh.lock_status
ps.gsi[1].uigeomrepr.lock_status
# --------------------------------------------------------------------------
def __init__(self, uidata=None):
    self.set_algorithm_hopping(uidata)
    self.set_s(uidata)
    self.set_kbt(self, uidata)
    self.boundary_condition_type = uidata['boundary_condition_type']
    self.NL = int(uidata['NL'])
    self.kineticity = uidata['kineticity']
    if any(self.__lock__.values()):
        self.__lock__['_'] = True

def set_algorithm_hopping(self, uidata):
    self.mcsteps = int(uidata['mcsteps'])
    self.mcstep_hops = []
    self.mcalg = str(int(uidata['mcalg']))
    self.algo_hop = True
    self.algo_hops = ((str(int(uidata['mcalg'])),
                       100
                       ),
                      )
    self.__वृत्ति१__ = tuple([_[0] for _ in self.algo_hops])
    self.validate_algo_hops(uidata)

def set_s(self, uidata):
    self.S = int(uidata['S'])
    self.state_sampling_scheme = uidata['state_sampling_scheme']

def set_kbt(self, uidata):
    self.consider_boltzmann_probability = bool(uidata['consider_boltzmann_probability'])
    self.s_boltz_prob = uidata['s_boltz_prob']
    self.boltzmann_temp_factor_max = uidata['boltzmann_temp_factor_max']



def validate_algo_hops(self, uidata):
    # --------------------------------------------------
    # mcalg_hops = self.algo_hops[0][0]
    # mcstep_hops = [[0, self.algo_hops[0][1]]]
    # AAA = self.algo_hops
    # AAA = (('200', 20), ('200', 40), ('200', 41))
    # AAA = (('200', 20), )
    # --------------------------------------------------
    PRINT_algo_hops = lambda: print(self.algo_hops) if self.DEV else None
    PRINT_mcsteps_lock = lambda: print(mcsteps_lock) if self.DEV else None
    # --------------------------------------------------
    PRINT_algo_hops()
    t = [_[1] for _ in self.algo_hops]
    t[0], t[-1] = 0, 100
    mcsteps_lock = [False for _ in t]

    for i in range(len(t)):
        if i > 0:
            if t[i] < t[i-1]:
                mcsteps_lock[i] = True

    PRINT_mcsteps_lock()

    if any(mcsteps_lock):
        # locked as True in it.
        self.__lock__['mcstep_hops'] = True
        print(f"{colored('Invalid algorithm hopping specification. LOCKED. ','red')}")
    else:
        # open as True not in it.
        mcstep_hops = []
        for i, ah in enumerate(self.algo_hops):
            if i == 0 and len(self.algo_hops) == 1:
                mcstep_hops = [0, 100]
            elif i == 0 and len(self.algo_hops) > 1:
                mcstep_hops.append([0, t[i]])
            elif i > 0:
                mcstep_hops.append([t[i-1]+1, t[i]])
        self.mcstep_hops = mcstep_hops




    __slots__ = ('S', 'mcsteps', 'nstates', 'solver', 'tgrad',
                 'default_mcalg',
                 'algo_hop', 'algo_hops', 'mcstep_hops', 'mcalg',
                 'save_at_mcsteps',
                 'state_sampling_scheme', 'consider_boltzmann_probability',
                 's_boltz_prob', 'boltzmann_temp_factor_max',
                 'boundary_condition_type',
                 'NL', 'kineticity', '__वृत्ति१__',
                 )
