from termcolor import colored
from color_specs import CPSPEC as CPS
import datatype_handlers as dth
import numpy as np
import re
import gops


class _UISIMPAR_MCGS_:

    __slots__ = ('S',
                 'mcsteps',
                 'nstates',
                 'solver',
                 'mctype',
                 'tgrad',
                 'default_mcalg',
                 'state_sampling_scheme',
                 'consider_boltzmann_probability',
                 's_boltz_prob',
                 'boltz_prob_values',
                 'boltzmann_temp_factor_max',
                 'boundary_condition_type',
                 'NL',
                 'kineticity',
                 'purge_previous',
                 'introduce_sub_grains',
                 'algo_hops_flag',
                 'alg_names',
                 'alg_epochs',
                 'save_intervals',
                 'mesh_intervals',
                 'algo_hop',
                 'algo_hops',
                 '__epoch_hops_mcsteps_pct__',
                 'epoch_hops_mcsteps',
                 'epoch_hops_algos',
                 '__mcstep_hop_locks__',
                 'mcalg',
                 'mcint_save_at_mcstep_interval',
                 'mcint_promt_display',
                 '__uiSP__',
                 'gsi',
                 )

    def __init__(self,
                 uiSIMPAR,
                 gsi=None,
                 ):
        '''
        Set the grain structure index, which could be:
            * Index of the current grain structure in the parameter sweep
            study. This is the same as the key value of the present grain
            structure in ps.gsi dictionary.
            * Slightly Perturbed Grain Structure. FUTURE FEATURE.
            NOTE: Work needed in on
            this in: (1) concept development (2) procedure development
            (3) interface development (4) code development (5) implementation
        '''
        self.gsi = gsi
        # --------------------------------------------------------------
        self.__uiSP__ = uiSIMPAR
        # --------------------------------------------------------------
        self.set_fundamental_parameters()
        self.set_algorithms_and_parameters()
        self.set_morph_effects()
        self.set_data_write_details()
        # --------------------------------------------------------------

    def show_slots(self):
        return gops.slots(self)

    def show_methods(self):
        return gops.methods(self)

    def show_dunders(self):
        return gops.dunders(self)

    def print_dict(self):
        print(self.__uiSP__)

    def reload(self):
        print("Please use ui.load_simpar()")

    def convert_strListOfNum_to_Numlist(self, strListOfNum):
        return list(map(int, re.findall(r'\d+', strListOfNum)))

    def set_fundamental_parameters(self):
        self.set_mcstates()
        self.set_mcsteps()
        self.set_default_save_intervals()
        self.set_default_mcint_promt_display()
        self.set_solver()
        self.set_mctype()

    def set_mcstates(self):
        self.S = int(self.__uiSP__['main']['nstates'])
        self.nstates = int(self.__uiSP__['main']['nstates'])

    def set_mcsteps(self):
        self.mcsteps = int(self.__uiSP__['main']['mcsteps'])

    def set_default_save_intervals(self):
        self.mcint_save_at_mcstep_interval = int(self.__uiSP__['main']['mcint_save_at_mcstep_interval'])

    def set_default_mcint_promt_display(self):
        self.mcint_promt_display = int(self.__uiSP__['main']['mcint_promt_display'])

    def set_solver(self):
        self.solver = self.__uiSP__['main']['solver']

    def set_mctype(self):
        self.mctype = self.__uiSP__['main']['MCTYPE']

    def set_algorithms_and_parameters(self):
        self.set_default_algorithm()
        self.set_state_sampling_scheme()
        self.set_kbt()
        self.set_boundary_condition()
        self.set_non_locality()
        self.set_kineticity()
        self.set_algo_hop_flag()
        self.set_alg_names()
        self.set_alg_epochs()
        self.set_epoch_hops_mcsteps()
        self.set_epoch_hops_algos()

    def set_default_algorithm(self):
        self.default_mcalg = self.__uiSP__['main']['default_mcalg']

    def set_kineticity(self):
        self.kineticity = self.__uiSP__['main']['kineticity']

    def set_state_sampling_scheme(self):
        self.state_sampling_scheme = self.__uiSP__['main']['state_sampling_scheme']

    def set_kbt(self):
        self.consider_boltzmann_probability = bool(self.__uiSP__['main']['consider_boltzmann_probability'])
        # ---------------------------------------
        self.boltzmann_temp_factor_max = self.__uiSP__['main']['boltzmann_temp_factor_max']
        # ---------------------------------------
        self.s_boltz_prob = self.__uiSP__['main']['s_boltz_prob']
        self.set_boltz_prob_values()
        # ---------------------------------------
        self.tgrad = None

    def set_boltz_prob_values(self):
        if self.s_boltz_prob == 'type-A':
            _a_ = np.random.random(size=self.nstates)
            kbf = self.boltzmann_temp_factor_max
            self.boltz_prob_values = np.exp(-kbf*_a_)
        elif self.s_boltz_prob == 'type-B':
            _a_ = np.arange(self.nstates)
            _a_ = self.boltzmann_temp_factor_max*_a_/_a_.max()
            _ = np.random.random(size=self.nstates)
            self.boltz_prob_values = np.exp(-_a_*_)
        elif self.s_boltz_prob == 'type-C':
            pass
        elif self.s_boltz_prob == 'type-D':
            pass
        elif self.s_boltz_prob == 'custom':
            print("Please use ui.sim.custom_boltz_prob_values(custom_kbt_list_data) to enter the custom data.")
            print("NOTE1: custom_kbt_list_data must be an iterable")
            print("NOTE2: len(custom_kbt_list_data) should be equal to ui.sim.nstates")
            print("NOTE3: Value set to: 'not yet set by user'")
            self.boltz_prob_values = 'not yet set by user'

    def custom_boltz_prob_values(self, custom_kbt_list_data):
        if type(custom_kbt_list_data) in dth.dt.ITERABLES:
            self.boltz_prob_values = custom_kbt_list_data
        else:
            print("Invalid type(custom_kbt_list_data). Only iterables allowed")

    def set_boundary_condition(self):
        self.boundary_condition_type = self.__uiSP__['main']['boundary_condition_type']

    def set_non_locality(self):
        self.NL = int(self.__uiSP__['main']['NL'])

    def set_morph_effects(self):
        self.set_introduce_sub_grains()

    def set_introduce_sub_grains(self):
        self.introduce_sub_grains = bool(self.__uiSP__['main']['introduce_sub_grains'])

    def set_algo_hop_flag(self):
        self.algo_hops_flag = bool(self.__uiSP__['main']['algo_hop'])

    def set_alg_names(self):
        self.alg_names = {'algo'+str(i): v for i, v in enumerate(self.__uiSP__['algNames'].values())}

    def set_alg_epochs(self):
        self.alg_epochs = {'epoch'+str(i): int(v) for i, v in enumerate(self.__uiSP__['algEpochs'].values())}

    def set_epoch_hops_algos(self):
        self.epoch_hops_algos = None

    def set_epoch_hops_mcsteps(self):
        self.epoch_hops_mcsteps = None

    def set_data_write_details(self):
        self.set_purge_previous()
        self.set_save_intervals()
        self.set_mesh_intervals()

    def set_purge_previous(self):
        self.purge_previous = bool(self.__uiSP__['main']['purge_previous'])

    def set_save_intervals(self):
        _ = list(self.__uiSP__['saveIntervals'].values())[:-1]
        self.save_intervals = {'save_at_simstart': bool(self.__uiSP__['saveIntervals']['save_at_simstart'])}
        for i, v in enumerate(_):
            self.save_intervals['sint'+str(i)] = v

    def set_mesh_intervals(self):
        _ = list(self.__uiSP__['meshIntervals'].values())[:-1]
        self.mesh_intervals = {'MESH_at_simstart': bool(self.__uiSP__['meshIntervals']['MESH_at_simstart'])}
        for i, v in enumerate(_):
            self.mesh_intervals['mint'+str(i)] = v

    def CPRINTN(self,
                STR1,
                STR2,
                START=' '*5,
                SEP12=': ',
                CLR1='red',
                CLR2='cyan',
                ATTRIBUTES1 = ["bold", "dark"],
                ATTRIBUTES2 = [],
                ):
        # ---------------------------------
        t1 = START + "\033[%dm%s" % (CPS.COLORS[CLR1], STR1) + "\033[0m"
        for attr in ATTRIBUTES1:
            t1 = "\033[%dm%s" % (CPS.ATTRIBUTES[attr], t1) + "\033[0m"
        # ---------------------------------
        t2 = "\033[%dm%s" % (CPS.COLORS[CLR2], STR2) + "\033[0m"
        for attr in ATTRIBUTES2:
            t2 = "\033[%dm%s" % (CPS.ATTRIBUTES[attr], t2) + "\033[0m"
        # ---------------------------------
        return t1 + SEP12 + t2 + "\n"

    def __repr__(self):
        _ = ' '*5
        retstr = "Attributes of Simulation parameters:\n"
        retstr += self.CPRINTN('GSI', self.gsi)
        retstr += _ + "--/"*12 + "\n"
        retstr += self.CPRINTN('SOLVER', self.solver)
        retstr += self.CPRINTN('MCTYPE', self.mctype)
        retstr += self.CPRINTN('DEFAULT_MCALG', self.default_mcalg)
        retstr += _ + "--/"*12 + "\n"
        retstr += self.CPRINTN('Num of MCSTEPS', self.mcsteps)
        retstr += self.CPRINTN('S', self.S)
        retstr += _ + "--/"*12 + "\n"
        retstr += self.CPRINTN('ALGO_HOP_FLAG', self.algo_hops_flag)
        retstr += self.CPRINTN('Algoithm hop: names of algos', self.alg_names)
        retstr += self.CPRINTN('Algoithm hop: epochs', self.alg_epochs)
        retstr += self.CPRINTN('EPOCH HOPS: MCSTEPS (%)', self.epoch_hops_mcsteps)
        retstr += self.CPRINTN('EPOCH HOPS: MCSTEPS', self.epoch_hops_mcsteps)
        retstr += self.CPRINTN('EPOCH HOPS: ALGORITHMS', self.epoch_hops_algos)
        retstr += _ + "   -"*8 + "\n"
        retstr += self.CPRINTN('SAVE_AT_MCSTEPS', self.save_intervals)
        retstr += self.CPRINTN('STATE_SAMPLING_SCHEME', self.state_sampling_scheme)
        retstr += _ + "--/"*12 + "\n"
        retstr += self.CPRINTN('CONSIDER BOLTZMANN PROBABILITY', self.consider_boltzmann_probability)
        retstr += self.CPRINTN('S BOLTZMANN PROBABILITY', self.s_boltz_prob)
        retstr += self.CPRINTN('BOLTZMANN TEMPERATURE FACTOR MAXIMUM', self.boltzmann_temp_factor_max)
        retstr += self.CPRINTN('BOUNDARY CONDITION TYPE', self.boundary_condition_type)
        retstr += self.CPRINTN('TGRAD', self.tgrad)
        retstr += _ + "--/"*12 + "\n"
        retstr += self.CPRINTN('NON LOCALITY', self.NL)
        retstr += self.CPRINTN('KINETICITY', self.kineticity)
        return retstr

    def set_algo_details(self, read_from_file):
        # --------------------------------------------
        self.set_algorithm_hopping()
        self.set_mcstep_hop_lock_and_epoch_hops_mcsteps_pct()
        self.set_epoch_hops_algos()

    def set_algorithm_hopping(self, read_from_file):
        self.algo_hops_names = self.__uiSP__['algNames']
        self.algo_hops_epochs = self.__uiSP__['algEpochs']

    def set_mcstep_hop_lock_and_epoch_hops_mcsteps_pct(self):
        """
        Takes in the info from validated algo_hops and build up
        two secondary lists which are easier to handle in algorithm
        selection. In the process, it may decide to perform additional
        validations which may be required.
        """
        # --------------------------------------------------
        ''' STEP 1    GENERATE MONTE-CARLO EPOCHS
        Each pair of adjacent entries denotes a hop range, which is basically
        the range of monte-carlo iterations within which an algorithm works.
        As to which algorithm is it, which would be working, is isolated
        later on in this method space, but the data is already present in the
        self.algo_hops. Example: If epochs is [0, 20, 100], then there would be
        two epoch hops, which are [0, 20] and [20, 100]. As you can see, its
        actually a range of iteration numbers. Note; To be accurate, the epoch
        hops would actually be [0, 20] and [21, 100].
        '''
        epoch = [_[1] for _ in self.algo_hops]
        epoch[0], epoch[-1] = 0, 100
        # --------------------------------------------------
        '''STEP 2    GENERATE TEMPORARY LOCKS FOR EACH EPOCH HOP.
        If the epoch hops contains startings and endings, which are invalid,
        then the mcsteps_lock will be set to True for the specific epoch hop.
        Obviously, there will be as many elements in it as there are
        epoch hops.
        '''
        self.__mcstep_hop_locks__ = [False for _ in epoch]
        for i in range(len(epoch)):
            if i > 0:
                if epoch[i] < epoch[i-1]:
                    self.__mcstep_hop_locks__[i] = True
        print(f"__mcstep_hop_locks__: {self.__mcstep_hop_locks__}")
        # --------------------------------------------------
        '''STEP 3: generate __epoch_hops_mcsteps_pct__ list for each algorithm.
        The outer list contains lists of all epoxh hops. Each inner list
        contains ther start and end of the monte-carlo iteration number. At
        mc iterartion number of srtart, the corresponding algorithm will be
        switched to, from the previous algorithm, if there was one active.
        If any locks in __mcstep_hop_locks__ have been locked in STEP 2, then
        'mcstep_hops' sublock of the global uisim lock gets locked and
        everything stops.
        '''
        if any(self.__mcstep_hop_locks__):
            # Branch: Locked if True in it.
            self.__lock__['mcstep_hops'] = True
            self.__epoch_hops_mcsteps_pct__ = ['invalid' for _ in self.algo_hops]
            self.epoch_hops_mcsteps = ['invalid' for _ in self.algo_hops]
            print(f"{colored('Invalid algorithm hopping specification. LOCKED. ','red')}")
        else:
            # Branch: Open if True not in it.
            self.__epoch_hops_mcsteps_pct__ = []
            for i, ah in enumerate(self.algo_hops):
                if i == 0 and len(self.algo_hops) == 1:
                    self.__epoch_hops_mcsteps_pct__.append([0, 100])
                elif i == 0 and len(self.algo_hops) > 1:
                    self.__epoch_hops_mcsteps_pct__.append([0, epoch[i]])
                elif i > 0:
                    self.__epoch_hops_mcsteps_pct__.append([epoch[i-1]+1, epoch[i]])
            '''Set the actual mcstep start end values including validity check.
            Validity check includes:
                If self.mcsteps is too small to accomodate the algo_hops,
                then uisim parameters get LOCKED.
            '''
            self.epoch_hops_mcsteps = []
            for _mcsteps_ in self.__epoch_hops_mcsteps_pct__:
                starting_mcstep = int(_mcsteps_[0]*self.mcsteps/100)
                c = int(_mcsteps_[1]*self.mcsteps/100)
                self.epoch_hops_mcsteps.append([starting_mcstep,
                                                starting_mcstep])
            self.epoch_hops_mcsteps[-1][1] = self.mcsteps
        print(f"__epoch_hops_mcsteps_pct__: {self.__epoch_hops_mcsteps_pct__}")

    #def set_epoch_hops_algos(self):
    #    if not self.__lock__['mcstep_hops']:
    #        self.epoch_hops_algos = [am[0] for am in self.algo_hops]
    #    else:
    #        self.epoch_hops_algos = ['invalid' for _ in self.algo_hops]
    #    print(f"epoch_hops_algos:  {self.epoch_hops_algos}")



    @property
    def lock_status(self):
        if self.__lock__['_']:
            print(f"{colored('Locked sub-locks exist', 'red', attrs=['bold'])}")
        return self.__lock__['_']

    @property
    def locks(self):
        return self.__lock__

    @property
    def lock_update(self):
        """
        Updates the lock. Outcome could be leave lock either in open or
        locked state. Returns None.
        """
        keys = list(self.locks.keys())
        keys[list(self.locks.keys()).index('_')] = -1
        keys.remove(-1)
        sublocks = [self.locks[key] for key in keys]
        if True in set(sublocks):
            self.__lock__['_'] = True
        else:
            self.__lock__['_'] = False
        return None
