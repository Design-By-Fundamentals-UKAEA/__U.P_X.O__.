from termcolor import colored
import datatype_handlers as dth
import numpy as np
import re
import gops


class _UIPIPELINE_MCGS_():
    _Primary_Study_options_ = ('cpfem', 'rqual', 'ps',
                               'ggk', '__dev__')
    _CPFEM_options = ('cpfem1', 'cpfem2', 'cpfem3',
                      'cpfem4', 'cpfem5', '__dev__')
    _RQUAL_options = ('rqual1', 'rqual2', 'rqual3',
                      'rqual4', 'rqual5', '__dev__')
    _Par_Sweep_options_ = ('ps1', 'ps2', '__dev__')
    _Grain_Growth_Kinetics_options_ = ('ggk1', 'ggk2', 'ggk3',
                                       'ggk4', '__dev__')
    __slots__ = ('primary_study',
                 'cpfem',
                 'rqual',
                 'par_sweep',
                 'grain_growth_kinetics',
                 '__LOCK_PS',
                 '__LOCK_CPFEM',
                 '__LOCK_RQUAL',
                 '__LOCK_PS',
                 '__LOCK_GGK',
                 '__uiPL__',
                 'gsi'
                 )

    def __init__(self, uiPL, gsi=None):
        self.gsi = gsi
        # --------------------------------------------------------------
        self.__uiPL__ = uiPL
        # --------------------------------------------------------------
        _primstudy_ = self.__uiPL__['main']['Primary_Study']
        _cpfem_ = self.__uiPL__['main']['CPFEM']
        _rqual_ = self.__uiPL__['main']['RQUAL']
        _ps_ = self.__uiPL__['main']['Par_Sweep']
        _ggk_ = self.__uiPL__['main']['Grain_Growth_Kinetics']
        # ----------------------------------------------
        # SET PRIMARY STUDY
        if _primstudy_ in _UIPIPELINE_MCGS_._Primary_Study_options_:
            self.primary_study = _primstudy_
        else:
            self.primary_study = '__dev__'
        # ----------------------------------------------
        # set cpfem work name
        if self.primary_study == 'cpfem':
            if _cpfem_ in _UIPIPELINE_MCGS_._CPFEM_options:
                self.cpfem = _cpfem_
            else:
                self.cpfem = 'cpfem1'
        else:
            self.cpfem = None
        # ----------------------------------------------
        # set rqual work name
        if self.primary_study == 'rqual':
            if _rqual_ in _UIPIPELINE_MCGS_._RQUAL_options:
                self.rqual = _rqual_
            else:
                self.rqual = 'rqual1'
        else:
            self.rqual = None
        # ----------------------------------------------
        # set parameter sweep work name
        if self.primary_study == 'ps':
            if _ps_ in _UIPIPELINE_MCGS_._Par_Sweep_options_:
                self.par_sweep = _ps_
            else:
                self.par_sweep = 'ps1'
        else:
            self.par_sweep = None
        # ----------------------------------------------
        # set parameter sweep work name
        if self.primary_study == 'ggk':
            if _ggk_ in _UIPIPELINE_MCGS_._Grain_Growth_Kinetics_options_:
                self.grain_growth_kinetics = _ggk_
            else:
                self.grain_growth_kinetics = 'ggk1'
        else:
            self.grain_growth_kinetics = None
        # ----------------------------------------------
        # set developer mode in primary study
        print(f"Primary study: {self.primary_study}")
        if self.primary_study == '__dev__':
            self.cpfem = '__dev__'
            print(f"CPFEM: {self.cpfem}")
            self.rqual = '__dev__'
            print(f"RQUAL: {self.rqual}")
            self.par_sweep = '__dev__'
            print(f"PAR SWEEP: {self.par_sweep}")
            self.grain_growth_kinetics = '__dev__'
            print(f"GRAIN GROWTH KINETICS: {self.grain_growth_kinetics}")

    def show_slots(self):
        return gops.slots(self)

    def show_methods(self):
        return gops.methods(self)

    def show_dunders(self):
        return gops.dunders(self)

    def print_dict(self):
        print(self.__uiPL__)

    def reload(self):
        print("Please use ui.load_pl()")

    def convert_strListOfNum_to_Numlist(self, strListOfNum):
        return list(map(int, re.findall(r'\d+', strListOfNum)))

    def __repr__(self):
        _ = ' '*5
        retstr = "Pipeline specifications: \n"
        retstr += _ + f"{colored('GSI', 'red', attrs=['bold'])}:  {colored(self.gsi, 'cyan')}\n"
        retstr += _ + f"{colored('primary_study', 'red', attrs=['bold'])}: {colored(self.primary_study, 'cyan')}\n"
        retstr += _ + f"{colored('cpfem', 'red', attrs=['bold'])}: {colored(self.cpfem, 'cyan')}\n"
        retstr += _ + f"{colored('rqual', 'red', attrs=['bold'])}: {colored(self.rqual, 'cyan')}\n"
        retstr += _ + f"{colored('par_sweep', 'red', attrs=['bold'])}: {colored(self.par_sweep, 'cyan')}\n"
        retstr += _ + f"{colored('grain_growth_kinetics', 'red', attrs=['bold'])}: {colored(self.grain_growth_kinetics, 'cyan')}\n"
        return retstr

    def unlock_all(self):
        self.unlock_ps()
        self.unlock_cpfem()
        self.unlock_rqual()
        self.unlock_ggk()

    def unlock_ps(self):
        self.__LOCK_PS = False

    def unlock_cpfem(self):
        self.__LOCK_CPFEM = False

    def unlock_rqual(self):
        self.__LOCK_RQUAL = False

    def unlock_ggk(self):
        self.__LOCK_GGK = False
