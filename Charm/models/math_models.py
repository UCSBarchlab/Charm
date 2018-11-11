from base.eval_functions import *
import logging
from mcerp import *
from uncertainties.core import AffineScalarFunc

class MathModel(object):
    #==========================================================================
    # define mapping between custom Symbols -> eval functions
    #==========================================================================
    custom_funcs = {'FABRIC': FABRIC, 'PERF': PERF,
            'POWER': POWER, 'SUM': SUM,
            'CONDMAX': CONDMAX, 'DPRINT': DPRINT}

    #==========================================================================
    # define all symbols in the system
    #==========================================================================
    index_syms = ['i']

    config_syms = ['core_design_size_i', 'core_design_num_i',
            'core_size_i', 'core_num_i', 'core_perf_i', 'core_power_i'
                    ]

    perf_syms = ['f', 'c', 't_s', 't_p'
                    ]

    power_syms = ['e_s', 'e_p', 'p_s', 'p_p'
                    ]

    stat_syms = ['execute_time', 'speedup', 'energy', 'energy_delay_product'
                    ]

    debug_syms = ['dbg_sum', 'dbg_sum_perf', 'dbg_condmax']

    mech_o3_syms = ['N_total', 'D', 'W', 'mil1', 'mil2', 'mbr', 'mdl2',
            'cil1', 'cl2', 'cdr', 'cfe']

    logca_syms = ['g', 'C', 'o', 'L', 'A', 'beta',
            'unaccelerated_time', 'accelerated_time', 'speedup', 'size']

    #============================================================================
    # define system equations
    #============================================================================
    
    mech_o3_exprs = [
            'execute_time = N_total/D + '
            '(D-1)*(mil1+mil2+mbr+mdl2)/(2*D) + '
            'mil1 * cil1 + mil2*cl2 + '
            'mbr * (cdr + cfe) + mdl2*cl2']

    gd_logca_exprs = [
            'unaccelerated_time = C * (g ** beta)',
            'accelerated_time = o + L * g + C * (g ** beta) / A',
            'A = PERF(size)',
            'speedup = unaccelerated_time / accelerated_time'
            ]

    gid_logca_exprs = [
            'unaccelerated_time = C * (g ** beta)',
            'accelerated_time = o + L + C * (g ** beta) / A',
            'A = PERF(size)',
            'speedup = unaccelerated_time / accelerated_time'
            ]

    common_exprs = [ 
            'execute_time = t_s + t_p',
            'speedup = 1/execute_time',
            ]
    
    debug_exprs = [
            'dbg_sum = SUM([core_num_i])',
            'dbg_sum_perf = SUM([core_num_i * core_perf_i])',
            'dbg_condmax = CONDMAX({core_num_i, core_perf_i})',
            ]

    hete_exprs = [
            'core_size_i = core_design_size_i',
            'core_num_i = FABRIC(core_size_i, core_design_num_i)',
            'core_perf_i = PERF(core_size_i)',
            't_s = (1-f+SUM([core_num_i])*c)/CONDMAX({core_num_i, core_perf_i})',
            't_p = f/SUM([core_perf_i * core_num_i])',
            ]

    hete_power_exprs = [
            'energy = e_s + e_p', # TODO: e_p is nan, but neither t_p nor p_p is nan, overflow?
            'energy_delay_product = energy * execute_time',
            'core_power_i = POWER(core_perf_i)',
            'p_s = CONDMAX({core_num_i, core_power_i})',
            'p_p = SUM([core_num_i * core_power_i])',
            'e_s = t_s * p_s',
            'e_p = t_p * p_p'
            ]

    symm_exprs = [
            'small_core_perf = sqrt(small_core_size)',
            't_s = (1 - f + small_core_num * c)/small_core_perf',
            't_p = f/(small_core_num * small_core_perf)',
            'small_core_num = area_total / small_core_size',
            ]

    symm_power_exprs = [
            'small_core_power = power_density * small_core_size',
            'e_s = small_core_power * t_s',
            'e_p = small_core_num * small_core_power * t_p',
            'small_core_perf = sqrt(small_core_size)',
            't_s = (1 - f + small_core_num * c) / small_core_perf',
            't_p = f / (small_core_num * small_core_perf)',
            'small_core_num = Min(area_total/small_core_size, power_total/small_core_power)'
            ]

    asymm_exprs = [
            'small_core_perf = sqrt(small_core_size)',
            't_s = (1 - f +  (1 + base_core_num) * c)/small_core_perf',
            't_p = f / (base_core_num + small_core_perf)',
            'base_core_num = area_total - small_core_size'
            ]

    asymm_power_exprs2 = [
            'small_core_power = power_density * small_core_size',
            'big_core_power = power_density * big_core_size',
            'e_p = (big_core_power + small_core_num * small_core_power) * t_p',
            'e_s = big_core_power * t_s',
            'small_core_perf = sqrt(small_core_size)',
            'big_core_perf = sqrt(big_core_size)',
            't_s = (1 - f + (1 + small_core_num) * c) / big_core_perf',
            't_p = f / (small_core_num * small_core_perf + big_core_perf)',
            'small_core_num = Min((area_total - big_core_size)/small_core_size, (power_total - big_core_power)/small_core_power)'
            ]

    asymm_exprs2 = [
            'small_core_perf = sqrt(small_core_size)',
            'big_core_perf = sqrt(big_core_size)',
            't_s = (1 - f + (1 + small_core_num) * c)/big_core_perf',
            't_p = f / (big_core_perf + small_core_perf * small_core_num)',
            'small_core_num = floor((area_total - big_core_size)/small_core_size)'
            ]

    dynamic_power_exprs = []

    extended_asymm_exprs = [
            'single_core_perf = sqrt(area_total)',
            'big_core_perf = sqrt(big_core_size)',
            'small_core_perf = sqrt(small_core_size)',
            't_s = (1 - f + small_core_num * c)/big_core_perf',
            't_p = f/(small_core_num * small_core_perf + big_core_num * big_core_perf)',
            'area_total = small_core_num * small_core_size + big_core_num * big_core_size'
            ]

    dynamic_exprs = [
            'small_core_perf = sqrt(small_core_size)',
            't_s = (1 - f + area_total * c)/small_core_perf',
            't_p = f/area_total'
            ]

    def get_numerical(self, result):
        if isinstance(result, UncertainFunction):
            return result.mean
        elif isinstance(result, AffineScalarFunc):
            return result.n
        else:
            return result

    def get_var(self, result):
        if isinstance(result, UncertainFunction):
            return result.var
        elif isinstance(result, AffineScalarFunc):
            return result.std_dev * result.std_dev
        else:
            logging.warn('Trying to get var on constant: {}'.format(result))
            return 0

    @staticmethod
    def names():
        return ['symmetric', 'asymmetric', 'hete', 'dynamic', 'mech_o3', 'gd_logca', 'gid_logca']
