from Charm.base.sheet import *
import matplotlib.pyplot as plt
import scipy.stats
from sympy import *
from mcerp import *

class PerformanceModel:
    #==========================================================================
    # define all symbols in the system
    #==========================================================================

    config_syms = [
            'symm_core_size', 'symm_core_num', 'big_core_size', 'big_core_num', 'small_core_size', \
                    'small_core_num', 'area_total' 
                    ]

    perf_syms = [
            'single_core_perf', 'symm_core_perf', 'big_core_perf', 'small_core_perf', 'f', 't_s', 't_p'
                    ]

    power_syms = [
            'symm_core_power', 'big_core_power', 'small_core_power', 'e_s', 'e_p'
                    ]

    stat_syms = [
            'execute_time', 'speedup', 'energy', 'energy_delay_product'
                    ]

    #============================================================================
    # define system equations, givens, intermediates (and their devs) and targets
    #============================================================================

    common_exprs = [ 
            'execute_time = t_s + t_p',
            #'speedup = (1/symm_core_perf)/execute_time',
            'speedup = (1/single_core_perf)/execute_time',
            'energy = e_s + e_p',
            'energy_delay_product = energy * execute_time',
            ]

    symm_exprs = [ 
            'single_core_perf = sqrt(area_total)',
            'symm_core_perf = sqrt(symm_core_size)',
            't_s = (1 - f)/symm_core_perf',
            't_p = f/(symm_core_num * symm_core_perf)',
            'symm_core_power = symm_core_size',
            'e_s = t_s * symm_core_power',
            'e_p = t_p * symm_core_power * symm_core_num',
            'area_total = symm_core_num * symm_core_size',
            ]

    given = [('area_total', '256'), ('symm_core_size', '16')]
    intermediates = [('symm_core_perf', '10%'), ('single_core_perf', '10%')]
    targets = ['speedup']

    fs = [('0.1', '0.01'), ('0.2', '0.01'), ('0.3', '0.01'), ('0.4', '0.01'), ('0.5', '0.01'), ('0.6', '0.01'), ('0.7', '0.01'), ('0.8', '0.01'), ('0.9', '0.01')]

    def __init__(self):
        """
        setup the computation
        """
        self.E_speedup = None
        self.sheet1 = Sheet()
        self.sheet1.addSyms(self.config_syms + self.perf_syms + self.stat_syms + self.power_syms)
        self.sheet1.addExprs(self.common_exprs + self.symm_exprs)
        self.sheet1.addPreds(given=self.given, intermediates=self.intermediates, response=self.targets)

    def SetPerformanceTarget(self, E_speedup):
        self.E_speedup = E_speedup

    def PrintLatex(self):
        self.sheet1.printLatex()

    def Eval(self):
        risk = []
        y = []
        y_err = []
        x = []
        x_err = []
        for f in self.fs:
            self.sheet1.addPreds(given=[('f', ) + f])
            result = self.sheet1.compute()['speedup']

            # storage for plotting
            if self.E_speedup:
                risk.append(scipy.stats.norm(result.mean, sqrt(result.var)).cdf(self.E_speedup))
            y.append(result.mean)
            y_err.append(sqrt(result.var))
            x.append(float(f[0]))
            x_err.append(float(f[1]))

        # plotting
        plt.figure()
        plt.xlabel('f')
        if self.E_speedup:
            plt.scatter(x, risk, color='blue')
        plt.errorbar(x, y, xerr = x_err, yerr = y_err)
        #plt.legend(['perf_lmlv', 'perf_hmhv', 'risk_lmlv', 'risk_hmhv'], loc='upper left')
        #plt.show()

model = PerformanceModel()
#model.SetPerformanceTarget(1.1)
model.Eval()
model.PrintLatex()
