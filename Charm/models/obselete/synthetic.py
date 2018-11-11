from base.sheet import *
import matplotlib.pyplot as plt
import scipy.stats
from sympy import *
from mcerp import *

class PerformanceModel:
    #==========================================================================
    # define all symbols in the system
    #==========================================================================

    config_syms = ['s']

    perf_syms = ['f']

    stat_syms = ['speedup']

    #============================================================================
    # define system equations, givens, intermediates (and their devs) and targets
    #============================================================================

    common_exprs = [ 
            'speedup = (1/(1-f+f/s))',
            ]

    given = [('f', '1'), ('s', '1')]
    intermediates = []
    targets = ['speedup']

    #ss = [('1', '0.2'), ('2', '0.4'), ('4', '0.8'), ('8', '1.6'), ('16', '3.2'), ('32', '6.4'), ('64', '12.8')]
    ss = ['1', '2', '4', '8', '16', '32', '64']
    fs = [('0.1', '0.03'), ('0.2', '0.03'), ('0.3', '0.03'), ('0.4', '0.03'), ('0.5', '0.03'), ('0.6', '0.03'), ('0.7', '0.03'), ('0.8', '0.03'), ('0.9', '0.03')]
    #fs = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']

    def __init__(self):
        """
        setup the computation
        """
        self.E_speedup = None
        self.sheet1 = Sheet()
        self.sheet1.addSyms(self.config_syms + self.perf_syms + self.stat_syms)
        self.sheet1.addExprs(self.common_exprs)
        self.sheet1.addPreds(given=self.given, intermediates=self.intermediates, response=self.targets)

    def SetPerformanceTarget(self, E_speedup):
        self.E_speedup = E_speedup

    def Eval(self):
        plt.figure()
        plt_legend = []
        for f in self.fs:
            risk = []
            y = []
            y_err = []
            x = []
            x_err = []
            x_ticks = []
            for s in self.ss:
                self.sheet1.addPreds(given=[('f', ) + f, ('s', s)])
                result = self.sheet1.compute()['speedup']

                # storage for plotting
                if self.E_speedup:
                    risk.append(scipy.stats.norm(result.mean, sqrt(result.var)).cdf(self.E_speedup))
                if isinstance(result, float):
                    y.append(result)
                    y_err.append(0)
                else:
                    y.append(result.mean)
                    y_err.append(sqrt(result.var))
                if isinstance(s, str):
                    x.append(float(s))
                    x_err.append(0)
                    x_ticks.append(s)
                else:
                    x.append(float(s[0]))
                    x_err.append(float(s[1]))
                    x_ticks.append(s[0])

            print x_err, y_err
            # plotting
            plt.xlabel('n')
            if self.E_speedup:
                plt.scatter(x, risk, color='blue')
            plt.errorbar(x, y, xerr = x_err, yerr = y_err)
            #plt.plot(x, y)
            #plt.semilogx(x, y, basex=2)
            plt.xticks(x, x_ticks, rotation=10)
            plt.xlim([0, 70])
            plt.ylim([-0.3, 15])
            plt_legend.append(f)
        plt.legend(plt_legend, loc='upper left')
        plt.show()

model = PerformanceModel()
#model.SetPerformanceTarget(1.1)
model.Eval()
