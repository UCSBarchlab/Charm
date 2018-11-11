from Charm.base.sheet import *
from sympy import *
from mcerp import *

class PerformanceModel:
    #==========================================================================
    # define all symbols in the system
    #==========================================================================

    config_syms = [
            'symm_core_size', 'symm_core_num', 'big_core_size', 'big_core_num', 'small_core_size', \
                    'small_core_num', 'area_total' , 'single_core_size'
                    ]

    perf_syms = [
            'single_core_perf', 'symm_core_perf', 'big_core_perf', 'small_core_perf', 'f', 't_s', 't_p', 'dark_silicon_ratio', 'c'
                    ]

    power_syms = [
            'power_total', 'power_density', 'symm_core_power', 'big_core_power', 'small_core_power', 'e_s', 'e_p'
                    ]

    stat_syms = [
            'execute_time', 'speedup', 'energy', 'energy_delay_product'
                    ]

    #============================================================================
    # define system equations, givens, intermediates (and their devs) and targets
    #============================================================================

    common_exprs = [ 
            'execute_time = t_s + t_p',
            'speedup = (1/single_core_perf)/execute_time',
            'dark_silicon_ratio = 1 - (small_core_num * small_core_size + big_core_num * big_core_size)/area_total',
            'energy = e_s + e_p',
            'energy_delay_product = energy * execute_time',
            ]

    asymm_exprs = [
            'single_core_perf = sqrt(single_core_size)',
            'big_core_perf = sqrt(big_core_size)',
            'small_core_perf = sqrt(small_core_size)',
            't_s = (1 - f)/(big_core_num * big_core_perf + (1-big_core_num) * small_core_perf)',
            't_p = f/(big_core_num * big_core_perf + small_core_num * small_core_perf)',
            ]

    extended_asymm_exprs = [
            'single_core_perf = sqrt(single_core_size)',
            'big_core_perf = sqrt(big_core_size)',
            'small_core_perf = sqrt(small_core_size)',
            't_s = (1 - f + small_core_num * c + big_core_num * c)/(big_core_num * big_core_perf + (1-big_core_num) * small_core_perf)',
            't_p = f/(big_core_num * big_core_perf + small_core_num * small_core_perf)',
            ]

    dark_silicon_exprs = [
            'single_core_size = Min(area_total, power_total / power_density)',
            'big_core_power = power_density * big_core_size',
            'big_core_perf = sqrt(big_core_size)',
            'small_core_power = power_density * small_core_size',
            'small_core_num = Max(0, Min((area_total - big_core_size)/small_core_size, (power_total - big_core_power)/small_core_power))',
            'big_core_num = Min(1, Max(0, ceiling(power_total - power_density * big_core_size)))'
            ]


    given = [('area_total', '256'), ('big_core_size', '16'), ('small_core_size', '4'), ('power_density', '1'), ('power_total', '0')]
    #intermediates = [('symm_core_perf', '10%'), ('single_core_perf', '10%')]
    intermediates = []
    targets = ['speedup']

    ds = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    def __init__(self):
        """
        setup the computation
        """
        self.E_speedup = None
        self.sheet1 = Sheet()
        self.sheet1.addSyms(self.config_syms + self.perf_syms + self.stat_syms + self.power_syms)
        self.sheet1.addExprs(self.common_exprs + self.extended_asymm_exprs + self.dark_silicon_exprs)
        self.sheet1.addPreds(given=self.given, intermediates=self.intermediates, response=self.targets)

    def _GetNumerical(self, result):
        if not isinstance(result, UncertainFunction):
            return result
        else:
            return result.mean

    #def SetPerformanceTarget(self, E_speedup):
    #    self.E_speedup = E_speedup

    def _ComputeBaseline(self, power_total, power_density, b_f, b_c=None):
        base_p = 0
        base_d = 0
        for d_s in self.ds:
            for d_b in self.ds:
                if d_b <= d_s:
                    continue
                if b_c:
                    self.sheet1.addPreds(given=[('f', str(b_f)), ('c', str(b_c)), ('small_core_size', str(d_s)), ('big_core_size', str(d_b)), ('power_total', str(power_total)), ('power_density', str(power_density))])
                else:
                    self.sheet1.addPreds(given=[('f', str(b_f)), ('small_core_size', str(d_s)), ('big_core_size', str(d_b)), ('power_total', str(power_total)), ('power_density', str(power_density))])
                result = self._GetNumerical(self.sheet1.compute()['speedup'])
                if result > base_p:
                    base_p = result
                    base_d = (d_s, d_b)
        return base_p, base_d

    def _EvalOnWorkloads(self, d, power_total, power_density, B_f, B_c=None):
        if B_c:
            if len(B_f) > len(B_c):
                B_f = B_f[:len(B_c)]
            if len(B_f) < len(B_c):
                B_c = B_c[:len(B_f)]
        assert(len(B_f) > 0)
        d_s = d[0]
        d_b = d[1]
        p_d = []
        p_d_norm = []
        if B_c:
            for f,c in zip(B_f, B_c):
                self.sheet1.addPreds(given=[('f', str(f)), ('c', str(c)), ('small_core_size', str(d_s)), ('big_core_size', str(d_b)), ('power_total', str(power_total)), ('power_density', str(power_density))])
                result = self._GetNumerical(self.sheet1.compute()['speedup'])
                base_p, base_d = self._ComputeBaseline(power_total, power_density, f, c)
                p_d.append(result)
                p_d_norm.append(result/base_p)
        else:
            for f in B_f:
                self.sheet1.addPreds(given=[('f', str(f)), ('small_core_size', str(d_s)), ('big_core_size', str(d_b)), ('power_total', str(power_total)), ('power_density', str(power_density))])
                result = self._GetNumerical(self.sheet1.compute()['speedup'])
                base_p, base_d = self._ComputeBaseline(power_total, power_density, f)
                p_d.append(result)
                p_d_norm.append(result/base_p)
        return p_d, p_d_norm

    def _FindOptimal(self, power_total, B_power_density, B_f, B_c=None):
        best_p = 0
        best_d = None

        for d_s in self.ds:
            for d_b in self.ds:
                if d_b <= d_s:
                    continue
                p_d, p_d_norm = [], []
                for power_density in B_power_density:
                    tmp_p_d, tmp_p_d_norm = self._EvalOnWorkloads((d_s, d_b), power_total, power_density, B_f, B_c)
                    p_d.extend(tmp_p_d)
                    p_d_norm.extend(tmp_p_d_norm)

                if sum(p_d)/len(p_d) > best_p:
                    best_p = sum(p_d)/len(p_d)
                    best_p_norm = sum(p_d_norm)/len(p_d_norm)
                    best_d = (d_s, d_b)

        return best_d, best_p, best_p_norm

    def EvalOnSamples(self, u_power_density, v_power_density, B_power_total, B_f, B_c=None):
        """
        expected: for each power_total, d = opt(u_power_density), p = P(d, shifted_D_power_density)
        risk-aware: for each power_total, d = opt(D_power_density), p = P(d, shifted_D_power_density)
        Oracle: for each power_total, d = opt(shifted_D_power_density), p = P(d, shifted_D_power_density)
        """

        N = 100 # sample size for training
        M = 30 # sample size for testing
        train_power_density = []
        train_power_density = np.random.normal(u_power_density, sqrt(v_power_density), N)
        train_power_density = [p for p in train_power_density if p>0]

        test_power_density = []
        test_power_density = np.random.normal(u_power_density, sqrt(v_power_density), M)
        test_power_density = [p for p in test_power_density if p>0]

        perf_expected, perf_risk_aware, perf_oracle = {}, {}, {}
        perf_expected_norm, perf_risk_aware_norm, perf_oracle_norm = {}, {}, {}

        for k in range(len(B_power_total)):
            power_total = B_power_total[k]
            print "Power Budget: {}".format(power_total)

            perf_expected[power_total], perf_risk_aware[power_total], perf_oracle[power_total] = [], [], []
            perf_expected_norm[power_total], perf_risk_aware_norm[power_total], perf_oracle_norm[power_total] = [], [], []

            d_expected, _, _ = self._FindOptimal(power_total, [u_power_density], B_f, B_c)
            print "d_expected: {}".format(d_expected)

            d_risk_aware, _, _ = self._FindOptimal(power_total, train_power_density, B_f, B_c)

            print "d_risk_aware: {}".format(d_risk_aware)

            for test_p in test_power_density:
                d_oracle, p_oracle, p_oracle_norm = self._FindOptimal(power_total, [test_p], B_f, B_c)
                print "d_oracle: {}".format(d_oracle)

                p_expected, p_expected_norm = self._EvalOnWorkloads(d_expected, power_total, test_p, B_f, B_c)
                if d_risk_aware == d_expected:
                    p_risk_aware, p_risk_aware_norm = p_expected, p_expected_norm
                else:
                    p_risk_aware, p_risk_aware_norm = self._EvalOnWorkloads(d_risk_aware, power_total, test_p, B_f, B_c)
                
                print "For power density {}: perf_oracle {}  perf_expected {}  perf_risk {}".format(test_p, p_oracle, np.mean(p_expected), np.mean(p_risk_aware))

                perf_expected[power_total].append(np.mean(p_expected))
                perf_expected_norm[power_total].append(np.mean(p_expected_norm))
                perf_risk_aware[power_total].append(np.mean(p_risk_aware))
                perf_risk_aware_norm[power_total].append(np.mean(p_risk_aware_norm))
                perf_oracle[power_total].append(p_oracle)
                perf_oracle_norm[power_total].append(p_oracle_norm)

        return perf_expected, perf_risk_aware, perf_oracle, perf_expected_norm, perf_risk_aware_norm, perf_oracle_norm
