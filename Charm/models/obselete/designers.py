from collections import defaultdict
import logging
import numpy as np

from regression_models import HillMartyModel
from scipy.stats import norm

class Designer(object):


    def __init__(self, perf_model):
        self.perf_model = perf_model

    def gen_perf_hist(self, test_fs, test_cs, test_pd=0, test_pt=0, top=0, fixed=0):
        """Generates the histogram of perf distribution on test set.
        """

        try:
            assert len(test_fs) == len(test_cs)
        except TypeError:
            test_fs = [test_fs]
            test_cs = [test_cs]
            assert len(test_fs) == len(test_cs)

        # d2perfs: d -> [perf...]
        d2perfs = defaultdict(list)
        for test_f, test_c in zip(test_fs, test_cs):
            d2perf, _ = self.perf_model.gen_perf_risk_space(
                    test_f, test_c, test_pd, test_pt, 0, fixed)
            for d, perf in d2perf.iteritems():
                d2perfs[d].append(perf)
        # Return perfs for given design.
        if fixed:
            fixed_d2perfs = {}
            fixed_d2perfs[fixed] = d2perfs[fixed]
            return fixed_d2perfs

        if top:
            # Only return the most permormant top designs.
            sorted_d2perfs = sorted(d2perfs.iteritems(),
                    key=lambda x: np.mean(x[1]), reverse=True)[:top]
            top_d2perfs = dict(sorted_d2perfs)
            return top_d2perfs

        # Else, return everything.
        return d2perfs

    def gen_perf_risk_space(self, test_bar, test_fs, test_cs,
            test_pd=0, test_pt=0, test_area=256, top=0, fixed=0, x=None):
        """Generates points in perf-risk space.
        """

        try:
            assert len(test_fs) == len(test_cs)
        except TypeError:
            test_fs = [test_fs]
            test_cs = [test_cs]
            assert len(test_fs) == len(test_cs)
        d2perfs = defaultdict(list)
        d2risks = defaultdict(list)
        for test_f, test_c in zip(test_fs, test_cs):
            d2perf, d2risk = self.perf_model.gen_perf_risk_space(test_f, test_c,
                    test_pd, test_pt, test_area, test_bar, fixed)
            for d, perf in d2perf.iteritems():
                d2perfs[d].append(perf)
            for d, risk in d2risk.iteritems():
                d2risks[d].append(risk)

        # -1 indicates using relative risk.
        if test_bar == -1:
            logging.debug("Using relative risk, x={}.".format(x))
            d2risks = self.perf_model.gen_relative_risks(d2perfs, x)

        if fixed:
            logging.debug("Using fixed design: {}".format(fixed))
            fixed_d2perfs = {}
            fixed_d2risks = {}
            fixed_d2perfs[fixed] = d2perfs[fixed]
            fixed_d2risks[fixed] = d2risks[fixed]
            return fixed_d2perfs, fixed_d2risks

        #if top == 0:
        # Hack to return all and filter later
        if top >= 0:
            logging.debug("Iterating all designs.")
            return d2perfs, d2risks

        if top == -1:
            logging.debug("Partial design iteration.")
            qualifying_d2perfs = {k:v for k,v in d2perfs.iteritems() if np.mean(v) >= test_bar}
            qualifying_d2risks = {}
            for d in qualifying_d2perfs.keys():
                qualifying_d2risks[d] = d2risks[d]
            return qualifying_d2perfs, qualifying_d2risks

        assert top > 0
        # Only return the most permormant top designs.
        sorted_d2perfs = sorted(d2perfs.iteritems(),
                key=lambda x: np.mean(x[1]), reverse=True)[:top]
        top_d2perfs = dict(sorted_d2perfs)
        top_d2risks = {}
        logging.debug("Getting top {}/{}".format(len(top_d2perfs.keys()), top))
        for d in top_d2perfs.keys():
            top_d2risks[d] = d2risks[d] 
        return top_d2perfs, top_d2risks

    def risk_func_name(self):
        return self.perf_model.risk_func.get_name()

    def get_design_candidates(self):
        return self.perf_model.ds

    def train(self, train_fs, train_cs, train_pd, train_pt, train_area, p_expected=None):
        """Finds best design for both risk-oblivious and risk-aware methods.
        """
        
        try:
            assert len(train_fs) == len(train_cs)
        except TypeError:
            train_fs = [train_fs]
            train_cs = [train_cs]
            assert len(train_fs) == len(train_cs)

        self.design, self.train_perf = self.perf_model.train(train_fs,
                train_cs, train_pd, train_pt, train_area, None)
        self.r_design, self.r_train_perf = self.perf_model.train(train_fs,
                train_cs, train_pd, train_pt, train_area, p_expected)
        logging.debug("{} -- design {}, risk-aware design {}".format(
            self.get_name(), self.design, self.r_design))
        return self.design, self.r_design

    def test(self, test_bar, test_fs, test_cs, test_pd=0, test_pt=0):
        """Tests performance against test sets.
        """

        try:
            assert len(test_fs) == len(test_cs)
        except TypeError:
            test_fs = [test_fs]
            test_cs = [test_cs]
            assert len(test_fs) == len(test_cs)

        # Vectors to hold results.
        v_perf, v_norm_perf, v_risk = [], [], []
        # Performance test for performance oriented design.
        for test_f, test_c in zip(test_fs, test_cs):
            (test_perf, test_norm_perf, test_risk) = self.perf_model.test(
                    self.design, test_f, test_c, test_pd, test_pt, test_bar)
            v_perf.append(test_perf)
            v_norm_perf.append(test_norm_perf)
            v_risk.append(test_risk)
        self.test_perf = (np.mean(v_perf), np.std(v_perf))
        self.norm_perf = (np.mean(v_norm_perf), np.std(v_norm_perf))
        self.risk = (np.mean(v_risk), np.std(v_risk))
        logging.debug("{} -- \n\ttest_perf {}\n\tnorm_perf {}\n\trisk {}".format(
            self.get_name(), self.test_perf, self.norm_perf, self.risk))

        # Reset the vectors.
        v_perf, v_norm_perf, v_risk = [], [], []
        # Performance test for risk oriented design.
        for test_f, test_c in zip(test_fs, test_cs):
            (r_test_perf, r_test_norm_perf, r_test_risk) = self.perf_model.test(
                    self.r_design, test_f, test_c, test_pd, test_pt, test_bar)
            v_perf.append(r_test_perf)
            v_norm_perf.append(r_test_norm_perf)
            v_risk.append(r_test_risk)
        self.r_test_perf = (np.mean(v_perf), np.std(v_perf))
        self.r_norm_perf = (np.mean(v_norm_perf), np.std(v_norm_perf))
        self.r_risk = (np.mean(v_risk), np.std(v_risk))
        logging.debug("{} -- \n\tr_test_perf {}\n\tr_norm_perf {}\n\tr_risk {}".format(
            self.get_name(), self.r_test_perf, self.r_norm_perf, self.r_risk))

    def get_perf_oriented_results(self):
        """Get test results using deisgn that maximize perf.

        Returns:
          self.test_perf: (mean performance, std) over test set.
          self.norm_perf: (mean norm performance, std) over test set.
          self.risk: (mean risk, std)  over test set.
        """

        return self.test_perf, self.norm_perf, self.risk

    def get_risk_oriented_results(self):
        """Get test results using design that minimize risk.
        """

        return self.r_test_perf, self.r_norm_perf, self.r_risk

    def get_name(self):
        return type(self).__name__[:-8]

class TraditionalDesigner(Designer):
    def __init__(self, perf_model):
        Designer.__init__(self, perf_model)

    def train(self, fs, cs, pd=0, pt=0, area=256, p_expected=None):
        train_fs = fs
        train_cs = cs
        if type(train_fs) is not list: train_fs = [train_fs]
        if type(train_cs) is not list: train_cs = [train_cs]
        return Designer.train(self, train_fs, train_cs, pd, pt, area, p_expected)

class AverageDesigner(Designer):
    def __init__(self, perf_model):
        Designer.__init__(self, perf_model)

    def train(self, fs, cs, pd=0, pt=0):
        train_fs = np.mean(fs)
        train_cs = np.mean(cs)
        if type(train_fs) is not list: train_fs = [train_fs]
        if type(train_cs) is not list: train_cs = [train_cs]
        # TODO(weil0ng): Add variance in train_fs, train_cs?
        Designer.train(self, train_fs, train_cs, pd, pt)

class DistributionAwareDesigner(Designer):
    def __init__(self, perf_model, sample_size):
        Designer.__init__(self, perf_model)
        self.sample_size = int(sample_size)

    def train(self, fs, cs, pd=0, pt=0, area=256, p_expected=None):
        mean_f, std_f = norm.fit(fs)
        mean_c, std_c = norm.fit(cs)
        train_fs = np.random.normal(mean_f, std_f, self.sample_size)
        train_cs = np.random.normal(mean_c, std_c, self.sample_size)
        train_fs = [f for f in train_fs if f>0 and f<1]
        train_cs = [c for c in train_cs if c>0]
        min_len = min(len(train_fs), len(train_cs))
        train_fs = train_fs[:min_len]
        train_cs = train_cs[:min_len]
        return Designer.train(self, train_fs, train_cs, pd, pt, area, p_expected)

class OracleDesigner(Designer):
    def __init__(self, perf_model):
        Designer.__init__(self, perf_model)

    def train(self, fs, cs, pd=0, pt=0):
        train_fs = fs
        train_cs = cs
        if type(train_fs) is not list: train_fs = [train_fs]
        if type(train_cs) is not list: train_cs = [train_cs]
        Designer.train(self, train_fs, train_cs, pd, pt)

class PowerDesigner():
    def __init__(self, base_designer):
        self.designer = base_designer

    def train(self, fs, cs, pds, pts):
        pass

    def test(self, test_fs, test_cs, test_pds, test_pts):
        pass
   
    def get_design_candidates(self):
        return self.designer.get_design_candidates()

    def risk_func_name(self):
        return self.designer.risk_func_name()

    def gen_perf_hist(self, test_fs, test_cs, test_pds, test_pt, top=0, fixed=0):
        # d -> [perf on pd]
        d2perfs = defaultdict(list)
        for test_pd in test_pds:
            # This iterates over workloads.
            d2perfs = self.designer.gen_perf_hist(test_fs, test_cs, test_pd, test_pt)
            pt2pd[test_pt].append(test_pd)
            pt2d2perfs[test_pt].append(d2perfs)
        return pt2pd, pt2d2perfs

    def gen_perf_risk_space(self, bar, test_fs, test_cs, test_pds, test_pt):
        # d -> [perf on pd]
        d2perfs = defaultdict(list)
        d2risks = defaultdict(list)
        for test_pd in test_pds:
            perfs_on_d, risks_on_d = self.designer.gen_perf_risk_space(bar,
                    test_fs, test_cs, test_pd, test_pt)
            for d, perfs in d2perfs:
                d2perfs[d].append(np.mean(perfs))
            for d, risks in d2risks:
                d2risks[d].append(np.mean(risks))
        d2risks = self.designer.perf_model.gen_relative_risks(d2risks)
        return d2perfs, d2risks
