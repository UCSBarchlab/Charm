import copy
import logging
import mcerp3 as mcerp
from mcerp3 import UncertainVariable
import numpy as np
import scipy.stats as ss
from Charm.utils.boxcox import BoxCox
from Charm.utils.kde import Transformations

def f(self):
    return hash(self.rv)

UncertainVariable.__hash__ = f

def Dummy():
    return Distribution.GetDummy()

def Gauss(mean, std, a=None, b=None):
    return Distribution.GaussianDistribution(mean, std, a=a, b=b)

class Distribution(object):
    """ Ddistributions used to express uncertainties.
    """

    NON_ZERO_FACTOR = 1e-6
    B_CACHE = {}
    DUMMY = mcerp.N(0, 1)

    @staticmethod
    def GetDummy():
        return copy.copy(Distribution.DUMMY)

    @staticmethod
    def ConstantDistribution(val):
        #X = Distribution.GetDummy()
        #X._mcpts = np.asarray([val] * mcerp.npts)
        return val

    @staticmethod
    def DistributionFromBoxCoxGaussian(
            target_mean, target_std, samples, lower=None, upper=None): 
        # Uncomment following codes to disable additional transformation.
        if lower is not None:
            lower = None
        if upper is not None:
            upper = None

        # Training sample size to use.
        k =50 
        seeds = np.random.choice(len(samples), replace=False, size=k)
        train_set = np.asarray([samples[s] for s in seeds])
        samples = train_set
        #logging.debug('Boxcox sample size: {}'.format(len(samples)))

        target_var = target_std * target_std
        if lower is not None and upper is None:
            samples = np.log(np.asarray(samples) - lower)
            reverse_func = lambda x: np.exp(x) + lower
            desired_mean = (np.log(target_mean - lower) -
                    (target_var/2) * (1/np.power(target_mean - lower, 2)))
            desired_var = target_var * np.power(1/(target_mean - lower), 2)
        elif lower is None and upper is not None:
            samples = np.log(-1. * np.asarray(samples) - (-1. * upper))
            reverse_func = lambda x: -1. * (np.exp(x) + (-1. * upper))
            desired_mean = (np.log(-1.*target_mean + upper) - 
                    (target_var/2) * (1/np.power(target_mean - upper, 2)))
            desired_var = target_var * np.power(1/(target_mean - upper), 2)
        elif lower is not None and upper is not None:
            assert lower < upper, 'Distribution -- Input variable bound error, lower >= upper'
            samples = (np.asarray(samples) - lower) / (upper - lower)
            assert np.amin(samples) > 0 and np.amax(samples) < 1
            samples = Transformations.logit(samples)
            reverse_func = lambda x: Transformations.sigmoid(x * (upper-lower) + lower)
            desired_mean = (Transformations.logit(target_mean) +
                    (target_var/2) *
                    ((2*target_mean-1)/(np.power(target_mean*(target_mean-1), 2))))
            desired_var = target_var * np.power(-1. / ((target_mean - 1) * target_mean), 2)
        else: 
            reverse_func = lambda x: x
            desired_mean = target_mean
            desired_var = target_var
        desired_std = np.sqrt(desired_var)

        # Shift data set to positive if needed.
        shift = np.amin(samples) - Distribution.NON_ZERO_FACTOR
        if shift < 0:
            samples = samples - shift
            desired_mean = desired_mean - shift
            bt_func = lambda x: reverse_func(x + shift)
        else:
            bt_func = reverse_func

        #BoxCox.test(samples, la=-40, lb=100)
        a = BoxCox.find_lambda(samples)
        #if a > -1. and a < 1.:
        #    a = 1. * round(a)
        logging.debug('Distribution -- BoxCox @ {} target dist: {}, {}'.format(
            a, target_mean, target_std))
        logging.debug('Distribution -- BoxCox @ shift {} desired dist: {}, {}'.format(
            shift, desired_mean, desired_std))

        if a == .0:
            bc_var = desired_var * np.power(1/desired_mean, 2)
            bc_mean = np.log(desired_mean) - desired_var/2 * np.power(1/desired_mean, 2)
            bc_std = np.sqrt(bc_var)
        elif a == 1.:
            bc_var = desired_var
            bc_mean = desired_mean - 1 + desired_var/2
            bc_std = np.sqrt(bc_var)
        else:
            bc_var = desired_var * np.power(np.power(desired_mean, a-1), 2)
            bc_mean = ((np.power(desired_mean, a) - 1) / a +
                    (desired_var / 2) * ((a-1) * np.power(desired_mean, a-2)))
            bc_std = np.sqrt(bc_var)

        # Compute bounds on box-cox transformed domain.
        bc_lower, bc_upper = None, None
        if a > 0:
            bc_lower = -1. / a
            bc_upper = 2 * bc_mean - bc_lower
        if a < 0:
            bc_upper = -1. / a
            bc_lower = 2 * bc_mean - bc_upper
        #logging.debug('CustomDist -- Transformed scale dist: {}, {} ([{}, {}])'.format(
        #    bc_mean, bc_std, bc_lower, bc_upper))
        Y = Distribution.GetDummy()
        max_trials = 20
        while (max_trials):
            # X: Gaussian in BoxCox transformed domain.
            X = Distribution.GaussianDistribution(bc_mean, bc_std, bc_lower, bc_upper)
            # Y: distribution in original domain.
            Y._mcpts = BoxCox.back_transform(X._mcpts, a)
            Y._mcpts = bt_func(Y._mcpts)
            if (Y._mcpts >= 0).all():
                logging.debug('Distribution -- BC-Generated dist: ({}, {})'.format(
                    Y.mean, np.sqrt(Y.var)))
                return Y
            max_trials += -1
        raise ValueError('Distribution -- Cannot generate proper BoxCox-Transformed distribution.')

    @staticmethod
    def DistributionFromSamplingFunction(sample_func, trans_func=None):
        x = Distribution.GetDummy()
        gen_vals = sample_func(mcerp.npts)
        gen_vals = gen_vals.reshape(gen_vals.shape[-1])
        if not trans_func:
            # Must convert to np array explicitly, or mcerp will complain.
            x._mcpts = np.asarray(gen_vals)
        else:
            assert(callable(trans_func))
            # Must convert to np array explicitly, or mcerp will complain.
            x._mcpts = np.asarray([trans_func(v) for v in gen_vals])
        return x

    @staticmethod
    def NormalizedBinomialDistribution(mean, std):
        if std == .0:
            logging.warn('CustomDist -- Trying to generate normalized Binomial with zero std.')
            return Distribution.ConstantDistribution(0)
        assert std > .0 and isinstance(std, float)
        n = int(mean * (1 - mean) / (std ** 2))
        assert n > 0
        #logging.debug('Normed Binomial N: {}'.format(N))
        X = mcerp.Binomial(n, mean) / n
        adjust_x = []
        for x in X._mcpts:
            assert x >= 0 and x <= 1
            if x == 0:
                y = x + Distribution.NON_ZERO_FACTOR
            elif x == 1:
                y = x - Distribution.NON_ZERO_FACTOR
            else:
                y = x
            adjust_x.append(y)
        X._mcpts = np.asarray(adjust_x)
        return X

    @staticmethod
    def HigherOrderBernoulli(p0, N):
        if (p0, N) not in Distribution.B_CACHE:
            Distribution.B_CACHE[(p0, N)] = (mcerp.Binomial(N, p0) if N > 0
                    else Distribution.ConstantDistribution(0))
        #return Binomial(N, p0) if N > 0 else Distribution.ConstantDistribution(0)
        return Distribution.B_CACHE[(p0, N)]

    @staticmethod
    def BinomialDistribution(mean, std, shift=0):
        if std == .0:
            logging.warn('CustomDist -- Trying to generate Binomial with zero std.')
            return Distribution.ConstantDistribution(mean)
        assert std > .0 and isinstance(std, float)
        mean = mean - shift
        var = std * std
        p = 1 - var/mean
        n = int(round(mean * mean / (mean - var)))
        assert p > .0 and n > .0, 'CustomDist -- p: {}, n: {}, mean: {}, std: {}, shift: {}'.format(
                p, n, mean, std, shift)
        return mcerp.Binomial(n, p)

    @staticmethod
    def LogNormalDistribution(mean, std):
        logging.debug('Lognorm -- Target LogN: {}, {}'.format(mean, std))
        if std == .0:
            return Distribution.ConstantDistribution(mean)
        var = std * std
        mean2 = mean * mean
        mu = np.log(mean) - (var/2)*(1/mean2)
        sigma = np.sqrt(var/mean2)
        # Have to construct LogN ourselves, the parameterizaion in mcerp is not correct.
        dist = UncertainVariable(ss.lognorm(sigma, scale=np.exp(mu)))
        logging.debug('Lognorm -- Gen LogN: ({}, {})'.format(dist.mean, np.sqrt(dist.var)))
        #PlotHelper.plot_hist(dist._mcpts)
        return dist

    @staticmethod
    def GaussianDistribution(mean, std, a=None, b=None):
        if std == .0:
            return Distribution.ConstantDistribution(mean)
        if a is None and b is None:
            # Unbounded Gaussian.
            return mcerp.N(mean, std*std)
        else:
            dist = UncertainVariable(ss.truncnorm(
                a = -np.inf if a is None else (a - mean) / std,
                b = np.inf if b is None else (b - mean) / std,
                loc = mean, scale = std))
            #logging.debug('CustomDist -- truncated gaussian: {}, {} [{}, {}]'.format(
            #    dist.mean, np.sqrt(dist.var), a, b))
            return dist

    @staticmethod
    def QLogNormalDistribution(p0, mean, std):
        if std == .0:
            return Distribution.ConstantDistribution(mean)
        assert(p0 >=0 and p0 <= 1)
        # Performance needs to be non zero, because num_core might be zero.
        result_dist = (mcerp.Bern(p0) * Distribution.LogNormalDistribution(mean, std) +
                Distribution.NON_ZERO_FACTOR)
        assert .0 not in result_dist._mcpts
        logging.debug('QLognorm -- {}'.format(result_dist))
        return result_dist

    @staticmethod
    def QGaussianDistribution(p0, mean, std, a = None, b = None):
        if std == .0:
            return Distribution.ConstantDistribution(mean)
        assert(p0 >= 0 and p0 <= 1)
        return (mcerp.Bern(p0) * Distribution.GaussianDistribution(mean, std, a, b) +
                Distribution.NON_ZERO_FACTOR)
    
    @staticmethod
    def RelatedMax(dists):
        """ @deprecated
        Return a rv x related with list of distributions.
        
        Each sample of x will be set to max(samples of dists) or non_zero_factor.
        """
        assert(dists)
        x = mcerp.N(0, 1) # dummy, could be any UncertainVariable
        mcpts = [None] * mcerp.npts
        for i in range(mcerp.npts):
            mcpts[i] = max([d._mcpts[i] for d in dists])
            mcpts[i] = mcpts[i] if mcpts[i] else Distribution.NON_ZERO_FACTOR
        x._mcpts = mcpts
        return x

    @staticmethod
    def ConditionalMax(ns, ps):
        """ @deprecated
        Return a rv x, consisting of the maximum of ps if corresponding ns hold.

        Each sample of x will be set to max(sample of p if corresponding n > 0) or non_zero_factor.

        Args:
            ns: conditionals, can be distributions or scalars.
            ps: candidates, can be distributions or scalars.
        Return:
            x: result distribution.
        """
        assert(ns and ps)

        if isinstance(ns[0], UncertainFunction) and isinstance(ps[0], UncertainFunction):
            logging.debug('CustomDist -- Distributional candidates with distributional conditions.')
            x = mcerp.N(0, 1) # dummy, could be any UncertainVariable
            mcpts = [None] * mcerp.npts
            for i in range(mcerp.npts):
                candidate = [p._mcpts[i] for n, p in zip(ns, ps) if n._mcpts[i] > 0]
                mcpts[i] = max(candidate) if candidate else Distribution.NON_ZERO_FACTOR
            x._mcpts = mcpts
            return x
        elif isinstance(ns[0], UncertainFunction) and not isinstance(ps[0], UncertainFunction):
            logging.debug('CustomDist -- Scalar candidates with distributional conditions.')
            x= mcerp.N(1, 1)
            mcpts = [None] * mcerp.npts
            for i in range(mcerp.npts):
                candidate = [p for n, p in zip(ns, ps) if n._mcpts[i] > 0]
                mcpts[i] = max(candidate) if candidate else Distribution.NON_ZERO_FACTOR
            x._mcpts = mcpts
            return x
        elif not isinstance(ns[0], UncertainFunction) and isinstance(ps[0], UncertainFunction):
            logging.debug('CustomDist -- Distributional candidates with scalar conditions.')
            x = mcerp.N(1, 1)
            mcpts = [None] * mcerp.npts
            for i in range(mcerp.npts):
                candidate = [p._mcpts[i] for n, p in zip(ns, ps) if n > 0]
                mcpts[i] = max(candidate) if candidate else Distribution.NON_ZERO_FACTOR
            x._mcpts = mcpts
            return x
        else:
            logging.debug('CustomDist -- Scalar candidates with scalar conditions.')
            candidate = [p for n, p in zip(ns, ps) if n > 0]
            return np.amax(candidate)
