import copy
import logging
import mcerp
import numpy as np
import scipy.stats as ss
from mcerp import *

def Dummy():
    return Distribution.GetDummy()

def Gauss(mean, std, a=None, b=None):
    return Distribution.GaussianDistribution(mean, std, a=a, b=b)

class Distribution(object):
    """ Ddistributions used to express uncertainties.
    """

    NON_ZERO_FACTOR = 1e-6
    B_CACHE = {}
    DUMMY = N(0, 1)

    @staticmethod
    def GetDummy():
        return copy.copy(Distribution.DUMMY)

    @staticmethod
    def ConstantDistribution(val):
        #X = Distribution.GetDummy()
        #X._mcpts = np.asarray([val] * mcerp.npts)
        return val

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
        X = Binomial(n, mean) / n
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
            Distribution.B_CACHE[(p0, N)] = (Binomial(N, p0) if N > 0
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
        return Binomial(n, p)

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
        return dist

    @staticmethod
    def GaussianDistribution(mean, std, a=None, b=None):
        if std == .0:
            return Distribution.ConstantDistribution(mean)
        if a is None and b is None:
            # Unbounded Gaussian.
            return N(mean, std*std)
        else:
            dist = UncertainVariable(ss.truncnorm(
                a = -np.inf if a is None else (a - mean) / std,
                b = np.inf if b is None else (b - mean) / std,
                loc = mean, scale = std))
            #logging.debug('CustomDist -- truncated gaussian: {}, {} [{}, {}]'.format(
            #    dist.mean, np.sqrt(dist.var), a, b))
            return dist
