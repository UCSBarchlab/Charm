import functools
from collections import defaultdict

from mcerp3 import *

from Charm.models import Distribution


# Get yield rate based on Possion model.
def get_yield(size, a, d): 
    return (1 + size * d / a) ** (-1 * a)

# Pollack's rule.
def get_pollack(size):
    return (size ** .5)

Bernoullis = defaultdict()

def many_bernoulli_fast(d, area, func):
    """ Cache previous result, stateful, much faster in some cases.
    """
    #logging.info('Yield for {}: {}'.format(d, func(d)))
    num = area / d 
    assert(num >= 0)
    if num == 0:
        Bernoullis[(d, 0)] = 0 
    if (d, num) not in Bernoullis:
        Bernoullis[(d, num)] = many_bernoulli_fast(d, area-d, func) + Bern(func(d))
    return Bernoullis[(d, num)]

def higher_order_bernoulli(d, num, func):
    return Distribution.HigherOrderBernoulli(func(d), num)

def pollack_uncertainty(d, perc):
    mean = get_pollack(d)
    std = perc * mean
    return Distribution.LogNormalDistribution(mean, std)

def pollack_gaussian_uncertainty(d, perc, a=None, b=None):
    mean = get_pollack(d)
    std = perc * mean
    return Distribution.GaussianDistribution(mean, std, a, b)

def pollack_boxcox_uncertainty(d, perc):
    mean = get_pollack(d)
    std = mean * perc
    gt_perf = pollack_uncertainty(d, perc)
    if std:
        return Distribution.DistributionFromBoxCoxGaussian(mean, std, gt_perf._mcpts, lower=0)
    else:
        return gt_perf

def design_pollack_uncertainty(d, p0, perc):
    mean = get_pollack(d)
    std = perc * mean
    return Distribution.QLogNormalDistribution(p0, mean, std)

def design_pollack_gaussian_uncertainty(d, p0, perc, a=None, b=None):
    mean = get_pollack(d)
    std = perc * mean
    return Distribution.QGaussianDistribution(p0, mean, std, a, b)

def design_pollack_boxcox_uncertainty(d, p0, perc):
    mean = get_pollack(d)
    std = mean * perc
    gt_perf = design_pollack_uncertainty(d, p0, perc)
    if std:
        tgt_mean = gt_perf.mean
        tgt_std = gt_perf.var ** .5
        return Distribution.DistributionFromBoxCoxGaussian(tgt_mean, tgt_std, gt_perf._mcpts, lower=0)
    else:
        return gt_perf

def fabrication_boxcox_uncertainty(d, n):
    if n > 0:
        func = UncertaintyModel.fabrication()
        gt = func(d, n)
        tgt_mean = gt.mean
        tgt_std = gt.var ** .5
        return Distribution.DistributionFromBoxCoxGaussian(tgt_mean, tgt_std, gt._mcpts, lower=0)
    else:
        return 0

class UncertaintyModel(object):
    """ This class defines the interface to insert uncertainty models_charm.
    """

    design_failure_rate = None
    perf_variation_rate = None
    fab_a = .5
    fab_d = .003

    @staticmethod
    def set_rates(d = None, p = None):
        """ Sets design failure rate and performance variation rate.
        """

        UncertaintyModel.design_failure_rate = d
        UncertaintyModel.perf_variation_rate = p

    @staticmethod
    def fabrication():
        """ Defines the fabrication uncertainty.

        Returns:
            A callable which evaluates 'real' core number distribution
            given core size and 'design' core number.
        """

        if not UncertaintyModel.perf_variation_rate:
            return lambda x, y: y
        chip_yield = functools.partial(get_yield, a=UncertaintyModel.fab_a,
                d=UncertaintyModel.fab_d)
        return functools.partial(higher_order_bernoulli, func=chip_yield)

    @staticmethod
    def fabrication_boxcox():
        if not UncertaintyModel.perf_variation_rate:
            return lambda x, y: y
        return functools.partial(fabrication_boxcox_uncertainty)

    @staticmethod
    def core_perf_lognorm():
        """ Defines the performance uncertainty based on a lognormal prior.

        Returns:
            A callable which evaluates 'real' performance distribution
            given core size based on a Log-Normal prior distribution.
        """

        if UncertaintyModel.perf_variation_rate is None:
            UncertaintyModel.perf_variation_rate = .0
        if not UncertaintyModel.design_failure_rate:
            return functools.partial(pollack_uncertainty,
                    perc=UncertaintyModel.perf_variation_rate)
        else:
            return functools.partial(design_pollack_uncertainty,
                    p0=1-UncertaintyModel.design_failure_rate,
                    perc=UncertaintyModel.perf_variation_rate)

    @staticmethod
    def core_perf_gaussian(p0, perc, a = None, b = None):
        """ Similar to above but based on a Gaussian prior distribution.
        """

        if UncertaintyModel.perf_variation_rate is None:
            UncertaintyModel.perf_variation_rate = .0
        if not UncertaintyModel.design_failure_rate:
            return functools.partial(pollack_gaussian_uncertainty,
                    perc=UncertaintyModel.perf_variation_rate, a=a, b=b)
        else:
            return functools.partial(design_pollack_gaussian_uncertainty,
                    p0=1-UncertaintyModel.design_failure_rate,
                    perc=UncertaintyModel.perf_variation_rate, a=a, b=b)

    @staticmethod
    def core_perf_boxcox():
        """ Similar to above but use box-cox bootstrapping as prior.
        """

        if UncertaintyModel.perf_variation_rate is None:
            UncertaintyModel.perf_variation_rate = .0
        if not UncertaintyModel.design_failure_rate:
            return functools.partial(pollack_boxcox_uncertainty,
                    perc=UncertaintyModel.perf_variation_rate)
        else:
            return functools.partial(design_pollack_boxcox_uncertainty,
                    p0=1-UncertaintyModel.design_failure_rate,
                    perc=UncertaintyModel.perf_variation_rate)
