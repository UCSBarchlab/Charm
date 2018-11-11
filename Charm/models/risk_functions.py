import numpy as np
from mcerp import *
from uncertainties.core import AffineScalarFunc

class RiskFunction(object):
    def get_risk(self, bar, p):
        """ Computes risk for perf array w.r.t. bar.

        Args:
            bar: reference performance bar.
            perfs: performance array-like.

        Returns:
            single float (mean risk)
        """
        if isinstance(p, UncertainFunction):
            return self.func(bar, p._mcpts)
        elif isinstance(p, AffineScalarFunc):
            #TODO: what should we return? How to define risk analytically?
            raise ValueError('Risk -- Undefined behavior.')
        else:
            return self.func(bar, [p])
    
    def get_name(self):
        name = type(self).__name__
        return name[:name.find('Function')]

class DollarValueFunction(RiskFunction):
    def dollar_function(self, bar, perf):
        value = .0
        for p in perf:
            normed_p = float(p)/bar
            if normed_p < .6:
                value += 100
            elif normed_p < .8:
                value += 200
            elif normed_p < .9:
                value += 300
            elif normed_p < 1.0:
                value += 600
            else:
                value += 1000
        return 1000 - value/len(perf)

    def __init__(self):
        self.func = self.dollar_function

class StepRiskFunction(RiskFunction):
    def step_function(self, bar, perf):
        return float(len([p for p in perf if p < bar]))/len(perf)

    def __init__(self):
        self.func = self.step_function

class LinearRiskFunction(RiskFunction):
    def linear_cutoff_function(self, bar, perf):
        # risk = a * (perf-bar)
        a = 1
        risk = []
        for p in perf:
            base = bar - p
            if base > 0:
                risk.append(a * base)
        return np.mean(risk) if risk else 0

    def __init__(self):
        self.func = self.linear_cutoff_function

class QuadraticRiskFunction(RiskFunction):
    def quadratic_cutoff_function(self, bar, perf):
        # risk = a * (perf-bar)**2 + b * (perf-bar) + c
        risk = []
        a = 4
        b = 0
        c = 0
        for p in perf:
            base = (bar - p)/bar
            if base > 0:
                risk.append(a*base**2 + b*base + c)
        return np.mean(risk) if risk else 0

    def __init__(self):
        self.func = self.quadratic_cutoff_function

class ExponentialRiskFunction(RiskFunction):
    def exponential_cutoff_function(self, bar, perf):
        # risk = a ** (perf-bar)
        risk = []
        a = 2.718
        for p in perf:
            base = (bar - p)/bar
            if base > 0:
                risk.append(a ** base)
        return np.mean(risk) if risk else 0

    def __init__(self):
        self.func = self.exponential_cutoff_function

class RiskFunctionCollection(object):
    funcs = {'step': StepRiskFunction(),
            'linear': LinearRiskFunction(),
            'quad': QuadraticRiskFunction(),
            'exp': ExponentialRiskFunction(),
            'dollar': DollarValueFunction()}
