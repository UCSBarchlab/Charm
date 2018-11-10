import numpy as np
import logging
from lmfit import minimize, Parameters, Parameter

class Model(object):
    """Base abstract model for regression.
    """

    def fit(self, *data_set):
        """Curve fitting for params.

        Args:
            data_set: training data set to fit.

        Returns:
            A dict of params, name -> value.
        """
        result = minimize(self.func, self.params, args=data_set, method='leastsq')
        labels = data_set[-1]
        rsq = 1 - result.residual.var() / np.var(labels)
        logging.debug('R2: {}'.format(rsq))
        return result.params.valuesdict()
    
class PollackModel(Model):
    """Pollack's Rule.
    """

    def res_func(self, params, transistors, data=None):
        p = params['p'].value
        model = np.power(transistors, p)
        if data is None:
            return model
        return model - data

    def __init__(self):
        self.func = self.res_func
        self.params = Parameters()
        self.params.add('p', value=.5, min=0, max=1)

class HillMartyModel(Model):
    """Hill and Marty's multicore amdahl's law.
    """

    def res_func(self, params, cores, data=None):
        f = params['f'].value
        model = 1 / (1 - f + f / cores)
        if data is None:
            return model
        return model - data

    def __init__(self):
        self.func = self.res_func
        self.params = Parameters()
        self.params.add('f', value=.9, min=0, max=.999999)

class ExtendedHillMartyModel(Model):
    """Extended Hill and Marty's multicore amdahl's law.
    """

    def res_func(self, params, cores, data=None):
        f = params['f'].value
        c = params['c'].value
        model = 1 / (1 - f + c * cores + f / cores)
        if data is None:
            return model
        return model - data

    def __init__(self):
        self.func = self.res_func
        self.params = Parameters()
        self.params.add('f', value=.9, min=0, max=.999999)
        self.params.add('c', value=.01, min=0)
