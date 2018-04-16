#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')

import functools
import logging
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import boxcox, boxcox_normplot, boxcox_normmax

class BoxCox(object):

    @staticmethod
    def transform(samples, shape):
        samples = np.asarray(samples)
        return boxcox(samples, shape)

    @staticmethod
    def back_transform(data, shape):
        data = np.asarray(data)
        if shape == .0:
            return np.exp(data)
        else:
            for d in data:
                if np.isnan(np.power(shape*d+1, 1/shape)):
                    assert False, 'data: {}'.format(d)
            trans_data = np.power(shape*data+1, 1/shape)
            return trans_data

    @staticmethod
    def find_lambda(samples):
        return boxcox_normmax(samples)

    @staticmethod
    def test(samples, la=-20, lb=20):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        prob = boxcox_normplot(samples, la, lb, plot=ax)
        best_lambda = boxcox_normmax(samples)
        ax.axvline(best_lambda, color='r')
        plt.show()
