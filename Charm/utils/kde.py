#!/usr/bin/env python
# matplotlib.use('Agg')

import math

from pandas import *
from scipy.stats.kde import gaussian_kde


class Transformations(object):

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def logit(x):
        return np.log(x) - np.log(1-x)

    @staticmethod
    def angle(v1, v2):
        def dotproduct(v1, v2):
            return sum(a*b for a, b in zip(v1, v2))

        def length(v):
            return dotproduct(v, v) ** .5

        return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

class KDE(object):

    @staticmethod
    def fit(samples, trans=lambda x: x):
        samples_trans = [trans(s) for s in samples]
        #logging.debug('KDE -- samples_trans: {}'.format(samples_trans))
        pdf = gaussian_kde(samples_trans)

        #size = 20
        #gen_fs = f_pdf.resample(size)[0]
        #gen_cs = c_pdf.resample(size)[0]

        #gen_fs_reversed = [sigmoid(f) for f in gen_fs]
        #gen_cs_reversed = [np.power(.01, c) for c in gen_cs]
        #logging.debug('gen_fs: {}'.format(gen_fs_reversed))
        #logging.debug('gen_cs: {}'.format(gen_cs_reversed))

        #fx = np.linspace(-1.5, 20, 100)
        #cx = np.linspace(-1.5, 1.5, 100)
        #PlotHelper.plot_KDE(fx, f_pdf, fs_logit_trans, gen_fs)
        #PlotHelper.plot_KDE(cx, c_pdf, cs_log_trans, gen_cs)

        return pdf
