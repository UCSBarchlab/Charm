import matplotlib
matplotlib.use('Agg')

import copy
import mixture
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.mlab import normpdf as pdf

def plot_mix(mix):
    bins = 80
    plt.hist(mix, bins, normed=True)

def plot_N(mu, sigma, col=None):
    x = np.linspace(-10+mu, 10+mu, 100)
    if col is not None:
        plt.plot(x, pdf(x, mu, sigma), color=(col, col, col))
    else:
        plt.plot(x, pdf(x, mu, sigma))

def get_moments(c):
    mu, sigma = None, None
    for dist in c:
        mu = dist.mu
        sigma = dist.sigma
    return mu, sigma

def plot_fitting_progress(dummy, n_comp, mix_data, inits):
    plot_mix(mix_data.dataMatrix)
    i = n_comp
    data = mix_data
    print('# of components: {}\n'.format(i))
    rand_peaks = inits
    print(rand_peaks)
    pi = [1./i] * i
    components = [dummy(p) for p in rand_peaks]
    m = mixture.MixtureModel(i, pi, copy.deepcopy(components))
    print('Initial: {}\n'.format(m))
    _, llh = m.EM(data, 40, .1)
    print('Final: {}\n'.format(m))
    for j in range(1, 41):
        print('Iter {}\n=======\n'.format(j))
        m = mixture.MixtureModel(i, pi, copy.deepcopy(components))
        print('Before:\n{}'.format(m))
        _, llh = m.EM(data, 1, .1)
        components = m.components
        pi = m.pi
        print('After:\n{}'.format(m))
        for m in m.components:
            plot_N(*get_moments(m), col=(40-j)/60.)
    plt.show()
