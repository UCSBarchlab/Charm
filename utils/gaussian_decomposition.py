import functools
import logging

import mixture
import numpy as np


# For plotting.

def gaussian_decomposition(dist, max_comp = 5, max_trials = 10, max_steps = 40):
    """ Performs Gaussian decomposition.

    Decompose the input distribution into best-fit Gaussian components.

    Args:
        dist: input distribution in 'mcerp' format.
        max_comp: maximum number of Gaussian components.
        max_trials: maximum number of trials beforing increasing # of comp. Larger is better/slower.
        max_steps: maximum steps in each fitting process. Larger is better/slower.

    Return:
        mixture: a list of tuple (mu, sigma) for the Gaussian components.
    """

    #mix = np.concatenate([np.random.normal(0, 1, [2000]), np.random.normal(6, 2, [4000]), np.random.normal(-3, 1.5, [1000])])
    mix = dist._mcpts

    data = mixture.DataSet()
    data.fromArray(mix)

    # TODO: what to set for init std? Sweep? Or some desired value for later analytical solving?
    std = 1.

    dummy = functools.partial(mixture.NormalDistribution, sigma = std)

    best_llh, best_peaks, best_mixture = None, None, None

    for i in range(1, 1 + max_comp):
        logging.debug('Gaussian Decomposition Iter: {}\n'.format(i))
        local_llh, local_peaks, local_mixture = None, None, None

        # Try max_trials times init sampling.
        for j in range(1, 1 + max_trials):
            pi = [1./i] * i
            rand_peaks = np.random.choice(mix, i)
            components = [dummy(p) for p in rand_peaks]
            m = mixture.MixtureModel(i, pi, components)
            # Fixed convergence cretiria here.
            _, llh = m.EM(data, max_steps, .1, True)
            if local_llh is None:
                local_llh = llh
                local_peaks = rand_peaks
                local_mixture = m
            else:
                if llh > local_llh:
                    local_llh = llh
                    local_peaks = rand_peaks
                    local_mixture = m

        if best_llh is None:
            best_llh = local_llh
            best_peaks = local_peaks
            best_mixture = local_mixture
        else:
            if local_llh > best_llh:
                best_llh = local_llh
                best_peaks = local_peaks
                best_mixture = local_mixture

    logging.debug('BEST MIXTURE ({}):\n{}'.format(best_llh, best_mixture))
    # Plot the progress of fitting, cool figure awaits!
    plot_fitting_progress(dummy, len(best_mixture.components), data, best_peaks)

    result = []
    for (comp, pi) in zip(best_mixture.components, best_mixture.pi):
        # comp is a ProductDistribution instance which may have (not in this case) multipul components too.
        assert(len(comp.distList) == 1)
        for d in comp:
            result.append((pi, d.mu, d.sigma))
    return result
