import numpy as np
import logging

# SoftMax on distributions?
def SoftMaximum(x, y):
    maximum = max(x, y)
    minimum = min(x, y)
    result = maximum + np.log(1.0 + np.exp(minimum - maximum))
    print 'SoftMax on {}, {}: {}'.format(x, y, result)
    return result
