#!/usr/bin/env python
# encoding: utf-8
"""
EM

History
-------
2011-09-28 - Created by

"""

__all__ = ['EM']

import numpy as np

_isqrt2pi = 1.0/np.sqrt(2*np.pi)
class MultiGaussian(object):
    def __init__(self, mu, var):
        self._mu  = mu
        self._var = var


def _normal(x,mu,var):
    return isqrt2pi * np.exp(-)

class EM(object):
    def __init__(self,k):
        self._k = k
        self._amplitudes = np.ones(k)/k

    def fit(self,data):
        """
        Fit the diven data using EM

        Parameters
        ----------
        data : numpy.ndarray (Npts, Ndim)
            Description

        """
        self._data  = data
        self._means = data[np.random.randint(data.shape[0],size=self._k),:]
        self._variances = np.var(data,axis=0)
        self._expectation()

    def _expectation(self):
        self._responsibilites = self._amplitudes *

    def _maximization(self):
        pass

if __name__ == '__main__':
    pass


