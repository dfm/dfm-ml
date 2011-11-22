#!/usr/bin/env python
# encoding: utf-8
"""
K-means

"""

__all__ = ['KMeans', 'KMeansConvergenceError']

import numpy as np

class KMeansConvergenceError(Exception):
    pass

class KMeans(object):
    def __init__(self,k):
        self._k = k

    @property
    def k(self):
        return self._k

    @k.setter
    def set_k(self, k):
        self._k = k
        self._means = None
        self._rs = None

    def fit(self, data, maxiter=200, tol=1e-8, verbose=True):
        """
        Fit the diven data using K-means

        Parameters
        ----------
        data : numpy.ndarray (Npts, Ndim)
            Description

        """
        self._data = data
        self._means = data[np.random.randint(data.shape[0],size=self._k),:]
        L = None
        for i in xrange(maxiter):
            dists = self._update_rs()
            self._update_means()
            newL = np.sum(self._rs * dists)
            if L is None:
                L = newL
            else:
                dL = np.abs(newL-L)
                if dL < tol:
                    break
                L = newL
        if i < maxiter-1:
            if verbose:
                print "K-Means converged after %d iterations"%(i)
        else:
            raise KMeansConvergenceError("K-means didn't converge")

    def _update_rs(self):
        dists = np.sum((self._data[None] - self._means[:,None])**2, axis=-1)
        self._rs = dists == np.min(dists,axis=0)
        return dists

    def _update_means(self):
        self._means = np.sum( self._rs[:,:,None] * self._data[None], axis=1 )/np.sum(self._rs, axis=1)[:,None]

if __name__ == '__main__':
    import pylab as pl
    np.random.seed(5)
    means     = [5.1,0.0,10]
    variances = [2.0,1.0,1.5]
    amplitudes= [5,1,2]
    factor    = 100
    data = np.zeros((1,2))
    for i in range(len(means)):
        data = np.concatenate((data,
            (means[i]+variances[i]*(np.random.randn(factor*amplitudes[i]))).reshape([-1,2])))
    data = data[1:,:]
    pl.plot(data[:,0],data[:,1],'+k')

    kmeans = KMeans(len(means))
    kmeans.fit(data)

    for i in range(kmeans.k):
        pl.plot(data[kmeans._rs[i,:],0],data[kmeans._rs[i,:],1],'+')

    for m in kmeans._means:
        pl.plot(m[0],m[1],'or')

    pl.show()


