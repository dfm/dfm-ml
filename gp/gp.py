#!/usr/bin/env python
# encoding: utf-8
"""
My Gaussian process object

History
-------
2011-09-07 - Created by Dan Foreman-Mackey

"""

from __future__ import division

__all__ = ['GaussianProcess']

import numpy as np
np.random.seed(2)
import scipy.sparse as sp
import scipy.sparse.linalg
import time

class GaussianProcess(object):
    """
    A Gaussian process object

    Parameters
    ----------
    kernel : function
        The kernel to use

    History
    -------
    2011-09-07 - Created by Dan Foreman-Mackey

    """
    def __init__(self,s2=1.0,a=1.0,l2=1.0):
        self._s2 = s2
        self._a  = a
        self._l2 = l2
        self._L,self._alpha = None,None

    def __repr__(self):
        return "GaussianProcess(s2=%f,a=%f,l2=%f)"%(self._s2,self._a,self._l2)

    def k(self,x1,x2,chi2max=25.0):
        d = (x1-x2)**2/self._l2
        k = self._a*np.exp(-0.5*d)
        k[d > chi2max] = 0.0
        return k

    def K(self,x,y=None):
        b = self.k(x[:,np.newaxis],x[np.newaxis,:]) \
            + self._s2*np.identity(len(x))
        return b

    def fit(self,x,y):
        Kxx = self.K(x,x)
        nnz = sp.lil_matrix(Kxx).nnz
        print "NNZ: ",nnz,nnz/np.product(np.shape(Kxx))

        strt = time.time()
        self._L = sp.linalg.factorized(sp.lil_matrix(Kxx).tocsc())
        self._alpha = self._L(y)
        print "sparse: ", time.time()-strt

    def test_sparse(self):
        N = 8000
        x = 1000*np.random.rand(N)
        y = np.sin(x) + 0.1*np.random.randn(N)

        # do it using dense algebra
        # Kxx = self.K(x,x)
        # strt = time.time()
        # alpha = np.linalg.solve(Kxx,y)
        # print "dense: ", time.time()-strt

        # sparse algebra
        self.fit(x,y)

        print np.linalg.norm(alpha-self._alpha)
        assert np.linalg.norm(alpha-self._alpha) < 1e-10

    def test_noiseless(self):
        self._s2 = 0.01
        self.test_sparse()

    def sample_prior(self,x):
        """
        Return N samples from the prior

        Parameters
        ----------
        x : numpy.ndarray
            Positions in parameter space

        Returns
        -------
        ret : type
            Description

        History
        -------
        2011-09-07 - Created by Dan Foreman-Mackey

        """
        return np.random.multivariate_normal(np.zeros(len(x)),self.K(x).todense())

    def sample(self,x):
        """
        Sample given some data

        Note
        ----
        This assumes that K is symmetric for now...

        History
        -------
        2011-09-07 - Created by Dan Foreman-Mackey

        """
        X = self._data[:,0]
        f = self._data[:,1]
        Kxx = self.full_kernel(X,X)
        invKxx = np.linalg.inv(Kxx)
        Kxs = self.full_kernel(X,x)
        Ksx = Kxs.T
        Kss = self.full_kernel(x,x)
        mean = self._mean(x)+np.dot(Ksx,np.dot(invKxx,f))
        cov = Kss - np.dot(Ksx,np.dot(invKxx,Kxs))
        return np.random.multivariate_normal(mean,cov)

if __name__ == '__main__':
    import sys
    import pylab as pl

    x = np.random.rand(5)
    y = np.random.rand(5)

    p = GaussianProcess()
    p.test_noiseless()

    sys.exit(0)
    for i in range(100):
        x = np.random.rand(100)*10-5
        y = p.sample_prior(x)
        pl.plot(x,y,'.k',alpha=0.3)
    pl.plot(data[:,0],data[:,1],'or')
    pl.show()

