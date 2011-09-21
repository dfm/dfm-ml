#!/usr/bin/env python
# encoding: utf-8
"""
My Gaussian process object

History
-------
2011-09-07 - Created by Dan Foreman-Mackey

"""

__all__ = ['GaussianProcess']

import numpy as np
np.random.seed()

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
    def __init__(self,**kwargs):
        def _setattr(attr,val):
            if attr in kwargs:
                return kwargs[attr]
            else:
                return val
        self._kernel = _setattr('kernel',self.default_kernel())
        self._mean   = _setattr('mean',self.default_mean())
        self._data   = _setattr('data',None)

    def __repr__(self):
        return "GaussianProcess(kernel=%s,mean=%s)"%(repr(self._kernel),
                repr(self._mean))

    @staticmethod
    def default_kernel(a=1.0,l=1.0):
        """
        A simple "squared exponential" kernel

        Returns
        -------
        k : function
            The kernel function

        History
        -------
        2011-09-07 - Created by Dan Foreman-Mackey

        """
        ml2_2 = -0.5*l**2
        return lambda x1,x2: a*np.exp(ml2_2*(x1-x2)**2)

    @staticmethod
    def default_mean():
        """
        My default mean value (... it's just zero!)

        Returns
        -------
        m : function
            The mean function

        History
        -------
        2011-09-07 - Created by Dan Foreman-Mackey

        """
        return lambda x: np.zeros(np.shape(x))

    def full_kernel(self,x,y):
        K = self._kernel(x[:,np.newaxis],y[np.newaxis,:])
        return K

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
        cov = self.full_kernel(x,x)
        mean = self._mean(x)
        return np.random.multivariate_normal(mean,cov)

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
    import pylab as pl
    data=2*np.random.randn(100).reshape((50,2))
    p = GaussianProcess(data=data)
    for i in range(100):
        x = np.random.rand(100)*10-5
        y = p.sample(x)
        pl.plot(x,y,'.k',alpha=0.3)
    pl.plot(data[:,0],data[:,1],'or')
    pl.show()

