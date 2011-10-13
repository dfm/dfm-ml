#!/usr/bin/env python
# encoding: utf-8

import numpy as np
np.random.seed(2)
import numpy.linalg

from gp import GaussianProcess

class TestGP:
    def setUp(self):
        self.gp = GaussianProcess()

    def test_sparse(self):
        N = 1000
        x = 1000*np.random.rand(N)
        y = np.sin(x) + 0.1*np.random.randn(N)

        # do it using dense algebra
        Kxx = self.gp.K(x,x).todense()
        alpha = np.linalg.solve(Kxx,y)

        # sparse algebra
        self.gp.fit(x,y)
        assert np.linalg.norm(alpha-self.gp._alpha) < 1e-10

if __name__ == '__main__':
    t = TestGP()
    t.setUp()
    t.test_sparse()

