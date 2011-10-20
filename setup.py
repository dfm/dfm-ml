#!/usr/bin/env python
# encoding: utf-8

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy.distutils.misc_util

setup(name='dfm-ml',
        version='0',
        description='Machine Learning by DFM',
        author='Daniel Foreman-Mackey',
        author_email='danfm@nyu.edu',
        packages=['gp'],
        cmdclass = {'build_ext': build_ext},
        ext_modules = [Extension("gp._gp", ["gp/_gp.pyx"])],
        include_dirs = numpy.distutils.misc_util.get_numpy_include_dirs(),
    )

