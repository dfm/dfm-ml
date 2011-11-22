default: all

all: gp/_gp.pyx
	python setup.py build_ext --inplace

