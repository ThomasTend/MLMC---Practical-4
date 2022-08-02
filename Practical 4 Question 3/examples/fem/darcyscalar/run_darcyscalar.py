#!/usr/bin/env python

# MLMC tests for a simple 2D Darcy problem
# with a random scalar diffusion.

from pymlmc import mlmc_test, mlmc_plot, mlmc_fn
from numpy.random import randn
from numpy import sqrt
from matplotlib import pyplot as plt
from darcyscalarproblem import DarcyScalarProblem

if __name__ == "__main__":
    N0 = 10 # initial samples on coarse levels
    Lmin = 2  # minimum refinement level
    Lmax = 7 # maximum refinement level
    M = 2 # refinement factor
    N = 200 # samples for convergence tests
    L = 7 # levels for convergence tests
    Eps = [0.000002, 0.000005, 0.00001, 0.00002, 0.00005, 0.0001]#, 0.0002, 0.0005]

    sig = 1.0 # standard deviation 
    def sampler(N, l):
        sample = sqrt(sig)*randn(N)
        return (sample, sample)
    l_range = range(Lmax+1)

    problems = [DarcyScalarProblem(l, M) for l in l_range]
    def darcyscalar_l(l, N):
        return mlmc_fn(l, N, problems, sampler = sampler)

    name = "Darcy with random scalar diffusion coefficent"
    filename = "darcyscalar1.txt"
    logfile = open(filename, "w")
    print('\n ---- ' + name + ' ---- \n')
    mlmc_test(darcyscalar_l, N, L, N0, Eps, Lmin, Lmax, logfile)
    del logfile
    mlmc_plot(filename, nvert=3)
    plt.savefig(filename.replace('.txt', '.eps'))
