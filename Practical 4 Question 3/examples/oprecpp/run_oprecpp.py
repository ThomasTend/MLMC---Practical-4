#!/usr/bin/env python

from pymlmc import mlmc_test, mlmc_plot, mlmc_fn
import matplotlib.pyplot as plt
from numpy import array
from numpy.random import randn
from math import sqrt
from opregbmproblem import OpreGBMProblem

class CallType(object):
    def __init__(self, name, M, N, L, Eps):
        self.name = name
        self.M = M # refinement cost factor
        self.N = N # samples for convergence tests
        self.L = L # levels for convergence tests
        self.Eps = Eps

calltypes = [CallType("European", 4, 2000000, 5, [0.005, 0.01, 0.02, 0.05, 0.1]),
             CallType("Asian",    4, 2000000, 5, [0.005, 0.01, 0.02, 0.05, 0.1]),
             CallType("Lookback", 4, 2000000, 5, [0.01, 0.02, 0.05, 0.1, 0.2]),
             CallType("Digital",  4, 3000000, 5, [0.02, 0.05, 0.1, 0.2, 0.5])]

if __name__ == "__main__":
    N0 = 1000 # initial samples on coarse levels
    Lmin = 2  # minimum refinement level
    Lmax = 6  # maximum refinement level


    for (i, calltype) in enumerate(calltypes):
        
        problems = [OpreGBMProblem(l, calltype.M, calltype.name) for l in range(Lmax+1)]

        def sampler(N,l):
            M = calltype.M
            nf = problems[l].nf
            hf = problems[l].hf
            samplef = sqrt(hf)*randn(nf, N)
            samplec = array([sum(samplef[i:i+M,:]) for i in range(0, nf, M)]) 
            return samplef, samplec

        def opre_l(l, N):
            return mlmc_fn(l, N, problems, sampler=sampler, N1 = 10000)

        filename = "opre_gbm%d.txt" % (i+1)
        logfile = open(filename, "w")
        print('\n ---- ' + calltype.name + ' Call ---- \n')
        mlmc_test(opre_l, calltype.N, calltype.L, N0, calltype.Eps, Lmin, Lmax, logfile)
        del logfile
        mlmc_plot(filename, nvert=3)
        plt.savefig(filename.replace('.txt', '.eps'))
