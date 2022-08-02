#!/usr/bin/env python

from pymlmc import mlmc_test, mlmc_plot
import matplotlib.pyplot as plt
import numpy
import numpy.random
from math import sqrt

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

def opre_gbm(l, N, calltype, randn=numpy.random.randn):
    M = calltype.M # refinement factor

    T   = 1.0  # interval
    r   = 0.05
    sig = 0.2
    K   = 100.0

    nf = M**l
    hf = T/nf

    nc = max(nf/M, 1)
    hc = T/nc

    sums = numpy.zeros(6)

    for N1 in range(1, N+1, 10000):
        N2 = min(10000, N - N1 + 1)

        X0 = K
        Xf = X0 * numpy.ones(N2)
        Xc = X0 * numpy.ones(N2)

        Af = 0.5 * hf * Xf
        Ac = 0.5 * hc * Xc

        Mf = numpy.array(Xf)
        Mc = numpy.array(Xc)

        if l == 0:
            dWf = sqrt(hf) * randn(1, N2)
            Xf[:] = Xf + r*Xf*hf + sig*Xf*dWf
            Af[:] = Af + 0.5*hf*Xf
            Mf[:] = numpy.minimum(Mf, Xf)
        else:
            for n in range(int(nc)):
                dWc = numpy.zeros((1, N2))

                for m in range(M):
                    dWf = sqrt(hf) * randn(1, N2)
                    dWc[:] = dWc + dWf
                    Xf[:] = (1.0 + r*hf)*Xf + sig*Xf*dWf
                    Af[:] = Af + hf*Xf
                    Mf[:] = numpy.minimum(Mf, Xf)

                Xc[:] = Xc + r*Xc*hc + sig*Xc*dWc
                Ac[:] = Ac + hc*Xc
                Mc[:] = numpy.minimum(Mc, Xc)

            Af[:] = Af - 0.5*hf*Xf
            Ac[:] = Ac - 0.5*hc*Xc

        if calltype.name == "European":
            Pf = numpy.maximum(0, Xf - K)
            Pc = numpy.maximum(0, Xc - K)
        elif calltype.name == "Asian":
            Pf = numpy.maximum(0, Af - K)
            Pc = numpy.maximum(0, Ac - K)
        elif calltype.name == "Lookback":
            beta = 0.5826 # special factor for offset correction
            Pf = Xf - Mf*(1 - beta*sig*sqrt(hf))
            Pc = Xc - Mc*(1 - beta*sig*sqrt(hc))
        elif calltype.name == "Digital":
            Pf = K * 0.5 * (numpy.sign(Xf - K) + 1)
            Pc = K * 0.5 * (numpy.sign(Xc - K) + 1)

        Pf = numpy.exp(-r*T)*Pf
        Pc = numpy.exp(-r*T)*Pc

        if l == 0:
            Pc = 0

        sums += numpy.array([numpy.sum(Pf - Pc),
                             numpy.sum((Pf - Pc)**2),
                             numpy.sum((Pf - Pc)**3),
                             numpy.sum((Pf - Pc)**4),
                             numpy.sum(Pf),
                             numpy.sum(Pf**2)])

        cost = N*nf # cost defined as number of fine timesteps

    return (numpy.array(sums), cost)

if __name__ == "__main__":
    N0 = 1000 # initial samples on coarse levels
    Lmin = 2  # minimum refinement level
    Lmax = 6  # maximum refinement level

    for (i, calltype) in enumerate(calltypes):
        def opre_l(l, N):
            return opre_gbm(l, N, calltype)

        filename = "opre_gbm%d.txt" % (i+1)
        logfile = open(filename, "w")
        print('\n ---- ' + calltype.name + ' Call ---- \n')
        mlmc_test(opre_l, calltype.N, calltype.L, N0, calltype.Eps, Lmin, Lmax, logfile)
        del logfile
        mlmc_plot(filename, nvert=3)
        plt.savefig(filename.replace('.txt', '.eps'))
