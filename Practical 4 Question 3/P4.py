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

calltypes = [CallType("OU", 2, 10000, 10, [0.5, 0.1, 0.05, 0.01, 0.001])]

def P4SingleSample(l, calltype, randn=numpy.random.randn):
    M = 2 #calltype.M # refinement factor
    delta = M**(-l)
    tf = -2*(l+1)*numpy.log(2)  # we know the smallest eigenvalue is at most 1, so we choose 1/2
    t = -2*(l+1)*numpy.log(2)
    tc = -2*l*numpy.log(2)
    hc = 0
    hf = 0
    dWc = 0
    dWf = 0
    Xc = 0
    Xf = 0
    while t<0:
        told=t
        t=min(tc, tf)
        dW=sqrt(t-told)*randn(1)
        dWc = dWc + dW
        if t==-2*l*numpy.log(2):
            dWc=0
        dWf=dWf+dW
        if t==tc:
            Xc=Xc-Xc*hc+dWc
            hc=2*delta
            hc=min(hc, -tc)
            tc=tc+hc
            dWc=0
        if t==tf:
            Xf=Xf-Xf*hf+dWf
            hf=delta
            hf=min(hf, -tf)
            tf=tf+hf
            dWf=0

    return Xf, Xc;
    
def P4(l, N, calltype, randn=numpy.random.randn):  
    temp1 = 0
    temp2 = 0
    temp3 = 0
    temp4 = 0
    temp5 = 0
    temp6 = 0
    for i in range(N):
        Xf, Xc = P4SingleSample(l, calltype, randn=numpy.random.randn)
        if l==0:
            Xc = 0
        temp1 += Xf**2-Xc**2
        temp2 += (Xf**2-Xc**2)**2
        temp3 += (Xf**2-Xc**2)**3
        temp4 += (Xf**2-Xc**2)**4
        temp5 += Xf**2
        temp6 += Xf**4

    sums = numpy.array([temp1, temp2, temp3, temp4, temp5, temp6])

    cost = N*2**l # cost defined as number of fine timesteps

    return (sums, cost)

if __name__ == "__main__":
    N0 = 1000 # initial samples on coarse levels
    Lmin = 2  # minimum refinement level
    Lmax = 11  # maximum refinement level

    for (i, calltype) in enumerate(calltypes):
        def P4_l(l, N):
            return P4(l, N, calltype)

        filename = "P4%d.txt" % (i+1)
        logfile = open(filename, "w")
        print('\n ---- ' + calltype.name + ' Call ---- \n')
        mlmc_test(P4_l, calltype.N, calltype.L, N0, calltype.Eps, Lmin, Lmax, logfile)
        del logfile
        mlmc_plot(filename, nvert=3)
        plt.savefig(filename.replace('.txt', '.eps'))
        plt.show()
