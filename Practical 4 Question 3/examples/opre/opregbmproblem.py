#!/usr/bin/env python

from numpy import ones, array, minimum, maximum, sqrt, sign, exp

class OpreGBMProblem(object):
    def __init__(self, l, calltype):
        self.calltype = calltype
        self.nf = calltype.M**l
        self.T = 1.0  # interval
        self.hf = self.T/self.nf
        self.cost = self.nf # cost defined as number of timesteps

    def evaluate(self, sample):
        T   = self.T  # interval
        r   = 0.05
        K   = 100.0
        sig = 0.2
        nf = self.nf
        hf = self.hf

        N = sample.shape[1]
        X0 = K
        Xf = X0 * ones(N)

        Af = 0.5 * hf * Xf

        Mf = array(Xf)

        for i in range(nf):
            dWf = sample[i,:]
            Xf[:] = (1.0 + r*hf)*Xf + sig*Xf*dWf
            Af[:] = Af + hf*Xf
            Mf[:] = minimum(Mf, Xf)

        Af[:] = Af - 0.5*hf*Xf

        if self.calltype.name == "European":
            Pf = maximum(0, Xf - K)
        elif self.calltype.name == "Asian":
            Pf = maximum(0, Af - K)
        elif self.calltype.name == "Lookback":
            beta = 0.5826 # special factor for offset correction
            Pf = Xf - Mf*(1 - beta*sig*sqrt(hf))
        elif self.calltype.name == "Digital":
            Pf = K * 0.5 * (sign(Xf - K) + 1)

        Pf = exp(-r*T)*Pf

        return Pf

