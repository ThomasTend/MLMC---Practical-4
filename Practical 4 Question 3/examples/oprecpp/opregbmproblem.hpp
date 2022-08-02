#include <cmath>
#include <string>

class OpreGBMProblem {
    public:
        int nf, M, l, cost;
        std::string name;
        double T, hf;
        OpreGBMProblem(int level, int refinement, std::string call_name) {
            l = level;
            M = refinement;
            name = call_name;
            nf = pow(M, l);
            cost = nf;
            T = 1.0;
            hf = T/nf;
        }
        double * evaluate(double* sample, int N)
        {
            double r, K, sig, Xf, Af, Mf, dWf, Pf, beta;
            r   = 0.05;
            K   = 100.0;
            sig = 0.2;

            double * Pff = new double[N];

            for (int j = 0; j < N; j++) {

                Xf = K;

                Af = 0.5* hf * Xf;

                Mf = Xf;

                for (int i = 0; i < nf; i++) {
                    dWf = sample[j + i*N];
                    Xf = (1.0+ r*hf)*Xf + sig*Xf*dWf;
                    Af = Af + hf*Xf;
                    Mf = std::min(Mf, Xf);
                }
                Af = Af - 0.5*hf*Xf;

                if (name == "European") {
                    Pf = std::max(0.0, Xf - K);
                } else if (name == "Asian") {
                    Pf = std::max(0.0, Af - K);
                } else if (name == "Lookback") {
                    beta = 0.5826; // special factor for offset correction
                    Pf = Xf - Mf*(1.0- beta*sig*sqrt(hf));
                } else if (name == "Digital") {
                    Pf = K * 0.5* (copysignf(1.0, Xf - K) + 1.0);
                }
                Pff[j] = exp(-r*T)*Pf;
            }


            return Pff;
        }

};

