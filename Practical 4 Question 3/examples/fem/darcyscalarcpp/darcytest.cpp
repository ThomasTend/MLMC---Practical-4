// Testing the DarcyScalarProblem
#include "darcyscalarproblem.hpp"

int main() {
    int l;
    cout << "Enter l: " << endl;
    cin >> l;
    cout << "l = " << l << endl;
    int M = 1;
    DarcyScalarProblem problem (l, M);

    double sample = 1.5;
    double P = problem.evaluate(sample);
    cout << "P is " << P << endl;

    return 0;
}
