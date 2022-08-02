
# For testing against MATLAB: have matlab generate
# exactly the same random numbers
import pymatlab
import opre

matsession = pymatlab.session_factory()
matsession.run("rng(1337, 'twister')")
def randn(x, y):
    matsession.run("A = randn(%s, %s)" % (x, y))
    A = matsession.getvalue("A")
    return A

if __name__ == "__main__":
    calltype = opre.CallType("European", 4, 20000, 5, [0.005, 0.01, 0.02, 0.05, 0.1])
    def opre_gbm(l, N):
        return opre.opre_gbm(l, N, calltype=calltype, randn=randn)

    (sums_0_20000, cost) = opre_gbm(0, 20000)
    print("sums_0_20000: ", sums_0_20000)
    # should be: [205386.911939833, 5360199.42917607, 175479869.680854, 6718310370.33331, 205386.911939833, 5360199.42917607]

    (sums_1_30000, cost) = opre_gbm(1, 30000)
    print("sums_1_30000: ", sums_1_30000)
    # should be: [5677.20406164657, 131080.869184932, 816005.191728655, 11962841.8093692, 311002.396579897, 9152933.57824117]
