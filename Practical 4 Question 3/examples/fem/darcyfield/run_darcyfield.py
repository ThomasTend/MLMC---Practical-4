#!/usr/bin/env python

# The finite element solution is done using Firedrake (and PETSc)
# See https://www.firedrakeproject.org
# Random sampling is done via Karhunen-Loeve (K-L) expansion
# For K-L, a generalized eigenvalue problem is solved. By default we use SLEPc.
# SLEPc can be installed using: python3 firedrake-install --slepc
# For large problems, 64-bit PETSc integers are needed for the eigenvalue problem.
# This is done by adding --petsc-int-type int64


from pymlmc import mlmc_test, mlmc_plot, mlmc_fn
import matplotlib.pyplot as plt
from numpy import vstack, ceil, zeros 
from numpy import exp as npexp
from numpy import real as npreal
from numpy import sqrt as npsqrt
from numpy.random import randn
from firedrake import *
from firedrake.petsc import PETSc
import petsc4py
import scipy
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import cKDTree

try:
    from slepc4py import SLEPc
except ImportError:
    import sys
    warning("Unable to import SLEPc, eigenvalue computation not possible (try firedrake-update --slepc)")
    sys.exit(0)
from time import time

def make_nested_mapping(outer_space, inner_space):
    # Maps the dofs between nested meshes
    outer_dof_coor = Function(VectorFunctionSpace(outer_space.mesh(), "CG", 1)).interpolate(SpatialCoordinate(outer_space.mesh())).vector()[:]
    inner_dof_coor = Function(VectorFunctionSpace(inner_space.mesh(), "CG", 1)).interpolate(SpatialCoordinate(inner_space.mesh())).vector()[:]

    tree = cKDTree(outer_dof_coor)
    _,mapping = tree.query(inner_dof_coor, k=1)
    return mapping

def covariance_func(X):
    # Gaussian covariance function
    # input X: array of size N_dofs X 2
    # output c: array of size N_dofs X N_dofs
    s = 1. # scaling parameter
    pairwise_sq_dists = squareform(pdist(X, 'sqeuclidean'))
    c = npexp(-pairwise_sq_dists / s**2)
    c = (c > 1e-16) * c
    return c

def karhunenloeve(problems, mappings, Nt):
    # Truncated Karhunen-Loeve expansion
    # We are solving the generalized eigenvalue problem M*C*M*phi= lmbda*M*phi
    # where M is the FEM mass matrix and C is the covariance matrix,
    # lmbda is an eigenvalue and phi the associated eigenmode.
    # Random field is approximated by sum_{i=1}^Nt sqrt(lmbda_i) phi_i 

    slepc = True

    V = problems[-1].V
    u = TrialFunction(V)
    v = TestFunction(V)
    a = u*v*dx
    petsc_m = assemble(a).M.handle

    n = V.dim()
    mesh = V.mesh()
    x = SpatialCoordinate(mesh)
    Vv = VectorFunctionSpace(mesh, "CG",1)
    coords = interpolate(x, Vv).dat.data

    C = covariance_func(coords)

    phi = [Function(V) for i in range(Nt)]

    # Using scipy for the Generalized Eigenvalue problem
    if not slepc:
        mi, mj, mv = petsc_m.getValuesCSR()
        M = scipy.sparse.csr_matrix( (mv, mj, mi), shape = C.shape).todense()
        A = M.dot(C.dot(M))
        lmbda,v = scipy.linalg.eigh(A, M, eigvals = (V.dim()-Nt,V.dim()-1))
        for i in range(Nt):
            phi[i].vector()[:] = v[:,i]
    # Using SLEPc for the Generalized Eigenvalue problem
    if slepc:
        CC = scipy.sparse.csr_matrix(C)
        I = CC.indptr
        J = CC.indices
        data = CC.data
        del CC
        petsc_c = PETSc.Mat();
        # Change to 'int64' if using int64 indices
        petsc_c.createAIJWithArrays([n,n], (I.astype('int32'), J.astype('int32'), data), comm = PETSc.COMM_WORLD);
        del I, J, data
        petsc_mc = petsc_m.matMult(petsc_c);
        del petsc_c
        petsc_a = petsc_mc.matMult(petsc_m);
        opts = PETSc.Options()
        opts.setValue("eps_gen_hermitian", None)
        opts.setValue("st_pc_factor_shift_type", "NONZERO")
        opts.setValue("eps_type", "krylovschur")
        opts.setValue("eps_largest_real", None)
        opts.setValue("eps_tol", 1e-10)

        es = SLEPc.EPS().create(comm=COMM_WORLD)
        es.setDimensions(Nt)
        es.setOperators(petsc_a, petsc_m)
        es.setFromOptions()
        es.solve()
        nconv = es.getConverged()
        lmbda = []
        vr, vi = petsc_a.getVecs()
        del petsc_a, petsc_m
        for i in range(Nt):
            lmbda.append(es.getEigenpair(i, vr, vi))
            phi[i].vector()[:] = vr

    lmbda = npreal(lmbda)
    print("Eigenvalue ratio: ", lmbda[0]/lmbda[-1])
    sqrtlmbda = npsqrt(lmbda)
    
    phis = [[Function(problem.V) for i in range(Nt)] for problem in problems[0:-1]]
    for l in range(len(phis)):
        for i in range(Nt):
            phis[l][i].vector()[:] = phi[i].vector()[mappings[l]]
    phis.append(phi)
    H = [vstack([phis[l][i].vector()[:] for i in range(Nt)]) for l in range(len(problems))]

    def generate_field(l, gaussians):
        Kw = Function(problems[l].V)
        Kw.vector()[:] = npexp((sqrtlmbda*gaussians).dot(H[l]))
        return Kw

    return generate_field

class DarcyProblem(object):
    # We are solve the Darcy problem -grad(K*grad(u)) = f using the Finite Element Method.
    # The weak form is: Find u in V s.t. (grad(K*u), grad(v)) = (f,v) for all v in V.

    # l: current level
    # M: refinement factor
    def __init__(self, l, M):
        self.init_problem(l, M)

    def init_problem(self, l, M):
        N = int(ceil(4*M**l))
        mesh = UnitSquareMesh(N, N)

        V = FunctionSpace(mesh, "CG", 1)
        self.V = V
        u = Function(V)
        v = TestFunction(V)
        x,y = SpatialCoordinate(mesh)
        f = Function(V).interpolate(sin(2.0*pi*x))

        K0 = Constant(1.) # Deterministic field
        self.Kw = Function(V) # Random field
        K = K0 + self.Kw # Diffusion coefficient

        bcs = [DirichletBC(V, Constant(1.0), [1]), DirichletBC(V, Constant(0.0), [2])]

        a = inner(K*grad(u), grad(v))*dx
        L = f*v*dx
        F = a-L

        self.u = u
        problem = NonlinearVariationalProblem(F, self.u, bcs)
        self.solver = NonlinearVariationalSolver(problem, solver_parameters = {"ksp_type": "preonly", "pc_type": "lu"})

    def _evaluate(self, sample):
        self.Kw.assign(sample)
        self.u.assign(Function(self.V))
        self.solver.solve()
        return norm(self.u)**2

    def evaluate(self, sample):
        if isinstance(sample, list):
            n = len(sample)
            P = zeros(n)
            for i in range(n):
                P[i] = self._evaluate(sample[i])
            return P
        else:
            return self._evaluate(sample)
        
if __name__ == "__main__":
    N0 = 10 # initial samples on coarse levels
    Lmin = 2  # minimum refinement level
    Lmax = 5 # maximum refinement level
    M = 2 # refinement factor
    N = 200 # samples for convergence tests
    L = 5 # levels for convergence tests
    Nt = 12 # Karhunen-Loeve truncation
    Eps = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005]
    filename = "darcysfield.txt"
    logfile = open(filename, "w")

    name = "Darcy with random field diffusion coefficent"
    print('\n ---- ' + name + ' ---- \n')
    l_range = range(Lmax+1)
    print('\n***** Generating problems *****\n')
    problems = [DarcyProblem(l, M) for l in l_range]
    print('\n***** Generating K-L expansion *****\n')
    twolevel_mappings = [None] + [make_nested_mapping(problems[l+1].V, problems[l].V) for l in l_range[0:-1]]
    top_mappings = [make_nested_mapping(problems[-1].V, problems[l].V) for l in l_range[0:-1]]
    generate_field = karhunenloeve(problems, top_mappings, Nt)
    sig = 0.5 # standard deviation
    def sampler(N, l):
        samplesf = []
        samplesc = []
        for i in range(N):
            gaussians = npsqrt(sig)*randn(Nt)
            samplef = generate_field(l, gaussians)
            samplesf.append(samplef)
            if l > 0:
                samplec = Function(problems[l-1].V)
                samplec.vector()[:] = samplef.vector()[twolevel_mappings[l]] # map field to coarse level
                samplesc.append(samplec)
        return samplesf, samplesc
    def darcyfield_l(l, N):
        return mlmc_fn(l, N, problems, sampler = sampler)

    mlmc_test(darcyfield_l, N, L, N0, Eps, Lmin, Lmax, logfile)
    del logfile
    mlmc_plot(filename, nvert=3)
    plt.savefig(filename.replace('.txt', '.eps'))
