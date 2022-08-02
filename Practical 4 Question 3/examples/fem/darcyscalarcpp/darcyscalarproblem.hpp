/* 
    Simple 2D Darcy Problem with a random scalar diffusion coefficient on a star mesh.
    It requires mfem compiled with SuiteSparse.
*/
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <math.h>

using namespace std;
using namespace mfem;

class DarcyScalarProblem {
    public:
        int M, l, argc;
        char** argv;
        DarcyScalarProblem(int level, int refinement, int argcc, char * argvv[]) {
            l = level;
            M = refinement; // currently unused
            argc = argcc;
            argv = argvv;
        }
        DarcyScalarProblem(int level, int refinement) {
            l = level;
            M = refinement;
        }
        double evaluate(double sample) {

            // 1. Options.
            const char *mesh_file = "star.mesh";
            int order = 1;
            bool static_cond = false;
            bool pa = false;
            const char *device_config = "cpu";
            bool visualization = false;

            // 2. Read the mesh from the given mesh file. 
            Mesh *mesh = new Mesh(mesh_file, 1, 1);
            int dim = mesh->Dimension();

            // 3. Refine the mesh to increase the resolution. In this example we do
            //    'ref_levels' of uniform refinement. We choose ref_levels=l
            {
               int ref_levels = l;
               for (int lev = 0; lev < ref_levels; lev++)
               {
                  mesh->UniformRefinement();
               }
            }

            // 4. Define a finite element space on the mesh. Here we use continuous
            //    Lagrange finite elements of the specified order.

            FiniteElementCollection *fec;
            fec = new H1_FECollection(order, dim);
            FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);

            // 5. Determine the list of true (i.e. conforming) essential boundary dofs.
            //    In this example, the boundary conditions are defined by marking all
            //    the boundary attributes from the mesh as essential (Dirichlet) and
            //    converting them to a list of true dofs.
            Array<int> ess_tdof_list;
            if (mesh->bdr_attributes.Size())
            {
               Array<int> ess_bdr(mesh->bdr_attributes.Max());
               ess_bdr = 1;
               fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
            }

            // 6. Set up the linear form b(.) which corresponds to the right-hand side of
            //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
            //    the basis functions in the finite element fespace.
            LinearForm *b = new LinearForm(fespace);
            ConstantCoefficient one(1.0);
            b->AddDomainIntegrator(new DomainLFIntegrator(one));
            b->Assemble();

            // 7. Define the solution vector x as a finite element grid function
            //    corresponding to fespace. Initialize x with initial guess of zero,
            //    which satisfies the boundary conditions.
            GridFunction x(fespace);
            x = 0.0;

            // 8. Set up the bilinear form a(.,.) on the finite element space
            //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
            //    domain integrator.
            Coefficient *alpha = new ConstantCoefficient(sample); // Diffusion Coefficient
            BilinearForm *a = new BilinearForm(fespace);
            if (pa) { a->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
            a->AddDomainIntegrator(new DiffusionIntegrator(*alpha));

            // 9. Assemble the bilinear form and the corresponding linear system,
            //     applying any necessary transformations such as: eliminating boundary
            //     conditions, applying conforming constraints for non-conforming AMR,
            //     static condensation, etc.
            if (static_cond) { a->EnableStaticCondensation(); }
            a->Assemble();

            OperatorPtr A;
            Vector B, X;
            a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

            // cout << "Size of linear system: " << A->Height() << endl;

            // 10. Solve the linear system A X = B.
            if (!pa)
            {
#ifndef MFEM_USE_SUITESPARSE
                // Use a simple symmetric Gauss-Seidel preconditioner with PCG.
                GSSmoother M((SparseMatrix&)(*A));
                PCG(*A, M, B, X, 1, 200, 1e-12, 0.0);
#else
                // If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
                UMFPackSolver umf_solver;
                umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
                umf_solver.SetOperator(*A);
                umf_solver.Mult(B, X);
#endif
            }
            else // Jacobi preconditioning in partial assembly mode
            {
               if (UsesTensorBasis(*fespace))
               {
                  OperatorJacobiSmoother M(*a, ess_tdof_list);
                  PCG(*A, M, B, X, 1, 400, 1e-12, 0.0);
               }
               else
               {
                  CG(*A, B, X, 1, 400, 1e-12, 0.0);
               }
            }

            // 11. Recover the solution as a finite element grid function.
            a->RecoverFEMSolution(X, *b, x);

            // 12. Save the refined mesh and the solution. This output can be viewed later
            //     using GLVis: "glvis -m refined.mesh -g sol.gf".
            ofstream mesh_ofs("refined.mesh");
            mesh_ofs.precision(8);
            mesh->Print(mesh_ofs);
            ofstream sol_ofs("sol.gf");
            sol_ofs.precision(8);
            x.Save(sol_ofs);

            // 13. Send the solution by socket to a GLVis server.
            if (visualization)
            {
               char vishost[] = "localhost";
               int  visport   = 19916;
               socketstream sol_sock(vishost, visport);
               sol_sock.precision(8);
               sol_sock << "solution\n" << *mesh << x << flush;
            }

            int order_quad = max(2, 2*order+1);
            const IntegrationRule *irs[Geometry::NumGeom];
            for (int i=0; i < Geometry::NumGeom; ++i)
            {
                irs[i] = &(IntRules.Get(i, order_quad));
            }

            // 14. Compute quantity of interest
            ConstantCoefficient zerocoeff(0.);
            double xnorm = x.ComputeL2Error(zerocoeff, irs);

            // 15. Free the used memory.
            delete a;
            delete b;
            delete fespace;
            if (order > 0) { delete fec; }
            delete mesh;

            return xnorm;
        }
};
