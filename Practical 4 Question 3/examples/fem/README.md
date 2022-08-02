## Finite Element examples

This directory contains examples of PDEs with random coefficients. They are solved using the Finite Element method. For the MLMC tests, the quantity of interest is the squared L2 norm of the solution. 

## Finite Element software

External Finite Element software is required for these examples. 

The examples darcyscalar and darcyfield require the installation of [Firedrake](https://www.firedrakeproject.org/download.html). For the darcyfield example, Firedrake needs to be installed with SLEPc (by adding `--slepc`). For larger darcyfield tests, Firedrake needs 64-bit PETSc integers indices (by adding `--petsc-int-type int64`). 

The darcyscalarcpp example requires the installation of the C++ software [MFEM](https://github.com/mfem/mfem) and [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse). MFEM should be compiled with the options `MFEM_USE_SUITESPARSE=YES MFEM_SHARED=YES`. To create Python bindings for the C++ code, we also require [pybind11](https://pybind11.readthedocs.io/en/stable/intro.html). 
