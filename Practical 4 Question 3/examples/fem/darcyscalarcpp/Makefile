#PYBIND11_DIR env variable needs to be defined
INCL := -I${PYBIND11_DIR}/include

#MFEM_DIR env variable needs to be defined. mfem needs to be built with umfpack and shared. For example
# make serial -j 4 MFEM_USE_SUITESPARSE=YES MFEM_SHARED=YES

# uncomment these lines for the GNU gcc compiler
CXXD	:= g++
FLAGSD	:= -std=c++17 -g -O0

# uncomment these lines for the Intel icpc compiler. LD_LIBRARY_PATH env variable needs to be defined
CXX	:= icpc
FLAGS	:= -std=c++14 -O3



default:
	$(CXX) $(FLAGS) $(INCL) -I${MFEM_DIR} -I${MFEM_DIR}/../SuiteSparse/include -shared -fPIC `python3 -m pybind11 --includes` python_bindings.cpp -o darcyscalarproblem`python3-config --extension-suffix` -L${MFEM_DIR} -lmfem -Wl,-rpath,${MFEM_DIR}/../SuiteSparse/lib -L${MFEM_DIR}/../SuiteSparse/lib -lklu -lbtf -lumfpack -lcholmod -lcolamd -lamd -lcamd -lccolamd -lsuitesparseconfig -lrt -L${MFEM_DIR}/../metis-4.0 -lmetis -llapack -lblas -lrt

debug:
	$(CXXD) $(FLAGSD) $(INCL) -I${MFEM_DIR} -I${MFEM_DIR}/../SuiteSparse/include -shared -fPIC `python3 -m pybind11 --includes` python_bindings.cpp -o darcyscalarproblem`python3-config --extension-suffix` -L${MFEM_DIR} -lmfem -Wl,-rpath,${MFEM_DIR}/../SuiteSparse/lib -L${MFEM_DIR}/../SuiteSparse/lib -lklu -lbtf -lumfpack -lcholmod -lcolamd -lamd -lcamd -lccolamd -lsuitesparseconfig -lrt -L${MFEM_DIR}/../metis-4.0 -lmetis -llapack -lblas -lrt

test:
	$(CXX) $(FLAGS) -I${MFEM_DIR} -I${MFEM_DIR}/../SuiteSparse/include darcytest.cpp -o darcytest -L${MFEM_DIR} -lmfem -Wl,-rpath,${MFEM_DIR}/../SuiteSparse/lib -L${MFEM_DIR}/../SuiteSparse/lib -lklu -lbtf -lumfpack -lcholmod -lcolamd -lamd -lcamd -lccolamd -lsuitesparseconfig -lrt -L${MFEM_DIR}/../metis-4.0 -lmetis -llapack -lblas -lrt
