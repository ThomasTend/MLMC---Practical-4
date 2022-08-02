#PYBIND11_DIR env variable needs to be defined
INCL := -I${PYBIND11_DIR}/include

# uncomment these lines for the GNU gcc compiler
CXXD	:= g++
FLAGSD	:= -std=c++17 -g -O0

# uncomment these lines for the Intel icpc compiler. LD_LIBRARY_PATH env variable needs to be defined
CXX	:= icpc
FLAGS	:= -std=c++14 -O3


default:
	$(CXX) $(FLAGS) $(INCL) -shared -fPIC `python3 -m pybind11 --includes` python_bindings.cpp -o opregbmproblem`python3-config --extension-suffix`

debug:
	$(CXXD) $(FLAGSD) $(INCL) -shared -fPIC `python3 -m pybind11 --includes` python_bindings.cpp -o opregbmproblem`python3-config --extension-suffix`
