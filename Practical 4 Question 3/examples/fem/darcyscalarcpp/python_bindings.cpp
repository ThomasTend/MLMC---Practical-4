#include "darcyscalarproblem.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;


PYBIND11_MODULE(darcyscalarproblem, m) {
    py::class_<DarcyScalarProblem>(m, "DarcyScalarProblem")
        .def(py::init<int, int>())
        .def("evaluate", &DarcyScalarProblem::evaluate);
}
