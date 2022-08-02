#include "opregbmproblem.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;


PYBIND11_MODULE(opregbmproblem, m) {
    py::class_<OpreGBMProblem>(m, "OpreGBMProblem")
        .def(py::init<int, int, std::string>())
        .def_readwrite("hf", &OpreGBMProblem::hf)
        .def_readwrite("nf", &OpreGBMProblem::nf)
        .def_readwrite("cost", &OpreGBMProblem::cost)
        .def("evaluate", [](OpreGBMProblem& self, py::array_t<double> array){
                double* p = self.evaluate(array.mutable_data(), array.shape(1));
                auto result = py::array_t<double>(array.shape(1));
                py::buffer_info buf = result.request();
                double* ptr = (double*) buf.ptr;
                for (int i=0; i<array.shape(1); i++) {
                    ptr[i] = p[i];
                }
                return result;
                });
}
