#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>

#include "operation.h"
#include <vector>
using namespace std;

// pybind11::array_t<double> get_ndarray_from_vector(pybind11::class_<Matrix> &data)
// {
//     // int data_size = data.nrow() * data.ncol()
//     pybind11::array_t<double> result = pybind11::array_t<double>({data.matrix.size()});
//     double *result_data = result.mutable_data();
//     copy(data.matrix.begin(), data.matrix.end(), result_data);
//     return result;
// }

PYBIND11_MODULE(mod, m)
{
    pybind11::class_<Matrix>(m, "Matrix")
        .def( pybind11::init<size_t, size_t>())
        .def("__getitem__", [](Matrix& m, pair<size_t, size_t> index) {
            return m(index.first, index.second);
        })
        .def("__setitem__", [](Matrix& m, pair<size_t, size_t> index, double value) {
            m(index.first, index.second) = value;
        })
        .def(pybind11::self == pybind11::self)
        .def_property_readonly("nrow", &Matrix::rows)
        .def_property_readonly("ncol", &Matrix::cols);
    
    m.def("multiply_naive", multiply_naive, "Matrix multiply with naive method.");
    m.def("multiply_tile", multiply_tile, "Matrix multiply with tile method.");
    m.def("multiply_mkl", multiply_mkl, "Matrix multiply with mkl method.");
    m.def("lu_Decomposition_naive", lu_Decomposition_naive, "naive LU decomposition");
    m.def("lu_Decomposition_mkl", lu_Decomposition_mkl, "LAPACK LU decomposition");
    // m.def("get_ndarray_from_vector", get_ndarray_from_vector, "Convert a C++ vector to a NumPy array");
}

