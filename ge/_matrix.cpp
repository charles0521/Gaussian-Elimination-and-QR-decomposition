#include <iostream>
using namespace std;

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>

#include "Matrix.h"
#include "mkl.h"



Matrix multiply_naive(const Matrix& m1, const Matrix& m2)
{
    if(m1.cols() != m2.rows())
    {
        throw out_of_range("the number of first matrix column differs from that of second matrix row");
    }
    Matrix m3(m1.rows(), m2.cols());

    for(size_t row=0; row<m1.rows(); ++row)
    {
        for(size_t col=0; col< m2.cols(); ++col)
        {
            for(size_t inner=0; inner<m1.cols(); ++inner)
            {
                m3(row, col) += m1(row, inner) * m2(inner, col);
            }
        }
    }
    return m3;
}

Matrix multiply_tile(const Matrix& m1, const Matrix& m2, size_t block_size)
{
    if(m1.cols() != m2.rows())
    {
        throw out_of_range("the number of first matrix column differs from that of second matrix row");
    }
    Matrix m3(m1.rows(), m2.cols());

    for(size_t row=0; row<m1.rows(); row+=block_size)
    {
        for(size_t col=0; col<m2.cols(); col+=block_size)
        {
            for(size_t inner=0; inner<m1.cols(); inner+=block_size)
            {
                for(size_t i=row; i<min(m1.rows(), row+block_size); ++i)
                {
                    for(size_t j=col; j<min(m2.cols(), col+block_size); ++j)
                    {
                        for(size_t k=inner; k<min(m1.cols(), inner+block_size); ++k)
                        {
                            m3(i, j) += m1(i, k) * m2(k, j);
                        }
                    }
                }
            }
        }
    }

    return m3;
}

Matrix multiply_mkl(Matrix& m1, Matrix& m2)
{
    if(m1.cols() != m2.rows())
    {
        throw out_of_range("the number of first matrix column differs from that of second matrix row");
    }

    Matrix m3(m1.rows(), m2.cols());

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m1.rows(), m2.cols(), m1.cols(), 1, m1.data(), m1.cols(), m2.data(), m2.cols(), 0, m3.data(), m3.cols());

    return m3;
}



PYBIND11_MODULE(_matrix, m)
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
}










// int main()
// {
//     Matrix<int> m1(4,4), m2(4,4);

//     m1(0, 0) = 1; m1(0, 1) = 2; m1(0, 2) = 2;
//     m1(1, 0) = -1; m1(1, 1) = 2; m1(1, 2) = -1;

//     m2(0, 0) = 1; m2(0, 1) = 2;
//     m2(1, 0) = 0; m2(1, 1) = -3;
//     m2(2, 0) = 1; m2(2, 1) = 2;

//     Matrix<int> m3(4,4);
//     m3.output();
//     // m1.output();
//     // cout << endl;
//     // m2.output();
//     // cout << endl;

//     m3 = multiply_tile(m1, m2, 2);
//     m3.output();
//     cout << endl;
//     m3 = multiply_naive(m1, m2);
//     m3.output();

    
    

//     return 0;
        
// }