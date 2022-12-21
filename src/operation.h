#include<iostream>
#include "Matrix.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>


void lu_Decomposition_naive(Matrix &A, pybind11::array_t<int> P);

void lu_Decomposition_mkl(Matrix &A, pybind11::array_t<int> P);

void swap_row(Matrix &m1, size_t r1, size_t r2);

Matrix multiply_tile(const Matrix& m1, const Matrix& m2, size_t block_size);

Matrix multiply_mkl(Matrix& m1, Matrix& m2);

Matrix multiply_naive(const Matrix& m1, const Matrix& m2);
