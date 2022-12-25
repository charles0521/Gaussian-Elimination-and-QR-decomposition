#pragma once
#include <iostream>
#include "Matrix.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
// #include <pybind11/operators.h>

pybind11::array_t<int> lu_Decomposition_naive(Matrix &A, pybind11::array_t<int> &P);

pybind11::array_t<int> lu_Decomposition_mkl(Matrix &A, pybind11::array_t<int> &P);

void qr_Decomposition_naive(Matrix &A, Matrix &Q, Matrix &R);

void qr_Decomposition_mkl(Matrix &A, Matrix &Q, Matrix &R);

void swap_row(Matrix &m1, size_t r1, size_t r2);

Matrix multiply_tile(const Matrix &m1, const Matrix &m2, size_t block_size);

Matrix multiply_mkl(Matrix &m1, Matrix &m2);

Matrix multiply_naive(const Matrix &m1, const Matrix &m2);

pybind11::array_t<double> lu_solve(Matrix &A, double *b, pybind11::array_t<int> &p);

pybind11::array_t<double> naive_lu_solver(Matrix &A, pybind11::array_t<double> B, pybind11::array_t<int> &P);

pybind11::array_t<double> mkl_lu_solver(Matrix &A, pybind11::array_t<double> B, pybind11::array_t<int> &P);

// basic operation
double *sub_vec(double *v1, double *v2, int n);
double *add_vec(double *v1, double *v2, int n);
void normalize_vec(double *x, int n);

Matrix sub_matrix(Matrix m1, Matrix m2);
Matrix add_matrix(Matrix m1, Matrix m2);
Matrix scale_matrix(Matrix m1, double size);
Matrix Identity(int n);
