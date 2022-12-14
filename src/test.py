import numpy as np
import blas
import pytest
from pytest import approx
import math
from scipy.linalg import lu_factor, lu_solve
import pybind11
import time

def generate_random_matrix(n, m, lower, upper):
    A = np.random.uniform(lower, upper, size=(n, m))
    return A

# def test_luDecomposition():
#     size = 4

#     m1 = blas.Matrix(size, size)
#     m2 = blas.Matrix(size, size)

#     nonsingular = generate_random_matrix(size, size, -1, 1)
#     m3 = nonsingular

#     for it in range(size):
#         for jt in range(size):
#             m1[it, jt] = nonsingular[it][jt]
#             m2[it, jt] = nonsingular[it][jt]

#     p1 = np.zeros(size, dtype=np.int)
#     p2 = np.zeros(size, dtype=np.int)


#     p1 = blas.lu_Decomposition_naive(m1, p1)
#     p2 = blas.lu_Decomposition_mkl(m2, p2)
#     LU, p3 = lu_factor(m3)


#     assert np.allclose(m1.array, m2.array)
#     assert np.allclose(m1.array, LU)
#     assert np.allclose(m2.array, LU)

#     print('naive:')
#     print(m1.array)
#     print(p1)
#     print('mkl')
#     print(m2.array)
#     print(p2)
#     print('numpy')
#     print(LU)
#     print(p3)

#     print()
#     print()
#     print('----------------')
#     m1.array[0][0] = -999
#     print('naive:')
#     print(m1.array)
#     print(p1)
#     print('mkl')
#     print(m2.array)
#     print(p2)
#     print('numpy')
#     print(LU)
#     print(p3)
#     assert np.allclose(m1.array, m2.array)
#     assert np.allclose(m1.array, LU)
#     assert np.allclose(m2.array, LU)

# test_luDecomposition()


# def test_qrDecomposition1():

#     size = 3
#     nonsingular_matrix = np.array(
#         [[-1, -1, 1], [1, 3, 3], [-1, -1, 5]], dtype=np.float64)

#     # matrix A Q R
#     A_qr_mkl = blas.Matrix(size, size)
#     Q_qr_mkl = blas.Matrix(size, size)
#     R_qr_mkl = blas.Matrix(size, size)

#     # matrix A Q R
#     A_qr_naive = blas.Matrix(size, size)
#     Q_qr_naive = blas.Matrix(size, size)
#     R_qr_naive = blas.Matrix(size, size)

#     # numpy
#     A_qr_numpy = nonsingular_matrix

#     for it in range(size):
#         for jt in range(size):
#             A_qr_naive[it, jt] = nonsingular_matrix[it][jt]
#             A_qr_mkl[it, jt] = nonsingular_matrix[it][jt]

#     # navie

#     blas.qr_Decomposition_naive(A_qr_naive, Q_qr_naive, R_qr_naive)
#     # mkl
#     blas.qr_Decomposition_mkl(A_qr_mkl, Q_qr_mkl, R_qr_mkl)

#     # numpy
#     Q_qr_numpy, R_qr_numpy = np.linalg.qr(A_qr_numpy)

#     naive = blas.Matrix(3, 3)
#     mkl = blas.Matrix(3, 3)

#     A_naive = blas.multiply_mkl(Q_qr_naive, R_qr_naive)
#     A_mkl = blas.multiply_mkl(Q_qr_mkl, R_qr_mkl)
#     A_numpy = np.matmul(Q_qr_numpy, R_qr_numpy)


#     assert np.allclose(nonsingular_matrix, A_naive.array)
#     assert np.allclose(nonsingular_matrix, A_mkl.array)
#     assert np.allclose(nonsingular_matrix, A_numpy)

# test_qrDecomposition1()


def test_lu_solver():
    size = 3

    # generate Ax = b
    A = np.random.rand(size, size)
    b = np.random.rand(size, 1)
    x = np.linalg.solve(A, b)
    AA = A
    bb = b

    lu_naive = blas.Matrix(size, size)
    lu_mkl = blas.Matrix(size, size)
    lu_numpy = A
    

    for it in range(size):
        for jt in range(size):
            lu_naive[it, jt] = A[it][jt]
            lu_mkl[it, jt] = A[it][jt]
    
    # pivot
    naive_pi = np.zeros(size)
    mkl_pi = np.zeros(size)
    # print(x)
    time1 = time.time()
    my_x = blas.naive_lu_solver(lu_naive, b, naive_pi)
    
    time2 = time.time()
    mkl_x = blas.mkl_lu_solver(lu_mkl, b, mkl_pi)
    time3 = time.time()
    x = np.linalg.solve(AA, bb)
    print(my_x)
    print(mkl_x)
    print(np.squeeze(x))
    time4 = time.time()
    # print('@@')
    # print(x)
    # with open("solve_equation.txt", "w") as f:
    #     f.write(f"size = {size}\n")
    #     f.write(f"naive_solve_equation: {time2-time1:.4f} seconds.\n")
    #     f.write(f"mkl_solve_equation: {time3-time2:.4f} seconds.\n")
    #     f.write(f"numpy_solve_equation: {time4-time3:.4f} seconds.\n")


    assert np.allclose(my_x, np.squeeze(x))
    assert np.allclose(mkl_x, np.squeeze(x))
test_lu_solver()