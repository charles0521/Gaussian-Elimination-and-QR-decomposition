import blas
import time
import pytest
import numpy as np
from scipy.linalg import lu_factor, lu_solve


# generate a nonsingular matrix
def generate_random_matrix(n, m, lower, upper):
    A = np.random.uniform(lower, upper, size=(n, m))
    return A

def test_matrix_multiply1():
    size = 500

    m1 = blas.Matrix(size, size)
    m2 = blas.Matrix(size, size)

    for it in range(size):
        for jt in range(size):
            m1[it, jt] = it * size + jt + 1
            m2[it, jt] = it * size + jt + 1

    m3_naive = blas.multiply_naive(m1, m2)
    m3_tile = blas.multiply_tile(m1, m2, 64)
    m3_mkl = blas.multiply_mkl(m1, m2)

    assert (m3_naive == m3_tile)
    assert (m3_naive == m3_mkl)

def test_matrix_multiply2():
    size = 1000

    m1 = blas.Matrix(size, size)
    m2 = blas.Matrix(size, size)
    

    for it in range(size):
        for jt in range(size):
            m1[it, jt] = it * size + jt + 1
            m2[it, jt] = it * size + jt + 1

    time1 = time.time()
    m3_naive = blas.multiply_naive(m1, m2)
    time2 = time.time()
    m3_tile = blas.multiply_tile(m1, m2, 64)
    time3 = time.time()
    m3_mkl = blas.multiply_mkl(m1, m2)
    time4 = time.time()

    # test time
    with open("performance_matrix_multiply.txt", "w") as f:
        f.write(f"size = {size}\n")
        f.write(f"multiply_naive: {time2-time1:.4f} seconds.\n")
        f.write(f"multiply_tile: {time3-time2:.4f} seconds.\n")
        f.write(f"multiply_mkl: {time4-time3:.4f} seconds.\n")

    assert (m3_naive == m3_tile)
    assert (m3_naive == m3_mkl)

def test_lu_Decomposition1():
    size = 3

    nonsingular_matrix = generate_random_matrix(size, size, -1, 1)

    lu_naive = blas.Matrix(size, size)
    lu_mkl = blas.Matrix(size, size)
    lu_numpy = nonsingular_matrix
    

    for it in range(size):
        for jt in range(size):
            lu_naive[it, jt] = nonsingular_matrix[it][jt]
            lu_mkl[it, jt] = nonsingular_matrix[it][jt]
    # pivot
    naive_pi = np.zeros(size)
    mkl_pi = np.zeros(size)

    # calculate mkl
    blas.lu_Decomposition_naive(lu_naive, naive_pi)
    blas.lu_Decomposition_mkl(lu_mkl, mkl_pi)
    lu_numpy, numpy_pi = lu_factor(lu_numpy)


    assert np.allclose(lu_naive.array, lu_mkl.array)
    assert np.allclose(lu_naive.array, lu_numpy)
    assert np.allclose(lu_numpy, lu_mkl.array)

def test_lu_Decomposition2():
    size = 2000

    nonsingular_matrix = generate_random_matrix(size, size, -1, 1)

    lu_naive = blas.Matrix(size, size)
    lu_mkl = blas.Matrix(size, size)
    lu_numpy = nonsingular_matrix
    

    for it in range(size):
        for jt in range(size):
            lu_naive[it, jt] = nonsingular_matrix[it][jt]
            lu_mkl[it, jt] = nonsingular_matrix[it][jt]
    # pivot
    naive_pi = np.zeros(size)
    mkl_pi = np.zeros(size)

    # calculate mkl
    time1 = time.time()
    blas.lu_Decomposition_naive(lu_naive, naive_pi)
    time2 = time.time()
    blas.lu_Decomposition_mkl(lu_mkl, mkl_pi)
    time3 = time.time()
    lu_numpy, numpy_pi = lu_factor(lu_numpy)
    time4 = time.time()

    with open("performance_lu.txt", "w") as f:
        f.write(f"size = {size}\n")
        f.write(f"lu_naive: {time2-time1:.4f} seconds.\n")
        f.write(f"lu_mkl: {time3-time2:.4f} seconds.\n")
        f.write(f"lu_numpy: {time4-time3:.4f} seconds.\n")


    assert np.allclose(lu_naive.array, lu_mkl.array)
    assert np.allclose(lu_naive.array, lu_numpy)
    assert np.allclose(lu_numpy, lu_mkl.array)


def test_qr_Decomposition1():

    size = 3
    nonsingular_matrix = np.array(
        [[-1, -1, 1], [1, 3, 3], [-1, -1, 5]], dtype=np.float64)

    # matrix A Q R
    A_qr_mkl = blas.Matrix(size, size)
    Q_qr_mkl = blas.Matrix(size, size)
    R_qr_mkl = blas.Matrix(size, size)

    # matrix A Q R
    A_qr_naive = blas.Matrix(size, size)
    Q_qr_naive = blas.Matrix(size, size)
    R_qr_naive = blas.Matrix(size, size)

    # numpy
    A_qr_numpy = nonsingular_matrix

    for it in range(size):
        for jt in range(size):
            A_qr_naive[it, jt] = nonsingular_matrix[it][jt]
            A_qr_mkl[it, jt] = nonsingular_matrix[it][jt]

    # navie

    blas.qr_Decomposition_naive(A_qr_naive, Q_qr_naive, R_qr_naive)
    # mkl
    blas.qr_Decomposition_mkl(A_qr_mkl, Q_qr_mkl, R_qr_mkl)

    # numpy
    Q_qr_numpy, R_qr_numpy = np.linalg.qr(A_qr_numpy)

    A_naive = blas.multiply_mkl(Q_qr_naive, R_qr_naive)
    A_mkl = blas.multiply_mkl(Q_qr_mkl, R_qr_mkl)
    A_numpy = np.matmul(Q_qr_numpy, R_qr_numpy)


    assert np.allclose(nonsingular_matrix, A_naive.array)
    assert np.allclose(nonsingular_matrix, A_mkl.array)
    assert np.allclose(nonsingular_matrix, A_numpy)

def test_qr_Decomposition2():

    size = 1000
    nonsingular_matrix = generate_random_matrix(size, size, -10, 10)

    # matrix A Q R
    A_qr_mkl = blas.Matrix(size, size)
    Q_qr_mkl = blas.Matrix(size, size)
    R_qr_mkl = blas.Matrix(size, size)

    # matrix A Q R
    A_qr_naive = blas.Matrix(size, size)
    Q_qr_naive = blas.Matrix(size, size)
    R_qr_naive = blas.Matrix(size, size)

    # numpy
    A_qr_numpy = nonsingular_matrix

    for it in range(size):
        for jt in range(size):
            A_qr_naive[it, jt] = nonsingular_matrix[it][jt]
            A_qr_mkl[it, jt] = nonsingular_matrix[it][jt]

    # naive
    time1 = time.time()
    blas.qr_Decomposition_naive(A_qr_naive, Q_qr_naive, R_qr_naive)
    # mkl
    time2 = time.time()
    blas.qr_Decomposition_mkl(A_qr_mkl, Q_qr_mkl, R_qr_mkl)
    time3 = time.time()

    # numpy
    Q_qr_numpy, R_qr_numpy = np.linalg.qr(A_qr_numpy)
    time4 = time.time()

    with open("performance_qr.txt", "w") as f:
        f.write(f"size = {size}\n")
        f.write(f"qr_naive: {time2-time1:.4f} seconds.\n")
        f.write(f"qr_mkl: {time3-time2:.4f} seconds.\n")
        f.write(f"qr_numpy: {time4-time3:.4f} seconds.\n")
    
    # A = Q*R
    A_naive = blas.multiply_mkl(Q_qr_naive, R_qr_naive)
    A_mkl = blas.multiply_mkl(Q_qr_mkl, R_qr_mkl)
    A_numpy = np.matmul(Q_qr_numpy, R_qr_numpy)

    # test
    assert np.allclose(nonsingular_matrix, A_naive.array)
    assert np.allclose(nonsingular_matrix, A_mkl.array)
    assert np.allclose(nonsingular_matrix, A_numpy)


# Ax = b
def test_lu_solver():
    size = 1000

    # generate Ax = b
    A = np.random.rand(size, size)
    b = np.random.rand(size, 1)
    x = np.linalg.solve(A, b)


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

    time1 = time.time()
    my_x = blas.naive_lu_solver(lu_naive, b, naive_pi)
    time2 = time.time()
    mkl_x = blas.mkl_lu_solver(lu_mkl, b, mkl_pi)
    time3 = time.time()
    # x = np.linalg.solve(A, b)
    # time4 = time.time()

    with open("solve_equation.txt", "w") as f:
        f.write(f"size = {size}\n")
        f.write(f"naive_solve_equation: {time2-time1:.4f} seconds.\n")
        f.write(f"mkl_solve_equation: {time3-time2:.4f} seconds.\n")
        # f.write(f"numpy_solve_equation: {time4-time3:.4f} seconds.\n")


    assert np.allclose(my_x, np.squeeze(x))
    assert np.allclose(mkl_x, np.squeeze(x))