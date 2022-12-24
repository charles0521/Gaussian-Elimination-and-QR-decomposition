import blas
import time
import pytest
import numpy as np
from scipy.linalg import lu_factor, lu_solve


# generate a nonsingular matrix
def generate_random_matrix(n, m, lower, upper):
    A = np.random.uniform(lower, upper, size=(n, m))
    return A

def test_correctness():
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

def test_correctness2():
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

def test_luDecomposition1():
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

def test_luDecomposition2():
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

def test_qrDecomposition1():

    size = 2000
    nonsingular_matrix = generate_random_matrix(size, size, -10, 10)


    # matrix A Q R
    A_qr_mkl = blas.Matrix(size, size)
    Q_qr_mkl = blas.Matrix(size, size)
    R_qr_mkl = blas.Matrix(size, size)


    A_qr_numpy = nonsingular_matrix

    for it in range(size):
        for jt in range(size):
            # qr_naive[it, jt] = nonsingular_matrix[it][jt]
            A_qr_mkl[it, jt] = nonsingular_matrix[it][jt]

    # mkl
    time1 = time.time()
    blas.qr_Decomposition_mkl(A_qr_mkl, Q_qr_mkl, R_qr_mkl)
    time2 = time.time()

    # numpy
    # Q_qr_numpy, R_qr_numpy = np.linalg.qr(A_qr_numpy, mode='full')
    Q_qr_numpy, R_qr_numpy = np.linalg.qr(A_qr_numpy)
    time3 = time.time()

    with open("performance_qr.txt", "w") as f:
        f.write(f"size = {size}\n")
        f.write(f"qr_mkl: {time2-time1:.4f} seconds.\n")
        f.write(f"qr_numpy: {time3-time2:.4f} seconds.\n")

    # matrix Q
    assert np.allclose(Q_qr_mkl.array, Q_qr_numpy)

    # matrix R
    assert np.allclose(R_qr_mkl.array, R_qr_numpy)

