import numpy as np
import blas
import pytest
from pytest import approx
import math
from scipy.linalg import lu_factor, lu_solve
import pybind11

def generate_random_matrix(n, m, lower, upper):
    A = np.random.uniform(lower, upper, size=(n, m))
    return A

def test_luDecomposition():
    size = 4

    m1 = blas.Matrix(size, size)
    m2 = blas.Matrix(size, size)

    nonsingular = generate_random_matrix(size, size, -1, 1)
    m3 = nonsingular

    for it in range(size):
        for jt in range(size):
            m1[it, jt] = nonsingular[it][jt]
            m2[it, jt] = nonsingular[it][jt]

    p1 = np.zeros(size, dtype=np.int)
    p2 = np.zeros(size, dtype=np.int)


    p1 = blas.lu_Decomposition_naive(m1, p1)
    p2 = blas.lu_Decomposition_mkl(m2, p2)
    LU, p3 = lu_factor(m3)
    

    assert np.allclose(m1.array, m2.array)
    assert np.allclose(m1.array, LU)
    assert np.allclose(m2.array, LU)

    print('naive:')
    print(m1.array)
    print(p1)
    print('mkl')
    print(m2.array)
    print(p2)
    print('numpy')
    print(LU)
    print(p3)

    print()
    print()
    print('----------------')
    m1.array[0][0] = -999
    print('naive:')
    print(m1.array)
    print(p1)
    print('mkl')
    print(m2.array)
    print(p2)
    print('numpy')
    print(LU)
    print(p3)
    assert np.allclose(m1.array, m2.array)
    assert np.allclose(m1.array, LU)
    assert np.allclose(m2.array, LU)

test_luDecomposition()
