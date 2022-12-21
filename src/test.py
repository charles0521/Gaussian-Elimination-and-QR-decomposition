import numpy as np
import mod
import pytest
from pytest import approx
import math
from scipy.linalg import lu_factor, lu_solve

def test_luDecomposition():
    size = 5

    m1 = mod.Matrix(size, size)
    m2 = mod.Matrix(size, size)
    m3 = np.zeros((size,size), dtype=np.float64)

    for it in range(size):
        for jt in range(size):
            m1[it, jt] = it * size + jt + 1
            m2[it, jt] = it * size + jt + 1
            m3[it][jt] = np.float64(it * size + jt + 1)

    p1 = np.zeros(size)
    p2 = np.zeros(size)

    # p1 = mod.get_array_data(p1)
    # p2 = mod.get_array_data(p2)

    mod.lu_Decomposition_naive(m1, p1)
    mod.lu_Decomposition_mkl(m2, p2)
    LU, _ = lu_factor(m3)

    ans1 = np.zeros((size, size))
    ans2 = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            ans1[i][j] = m1[i, j]
            ans2[i][j] = m2[i, j]
    
    print('naive:')
    print(ans1)
    print('mkl')
    print(ans2)
    print('numpy')
    print(LU)

    print('-----------numpy == navie-------------')
    print(np.allclose(ans1, LU))
    print('-----------mkl == navie-------------')
    print(np.allclose(ans2, LU))
    print('-----------navie == mkl-------------')
    print(np.allclose(ans1, ans2))

test_luDecomposition()
