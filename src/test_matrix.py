import mod
import time
import pytest
import numpy as np


def test_correctness():
    size = 1000

    m1 = mod.Matrix(size, size)
    m2 = mod.Matrix(size, size)

    for it in range(size):
        for jt in range(size):
            m1[it, jt] = it * size + jt + 1
            m2[it, jt] = it * size + jt + 1

    m3_naive = mod.multiply_naive(m1, m2)
    m3_tile = mod.multiply_tile(m1, m2, 64)
    m3_mkl = mod.multiply_mkl(m1, m2)

    assert (m3_naive == m3_tile)
    assert (m3_naive == m3_mkl)


def test_correctness2():
    size = 1000

    m1 = mod.Matrix(size, size)
    m2 = mod.Matrix(size, size)

    for it in range(size):
        for jt in range(size):
            m1[it, jt] = it * size + jt + 1
            m2[it, jt] = it * size + jt + 1

    time1 = time.time()
    m3_naive = mod.multiply_naive(m1, m2)
    time2 = time.time()
    m3_tile = mod.multiply_tile(m1, m2, 16)
    time3 = time.time()
    m3_mkl = mod.multiply_mkl(m1, m2)
    time4 = time.time()

    # test time
    # with open("performance.txt", "w") as f:
    #     f.write(f"multiply_naive: {time2-time1:.4f} seconds.\n")
    #     f.write(f"multiply_tile: {time3-time2:.4f} seconds.\n")
    #     f.write(f"multiply_mkl: {time4-time3:.4f} seconds.\n")

    assert (m3_naive == m3_tile)
    assert (m3_naive == m3_mkl)


def test_luDecomposition1():
    size = 3

    m1 = mod.Matrix(size, size)
    m2 = mod.Matrix(size, size)

    for it in range(size):
        for jt in range(size):
            m1[it, jt] = it * size + jt + 1
            m2[it, jt] = it * size + jt + 1
    p1 = np.zeros(size)
    p2 = np.zeros(size)

    # p1 = mod.get_array_data(p1)
    # p2 = mod.get_array_data(p2)

    mod.lu_Decomposition_naive(m1, p1)
    mod.lu_Decomposition_mkl(m2, p2)

    lu_naive_array = np.zeros((size, size))
    lu_mkl_array = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            lu_naive_array[i][j] = m1[i, j]
            lu_mkl_array[i][j] = m2[i, j]

    assert np.allclose(lu_naive_array, lu_mkl_array)

def test_luDecomposition2():
    size = 1000

    m1 = mod.Matrix(size, size)
    m2 = mod.Matrix(size, size)

    for it in range(size):
        for jt in range(size):
            m1[it, jt] = it * size + jt + 1
            m2[it, jt] = it * size + jt + 1
    p1 = np.zeros(size)
    p2 = np.zeros(size)

    # p1 = mod.get_array_data(p1)
    # p2 = mod.get_array_data(p2)

    mod.lu_Decomposition_naive(m1, p1)
    mod.lu_Decomposition_mkl(m2, p2)

    lu_naive_array = np.zeros((size, size))
    lu_mkl_array = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            lu_naive_array[i][j] = m1[i, j]
            lu_mkl_array[i][j] = m2[i, j]

    assert np.allclose(lu_naive_array, lu_mkl_array)