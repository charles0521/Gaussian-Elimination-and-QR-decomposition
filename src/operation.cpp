#include <iostream>
// #include <Matrix.h>
#include "operation.h"

#include "mkl.h"

using namespace std;

void swap_row(Matrix &m1, size_t r1, size_t r2)
{
    double tmp;
    if (r1 >= m1.rows() || r2 >= m1.rows())
    {
        throw out_of_range("the number of first matrix column differs from that of second matrix row");
    }
    for (size_t i = 0; i < m1.cols(); ++i)
    {
        tmp = m1(r1, i);
        m1(r1, i) = m1(r2, i);
        m1(r2, i) = tmp;
    }
}

void lu_Decomposition_mkl(Matrix &A, pybind11::array_t<int> P)
{
    if (A.rows() != A.cols())
    {
        throw out_of_range("Not a square Matrix!");
    }
    size_t n = A.rows();
    int* P_data = P.mutable_data();
    LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, A.data(), n, P_data);
}

void lu_Decomposition_naive(Matrix &A, pybind11::array_t<int> P)
{
    if (A.rows() != A.cols())
    {
        throw out_of_range("Not a square Matrix!");
    }

    size_t n = A.rows();
    int* P_data = P.mutable_data();
    for (size_t i = 0; i < n; ++i)
    {
        P_data[i] = i;
    }

    for (size_t k = 0; k < n; k++)
    {
        // Find the pivot element
        size_t pivot_row = k;
        double pivot = A(k, k);
        for (size_t i = k + 1; i < n; i++)
        {
            if (abs(A(i, k)) > abs(pivot))
            {
                pivot = A(i, k);
                pivot_row = i;
            }
        }

        // Swap rows to place the pivot element on the diagonal
        swap_row(A, k, pivot_row);
        swap(P_data[k], P_data[pivot_row]);

        // Compute the factor for each row
        for (size_t i = k + 1; i < n; i++)
        {
            double factor = A(i, k) / pivot;
            A(i, k) = factor;
            for (size_t j = k + 1; j < n; j++)
            {
                A(i, j) -= factor * A(k, j);
            }
        }
    }
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