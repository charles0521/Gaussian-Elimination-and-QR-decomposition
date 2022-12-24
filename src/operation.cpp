
#include <iostream>
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

pybind11::array_t<int> lu_Decomposition_mkl(Matrix &A, pybind11::array_t<int> &P)
{
    if (A.rows() != A.cols())
    {
        throw out_of_range("Not a square Matrix!");
    }
    size_t n = A.rows();
    int *P_data = P.mutable_data();
    LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, A.data(), n, P_data);

    // cout << endl << endl << "mkl P_data:" << endl;
    // cout << P_data[0] << " " << P_data[1] << " " << P_data[2] << endl << endl;
    return pybind11::array_t<int>(
        {n},           // shape
        {sizeof(int)}, // C-style contiguous strides for double
        P_data         // the data pointer
    );                 // numpy array references this parent
}

pybind11::array_t<int> lu_Decomposition_naive(Matrix &A, pybind11::array_t<int> &P)
{
    if (A.rows() != A.cols())
    {
        throw out_of_range("Not a square Matrix!");
    }

    size_t n = A.rows();
    int *P_data = P.mutable_data();
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
    return pybind11::array_t<int>(
        {n},           // shape
        {sizeof(int)}, // C-style contiguous strides for double
        P_data         // the data pointer
    );
}

Matrix multiply_tile(const Matrix &m1, const Matrix &m2, size_t block_size)
{
    if (m1.cols() != m2.rows())
    {
        throw out_of_range("the number of first matrix column differs from that of second matrix row");
    }
    Matrix m3(m1.rows(), m2.cols());

    for (size_t row = 0; row < m1.rows(); row += block_size)
    {
        for (size_t col = 0; col < m2.cols(); col += block_size)
        {
            for (size_t inner = 0; inner < m1.cols(); inner += block_size)
            {
                for (size_t i = row; i < min(m1.rows(), row + block_size); ++i)
                {
                    for (size_t j = col; j < min(m2.cols(), col + block_size); ++j)
                    {
                        for (size_t k = inner; k < min(m1.cols(), inner + block_size); ++k)
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

Matrix multiply_mkl(Matrix &m1, Matrix &m2)
{
    if (m1.cols() != m2.rows())
    {
        throw out_of_range("the number of first matrix column differs from that of second matrix row");
    }

    Matrix m3(m1.rows(), m2.cols());

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m1.rows(), m2.cols(), m1.cols(), 1, m1.data(), m1.cols(), m2.data(), m2.cols(), 0, m3.data(), m3.cols());

    return m3;
}

Matrix multiply_naive(const Matrix &m1, const Matrix &m2)
{
    if (m1.cols() != m2.rows())
    {
        throw out_of_range("the number of first matrix column differs from that of second matrix row");
    }
    Matrix m3(m1.rows(), m2.cols());

    for (size_t row = 0; row < m1.rows(); ++row)
    {
        for (size_t col = 0; col < m2.cols(); ++col)
        {
            for (size_t inner = 0; inner < m1.cols(); ++inner)
            {
                m3(row, col) += m1(row, inner) * m2(inner, col);
            }
        }
    }
    return m3;
}

void qr_Decomposition_mkl(Matrix &A, Matrix &Q, Matrix &R)
{
    int row = A.rows();
    int col = A.cols();

    int lda = row;

    // Workspace array (required by LAPACKE_dgeqrf)
    double *tau = (double *)malloc(row * sizeof(double));

    int info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, row, col,  A.data(), lda, tau);
    if (info != 0)
    {
        printf("Error: LAPACKE_dgeqrf returned error code %d\n", info);
        return;
    }

    // Copy the upper triangle of A (which now contains the R matrix) into R
    for (int i = 0; i < row; i++)
    {
        for (int j = i; j < col; j++)
        {
            R(i, j) = A(i, j);
        }
    }

    // Generate the Q matrix from the output of LAPACKE_dgeqrf using LAPACKE_dorgqr
    info = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, row, row, col, A.data(), lda, tau);
    if (info != 0)
    {
        printf("Error: LAPACKE_dorgqr returned error code %d\n", info);
        return ;
    }

    // Copy the Q matrix into Q
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            Q(i,j) = A(i,j);
        }
    }
    free(tau);
}

void qr_Decomposiotn_naive(Matrix &A, Matrix &Q, Matrix &R)
{
    
}