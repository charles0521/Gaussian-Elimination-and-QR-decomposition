#include <iostream>
using namespace std;

#include "Matrix.h"
#include "mkl.h"

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

size_t forward_elim(Matrix &m1)
{
    size_t rows = m1.rows();
    size_t cols = m1.cols();

    for (size_t i = 0; i < cols; ++i)
    {
        size_t max_index = i;
        double max_pivot = m1(max_index, i);

        for (size_t j = i + 1; j < rows; ++j)
        {
            if (abs(m1(j, i)) > max_pivot)
            {
                max_pivot = m1(j, i);
                max_index = j;
            }
        }

        // Matrix is singular
        // if(!max_pivot)
        //     continue;
        if(!max_pivot)
            return i;
        
        if(max_index != i)
            swap_row(m1, i, max_index);
        
        for(size_t j=i+1; j<rows; ++j)
        {
            double f = m1(j, i)/m1(i, i);

            for(size_t col=i+1; col<cols; ++col)
            {
                m1(j, col) -= f * m1(i, col);
            }
            m1(j, i) = 0;
        }

    }
    return -1;
}

// Matrix backward_elim(Matrix m1)
// {

// }

Matrix gaussian_elim(Matrix m1)
{
    int singular_flag = forward_elim(m1);

    if (singular_flag != -1)
    {
        cout << "Singular Matrix." << endl;
        throw out_of_range("the number of first matrix column differs from that of second matrix row");
    }

    return m1;
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

int main()
{
    size_t lines = 3;
    Matrix m1(lines, lines), m2(4, 4);

    for(size_t i=0; i<lines; ++i)
    {
        for(size_t j=0; j<lines; ++j)
        {
            m1(i, j) = i*3 + j;
        }
    }

    cout << "original:" << endl;
    m1.output();
    cout << "forward" << endl;
    cout << forward_elim(m1) << endl;
    m1.output();

    // m1.output();
    // cout << endl;
    // m2.output();
    // cout << endl;

    // m3 = multiply_tile(m1, m2, 2);
    // m3.output();
    // cout << endl;
    // m3 = multiply_naive(m1, m2);
    // m3.output();

    return 0;
}