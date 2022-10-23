# Gaussian-Elimination and QR-decomposition
## Basic Information
In this project, I am going to solve over-constraint problems by using the Least Square Methods, employing the Gaussian elimination and QR-decomposition solvers.

Github repository: https://github.com/charles0521/Gaussian-Elimination-and-QR-decomposition

## Problem to Solve(example)
*  Define $p(x) = x^0 + x^1 + x^2 + x^3 + x^4 + x^5 + x^6 + x^7$
*  Generate a data set, there are 11 sample points. ${(x_i, y_i)}, y = p(x_i), 0 \leq i \leq 10. x_i = 2.0, 2.2, ...,4.0$

*  Fitting the data by using a polynomial of degree 7 $q(x) = a_0x^0 + a_1x^1 + a_2x^2 + a_3x^3 + a_4x^4 + a_5x^5 + a_6x^6 + a_7x^7$. At first form the following system $A\vec{c} = \vec{y} $, where
![image](https://user-images.githubusercontent.com/56105794/197404654-510ec5e3-8851-4387-8485-0b1d68d3ba56.png)

* Form a new system ${A^T}A\vec{c} = {A^T}\vec{y} $, which is theoretically solvable. Call it $B\vec{c} = \vec{d}$, which is an 8 by 8 system.
* I will implement three solutions and compare the precision and speed of three solutions.

    1.  Solve the new system by using the Gaussian elimination solver.
    2.  Solve the new system by using the QR-decomposition solver.
    3.  Solve the original system by using the QR-decomposition solver.

## Goal
* Implement Gaussian-elimination method.
* Implement QR-decomposition method.
* Comparing the precision and speed of the three solutions

## Engineering Infrastructure
* Automatic build system: make
* Version control: git
* Testing framework: pytest
* Documentation: GitHub [[README.md]](https://github.com/charles0521/Gaussian-Elimination-and-QR-decomposition/blob/main/README.md)

