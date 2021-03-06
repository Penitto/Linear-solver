﻿# Linear solver

This repo includes 4 implementations of linear solver based on conjugate gradient method using CUDA.

1. [Conjugate gradient method](https://en.wikipedia.org/wiki/Conjugate_gradient_method)

2. Preconditioned conjugate gradient method

3. [Biconjugate gradient stabilized method](https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method)

4. Preconditioned biconjugate stabilized method

Preconditioner is the matrix with diagonal's elements of matrix A.

Matrix A is presented in code in [CSR format](https://en.wikipedia.org/wiki/Sparse_matrix) and vector of diagonal elements. Vector x is vector of unknowns. Vector b is vector with constant terms.

## Project

Project consists of 2 parts: starter(.exe) and solver(.dll). Starter parses matrix from .mtx file, fills vector b one and sends all this data to solver. You can take matrixes [here](https://sparse.tamu.edu).

