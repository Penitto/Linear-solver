# Linear solver

This repo includes 5 projects that solve linear system of linear algebraic equations Ax = b using CUDA with conjugate gradient method realized. 

## Theory

Matrix A is sparse, symmetrix, positive-definite matrix. Presented in code in [CSR format](https://en.wikipedia.org/wiki/Sparse_matrix) and vector of diagonal elements. Vector x vector of unknowns. Vector b is vector with constant terms. Algorithm for conjugate gradient method you can find [here](https://ru.wikipedia.org/wiki/Метод_сопряжённых_градиентов_(для_решения_СЛАУ)) (sorry, it's Russian). 

### Algorithm

1st iteration:

![alt text](https://github.com/Penitto/Linear-solver/blob/master/common/eq1_1.gif)

![alt text](https://github.com/Penitto/Linear-solver/blob/master/common/eq1_2.gif)

![alt text](https://github.com/Penitto/Linear-solver/blob/master/common/eq1_3.gif)

k-th iteration:

![alt text](https://github.com/Penitto/Linear-solver/blob/master/common/eq2_1.gif)

![alt text](https://github.com/Penitto/Linear-solver/blob/master/common/eq2_2.gif)

![alt text](https://github.com/Penitto/Linear-solver/blob/master/common/eq2_3.gif)

![alt text](https://github.com/Penitto/Linear-solver/blob/master/common/eq2_4.gif)

![alt text](https://github.com/Penitto/Linear-solver/blob/master/common/eq2_5.gif)

![alt text](https://github.com/Penitto/Linear-solver/blob/master/common/eq2_6.gif)

## Projects

All projects consist of 2 parts: starter and solver. 2 of them has starter that parse matrix, others create it by themselves. As source of matrixes you can use [this one](https://sparse.tamu.edu).

### GPU_vector

This project creates matrix by itself and use `<vector>` as container for matrix A in CSR format and vectors x and b. This project is first version of solver.

### GPU_pointer

This project is the second version. Simple C pointers are used as containers in this project. This is the difference between `GPU_vector` and `GPU_pointer`.

### GPU_prepointer

This is the third version of solver. The main difference between `GPU_pointer` and `GPU_prepointer` is in algorithm. Preconditioning matrix P are used here. It consists of elements inverse to the diagonal elements of the matrix A. This elements are situated at diagonal.

#### Algorithm

1st iteration:

![alt text](https://github.com/Penitto/Linear-solver/blob/master/common/eq3_1.gif)

![alt text](https://github.com/Penitto/Linear-solver/blob/master/common/eq3_2.gif)

![alt text](https://github.com/Penitto/Linear-solver/blob/master/common/eq3_3.gif)

k-th iteration:

![alt text](https://github.com/Penitto/Linear-solver/blob/master/common/eq4_1.gif)

![alt text](https://github.com/Penitto/Linear-solver/blob/master/common/eq4_2.gif)

![alt text](https://github.com/Penitto/Linear-solver/blob/master/common/eq4_3.gif)

![alt text](https://github.com/Penitto/Linear-solver/blob/master/common/eq4_4.gif)

![alt text](https://github.com/Penitto/Linear-solver/blob/master/common/eq4_5.gif)

![alt text](https://github.com/Penitto/Linear-solver/blob/master/common/eq4_6.gif)

![alt text](https://github.com/Penitto/Linear-solver/blob/master/common/eq4_7.gif)

### GPU_pointer_mtx and GPU_prepointer_mtx

This projects are equal two last projects but they have parser for matrixes.
