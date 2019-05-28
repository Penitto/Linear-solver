#pragma once

#ifdef _MSC_VER // Visual Studio specific macro
#ifdef BUILDING_DLL
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT __declspec(dllimport)
#endif
#define DLLLOCAL 
#else 
#define DLLEXPORT __attribute__ ((visibility("default")))
#define DLLLOCAL   __attribute__ ((visibility("hidden")))
#endif 

class SolverGPU {

public:

	DLLEXPORT double* conjugateGradientMethod(const double *val, const int *col, const int *row, const double *right, const double *diag, const int non_zero, const int size); //Conjugate gradient method
	DLLEXPORT double* preConjugateGradientMethod(const double *val, const int *col, const int *row, const double *right, const double *diag, const int non_zero, const int size); //Preconditioned conjugate gradient method
	DLLEXPORT double* biconjugateStabGradientMethod(const double *val, const int *col, const int *row, const double *right, const double *diag, const int non_zero, const int size); //Biconjugate gradient stabilized method
	DLLEXPORT double* preBiconjugateStabGradientMethod(const double *val, const int *col, const int *row, const double *right, const double *diag, const int non_zero, const int size); //Preconditioned biconjugate gradient stabilized method
};