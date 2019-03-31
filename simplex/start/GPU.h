#pragma once

using namespace std;

#define CHECKER false
#define DEBUGGER false
#define HARDCODE false
#define FILE_CHECK false
#define TIMER true

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

class gpu_solver {

public:

	DLLEXPORT double* GPU_CG(double *val, int *col, int *row, double *right, double *diag, int non_zero, int size); //Conjugate gradient method
	DLLEXPORT double* GPU_PCG(double *val, int *col, int *row, double *right, double *diag, int non_zero, int size); //Preconditioned conjugate gradient method
	DLLEXPORT double* GPU_BiCGSTAB(double *val, int *col, int *row, double *right, double *diag, int non_zero, int size); //Biconjugate gradient stabilized method
	DLLEXPORT double* GPU_PBiCGSTAB(double *val, int *col, int *row, double *right, double *diag, int non_zero, int size); //Preconditioned biconjugate gradient stabilized method
};