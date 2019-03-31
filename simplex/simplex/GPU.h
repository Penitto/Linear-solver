#pragma once

#include <locale.h>
#include <iostream>
#include <omp.h>
#include <ctype.h>
#include <algorithm>
#include <vector>
#include <assert.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <cmath>
#include <ctime>

using namespace std;

#define RESIDUE false //Absolute or relative residue
#define STAB false //Check divisions
#define OMEGA true //Check omega
#define MAXITER 100000
#define MAXACC 1e-10
#define GPU_BLOCKS(sz, thrdsPrBlck) int(sz/thrdsPrBlck) + 1

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