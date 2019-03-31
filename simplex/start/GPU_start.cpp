#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <cmath>
#include <ctime>
#include <locale.h>
#include <iostream>
#include <omp.h>
#include <ctype.h>
#include <algorithm>
#include <vector>
#include <assert.h>

#include "mmio_wrapper.h"
#include "GPU.h"

using namespace std;

double* GPU_solve1(double *val, int *col, int *row, double *right, double *diag, int non_zero, int size)
{
	gpu_solver obj;

	return obj.GPU_CG(val, col, row, right, diag, non_zero, size);
}

double* GPU_solve2(double *val, int *col, int *row, double *right, double *diag, int non_zero, int size)
{
	gpu_solver obj;

	return obj.GPU_BiCGSTAB(val, col, row, right, diag, non_zero, size);
}

double* GPU_solve3(double *val, int *col, int *row, double *right, double *diag, int non_zero, int size)
{
	gpu_solver obj;

	return obj.GPU_PCG(val, col, row, right, diag, non_zero, size);
}

double* GPU_solve4(double *val, int *col, int *row, double *right, double *diag, int non_zero, int size)
{
	gpu_solver obj;

	return obj.GPU_PBiCGSTAB(val, col, row, right, diag, non_zero, size);
}

int main(int argc, char *argv[])
{
	int n;
	int m;
	int nnz;
	double *val;
	int *col;
	int *row;
#if !HARDCODE
	loadMMSparseMatrix("../../Mtx/lap2D_5pt_n100.mtx", 'd', true, &m, &n, &nnz, &val, &row, &col);
#else
	m = 10;
	n = 10;
	nnz = 12;
	val = new double[nnz];
	col = new int[nnz];
	row = new int[m + 1];
	val[0] = 1;
	val[1] = 2;
	val[2] = 3;
	val[3] = 4;
	val[4] = 5;
	val[5] = 6;
	val[6] = 6;
	val[7] = 5;
	val[8] = 4;
	val[9] = 1;
	val[10] = 2;
	val[11] = 3;
	col[0] = 9;
	col[1] = 9;
	col[2] = 9;
	col[3] = 8;
	col[4] = 7;
	col[5] = 6;
	col[6] = 5;
	col[7] = 4;
	col[8] = 3;
	col[9] = 0;
	col[10] = 1;
	col[11] = 2;
	for (int i = 0; i < m; i++)
	{
		row[i] = i;
	}
	row[m] = nnz;
#endif

#if FILE_CHECK
	cout << "File data: \n";
	for (int i = 0; i < nnz; i++)
	{
		cout << val[i] << " " << col[i] << " " << row[i] << endl;
	}
#endif
	double* diag = new double[n];
	double* right = new double[n];
	double* sol = new double[n];
	for (int i = 0; i < n; i++)
	{
		right[i] = 1;
	}

	if (row[0] == 1)
	{
		for (int i = 0; i < nnz; i++)
		{
			col[i] = col[i] - 1;
		}
		for (int i = 0; i <= n; i++)
		{
			row[i] = row[i] - 1;
		}

	}
#if !HARDCODE
	for (int i = 0; i < n; i++)
	{
		int k = row[i]; //  
		int r = 0;//смещение 
		while (col[k + r] != i)
		{
			r++;
		}
		diag[i] = val[k + r];
		val[k + r] = 0;
	}
#else
	for (int i = 0; i < m; i++)
	{
		diag[i] = i + 1;
	}
#endif
#if FILE_CHECK
	cout << "After transform\n";
	for (int i = 0; i < nnz; i++)
	{
		cout << val[i] << " " << col[i] << " " << row[i] << endl;
	}

	cout << "Diag:\n";
	for (int i = 0; i < n; i++)
	{
		cout << diag[i] << " ";
	}
#endif

	//Testing CG
	cout << endl << "Testing CG" << endl;
	clock_t int1 = clock();
	sol = GPU_solve1(val, col, row, right, diag, nnz, n);
	clock_t int2 = clock();
#if TIMER
	cout << "TIME: " << double(int2 - int1) / 1000.0 << endl;
#endif

#if CHECKER
	cout << "Solution: " << endl;
	for (int i = 0; i < n; i++)
	{
		cout << sol[i] << " ";
	}
	cout << endl;
#endif
	cout << "Solution: " << sol[0] << " " << sol[1] << " " << sol[2] << " " << sol[3] << " " << sol[4] << " " << sol[5] << " " << sol[6] << endl;

	//Testing PCG
	cout << endl << "Testing PCG" << endl;
	clock_t int5 = clock();
	sol = GPU_solve3(val, col, row, right, diag, nnz, n);
	clock_t int6 = clock();
#if TIMER
	cout << "TIME: " << double(int6 - int5) / 1000.0 << endl;
#endif
#if CHECKER
	cout << "Solution: " << endl;
	for (int i = 0; i < n; i++)
	{
		cout << sol[i] << " ";
	}
	cout << endl;
#endif
	cout << "Solution: " << sol[0] << " " << sol[1] << " " << sol[2] << " " << sol[3] << " " << sol[4] << " " << sol[5] << " " << sol[6] << endl;

	//Testing BiCGSTAB
	cout << endl << "Testing BiCGSTAB" << endl;
	clock_t int3 = clock();
	sol = GPU_solve2(val, col, row, right, diag, nnz, n);
	clock_t int4 = clock();
#if TIMER
	cout << "TIME: " << double(int4 - int3) / 1000.0 << endl;
#endif
#if CHECKER
	cout << "Solution: " << endl;
	for (int i = 0; i < n; i++)
	{
		cout << sol[i] << " ";
	}
	cout << endl;
#endif
	cout << "Solution: " << sol[0] << " " << sol[1] << " " << sol[2] << " " << sol[3] << " " << sol[4] << " " << sol[5] << " " << sol[6] << endl;

	//Testing PBiCGSTAB
	cout << endl << "Testing PBiCGSTAB" << endl;
	clock_t int7 = clock();
	sol = GPU_solve4(val, col, row, right, diag, nnz, n);
	clock_t int8 = clock();
#if TIMER
	cout << "TIME: " << double(int8 - int7) / 1000.0 << endl;
#endif
#if CHECKER
	cout << "Solution: " << endl;
	for (int i = 0; i < n; i++)
	{
		cout << sol[i] << " ";
	}
	cout << endl;
#endif
	cout << "Solution: " << sol[0] << " " << sol[1] << " " << sol[2] << " " << sol[3] << " " << sol[4] << " " << sol[5] << " " << sol[6] << endl;

	system("pause");

	return 0;
}