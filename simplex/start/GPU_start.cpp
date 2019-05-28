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
#define THR 2
#define CHECKER false
#define DEBUGGER false
#define HARDCODE false
#define FILE_CHECK false
#define TIMER true

void Multiplicate(int size, int nnz, double* val, int* row, int* col, vector<double>& x, vector<double>& res)
{
	vector<double> b(size, 0.0);
	//#pragma omp parallel for num_threads(thr)
	for (int i = 0; i < size; i++)
	{
		for (int j = row[i]; j < row[i + 1]; j++)
			b[i] += val[j] * x[col[j]];
	}
	res = b;
}

double scPl(vector<double>& v1, vector<double>&v2)
{
	double res = 0;
	//#pragma omp parallel for num_threads(THR) reduction(+:res)
	for (int i = 0; i < v1.size(); i++)
	{
		res += v1[i] * v2[i];
	}
	return res;
}

void sumV(vector<double>& v1, vector<double>&v2, vector<double>& res)
{
	//#pragma omp parallel for num_threads(THR)
	for (int i = 0; i < v1.size(); i++)
	{
		res[i] = v1[i] + v2[i];
	}
}

void subV(vector<double>& v1, vector<double>&v2, vector<double>& res)
{
	//#pragma omp parallel for num_threads(THR)
	for (int i = 0; i < v1.size(); i++)
	{
		res[i] = v1[i] - v2[i];
	}
}

void prV(vector<double>& v1, double num, vector<double>& res)
{
	//#pragma omp parallel for num_threads(THR)
	for (int i = 0; i < v1.size(); i++)
	{
		res[i] = num * v1[i];
	}
}

void prV(vector<double>& v1, vector<double>& v2, vector<double>& res)
{
	//#pragma omp parallel for num_threads(THR)
	for (int i = 0; i < v1.size(); i++)
	{
		res[i] = v2[i] * v1[i];
	}
}

void prV(vector<double>& v1, double* v2, vector<double>& res)
{
	//#pragma omp parallel for num_threads(THR)
	for (int i = 0; i < v1.size(); i++)
	{
		res[i] = v2[i] * v1[i];
	}
}

void rDiag(double* diag, int size, vector<double>& res)
{
	for (int i = 0; i < size; i++)
	{
		res[i] = 1 / diag[i];
	}
}

double* CPU(double *val, int *col, int *row, double *right, double *diag, int non_zero, int size)
{
	int max = 50000;
	int step = 0;
	bool flag = false;
	double acc = 1e-10;
	vector<double> r0;
	r0.assign(right, right + size);
	double alpha = 1, omega0 = 1, rho0 = 1;
	vector<double> rT(r0), nu0(size, 0.0), p0(size, 0.0),
		x0(size, 0.0), pK(size, 0.0), y(size, 0.0),
		nuK(size, 0.0), h(size, 0.0), z(size, 0.0),
		t(size, 0.0), xK(size, 0.0), rK(size, 0.0),
		s(size, 0.0), temp(size, 0.0), temp1(size, 0.0);
	double beta, omegaK, rhoK;
	double minus = -1;

	do {

		if (flag)
		{
			rho0 = rhoK;
			omega0 = omegaK;
			x0 = xK;
			nu0 = nuK;
			p0 = pK;
			r0 = rK;
		}

		rhoK = scPl(rT, r0);

		beta = (rhoK / rho0)*(alpha / omega0);

		prV(nu0, omega0, temp);
		subV(p0, temp, temp);
		prV(temp, beta, temp);
		sumV(r0, temp, pK);

		rDiag(diag, size, temp);
		prV(pK, temp, y);

		Multiplicate(size, non_zero, val, row, col, y, nuK);
		prV(y, diag, temp);
		sumV(nuK, temp, nuK);

		alpha = rhoK / scPl(rT, nuK);

		prV(y, alpha, temp);
		sumV(x0, temp, h);

		prV(nuK, alpha, temp);
		subV(r0, temp, s);

		rDiag(diag, size, temp);
		prV(s, temp, z);

		Multiplicate(size, non_zero, val, row, col, z, t);
		prV(z, diag, temp);
		sumV(t, temp, t);

		rDiag(diag, size, temp);
		prV(t, temp, temp1);
		prV(s, temp, temp);
		omegaK = scPl(temp, temp1)
			/ scPl(temp1, temp1);

		prV(z, omegaK, temp);
		sumV(h, temp, xK);

		prV(t, omegaK, temp);
		subV(s, temp, rK);

		if (sqrt(scPl(rK, rK)) < acc)
			break;

		if (step % 20 == 0)
		{
			cout << "res: " << sqrt(scPl(rK, rK)) << endl;
		}
		step++;
		flag = true;

	} while (step < max);

	double* res = new double[size];
	for (int i = 0; i < xK.size(); i++)
	{
		res[i] = xK[i];
	}

	return res;
}

double* GPU_solve1(double *val, int *col, int *row, double *right, double *diag, int non_zero, int size)
{
	SolverGPU obj;

	return obj.conjugateGradientMethod(val, col, row, right, diag, non_zero, size);
}

double* GPU_solve2(double *val, int *col, int *row, double *right, double *diag, int non_zero, int size)
{
	SolverGPU obj;

	return obj.biconjugateStabGradientMethod(val, col, row, right, diag, non_zero, size);
}

double* GPU_solve3(double *val, int *col, int *row, double *right, double *diag, int non_zero, int size)
{
	SolverGPU obj;

	return obj.preConjugateGradientMethod(val, col, row, right, diag, non_zero, size);
}

double* GPU_solve4(double *val, int *col, int *row, double *right, double *diag, int non_zero, int size)
{
	SolverGPU obj;

	return obj.preBiconjugateStabGradientMethod(val, col, row, right, diag, non_zero, size);
}

int main(int argc, char *argv[])
{
	int n;
	int m;
	int nnz;
	double *val;
	int *col;
	int *row;

	loadMMSparseMatrix("lap2D_5pt_n100.mtx", 'd', true, &m, &n, &nnz, &val, &row, &col);

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

	//Testing CG
	cout << endl << "Testing CG" << endl;
	clock_t int1 = clock();
	sol = GPU_solve1(val, col, row, right, diag, nnz, n);
	clock_t int2 = clock();
#if TIMER
	cout << "TIME: " << double(int2 - int1) / 1000.0 << endl;
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
	cout << "Solution: " << sol[0] << " " << sol[1] << " " << sol[2] << " " << sol[3] << " " << sol[4] << " " << sol[5] << " " << sol[6] << endl;

	//Testing BiCGSTAB
	cout << endl << "Testing BiCGSTAB" << endl;
	clock_t int3 = clock();
	sol = GPU_solve2(val, col, row, right, diag, nnz, n);
	clock_t int4 = clock();
#if TIMER
	cout << "TIME: " << double(int4 - int3) / 1000.0 << endl;
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
	cout << "Solution: " << sol[0] << " " << sol[1] << " " << sol[2] << " " << sol[3] << " " << sol[4] << " " << sol[5] << " " << sol[6] << endl;

	//Testing CPU
	cout << endl << "Testing CPU" << endl;
	clock_t int9 = clock();
	sol = CPU(val, col, row, right, diag, nnz, n);
	clock_t int10 = clock();
	cout << "TIME: " << float(int10 - int9) / CLOCKS_PER_SEC << endl;
	cout << "Solution: " << sol[0] << " " << sol[1] << " " << sol[2] << " " << sol[3] << " " << sol[4] << " " << sol[5] << " " << sol[6] << endl;

	system("pause");

	return 0;
}