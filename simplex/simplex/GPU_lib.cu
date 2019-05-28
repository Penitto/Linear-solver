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

#include <cuda.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <cublas_api.h>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>

#include <device_launch_parameters.h>

#include "GPU.h"

const bool residue = false; //true for absolute, false for relative
const bool stab = false;
const bool omega = false; //extra check with omega
const double maxiter = 50000;
const double maxacc = 1e-10;
const bool myCheck = true;
const bool myBest = false; //looking for best solution

#define GPU_BLOCKS(sz, thrdsPrBlck) (int((sz)/(thrdsPrBlck)) + 1)

using namespace std;

int returnString(int p_number, const int* p_C)
{
	int i = 0;
	while (p_C[i] <= p_number) {
		i++;
	}
	return i;
}

__global__ void diVec(int p_size, double *p_diag, double *p_vec, double *o_rez)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < p_size) {
		o_rez[i] += p_diag[i] * p_vec[i];
	}
}

__global__ void diRev(int p_size, double *p_diag, double *o_rez)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < p_size) {
		o_rez[i] = 1 / p_diag[i];
	}
}

__global__ void vec2vec(int p_size, double *p_vec1, double *p_vec2, double *o_rez)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < p_size) {
		o_rez[i] = p_vec1[i] * p_vec2[i];
	}
}

int sgn(double p_val)
{
	if (p_val == 0.) {
		return 0;
	}
	if (p_val > 0.) {
		return 1;
	}
	else {
		return -1;
	}
}

int* splitMatrix(int p_gpu_amount,
	const double* p_aelem,
	const int* p_jptr,
	const int* p_iptr,
	const int p_size,
	const int p_non_zero,
	double **o_d_aelem,
	int ** o_d_jptr,
	int **o_d_iptr)
{
	int mod = p_non_zero / p_gpu_amount;
	int rest = p_non_zero - mod * (p_gpu_amount - 1);
	int first_position;
	int last_position;
	int first_string;
	int last_string;
	double *aelem_;
	int *jptr_;
	int *iptr_;

	int *temp = new int[p_gpu_amount];
	int nsize;

	//firstr position = n, last_position = k
	//first_string = i, last_string = j
	for (int number = 0; number < p_gpu_amount; number++) {
		if (number == p_gpu_amount - 1) {
			int in1 = 0;
			int in2 = 0;
			first_position = number * mod;
			last_position = p_non_zero - 1;
			first_string = returnString(number*mod, p_iptr) - 1;
			last_string = returnString(p_non_zero - 1, p_iptr) - 1;
			nsize = rest + first_string + p_size - 1 - last_string;

			aelem_ = new double[nsize];
			for (int i = 0; i < nsize; i++) {
				if (i < first_string) {
					aelem_[i] = 0;
				}
				else {
					aelem_[i] = p_aelem[first_position + in1];
					in1++;
				}
			}

			jptr_ = new int[nsize];
			for (int i = 0; i < nsize; i++) {
				if (i < first_string) {
					jptr_[i] = i;
				}
				else {
					jptr_[i] = p_jptr[first_position + in2];
					in2++;
				}
			}

			iptr_ = new int[p_size + 1];

			for (int i = 0; i < first_string; i++) {
				iptr_[i] = i;
			}
			for (int count = first_string; count <= last_string; count++) {
				iptr_[count] = p_iptr[count] - first_position + first_string;
				if (p_iptr[count] - first_position < 0) iptr_[count] = first_string;
			}
			iptr_[p_size] = nsize;

		}
		else {
			int in1 = 0;
			int in2 = 0;
			first_position = number * mod;
			last_position = (number + 1)*mod - 1;
			first_string = returnString(number*mod, p_iptr) - 1;
			last_string = returnString((number + 1)*mod - 1, p_iptr) - 1;
			nsize = mod + first_string + p_size - 1 - last_string;

			aelem_ = new double[nsize];
			for (int i = 0; i < nsize; i++) {
				if ((i < first_string) || (i > first_string + mod - 1)) {
					aelem_[i] = 0;
				}
				else {
					aelem_[i] = p_aelem[first_position + in1];
					in1++;
				}
			}

			jptr_ = new int[nsize];

			int inn = 1;
			for (int i = 0; i < nsize; i++) {
				if (i < first_string) {
					jptr_[i] = i;
				}
				else if (i < first_string + mod) {
					jptr_[i] = p_jptr[first_position + in2];
					in2++;
				}
				else {
					jptr_[i] = last_string + inn;
					inn++;
				}
			}

			iptr_ = new int[p_size + 1];

			for (int i = 0; i < first_string; i++) {
				iptr_[i] = i;
			}
			for (int count = first_string; count <= last_string; count++) {
				iptr_[count] = p_iptr[count] - first_position + first_string;
				if (p_iptr[count] - first_position < 0) {
					iptr_[count] = first_string;
				}
			}
			int l = 1;
			for (int i = last_string + 1; i < p_size; i++) {
				iptr_[i] = first_string + last_position - first_position + l;
				l++;
			}
			iptr_[p_size] = nsize;
		}

		temp[number] = nsize;

		cudaSetDevice(number);
		cudaMalloc((void **)&o_d_aelem[number], sizeof(double)*nsize);
		cudaMalloc((void **)&o_d_jptr[number], sizeof(int)*nsize);
		cudaMalloc((void **)&o_d_iptr[number], sizeof(int)*(p_size + 1));
		cudaMemcpy(o_d_aelem[number], aelem_, sizeof(double)*nsize, cudaMemcpyHostToDevice);
		cudaMemcpy(o_d_jptr[number], jptr_, sizeof(int)*nsize, cudaMemcpyHostToDevice);
		cudaMemcpy(o_d_iptr[number], iptr_, sizeof(int)*(p_size + 1), cudaMemcpyHostToDevice);

		delete[] aelem_;
		delete[] jptr_;
		delete[] iptr_;
	}
	return temp;
}

double* SolverGPU::conjugateGradientMethod(const double *p_aelem,
	const int *p_jptr,
	const int *p_iptr,
	const double *p_right,
	const double *p_diag,
	const int p_nnz,
	const int p_size)
{
	//Count amount of devices
	int gpu;
	cudaGetDeviceCount(&gpu);

	//Arrays for devices
	double ** d_aelem = new  double *[gpu];
	int ** d_jptr = new int *[gpu];
	int ** d_iptr = new int *[gpu];

	//Array with device array's sizes
	int *sizes = new int[gpu];
	sizes = splitMatrix(gpu, p_aelem, p_jptr, p_iptr, p_size, p_nnz, d_aelem, d_jptr, d_iptr);

	//Other
	bool flag = true;
	int step = 0;
	double *minus = new double;
	double *zero = new double;
	double *one = new double;
	*minus = -1.0;
	*zero = 0.0;
	*one = 1.0;

	//Set device 0. It's main device that would calculate everything
	cudaSetDevice(0);

	//Preparing CUBLAS handle
	cublasHandle_t cublas_handle = NULL;
	cublasCreate(&cublas_handle);

	//Preparing CUSPARSE handle
	cusparseHandle_t cusparse_handle = NULL;
	cusparseCreate(&cusparse_handle);

	//Preparing matrix descriptor
	cusparseMatDescr_t matDescr = NULL;
	cusparseCreateMatDescr(&matDescr);

	//Set base for matrix
	if (p_iptr[0]) {
		cusparseSetMatIndexBase(matDescr, CUSPARSE_INDEX_BASE_ONE);
	}
	else {
		cusparseSetMatIndexBase(matDescr, CUSPARSE_INDEX_BASE_ZERO);
	}

	//Prepating device information
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);

	//Preparing x0
	double *x0;
	cudaMalloc((void **)&x0, sizeof(double)*(p_size));
	cudaMemset(x0, 0, sizeof(double)*(p_size));

	//Preparing r0 and p0
	double *r0, *p0;
	cudaMalloc((void **)&r0, sizeof(double)*(p_size));
	cudaMallocManaged((void **)&p0, sizeof(double)*(p_size));
	cudaMemcpy(r0, p_right, sizeof(double)*(p_size), cudaMemcpyHostToDevice);
	cudaMemcpy(p0, p_right, sizeof(double)*(p_size), cudaMemcpyHostToDevice);

	//Preparing alpha and beta
	double *alpha = new double, *beta = new double, *negalpha = new double;

	//Preparing pK, nuK, s, t, xK, rK, h
	double **vec_temp1 = new double *[gpu],
		**vec_temp5 = new double *[gpu],
		*xK, *rK, *pK;
#pragma omp parallel num_threads(gpu)
	{
		cudaSetDevice(omp_get_thread_num());
		cudaMalloc((void **)&vec_temp1[omp_get_thread_num()], sizeof(double)*(p_size));
		cudaMalloc((void **)&vec_temp5[omp_get_thread_num()], sizeof(double)*(p_size));
	}
	cudaSetDevice(0);
	cudaMalloc((void **)&xK, sizeof(double)*(p_size));
	cudaMalloc((void **)&rK, sizeof(double)*(p_size));
	cudaMallocManaged((void **)&pK, sizeof(double)*(p_size));

	//Preparing diag for devices
	double *d_diag;
	cudaMalloc((void **)&d_diag, sizeof(double)*(p_size));
	cudaMemcpy(d_diag, p_diag, sizeof(double)*(p_size), cudaMemcpyHostToDevice);

	//Preparing temporary variables.
	//There are 2 types of temporary variables: vec_ and num_
	double *r0nrm = new double,
		*rKnrm = new double,
		*num_temp1 = new double;
	double *vec_temp2, *vec_temp3, *vec_temp4 = new double[p_size];
	cudaMalloc((void **)&vec_temp2, sizeof(double)*(p_size));
	cudaMalloc((void **)&vec_temp3, sizeof(double)*(p_size));

	double *bnrm = new double;
	if (!residue) {
		cublasDnrm2(cublas_handle, p_size, r0, 1, bnrm);
	}

	double *best;
	double *bestNrm = new double;
	int bestSt = 0;
	if (myBest) {
		cudaMalloc((void **)&best, sizeof(double)*(p_size));
		*bestNrm = 1000.0;
	}

	do {
		if (!flag) {
			p0 = pK;
		}

		//Step 1
		//A = A1+A2+...+An => A*x = A1*x+A2*x+...+An*x
		//We use all GPU's but then we send everything to first one
		//And don't forget to add diag
		cublasDdot(cublas_handle, p_size, r0, 1, r0, 1, alpha);

#pragma omp parallel num_threads(gpu)
		{
			cudaSetDevice(omp_get_thread_num());
			cudaMemcpy(vec_temp5[omp_get_thread_num()], p0, sizeof(double)*(p_size), cudaMemcpyDefault);
			cusparseDcsrmv(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
				p_size, p_size, sizes[omp_get_thread_num()],
				one, matDescr, d_aelem[omp_get_thread_num()],
				d_iptr[omp_get_thread_num()], d_jptr[omp_get_thread_num()],
				vec_temp5[omp_get_thread_num()], zero, vec_temp1[omp_get_thread_num()]);
		}

		for (int i = 1; i < gpu; i++) {
			cudaSetDevice(i);
			cudaMemcpy(vec_temp4, vec_temp1[i], sizeof(double)*(p_size), cudaMemcpyDefault);
			cudaSetDevice(0);
			cudaMemcpy(vec_temp3, vec_temp4, sizeof(double)*(p_size), cudaMemcpyHostToDevice);
			cublasDaxpy(cublas_handle, p_size, one, vec_temp3, 1, vec_temp1[0], 1);
		}

		diVec << <GPU_BLOCKS(p_size, deviceProp.maxThreadsPerBlock), deviceProp.maxThreadsPerBlock >> > (p_size, d_diag, p0, vec_temp1[0]);
		cudaDeviceSynchronize();
		cublasDdot(cublas_handle, p_size, vec_temp1[0], 1, p0, 1, num_temp1);
		*alpha /= (*num_temp1);
		*negalpha = -(*alpha);

		//Step 2
		cublasDaxpy(cublas_handle, p_size, alpha, p0, 1, x0, 1);
		xK = x0;

		//Step 3
		cublasDnrm2(cublas_handle, p_size, r0, 1, r0nrm);
		cublasDaxpy(cublas_handle, p_size, negalpha, vec_temp1[0], 1, r0, 1);
		rK = r0;

		//Step 4 (check)
		cublasDnrm2(cublas_handle, p_size, rK, 1, rKnrm);

		if (myBest) {
			if (residue) {
				if (*bestNrm > *rKnrm) {
					*bestNrm = *rKnrm;
				}
			}
			else {
				if (*bestNrm > (*rKnrm) / (*bnrm)) {
					*bestNrm = (*rKnrm) / (*bnrm);
				}
			}
			cublasDcopy(cublas_handle, p_size, xK, 1, best, 1);
			bestSt = step;
		}

		if (myCheck) {
			if (*rKnrm != *rKnrm) {
				cout << "Nan detected. Aborting..." << endl;
				break;
			}

			if (step % 20 == 0) {
				if (myBest) {
					if (residue) {
						cout << "Abs res: " << *rKnrm << " with abs best: " << *bestNrm << " on " << bestSt << " step" << endl;
					}
					else {
						cout << "Rel res: " << (*rKnrm) / (*bnrm) << " with rel best: " << *bestNrm << " on " << bestSt << " step" << endl;
					}
				}
				else {
					if (residue) {
						cout << "Abs res: " << *rKnrm << endl;
					}
					else {
						cout << "Rel res: " << (*rKnrm) / (*bnrm) << endl;
					}
				}
			}
		}

		if (residue)
		{
			if ((*rKnrm) < maxacc) {
				break;
			}
		}
		else {
			if ((*rKnrm) / (*bnrm) < maxacc) {
				break;
			}
		}

		//Step 5
		*beta = pow((*rKnrm), 2) / pow((*r0nrm), 2);

		//Step 6
		cublasDcopy(cublas_handle, p_size, rK, 1, vec_temp2, 1);
		cublasDaxpy(cublas_handle, p_size, beta, p0, 1, vec_temp2, 1);
		cublasDcopy(cublas_handle, p_size, vec_temp2, 1, pK, 1);

		flag = false;
		step++;

	} while (step <= maxiter);

	if (myBest) {
		cublasDcopy(cublas_handle, p_size, best, 1, xK, 1);
	}

	//Check || Ax - b || or || Ax - b || / || b ||
	if (myCheck) {
		cudaMemcpy(vec_temp4, xK, sizeof(double)*(p_size), cudaMemcpyDefault);
#pragma omp parallel num_threads(gpu)
		{
			cudaSetDevice(omp_get_thread_num());
			cudaMemcpy(vec_temp5[omp_get_thread_num()], vec_temp4, sizeof(double)*(p_size), cudaMemcpyDefault);
			cusparseDcsrmv(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
				p_size, p_size, sizes[omp_get_thread_num()],
				one, matDescr, d_aelem[omp_get_thread_num()],
				d_iptr[omp_get_thread_num()], d_jptr[omp_get_thread_num()],
				vec_temp5[omp_get_thread_num()], zero, vec_temp1[omp_get_thread_num()]);

		}
		for (int i = 1; i < gpu; i++) {
			cudaSetDevice(i);
			cudaMemcpy(vec_temp4, vec_temp1[i], sizeof(double)*(p_size), cudaMemcpyDeviceToHost);
			cudaSetDevice(0);
			cudaMemcpy(vec_temp3, vec_temp4, sizeof(double)*(p_size), cudaMemcpyHostToDevice);
			cublasDaxpy(cublas_handle, p_size, one, vec_temp3, 1, vec_temp1[0], 1);
		}
		diVec << <GPU_BLOCKS(p_size, deviceProp.maxThreadsPerBlock), deviceProp.maxThreadsPerBlock >> > (p_size, d_diag, xK, vec_temp1[0]);
		cudaDeviceSynchronize();
		cudaMemcpy(vec_temp3, p_right, sizeof(double)*(p_size), cudaMemcpyHostToDevice);
		cublasDaxpy(cublas_handle, p_size, minus, vec_temp3, 1, vec_temp1[0], 1);
		cublasDnrm2(cublas_handle, p_size, vec_temp1[0], 1, rKnrm);

		if (residue) {
			cout << "||Ax-b||: " << *rKnrm;
		}
		else {
			cout << "||Ax-b|| / ||b||: " << (*rKnrm) / (*bnrm);
		}

		cout << endl << "Steps: " << step << endl;

		if (myBest) {
			cout << "Best result on " << bestSt << " step" << endl;
		}
	}

	//Transferring result from device to host
	double *res = new double[p_size];
	cudaMemcpy(res, xK, sizeof(double)*p_size, cudaMemcpyDeviceToHost);

	//Clean all
	cublasDestroy(cublas_handle);
	cusparseDestroy(cusparse_handle);
	cusparseDestroyMatDescr(matDescr);

	delete minus, one, zero, alpha, beta, negalpha, r0nrm, rKnrm, num_temp1;

	cudaFree(x0);
	cudaFree(r0);
	cudaFree(p0);
	cudaFree(vec_temp2);
	cudaFree(vec_temp3);
	cudaFree(d_diag);

#pragma omp parallel num_threads(gpu)
	{
		cudaSetDevice(omp_get_thread_num());
		cudaFree(vec_temp1[omp_get_thread_num()]);
		cudaFree(vec_temp5[omp_get_thread_num()]);
		cudaFree(d_aelem[omp_get_thread_num()]);
		cudaFree(d_jptr[omp_get_thread_num()]);
		cudaFree(d_iptr[omp_get_thread_num()]);

	}

	delete[] sizes, d_aelem, d_iptr, d_jptr, vec_temp4, vec_temp1, vec_temp5;

	return res;
}

double* SolverGPU::preConjugateGradientMethod(const double *p_aelem,
	const int *p_jptr,
	const int *p_iptr,
	const double *p_right,
	const double *p_diag,
	const int p_nnz,
	const int p_size)
{
	//Count amount of devices
	int gpu;
	cudaGetDeviceCount(&gpu);

	//Arrays for devices
	double ** d_aelem = new  double *[gpu];
	int ** d_jptr = new int *[gpu];
	int ** d_iptr = new int *[gpu];

	//Array with device array's sizes
	int *sizes = new int[gpu];
	sizes = splitMatrix(gpu, p_aelem, p_jptr, p_iptr, p_size, p_nnz, d_aelem, d_jptr, d_iptr);

	//Other
	bool flag = true;
	int step = 0;
	double *minus = new double;
	double *zero = new double;
	double *one = new double;
	*minus = -1.0;
	*zero = 0.0;
	*one = 1.0;

	//Set device 0. It's main device that would calculate everything
	cudaSetDevice(0);

	//Preparing CUBLAS handle
	cublasHandle_t cublas_handle = NULL;
	cublasCreate(&cublas_handle);

	//Preparing CUSPARSE handle
	cusparseHandle_t cusparse_handle = NULL;
	cusparseCreate(&cusparse_handle);

	//Preparing matrix descriptor
	cusparseMatDescr_t matDescr = NULL;
	cusparseCreateMatDescr(&matDescr);

	//Set base for matrix
	if (p_iptr[0]) {
		cusparseSetMatIndexBase(matDescr, CUSPARSE_INDEX_BASE_ONE);
	}
	else {
		cusparseSetMatIndexBase(matDescr, CUSPARSE_INDEX_BASE_ZERO);
	}

	//Prepating device information
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);

	//Preparing diag
	double *d_diag, *d_revDiag;
	cudaMalloc((void **)&d_diag, sizeof(double)*(p_size));
	cudaMalloc((void **)&d_revDiag, sizeof(double)*(p_size));
	cudaMemcpy(d_diag, p_diag, sizeof(double)*(p_size), cudaMemcpyHostToDevice);
	diRev << <GPU_BLOCKS(p_size, deviceProp.maxThreadsPerBlock), deviceProp.maxThreadsPerBlock >> > (p_size, d_diag, d_revDiag);
	cudaDeviceSynchronize();

	//Preparing x0
	double *x0;
	cudaMalloc((void **)&x0, sizeof(double)*(p_size));
	cudaMemset(x0, 0, sizeof(double)*(p_size));

	//Preparing r0 and p0
	double *r0, *p0, *z0;
	cudaMalloc((void **)&r0, sizeof(double)*(p_size));
	cudaMallocManaged((void **)&p0, sizeof(double)*(p_size));
	cudaMalloc((void **)&z0, sizeof(double)*(p_size));
	cudaMemcpy(r0, p_right, sizeof(double)*(p_size), cudaMemcpyHostToDevice);
	vec2vec << <GPU_BLOCKS(p_size, deviceProp.maxThreadsPerBlock), deviceProp.maxThreadsPerBlock >> > (p_size, d_revDiag, r0, z0);
	cudaDeviceSynchronize();
	cudaMemcpy(p0, z0, sizeof(double)*(p_size), cudaMemcpyHostToDevice);

	//Preparing alpha and beta
	double *alpha = new double, *beta = new double, *negalpha = new double;

	//Preparing pK, nuK, s, t, xK, rK, h
	double **vec_temp1 = new double *[gpu],
		**vec_temp5 = new double *[gpu],
		*xK, *rK, *pK, *zK;
#pragma omp parallel num_threads(gpu)
	{
		cudaSetDevice(omp_get_thread_num());
		cudaMalloc((void **)&vec_temp1[omp_get_thread_num()], sizeof(double)*(p_size));
		cudaMalloc((void **)&vec_temp5[omp_get_thread_num()], sizeof(double)*(p_size));

	}
	cudaSetDevice(0);
	cudaMalloc((void **)&xK, sizeof(double)*(p_size));
	cudaMalloc((void **)&rK, sizeof(double)*(p_size));
	cudaMallocManaged((void **)&pK, sizeof(double)*(p_size));
	cudaMalloc((void **)&zK, sizeof(double)*(p_size));

	//Preparing temporary variables.
	//There are 2 types of temporary variables: vec_ and num_
	double *num_temp1 = new double,
		*rKnrm = new double,
		*num_temp3 = new double;
	double *vec_temp2, *vec_temp3, *vec_temp4 = new double[p_size];
	cudaMalloc((void **)&vec_temp2, sizeof(double)*(p_size));
	cudaMalloc((void **)&vec_temp3, sizeof(double)*(p_size));

	double *bnrm = new double;
	if (!residue) {
		cublasDnrm2(cublas_handle, p_size, r0, 1, bnrm);
	}

	double *best;
	double *bestNrm = new double;
	int bestSt = 0;
	if (myBest) {
		cudaMalloc((void **)&best, sizeof(double)*(p_size));
		*bestNrm = 1000.0;
	}

	do {
		if (!flag) {
			p0 = pK;
			z0 = zK;
		}

		//Step 1
		//A = A1+A2+...+An => A*x = A1*x+A2*x+...+An*x
		//We use all GPU's but then we send everything to first one
		//And don't forget to add diag
		cublasDdot(cublas_handle, p_size, r0, 1, z0, 1, alpha);
#pragma omp parallel num_threads(gpu)
		{
			cudaSetDevice(omp_get_thread_num());
			cudaMemcpy(vec_temp5[omp_get_thread_num()], p0, sizeof(double)*(p_size), cudaMemcpyDefault);
			cusparseDcsrmv(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
				p_size, p_size, sizes[omp_get_thread_num()],
				one, matDescr, d_aelem[omp_get_thread_num()],
				d_iptr[omp_get_thread_num()], d_jptr[omp_get_thread_num()],
				vec_temp5[omp_get_thread_num()], zero, vec_temp1[omp_get_thread_num()]);

		}
		for (int i = 1; i < gpu; i++) {
			cudaSetDevice(i);
			cudaMemcpy(vec_temp4, vec_temp1[i], sizeof(double)*(p_size), cudaMemcpyDeviceToHost);
			cudaSetDevice(0);
			cudaMemcpy(vec_temp3, vec_temp4, sizeof(double)*(p_size), cudaMemcpyHostToDevice);
			cublasDaxpy(cublas_handle, p_size, one, vec_temp3, 1, vec_temp1[0], 1);
		}

		diVec << <GPU_BLOCKS(p_size, deviceProp.maxThreadsPerBlock), deviceProp.maxThreadsPerBlock >> > (p_size, d_diag, p0, vec_temp1[0]);
		cudaDeviceSynchronize();
		cublasDdot(cublas_handle, p_size, vec_temp1[0], 1, p0, 1, num_temp1);
		*alpha /= (*num_temp1);
		*negalpha = -(*alpha);

		//Step 2
		cublasDaxpy(cublas_handle, p_size, alpha, p0, 1, x0, 1);
		xK = x0;

		//Step 3
		cublasDdot(cublas_handle, p_size, z0, 1, r0, 1, num_temp3);
		cublasDaxpy(cublas_handle, p_size, negalpha, vec_temp1[0], 1, r0, 1);
		rK = r0;

		//Step 4 (check)
		cublasDnrm2(cublas_handle, p_size, rK, 1, rKnrm);

		if (myBest) {
			if (residue) {
				if (*bestNrm > *rKnrm) {
					*bestNrm = *rKnrm;
				}
			}
			else {
				if (*bestNrm > (*rKnrm) / (*bnrm)) {
					*bestNrm = (*rKnrm) / (*bnrm);
				}
			}
			cublasDcopy(cublas_handle, p_size, xK, 1, best, 1);
			bestSt = step;
		}

		if (myCheck) {
			if (*rKnrm != *rKnrm) {
				cout << "Nan detected. Aborting..." << endl;
				break;
			}

			if (step % 20 == 0) {
				if (myBest) {
					if (residue) {
						cout << "Abs res: " << *rKnrm << " with abs best: " << *bestNrm << " on " << bestSt << " step" << endl;
					}
					else {
						cout << "Rel res: " << (*rKnrm) / (*bnrm) << " with rel best: " << *bestNrm << " on " << bestSt << " step" << endl;
					}
				}
				else {
					if (residue) {
						cout << "Abs res: " << *rKnrm << endl;
					}
					else {
						cout << "Rel res: " << (*rKnrm) / (*bnrm) << endl;
					}
				}
			}
		}

		if (residue)
		{
			if ((*rKnrm) < maxacc) {
				break;
			}
		}
		else {
			if ((*rKnrm) / (*bnrm) < maxacc) {
				break;
			}
		}

		//Step 5
		vec2vec << <GPU_BLOCKS(p_size, deviceProp.maxThreadsPerBlock), deviceProp.maxThreadsPerBlock >> > (p_size, d_revDiag, rK, zK);
		cudaDeviceSynchronize();

		//Step 6
		cublasDdot(cublas_handle, p_size, rK, 1, zK, 1, beta);
		*beta /= (*num_temp3);

		//Step 7
		cublasDcopy(cublas_handle, p_size, zK, 1, vec_temp2, 1);
		cublasDaxpy(cublas_handle, p_size, beta, p0, 1, vec_temp2, 1);
		cublasDcopy(cublas_handle, p_size, vec_temp2, 1, pK, 1);

		flag = false;
		step++;

	} while (step <= maxiter);

	if (myBest) {
		cublasDcopy(cublas_handle, p_size, best, 1, xK, 1);
	}

	//Check || Ax - b || or || Ax - b || / || b ||
	if (myCheck) {
		cudaMemcpy(vec_temp4, xK, sizeof(double)*(p_size), cudaMemcpyDefault);
#pragma omp parallel num_threads(gpu)
		{
			cudaSetDevice(omp_get_thread_num());
			cudaMemcpy(vec_temp5[omp_get_thread_num()], vec_temp4, sizeof(double)*(p_size), cudaMemcpyDefault);
			cusparseDcsrmv(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
				p_size, p_size, sizes[omp_get_thread_num()],
				one, matDescr, d_aelem[omp_get_thread_num()],
				d_iptr[omp_get_thread_num()], d_jptr[omp_get_thread_num()],
				vec_temp5[omp_get_thread_num()], zero, vec_temp1[omp_get_thread_num()]);

		}
		for (int i = 1; i < gpu; i++) {
			cudaSetDevice(i);
			cudaMemcpy(vec_temp4, vec_temp1[i], sizeof(double)*(p_size), cudaMemcpyDeviceToHost);
			cudaSetDevice(0);
			cudaMemcpy(vec_temp3, vec_temp4, sizeof(double)*(p_size), cudaMemcpyHostToDevice);
			cublasDaxpy(cublas_handle, p_size, one, vec_temp3, 1, vec_temp1[0], 1);
		}
		diVec << <GPU_BLOCKS(p_size, deviceProp.maxThreadsPerBlock), deviceProp.maxThreadsPerBlock >> > (p_size, d_diag, xK, vec_temp1[0]);
		cudaDeviceSynchronize();
		cudaMemcpy(vec_temp3, p_right, sizeof(double)*(p_size), cudaMemcpyHostToDevice);
		cublasDaxpy(cublas_handle, p_size, minus, vec_temp3, 1, vec_temp1[0], 1);
		cublasDnrm2(cublas_handle, p_size, vec_temp1[0], 1, rKnrm);

		if (residue) {
			cout << "||Ax-b||: " << *rKnrm;
		}
		else {
			cout << "||Ax-b|| / ||b||: " << (*rKnrm) / (*bnrm);
		}

		cout << endl << "Steps: " << step << endl;

		if (myBest) {
			cout << "Best result on " << bestSt << " step" << endl;
		}
	}

	//Transferring result from device to host
	double *res = new double[p_size];
	cudaMemcpy(res, xK, sizeof(double)*p_size, cudaMemcpyDeviceToHost);

	//Clean all
	cublasDestroy(cublas_handle);
	cusparseDestroy(cusparse_handle);
	cusparseDestroyMatDescr(matDescr);

	delete minus, one, zero, alpha, beta, negalpha, rKnrm, num_temp1, num_temp3;

	cudaFree(x0);
	cudaFree(r0);
	cudaFree(p0);
	cudaFree(z0);
	cudaFree(vec_temp2);
	cudaFree(vec_temp3);
	cudaFree(d_diag);
	cudaFree(d_revDiag);

#pragma omp parallel num_threads(gpu)
	{
		cudaSetDevice(omp_get_thread_num());
		cudaFree(vec_temp1[omp_get_thread_num()]);
		cudaFree(vec_temp5[omp_get_thread_num()]);
		cudaFree(d_aelem[omp_get_thread_num()]);
		cudaFree(d_jptr[omp_get_thread_num()]);
		cudaFree(d_iptr[omp_get_thread_num()]);

	}

	delete[] sizes, d_aelem, d_iptr, d_jptr, vec_temp4, vec_temp1, vec_temp5;

	return res;
}

double* SolverGPU::biconjugateStabGradientMethod(const double *p_aelem,
	const int *p_jptr,
	const int *p_iptr,
	const double *p_right,
	const double *p_diag,
	const int p_nnz,
	const int p_size)
{
	//Count amount of devices
	int gpu;
	cudaGetDeviceCount(&gpu);

	//Arrays for devices
	double ** d_aelem = new  double *[gpu];
	int ** d_jptr = new int *[gpu];
	int ** d_iptr = new int *[gpu];

	//Array with device array's sizes
	int *sizes = new int[gpu];
	sizes = splitMatrix(gpu, p_aelem, p_jptr, p_iptr, p_size, p_nnz, d_aelem, d_jptr, d_iptr);

	//Other
	int step = 0;
	bool flag = true;
	double *minus = new double;
	double *zero = new double;
	double *one = new double;
	*minus = -1.0;
	*zero = 0.0;
	*one = 1.0;

	//Set device 0. It's main device that would calculate everything
	cudaSetDevice(0);

	//Preparing CUBLAS handle
	cublasHandle_t cublas_handle = NULL;
	cublasCreate(&cublas_handle);

	//Preparing CUSPARSE handle
	cusparseHandle_t cusparse_handle = NULL;
	cusparseCreate(&cusparse_handle);

	//Preparing matrix descriptor
	cusparseMatDescr_t matDescr = NULL;
	cusparseCreateMatDescr(&matDescr);

	//Set base for matrix
	if (p_iptr[0]) {
		cusparseSetMatIndexBase(matDescr, CUSPARSE_INDEX_BASE_ONE);
	}
	else {
		cusparseSetMatIndexBase(matDescr, CUSPARSE_INDEX_BASE_ZERO);
	}

	//Prepating device information
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);

	//Preparing x0
	double *x0;
	cudaMalloc((void **)&x0, sizeof(double)*(p_size));
	cudaMemset(x0, 0, sizeof(double)*(p_size));

	//Preparing r0 and rT
	double *r0, *rT;

	cudaMallocManaged((void **)&r0, sizeof(double)*(p_size));
	cudaMalloc((void **)&rT, sizeof(double)*(p_size));
	cudaMemcpy(r0, p_right, sizeof(double)*(p_size), cudaMemcpyHostToDevice);
	cudaMemcpy(rT, p_right, sizeof(double)*(p_size), cudaMemcpyHostToDevice);

	//Preparing rho0, alpha, omega0
	double *rho0 = new double;
	double *alpha = new double;
	double *omega0 = new double;
	*rho0 = 1.0;
	*alpha = 1.0;
	*omega0 = 1.0;

	//Praparing nu0,p0
	double *nu0, *p0;
	cudaMalloc((void **)&nu0, sizeof(double)*(p_size));
	cudaMalloc((void **)&p0, sizeof(double)*(p_size));
	cudaMemset(nu0, 0, sizeof(double)*(p_size));
	cudaMemset(p0, 0, sizeof(double)*(p_size));

	//Preparing rhoK, omegaK, beta
	double *rhoK = new double;
	double *omegaK = new double;
	double *beta = new double;

	//Preparing pK, nuK, s, t, xK, rK, h
	double **nuK = new double *[gpu],
		**t = new double *[gpu],
		**vec_temp5 = new double *[gpu],
		*xK, *rK, *h, *s, *pK;

#pragma omp parallel num_threads(gpu)
	{
		cudaSetDevice(omp_get_thread_num());
		cudaMalloc((void **)&nuK[omp_get_thread_num()], sizeof(double)*(p_size));
		cudaMalloc((void **)&t[omp_get_thread_num()], sizeof(double)*(p_size));
		cudaMalloc((void **)&vec_temp5[omp_get_thread_num()], sizeof(double)*(p_size));
	}

	cudaSetDevice(0);
	cudaMalloc((void **)&xK, sizeof(double)*(p_size));
	cudaMalloc((void **)&rK, sizeof(double)*(p_size));
	cudaMalloc((void **)&h, sizeof(double)*(p_size));
	cudaMallocManaged((void **)&s, sizeof(double)*(p_size));
	cudaMallocManaged((void **)&pK, sizeof(double)*(p_size));


	//Preparing diag for devices
	double *d_diag;
	cudaMalloc((void **)&d_diag, sizeof(double)*(p_size));
	cudaMemcpy(d_diag, p_diag, sizeof(double)*(p_size), cudaMemcpyHostToDevice);

	//Preparing temporary variables.
	//There are 2 types of temporary variables: vec_ and num_
	double *num_temp1 = new double,
		*num_temp2 = new double,
		*num_temp6 = new double,
		*rKnrm = new double;
	double *vec_temp1, *vec_temp2, *vec_temp3, *vec_temp4 = new double[p_size];
	cudaMalloc((void **)&vec_temp1, sizeof(double)*(p_size));
	cudaMalloc((void **)&vec_temp2, sizeof(double)*(p_size));
	cudaMalloc((void **)&vec_temp3, sizeof(double)*(p_size));

	//Others
	double *negalpha = new double;
	double *negomega = new double;
	*negomega = -(*omega0);

	double *bnrm = new double;
	if (!residue) {
		cublasDnrm2(cublas_handle, p_size, r0, 1, bnrm);
	}

	double *best;
	double *bestNrm = new double;
	int bestSt = 0;
	if (myBest) {
		cudaMalloc((void **)&best, sizeof(double)*(p_size));
		*bestNrm = 1000.0;
	}

	double *temp = new double[p_size];

	do {
		if (!flag) {
			nu0 = nuK[0];
			p0 = pK;
			omega0 = omegaK;
			*negomega = -(*omega0);
			*rho0 = *rhoK;
		}

		//Step 1
		cublasDdot(cublas_handle, p_size, rT, 1, r0, 1, rhoK);

		//Step 2
		if (stab) {
			if ((abs(*rho0) < maxacc) || (abs(*omega0) < maxacc)) {
				*rho0 = sgn(*rho0) * 1e-7;
				*omega0 = sgn(*omega0) * 1e-7;
			}
		}
		*beta = ((*rhoK) * (*alpha)) / ((*rho0) * (*omega0));

		//Step 3
		cublasDaxpy(cublas_handle, p_size, negomega, nu0, 1, p0, 1);
		cublasDcopy(cublas_handle, p_size, r0, 1, vec_temp1, 1);
		cublasDaxpy(cublas_handle, p_size, beta, p0, 1, vec_temp1, 1);
		cublasDcopy(cublas_handle, p_size, vec_temp1, 1, pK, 1);

		cudaMemcpy(temp, pK, sizeof(double)*(p_size), cudaMemcpyDefault);

		//Step 4
		//A = A1+A2+...+An => A*x = A1*x+A2*x+...+An*x
		//We use all GPU's but then we send everything to first one
		//And don't forget to add diag
#pragma omp parallel num_threads(gpu)
		{
			cudaSetDevice(omp_get_thread_num());
			cudaMemcpy(vec_temp5[omp_get_thread_num()], pK, sizeof(double)*(p_size), cudaMemcpyDefault);
			cusparseDcsrmv(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
				p_size, p_size, sizes[omp_get_thread_num()],
				one, matDescr, d_aelem[omp_get_thread_num()],
				d_iptr[omp_get_thread_num()], d_jptr[omp_get_thread_num()],
				vec_temp5[omp_get_thread_num()], zero, nuK[omp_get_thread_num()]);
		}
		for (int i = 1; i < gpu; i++) {
			cudaSetDevice(i);
			cudaMemcpy(vec_temp4, nuK[i], sizeof(double)*(p_size), cudaMemcpyDeviceToHost);
			cudaSetDevice(0);
			cudaMemcpy(vec_temp3, vec_temp4, sizeof(double)*(p_size), cudaMemcpyHostToDevice);
			cublasDaxpy(cublas_handle, p_size, one, vec_temp3, 1, nuK[0], 1);
		}

		diVec << <GPU_BLOCKS(p_size, deviceProp.maxThreadsPerBlock), deviceProp.maxThreadsPerBlock >> > (p_size, d_diag, pK, nuK[0]);
		cudaDeviceSynchronize();
		cudaMemcpy(temp, nuK[0], sizeof(double)*(p_size), cudaMemcpyDefault);

		//Step 5
		cublasDdot(cublas_handle, p_size, rT, 1, nuK[0], 1, num_temp1);
		if (stab) {
			if (abs(*num_temp1) < maxacc) {
				*num_temp1 = sgn(*num_temp1) * 1e-7;
			}
		}
		*alpha = (*rhoK) / (*num_temp1);
		*negalpha = -(*alpha);

		//Step 6
		cublasDaxpy(cublas_handle, p_size, alpha, pK, 1, x0, 1);
		h = x0;
		cudaMemcpy(temp, h, sizeof(double)*(p_size), cudaMemcpyDefault);

		//Step 7
		cublasDaxpy(cublas_handle, p_size, negalpha, nuK[0], 1, r0, 1);
		s = r0;
		cudaMemcpy(temp, s, sizeof(double)*(p_size), cudaMemcpyDefault);

		//Step 8
		//A = A1+A2+...+An => A*x = A1*x+A2*x+...+An*x
		//We use all GPU's but then we send everything to first one
		//And don't forget to add diag
#pragma omp parallel num_threads(gpu)
		{
			cudaSetDevice(omp_get_thread_num());
			cudaMemcpy(vec_temp5[omp_get_thread_num()], s, sizeof(double)*(p_size), cudaMemcpyDefault);
			cusparseDcsrmv(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
				p_size, p_size, sizes[omp_get_thread_num()],
				one, matDescr, d_aelem[omp_get_thread_num()],
				d_iptr[omp_get_thread_num()], d_jptr[omp_get_thread_num()],
				vec_temp5[omp_get_thread_num()], zero, t[omp_get_thread_num()]);
		}
		for (int i = 1; i < gpu; i++) {
			cudaSetDevice(i);
			cudaMemcpy(vec_temp4, t[i], sizeof(double)*(p_size), cudaMemcpyDeviceToHost);
			cudaSetDevice(0);
			cudaMemcpy(vec_temp3, vec_temp4, sizeof(double)*(p_size), cudaMemcpyHostToDevice);
			cublasDaxpy(cublas_handle, p_size, one, vec_temp3, 1, t[0], 1);
		}
		diVec << <GPU_BLOCKS(p_size, deviceProp.maxThreadsPerBlock), deviceProp.maxThreadsPerBlock >> > (p_size, d_diag, s, t[0]);
		cudaDeviceSynchronize();
		cudaMemcpy(temp, t[0], sizeof(double)*(p_size), cudaMemcpyDefault);

		//Step 9
		cublasDdot(cublas_handle, p_size, t[0], 1, s, 1, num_temp6);
		cublasDdot(cublas_handle, p_size, t[0], 1, t[0], 1, num_temp2);
		if (stab) {
			if (abs(*num_temp2) < maxacc) {
				*num_temp2 = sgn(*num_temp2) * 1e-7;
			}
		}
		*omegaK = (*num_temp6) / (*num_temp2);
		*negomega = -(*omegaK);

		//Step 10
		cublasDaxpy(cublas_handle, p_size, omegaK, s, 1, h, 1);
		xK = h;
		cudaMemcpy(temp, xK, sizeof(double)*(p_size), cudaMemcpyDefault);

		//Step 11
		cublasDaxpy(cublas_handle, p_size, negomega, t[0], 1, s, 1);
		rK = s;

		cudaMemcpy(temp, rK, sizeof(double)*(p_size), cudaMemcpyDefault);

		//Step 12 (check)
		cublasDnrm2(cublas_handle, p_size, rK, 1, rKnrm);

		if (myBest) {
			if (residue) {
				if (*bestNrm > *rKnrm) {
					*bestNrm = *rKnrm;
				}
			}
			else {
				if (*bestNrm > (*rKnrm) / (*bnrm)) {
					*bestNrm = (*rKnrm) / (*bnrm);
				}
			}
			cublasDcopy(cublas_handle, p_size, xK, 1, best, 1);
			bestSt = step;
		}

		if (myCheck) {
			if (*rKnrm != *rKnrm) {
				cout << "Nan detected. Aborting..." << endl;
				break;
			}

			if (step % 20 == 0) {
				if (myBest) {
					if (residue) {
						cout << "Abs res: " << *rKnrm << " with abs best: " << *bestNrm << " on " << bestSt << " step" << endl;
					}
					else {
						cout << "Rel res: " << (*rKnrm) / (*bnrm) << " with rel best: " << *bestNrm << " on " << bestSt << " step" << endl;
					}
				}
				else {
					if (residue) {
						cout << "Abs res: " << *rKnrm << endl;
					}
					else {
						cout << "Rel res: " << (*rKnrm) / (*bnrm) << endl;
					}
				}
			}
		}

		if (residue) {
			if (omega) {
				if (((*rKnrm) < maxacc) || (abs(*omegaK) <= maxacc)) {
					break;
				}
			}
			else {
				if ((*rKnrm) < maxacc) {
					break;
				}
			}
		}
		else {
			if (omega) {
				if (((*rKnrm) / (*bnrm) < maxacc) || (abs(*omegaK) <= maxacc)) {
					break;
				}
			}
			else {
				if ((*rKnrm) / (*bnrm) < maxacc) {
					break;
				}
			}
		}

		flag = false;
		step++;

	} while (step <= maxiter);

	if (myBest) {
		cublasDcopy(cublas_handle, p_size, best, 1, xK, 1);
	}

	//Check || Ax - b || or || Ax - b || / || b ||
	if (myCheck) {
		cudaMemcpy(vec_temp4, xK, sizeof(double)*(p_size), cudaMemcpyDefault);
#pragma omp parallel num_threads(gpu)
		{
			cudaSetDevice(omp_get_thread_num());
			cudaMemcpy(vec_temp5[omp_get_thread_num()], vec_temp4, sizeof(double)*(p_size), cudaMemcpyDefault);
			cusparseDcsrmv(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
				p_size, p_size, sizes[omp_get_thread_num()],
				one, matDescr, d_aelem[omp_get_thread_num()],
				d_iptr[omp_get_thread_num()], d_jptr[omp_get_thread_num()],
				vec_temp5[omp_get_thread_num()], zero, t[omp_get_thread_num()]);
		}
		for (int i = 1; i < gpu; i++) {
			cudaSetDevice(i);
			cudaMemcpy(vec_temp4, t[i], sizeof(double)*(p_size), cudaMemcpyDeviceToHost);
			cudaSetDevice(0);
			cudaMemcpy(vec_temp3, vec_temp4, sizeof(double)*(p_size), cudaMemcpyHostToDevice);
			cublasDaxpy(cublas_handle, p_size, one, vec_temp3, 1, t[0], 1);
		}
		diVec << <GPU_BLOCKS(p_size, deviceProp.maxThreadsPerBlock), deviceProp.maxThreadsPerBlock >> > (p_size, d_diag, xK, t[0]);
		cudaDeviceSynchronize();
		cudaMemcpy(vec_temp3, p_right, sizeof(double)*(p_size), cudaMemcpyHostToDevice);
		cublasDaxpy(cublas_handle, p_size, minus, vec_temp3, 1, t[0], 1);
		cublasDnrm2(cublas_handle, p_size, t[0], 1, rKnrm);

		if (residue) {
			cout << "||Ax-b||: " << *rKnrm;
		}
		else {
			cout << "||Ax-b|| / ||b||: " << (*rKnrm) / (*bnrm);
		}

		cout << endl << "Steps: " << step << endl;

		if (myBest) {
			cout << "Best result on " << bestSt << " step" << endl;
		}
	}

	//Transferring result from device to host
	double *res = new double[p_size];
	cudaMemcpy(res, xK, sizeof(double)*p_size, cudaMemcpyDeviceToHost);

	//Clean all
	cublasDestroy(cublas_handle);
	cusparseDestroy(cusparse_handle);
	cusparseDestroyMatDescr(matDescr);

	cudaFree(x0);
	cudaFree(r0);
	cudaFree(rT);
	cudaFree(nu0);
	cudaFree(pK);
	cudaFree(d_diag);
	cudaFree(vec_temp1);
	cudaFree(vec_temp2);
	cudaFree(vec_temp3);

	delete minus, one, zero, rhoK,
		beta, alpha, omega0, omegaK,
		negalpha, negomega, rho0, num_temp1,
		num_temp2, num_temp6, rKnrm;

#pragma omp parallel num_threads(gpu)
	{
		cudaSetDevice(omp_get_thread_num());
		cudaFree(t[omp_get_thread_num()]);
		cudaFree(d_aelem[omp_get_thread_num()]);
		cudaFree(d_jptr[omp_get_thread_num()]);
		cudaFree(d_iptr[omp_get_thread_num()]);
		cudaFree(vec_temp5[omp_get_thread_num()]);
	}

	if (gpu > 1) {
#pragma omp parallel num_threads(gpu - 1)
		{
			cudaSetDevice(omp_get_thread_num() + 1);
			cudaFree(nuK[omp_get_thread_num() + 1]);
		}
	}

	delete[] sizes, d_aelem, d_iptr, d_jptr, t, nuK, vec_temp4, vec_temp5;

	return res;
}

double* SolverGPU::preBiconjugateStabGradientMethod(const double *p_aelem,
	const int *p_jptr,
	const int *p_iptr,
	const double *p_right,
	const double *p_diag,
	const int p_nnz,
	const int p_size)
{

	//Count amount of devices
	int gpu;
	cudaGetDeviceCount(&gpu);

	//Arrays for devices
	double ** d_aelem = new  double *[gpu];
	int ** d_jptr = new int *[gpu];
	int ** d_iptr = new int *[gpu];

	//Array with device array's sizes
	int *sizes = new int[gpu];
	sizes = splitMatrix(gpu, p_aelem, p_jptr, p_iptr, p_size, p_nnz, d_aelem, d_jptr, d_iptr);

	//Other
	int step = 0;
	bool flag = true;
	double *minus = new double;
	double *zero = new double;
	double *one = new double;
	*minus = -1.0;
	*zero = 0.0;
	*one = 1.0;

	//Set device 0. It's main device that would calculate everything
	cudaSetDevice(0);

	//Preparing CUBLAS handle
	cublasHandle_t cublas_handle = NULL;
	cublasCreate(&cublas_handle);

	//Preparing CUSPARSE handle
	cusparseHandle_t cusparse_handle = NULL;
	cusparseCreate(&cusparse_handle);

	//Preparing matrix descriptor
	cusparseMatDescr_t matDescr = NULL;
	cusparseCreateMatDescr(&matDescr);

	//Set base for matrix
	if (p_iptr[0]) {
		cusparseSetMatIndexBase(matDescr, CUSPARSE_INDEX_BASE_ONE);
	}
	else {
		cusparseSetMatIndexBase(matDescr, CUSPARSE_INDEX_BASE_ZERO);
	}

	//Prepating device information
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);

	// cerr << "GPU: " << gpu << " RESIDUE: " << RESIDUE << " STAB: " << STAB << " OMEGA: " << OMEGA << " BEST: "<< BEST << "\n"
	// 	<< "Total amount of threads: " << GPU_BLOCKS(p_size, deviceProp.maxThreadsPerBlock)*deviceProp.maxThreadsPerBlock << "\n";

	//Preparing x0
	double *x0;
	cudaMalloc((void **)&x0, sizeof(double)*(p_size));
	cudaMemset(x0, 0, sizeof(double)*(p_size));

	//Preparing r0 and rT
	double *r0, *rT;
	cudaMalloc((void **)&r0, sizeof(double)*(p_size));
	cudaMalloc((void **)&rT, sizeof(double)*(p_size));
	cudaMemcpy(r0, p_right, sizeof(double)*(p_size), cudaMemcpyHostToDevice);
	cudaMemcpy(rT, p_right, sizeof(double)*(p_size), cudaMemcpyHostToDevice);

	//Preparing rho0, alpha, omega0
	double *rho0 = new double;
	double *alpha = new double;
	double *omega0 = new double;
	*rho0 = 1.0;
	*alpha = 1.0;
	*omega0 = 1.0;

	//Praparing nu0,p0
	double *nu0, *p0;
	cudaMalloc((void **)&nu0, sizeof(double)*(p_size));
	cudaMalloc((void **)&p0, sizeof(double)*(p_size));
	cudaMemset(nu0, 0, sizeof(double)*(p_size));
	cudaMemset(p0, 0, sizeof(double)*(p_size));

	//Preparing rhoK, omegaK, beta
	double *rhoK = new double;
	double *omegaK = new double;
	double *beta = new double;

	//Preparing pK, nuK, s, t, xK, rK, h
	double **nuK = new double *[gpu],
		**t = new double *[gpu],
		**vec_temp6 = new double *[gpu],
		*xK, *rK, *h, *s, *pK, *y, *z;
#pragma omp parallel num_threads(gpu)
	{
		cudaSetDevice(omp_get_thread_num());
		cudaMalloc((void **)&nuK[omp_get_thread_num()], sizeof(double)*(p_size));
		cudaMalloc((void **)&t[omp_get_thread_num()], sizeof(double)*(p_size));
		cudaMalloc((void **)&vec_temp6[omp_get_thread_num()], sizeof(double)*(p_size));
	}

	cudaSetDevice(0);
	cudaMalloc((void **)&s, sizeof(double)*(p_size));
	cudaMalloc((void **)&pK, sizeof(double)*(p_size));
	cudaMalloc((void **)&xK, sizeof(double)*(p_size));
	cudaMalloc((void **)&rK, sizeof(double)*(p_size));
	cudaMalloc((void **)&h, sizeof(double)*(p_size));
	cudaMallocManaged((void **)&y, sizeof(double)*(p_size));
	cudaMallocManaged((void **)&z, sizeof(double)*(p_size));

	//Preparing diag and "reversed" diag
	double *d_diag, *d_revDiag;
	cudaMalloc((void **)&d_diag, sizeof(double)*(p_size));
	cudaMalloc((void **)&d_revDiag, sizeof(double)*(p_size));
	cudaMemcpy(d_diag, p_diag, sizeof(double)*(p_size), cudaMemcpyHostToDevice);
	diRev << <GPU_BLOCKS(p_size, deviceProp.maxThreadsPerBlock), deviceProp.maxThreadsPerBlock >> > (p_size, d_diag, d_revDiag);
	cudaDeviceSynchronize();

	//Preparing temporary variables.
	//There are 2 types of temporary variables: vec_ and num_
	double *num_temp1 = new double,
		*num_temp2 = new double,
		*num_temp6 = new double,
		*rKnrm = new double;
	double *vec_temp1, *vec_temp2, *vec_temp3, *vec_temp4, *vec_temp5 = new double[p_size];
	cudaMalloc((void **)&vec_temp1, sizeof(double)*(p_size));
	cudaMalloc((void **)&vec_temp2, sizeof(double)*(p_size));
	cudaMalloc((void **)&vec_temp3, sizeof(double)*(p_size));
	cudaMalloc((void **)&vec_temp4, sizeof(double)*(p_size));

	//Others
	double *negalpha = new double;
	double *negomega = new double;
	*negomega = -(*omega0);

	double *bnrm = new double;
	if (!residue) {
		cublasDnrm2(cublas_handle, p_size, r0, 1, bnrm);
	}

	double *best;
	double *bestNrm = new double;
	int bestSt = 0;
	if (myBest) {
		cudaMalloc((void **)&best, sizeof(double)*(p_size));
		*bestNrm = 1000.0;
	}

	do {
		if (!flag) {
			nu0 = nuK[0];
			p0 = pK;
			omega0 = omegaK;
			*negomega = -(*omega0);
			*rho0 = *rhoK;
		}

		//Step 1
		cublasDdot(cublas_handle, p_size, rT, 1, r0, 1, rhoK);

		//Step 2
		if (stab) {
			if ((abs(*rho0) < maxacc) || (abs(*omega0) < maxacc)) {
				*rho0 = sgn(*rho0) * 1e-7;
				*omega0 = sgn(*omega0) * 1e-7;
			}
		}
		*beta = ((*rhoK) * (*alpha)) / ((*rho0) * (*omega0));

		//Step 3
		cublasDaxpy(cublas_handle, p_size, negomega, nu0, 1, p0, 1);
		cublasDcopy(cublas_handle, p_size, r0, 1, vec_temp1, 1);
		cublasDaxpy(cublas_handle, p_size, beta, p0, 1, vec_temp1, 1);
		cublasDcopy(cublas_handle, p_size, vec_temp1, 1, pK, 1);

		//Step 4
		vec2vec << <GPU_BLOCKS(p_size, deviceProp.maxThreadsPerBlock), deviceProp.maxThreadsPerBlock >> > (p_size, d_revDiag, pK, y);
		cudaDeviceSynchronize();
		if (stab) {
			if (*y < maxacc) {
				*y = sgn(*y) * 1e-7;
			}
		}

		//Step 5
		//A = A1+A2+...+An => A*x = A1*x+A2*x+...+An*x
		//We use all GPU's but then we send everything to first one
		//And don't forget to add diag
#pragma omp parallel num_threads(gpu)
		{
			cudaSetDevice(omp_get_thread_num());
			cudaMemcpy(vec_temp6[omp_get_thread_num()], y, sizeof(double)*(p_size), cudaMemcpyDefault);
			cusparseDcsrmv(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
				p_size, p_size, sizes[omp_get_thread_num()],
				one, matDescr, d_aelem[omp_get_thread_num()],
				d_iptr[omp_get_thread_num()], d_jptr[omp_get_thread_num()],
				vec_temp6[omp_get_thread_num()], zero, nuK[omp_get_thread_num()]);
		}

		for (int i = 1; i < gpu; i++) {
			cudaSetDevice(i);
			cudaMemcpy(vec_temp5, nuK[i], sizeof(double)*p_size, cudaMemcpyDeviceToHost);
			cudaSetDevice(0);
			cudaMemcpy(vec_temp4, vec_temp5, sizeof(double)*p_size, cudaMemcpyHostToDevice);
			cublasDaxpy(cublas_handle, p_size, one, vec_temp4, 1, nuK[0], 1);
		}

		//I don't know how match blocks and threads I can launch
		diVec << <GPU_BLOCKS(p_size, deviceProp.maxThreadsPerBlock), deviceProp.maxThreadsPerBlock >> > (p_size, d_diag, y, nuK[0]);
		cudaDeviceSynchronize();

		//Step 6
		cublasDdot(cublas_handle, p_size, rT, 1, nuK[0], 1, num_temp1);
		if (stab) {
			if (abs(*num_temp1) < maxacc) {
				*num_temp1 = sgn(*num_temp1) * 1e-7;
			}
		}
		*alpha = (*rhoK) / (*num_temp1);
		*negalpha = -(*alpha);

		//Step 7
		cublasDaxpy(cublas_handle, p_size, alpha, y, 1, x0, 1);
		h = x0;

		//Step 8
		cublasDaxpy(cublas_handle, p_size, negalpha, nuK[0], 1, r0, 1);
		s = r0;

		//Step 9
		vec2vec << <GPU_BLOCKS(p_size, deviceProp.maxThreadsPerBlock), deviceProp.maxThreadsPerBlock >> > (p_size, d_revDiag, s, z);
		cudaDeviceSynchronize();

		//Step 10
		//A = A1+A2+...+An => A*x = A1*x+A2*x+...+An*x
		//We use all GPU's but then we send everything to first one
		//And don't forget to add diag
#pragma omp parallel num_threads(gpu)
		{
			cudaSetDevice(omp_get_thread_num());
			cudaMemcpy(vec_temp6[omp_get_thread_num()], z, sizeof(double)*(p_size), cudaMemcpyDefault);
			cusparseDcsrmv(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
				p_size, p_size, sizes[omp_get_thread_num()],
				one, matDescr, d_aelem[omp_get_thread_num()],
				d_iptr[omp_get_thread_num()], d_jptr[omp_get_thread_num()],
				vec_temp6[omp_get_thread_num()], zero, t[omp_get_thread_num()]);
		}

		for (int i = 1; i < gpu; i++) {
			cudaSetDevice(i);
			cudaMemcpy(vec_temp5, t[i], sizeof(double)*p_size, cudaMemcpyDeviceToHost);
			cudaSetDevice(0);
			cudaMemcpy(vec_temp4, vec_temp5, sizeof(double)*p_size, cudaMemcpyHostToDevice);
			cublasDaxpy(cublas_handle, p_size, one, vec_temp4, 1, t[0], 1);
		}
		diVec << <GPU_BLOCKS(p_size, deviceProp.maxThreadsPerBlock), deviceProp.maxThreadsPerBlock >> > (p_size, d_diag, z, t[0]);
		cudaDeviceSynchronize();
		if (stab) {
			if (*z < maxacc) {
				*z = sgn(*z) * 1e-7;
			}
		}

		//Step 11
		vec2vec << <GPU_BLOCKS(p_size, deviceProp.maxThreadsPerBlock), deviceProp.maxThreadsPerBlock >> > (p_size, d_revDiag, t[0], vec_temp3);
		cudaDeviceSynchronize();
		cublasDdot(cublas_handle, p_size, vec_temp3, 1, z, 1, num_temp6);
		cublasDdot(cublas_handle, p_size, vec_temp3, 1, vec_temp3, 1, num_temp2);
		if (stab) {
			if (abs(*num_temp2) < maxacc) {
				*num_temp2 = sgn(*num_temp2) * 1e-7;
			}
		}
		*omegaK = (*num_temp6) / (*num_temp2);
		*negomega = -(*omegaK);

		//Step 12
		cublasDaxpy(cublas_handle, p_size, omegaK, z, 1, h, 1);
		xK = h;

		//Step 13
		cublasDaxpy(cublas_handle, p_size, negomega, t[0], 1, s, 1);
		rK = s;

		//Step 14 (check)
		cublasDnrm2(cublas_handle, p_size, rK, 1, rKnrm);

		if (myBest) {
			if (residue) {
				if (*bestNrm > *rKnrm) {
					*bestNrm = *rKnrm;
				}
			}
			else {
				if (*bestNrm > (*rKnrm) / (*bnrm)) {
					*bestNrm = (*rKnrm) / (*bnrm);
				}
			}
			cublasDcopy(cublas_handle, p_size, xK, 1, best, 1);
			bestSt = step;
		}

		if (myCheck) {
			if (*rKnrm != *rKnrm) {
				cout << "Nan detected. Aborting..." << endl;
				break;
			}

			if (step % 20 == 0) {
				if (myBest) {
					if (residue) {
						cout << "Abs res: " << *rKnrm << " with abs best: " << *bestNrm << " on " << bestSt << " step" << endl;
					}
					else {
						cout << "Rel res: " << (*rKnrm) / (*bnrm) << " with rel best: " << *bestNrm << " on " << bestSt << " step" << endl;
					}
				}
				else {
					if (residue) {
						cout << "Abs res: " << *rKnrm << endl;
					}
					else {
						cout << "Rel res: " << (*rKnrm) / (*bnrm) << endl;
					}
				}
			}
		}

		if (residue) {
			if (omega) {
				if (((*rKnrm) < maxacc) || (abs(*omegaK) <= maxacc)) {
					break;
				}
			}
			else {
				if ((*rKnrm) < maxacc) {
					break;
				}
			}
		}
		else {
			if (omega) {
				if (((*rKnrm) / (*bnrm) < maxacc) || (abs(*omegaK) <= maxacc)) {
					break;
				}
			}
			else {
				if ((*rKnrm) / (*bnrm) < maxacc) {
					break;
				}
			}
		}

		flag = false;
		step++;

	} while (step <= maxiter);

	if (myBest) {
		cublasDcopy(cublas_handle, p_size, best, 1, xK, 1);
	}

	//Check || Ax - b || or || Ax - b || / || b ||
	if (myCheck) {
		cudaMemcpy(vec_temp5, xK, sizeof(double)*(p_size), cudaMemcpyDefault);
#pragma omp parallel num_threads(gpu)
		{
			cudaSetDevice(omp_get_thread_num());
			cudaMemcpy(vec_temp6[omp_get_thread_num()], vec_temp5, sizeof(double)*(p_size), cudaMemcpyDefault);
			cusparseDcsrmv(cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
				p_size, p_size, sizes[omp_get_thread_num()],
				one, matDescr, d_aelem[omp_get_thread_num()],
				d_iptr[omp_get_thread_num()], d_jptr[omp_get_thread_num()],
				vec_temp6[omp_get_thread_num()], zero, t[omp_get_thread_num()]);
		}
		for (int i = 1; i < gpu; i++) {
			cudaSetDevice(i);
			cudaMemcpy(vec_temp4, t[i], sizeof(double)*(p_size), cudaMemcpyDeviceToHost);
			cudaSetDevice(0);
			cudaMemcpy(vec_temp3, vec_temp4, sizeof(double)*(p_size), cudaMemcpyHostToDevice);
			cublasDaxpy(cublas_handle, p_size, one, vec_temp3, 1, t[0], 1);
		}
		diVec << <GPU_BLOCKS(p_size, deviceProp.maxThreadsPerBlock), deviceProp.maxThreadsPerBlock >> > (p_size, d_diag, xK, t[0]);
		cudaDeviceSynchronize();
		cudaMemcpy(vec_temp3, p_right, sizeof(double)*(p_size), cudaMemcpyHostToDevice);
		cublasDaxpy(cublas_handle, p_size, minus, vec_temp3, 1, t[0], 1);
		cublasDnrm2(cublas_handle, p_size, t[0], 1, rKnrm);

		if (residue) {
			cout << "||Ax-b||: " << *rKnrm;
		}
		else {
			cout << "||Ax-b|| / ||b||: " << (*rKnrm) / (*bnrm);
		}

		cout << endl << "Steps: " << step << endl;

		if (myBest) {
			cout << "Best result on " << bestSt << " step" << endl;
		}
	}

	//Transferring result from device to host
	double *res = new double[p_size];
	cudaMemcpy(res, xK, sizeof(double)*p_size, cudaMemcpyDeviceToHost);

	//Clean all
	cublasDestroy(cublas_handle);
	cusparseDestroy(cusparse_handle);
	cusparseDestroyMatDescr(matDescr);

	cudaFree(x0);
	cudaFree(r0);
	cudaFree(rT);
	cudaFree(nu0);
	cudaFree(pK);
	cudaFree(d_diag);
	cudaFree(d_revDiag);
	cudaFree(vec_temp1);
	cudaFree(vec_temp2);
	cudaFree(vec_temp3);
	cudaFree(vec_temp4);
	cudaFree(z);
	cudaFree(y);

	delete minus, one, zero, rhoK,
		beta, alpha, omega0, omegaK,
		negalpha, negomega, rho0, num_temp1,
		num_temp2, num_temp6, rKnrm;

#pragma omp parallel num_threads(gpu)
	{
		cudaSetDevice(omp_get_thread_num());
		cudaFree(t[omp_get_thread_num()]);
		cudaFree(d_aelem[omp_get_thread_num()]);
		cudaFree(d_jptr[omp_get_thread_num()]);
		cudaFree(d_iptr[omp_get_thread_num()]);
		cudaFree(vec_temp6[omp_get_thread_num()]);
	}

	if (gpu > 1) {
#pragma omp parallel num_threads(gpu - 1)
		{
			cudaSetDevice(omp_get_thread_num() + 1);
			cudaFree(nuK[omp_get_thread_num() + 1]);
		}
	}

	delete[] sizes, d_aelem, d_iptr, d_jptr, t, nuK, vec_temp5, vec_temp6;

	return res;
}
