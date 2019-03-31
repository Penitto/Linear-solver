#include <cuda.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <cublas_api.h>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>

#include <helper_cuda.h>
//#include <helper_cuda_drvapi.h>
#include <helper_functions.h>
#include <helper_cusolver.h>

#include <device_launch_parameters.h>

#include "GPU.h"

int* split(int gpu_amount, double* A, int* B, int* C, int size, int non_zero, double **d_A, int ** d_B, int **d_C);

int return_string(int number, int* C)
{
	int i = 0;
	while (C[i] <= number)
		i++;
	return i;
}

__global__ void diVec(int size, double *diag, double *vec, double *rez)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < size)
	{
		rez[i] += diag[i] * vec[i];
	}
}

__global__ void diRev(int size, double *diag, double *rez)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < size)
	{
		rez[i] = 1 / diag[i];
	}
}

__global__ void vec2vec(int size, double *vec1, double *vec2, double *rez)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < size)
	{
		rez[i] = vec1[i] * vec2[i];
	}
}

int sgn(double val)
{
	if (val == 0.)  return 0;
	if (val > 0.)  return 1;
	else return -1;
}

int* split(int gpu_amount, double* val, int* col, int* row, int size, int non_zero, double **d_val, int ** d_col, int **d_row) // Костляво
{
	int mod = non_zero / gpu_amount; // уходит на все 
	int rest = non_zero - mod*(gpu_amount - 1); //уходит на последнюю 
	int first_position;
	int last_position;
	int first_string;
	int last_string;
	double *val_;
	int *col_;
	int *row_;

	int *temp = new int[gpu_amount];
	int nsize;

	for (int number = 0; number < gpu_amount; number++)
	{
		if (number == gpu_amount - 1)
		{
			int in1 = 0;
			int in2 = 0;
			first_position = number*mod;//n 
			last_position = non_zero - 1;//k 
			first_string = return_string(number*mod, row) - 1; //i 
			last_string = return_string(non_zero - 1, row) - 1;//j
			nsize = rest + first_string + size - 1 - last_string;

			val_ = new double[nsize]; // definition 
			for (int i = 0; i < nsize; i++)
			{
				if (i < first_string)
				{
					val_[i] = 0;
				}
				else
				{
					val_[i] = val[first_position + in1];
					in1++;
				}
			}

			col_ = new int[nsize];
			for (int i = 0; i < nsize; i++)
			{
				if (i < first_string)
				{
					col_[i] = i;
				}
				else
				{
					col_[i] = col[first_position + in2];
					in2++;
				}
			}

			row_ = new int[size + 1];

			for (int i = 0; i < first_string; i++) //0123..C..000 
				row_[i] = i;
			for (int count = first_string; count <= last_string; count++)
			{
				row_[count] = row[count] - first_position + first_string;
				if (row[count] - first_position < 0) row_[count] = first_string;
			}
			row_[size] = nsize;

		}
		else
		{
			int in1 = 0;
			int in2 = 0;
			first_position = number*mod;// n 
			last_position = (number + 1)*mod - 1;//k 
			first_string = return_string(number*mod, row) - 1; //i 
			last_string = return_string((number + 1)*mod - 1, row) - 1;//j 
			nsize = mod + first_string + size - 1 - last_string;

			val_ = new double[nsize]; // definition 
			for (int i = 0; i < nsize; i++)
			{
				if ((i < first_string) || (i > first_string + mod - 1))
				{
					val_[i] = 0;
				}
				else
				{
					val_[i] = val[first_position + in1];
					in1++;
				}
			}

			col_ = new int[nsize];

			int inn = 1;
			for (int i = 0; i < nsize; i++)
			{
				if (i < first_string)
				{
					col_[i] = i;
				}
				else if (i < first_string + mod)
				{
					col_[i] = col[first_position + in2];
					in2++;
				}
				else
				{
					col_[i] = last_string + inn;
					inn++;
				}
			}

			row_ = new int[size + 1];

			for (int i = 0; i < first_string; i++) //0123..C..000 
				row_[i] = i;
			for (int count = first_string; count <= last_string; count++)
			{
				row_[count] = row[count] - first_position + first_string;
				if (row[count] - first_position < 0) row_[count] = first_string;
			}
			int l = 1;
			for (int i = last_string + 1; i < size; i++) //0123..C..n.. 
			{
				row_[i] = first_string + last_position - first_position + l;
				l++;
			}
			row_[size] = nsize;

		}

		temp[number] = nsize;

		checkCudaErrors(cudaSetDevice(number));
		checkCudaErrors(cudaMalloc((void **)&d_val[number], sizeof(double)*nsize));
		checkCudaErrors(cudaMalloc((void **)&d_col[number], sizeof(int)*nsize));
		checkCudaErrors(cudaMalloc((void **)&d_row[number], sizeof(int)*(size + 1)));
		checkCudaErrors(cudaMemcpy(d_val[number], val_, sizeof(double)*nsize, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_col[number], col_, sizeof(int)*nsize, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_row[number], row_, sizeof(int)*(size + 1), cudaMemcpyHostToDevice));

		delete[] val_;
		delete[] col_;
		delete[] row_;
	}
	return temp;
}

double* gpu_solver::GPU_CG(double *val, int *col, int *row, double *right, double *diag, int nnz, int size)
{
	//Count amount of devices
	int gpu;
	checkCudaErrors(cudaGetDeviceCount(&gpu));

	//Arrays for devices
	double ** d_val = new  double *[gpu];
	int ** d_col = new int *[gpu];
	int ** d_row = new int *[gpu];

	//Array with device array's sizes
	int *sizes = new int[gpu];
	sizes = split(gpu, val, col, row, size, nnz, d_val, d_col, d_row);

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
	checkCudaErrors(cudaSetDevice(0));

	//Preparing CUBLAS handle
	cublasHandle_t cublasHandle = NULL;
	checkCudaErrors(cublasCreate(&cublasHandle));

	//Preparing CUSPARSE handle
	cusparseHandle_t cusparseHandle = NULL;
	checkCudaErrors(cusparseCreate(&cusparseHandle));

	//Preparing matrix descriptor
	cusparseMatDescr_t matDescr = NULL;
	checkCudaErrors(cusparseCreateMatDescr(&matDescr));

	//Set base for matrix
	if (row[0]) {
		checkCudaErrors(cusparseSetMatIndexBase(matDescr, CUSPARSE_INDEX_BASE_ONE));
	} else {
		checkCudaErrors(cusparseSetMatIndexBase(matDescr, CUSPARSE_INDEX_BASE_ZERO));
	}

	//Prepating device information
	cudaDeviceProp deviceProp;
	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, 0));
	//deviceProp.maxThreadsPerBlock
	//deviceProp.maxGridSize[0]
	//blocks, threads

	//Preparing x0
	double *x0;
	checkCudaErrors(cudaMalloc((void **)&x0, sizeof(double)*(size)));
	checkCudaErrors(cudaMemset(x0, 0.0, sizeof(double)*(size)));

	//Preparing r0 and p0
	double *r0, *p0;
	checkCudaErrors(cudaMalloc((void **)&r0, sizeof(double)*(size)));
	checkCudaErrors(cudaMallocManaged((void **)&p0, sizeof(double)*(size)));
	//checkCudaErrors(cublasDcopy(cublasHandle, size, right));
	checkCudaErrors(cudaMemcpy(r0, right, sizeof(double)*(size), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(p0, right, sizeof(double)*(size), cudaMemcpyHostToDevice));

	//Preparing alpha and beta
	double *alpha = new double, *beta = new double, *negalpha = new double;

	//Preparing pK, nuK, s, t, xK, rK, h
	double **vec_temp1 = new double *[gpu],
		**vec_temp5 = new double *[gpu],
		*xK, *rK, *pK;
#pragma omp parallel num_threads(gpu)
	{
		checkCudaErrors(cudaSetDevice(omp_get_thread_num()));
		checkCudaErrors(cudaMalloc((void **)&vec_temp1[omp_get_thread_num()], sizeof(double)*(size)));
		checkCudaErrors(cudaMalloc((void **)&vec_temp5[omp_get_thread_num()], sizeof(double)*(size)));
	}

	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaMalloc((void **)&xK, sizeof(double)*(size)));
	checkCudaErrors(cudaMalloc((void **)&rK, sizeof(double)*(size)));
	checkCudaErrors(cudaMallocManaged((void **)&pK, sizeof(double)*(size)));

	//Preparing diag for devices
	double *d_diag;
	checkCudaErrors(cudaMalloc((void **)&d_diag, sizeof(double)*(size)));
	checkCudaErrors(cudaMemcpy(d_diag, diag, sizeof(double)*(size), cudaMemcpyHostToDevice));

	//Preparing temporary variables.
	//There are 2 types of temporary variables: vec_ and num_
	double *r0nrm = new double,
		*rKnrm = new double,
		*num_temp1 = new double;
	double *vec_temp2, *vec_temp3, *vec_temp4 = new double[size];
	checkCudaErrors(cudaMalloc((void **)&vec_temp2, sizeof(double)*(size)));
	checkCudaErrors(cudaMalloc((void **)&vec_temp3, sizeof(double)*(size)));

#if !RESIDUE
	double *bnrm = new double;
	checkCudaErrors(cublasDnrm2(cublasHandle, size, r0, 1, bnrm));
#endif

	do
	{
		if (!flag)
		{
			p0 = pK;
		}

		//Step 1
		//A = A1+A2+...+An => A*x = A1*x+A2*x+...+An*x
		//We use all GPU's but then we send everything to first one
		//And don't forget to add diag
		checkCudaErrors(cublasDdot(cublasHandle, size, r0, 1, r0, 1, alpha));

#pragma omp parallel num_threads(gpu)
		{
			checkCudaErrors(cudaSetDevice(omp_get_thread_num()));
			checkCudaErrors(cudaMemcpy(vec_temp5[omp_get_thread_num()], p0, sizeof(double)*(size), cudaMemcpyDefault));
			checkCudaErrors(cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
				size, size, sizes[omp_get_thread_num()],
				one, matDescr, d_val[omp_get_thread_num()],
				d_row[omp_get_thread_num()], d_col[omp_get_thread_num()],
				vec_temp5[omp_get_thread_num()], zero, vec_temp1[omp_get_thread_num()]));
		}

		for (int i = 1; i < gpu; i++)
		{
			checkCudaErrors(cudaSetDevice(i));
			//checkCudaErrors(cublasDcopy(cublasHandle, size, vec_temp1[i], 1, vec_temp4, 1));
			checkCudaErrors(cudaMemcpy(vec_temp4, vec_temp1[i], sizeof(double)*(size), cudaMemcpyDefault));
			checkCudaErrors(cudaSetDevice(0));
			checkCudaErrors(cudaMemcpy(vec_temp3, vec_temp4, sizeof(double)*(size), cudaMemcpyHostToDevice));
			checkCudaErrors(cublasDaxpy(cublasHandle, size, one, vec_temp3, 1, vec_temp1[0], 1));
		}

		//I don't know how match blocks and threads I can launch
		diVec << <GPU_BLOCKS(size, deviceProp.maxThreadsPerBlock), deviceProp.maxThreadsPerBlock >> >(size, d_diag, p0, vec_temp1[0]);
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cublasDdot(cublasHandle, size, vec_temp1[0], 1, p0, 1, num_temp1));
		*alpha /= (*num_temp1);
		*negalpha = -(*alpha);

		//Step 2
		checkCudaErrors(cublasDaxpy(cublasHandle, size, alpha, p0, 1, x0, 1));
		xK = x0;

		//Step 3
		checkCudaErrors(cublasDnrm2(cublasHandle, size, r0, 1, r0nrm));
		checkCudaErrors(cublasDaxpy(cublasHandle, size, negalpha, vec_temp1[0], 1, r0, 1));
		rK = r0;

		//Step 4 (check)
		checkCudaErrors(cublasDnrm2(cublasHandle, size, rK, 1, rKnrm));

#if RESIDUE
		if ((*rKnrm) < MAXACC)
#else
		if ((*rKnrm)/(*bnrm) < MAXACC)
#endif
		{
			break;
		}

		//Step 5
		*beta = pow((*rKnrm), 2) / pow((*r0nrm), 2);

		//Step 6
		checkCudaErrors(cublasDcopy(cublasHandle, size, rK, 1, vec_temp2, 1));
		checkCudaErrors(cublasDaxpy(cublasHandle, size, beta, p0, 1, vec_temp2, 1));
		checkCudaErrors(cublasDcopy(cublasHandle, size, vec_temp2, 1, pK, 1));

		flag = false;
		step++;

	} while (step <= MAXITER);

	//Check || Ax - b ||
	checkCudaErrors(cudaMemcpy(vec_temp4, xK, sizeof(double)*(size), cudaMemcpyDefault));
#pragma omp parallel num_threads(gpu)
	{
		checkCudaErrors(cudaSetDevice(omp_get_thread_num()));
		checkCudaErrors(cudaMemcpy(vec_temp5[omp_get_thread_num()], vec_temp4, sizeof(double)*(size), cudaMemcpyDefault));
		checkCudaErrors(cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
			size, size, sizes[omp_get_thread_num()],
			one, matDescr, d_val[omp_get_thread_num()],
			d_row[omp_get_thread_num()], d_col[omp_get_thread_num()],
			vec_temp5[omp_get_thread_num()], zero, vec_temp1[omp_get_thread_num()]));

	}
	for (int i = 1; i < gpu; i++)
	{
		checkCudaErrors(cudaSetDevice(i));
		checkCudaErrors(cudaMemcpy(vec_temp4, vec_temp1[i], sizeof(double)*(size), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaSetDevice(0));
		checkCudaErrors(cudaMemcpy(vec_temp3, vec_temp4, sizeof(double)*(size), cudaMemcpyHostToDevice));
		checkCudaErrors(cublasDaxpy(cublasHandle, size, one, vec_temp3, 1, vec_temp1[0], 1));
	}
	diVec << <GPU_BLOCKS(size, deviceProp.maxThreadsPerBlock), deviceProp.maxThreadsPerBlock >> >(size, d_diag, xK, vec_temp1[0]);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(vec_temp3, right, sizeof(double)*(size), cudaMemcpyHostToDevice));
	checkCudaErrors(cublasDaxpy(cublasHandle, size, minus, vec_temp3, 1, vec_temp1[0], 1));
	checkCudaErrors(cublasDnrm2(cublasHandle, size, vec_temp1[0], 1, rKnrm));

	cout << "Residue: " << *rKnrm;
	cout << endl << "Steps: " << step << endl;

	double *res = new double[size];
	checkCudaErrors(cudaMemcpy(res, xK, sizeof(double)*size, cudaMemcpyDeviceToHost));

	checkCudaErrors(cublasDestroy(cublasHandle));
	checkCudaErrors(cusparseDestroy(cusparseHandle));
	checkCudaErrors(cusparseDestroyMatDescr(matDescr));

	delete minus, one, zero, alpha, beta, negalpha, r0nrm, rKnrm, num_temp1;

	checkCudaErrors(cudaFree(x0));
	checkCudaErrors(cudaFree(r0));
	checkCudaErrors(cudaFree(p0));
	checkCudaErrors(cudaFree(vec_temp2));
	checkCudaErrors(cudaFree(vec_temp3));
	checkCudaErrors(cudaFree(d_diag));

#pragma omp parallel num_threads(gpu)
	{
		checkCudaErrors(cudaSetDevice(omp_get_thread_num()));
		checkCudaErrors(cudaFree(vec_temp1[omp_get_thread_num()]));
		checkCudaErrors(cudaFree(vec_temp5[omp_get_thread_num()]));
		checkCudaErrors(cudaFree(d_val[omp_get_thread_num()]));
		checkCudaErrors(cudaFree(d_col[omp_get_thread_num()]));
		checkCudaErrors(cudaFree(d_row[omp_get_thread_num()]));

	}

	delete[] sizes, d_val, d_row, d_col, vec_temp4, vec_temp1, vec_temp5;

	return res;
}

double* gpu_solver::GPU_PCG(double *val, int *col, int *row, double *right, double *diag, int nnz, int size)
{
	//Count amount of devices
	int gpu;
	checkCudaErrors(cudaGetDeviceCount(&gpu));

	//Arrays for devices
	double ** d_val = new  double *[gpu];
	int ** d_col = new int *[gpu];
	int ** d_row = new int *[gpu];

	//Array with device array's sizes
	int *sizes = new int[gpu];
	sizes = split(gpu, val, col, row, size, nnz, d_val, d_col, d_row);

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
	checkCudaErrors(cudaSetDevice(0));

	//Preparing CUBLAS handle
	cublasHandle_t cublasHandle = NULL;
	checkCudaErrors(cublasCreate(&cublasHandle));

	//Preparing CUSPARSE handle
	cusparseHandle_t cusparseHandle = NULL;
	checkCudaErrors(cusparseCreate(&cusparseHandle));

	//Preparing matrix descriptor
	cusparseMatDescr_t matDescr = NULL;
	checkCudaErrors(cusparseCreateMatDescr(&matDescr));

	//Set base for matrix
	if (row[0]) {
		checkCudaErrors(cusparseSetMatIndexBase(matDescr, CUSPARSE_INDEX_BASE_ONE));
	}
	else {
		checkCudaErrors(cusparseSetMatIndexBase(matDescr, CUSPARSE_INDEX_BASE_ZERO));
	}

	//Prepating device information
	cudaDeviceProp deviceProp;
	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, 0));
	//deviceProp.maxThreadsPerBlock
	//deviceProp.maxGridSize[0]
	//blocks, threads

	//Preparing diag
	double *d_diag, *d_revDiag;
	checkCudaErrors(cudaMalloc((void **)&d_diag, sizeof(double)*(size)));
	checkCudaErrors(cudaMalloc((void **)&d_revDiag, sizeof(double)*(size)));
	checkCudaErrors(cudaMemcpy(d_diag, diag, sizeof(double)*(size), cudaMemcpyHostToDevice));
	diRev << <GPU_BLOCKS(size, deviceProp.maxThreadsPerBlock), deviceProp.maxThreadsPerBlock >> > (size, d_diag, d_revDiag);
	checkCudaErrors(cudaDeviceSynchronize());

	//Preparing x0
	double *x0;
	checkCudaErrors(cudaMalloc((void **)&x0, sizeof(double)*(size)));
	checkCudaErrors(cudaMemset(x0, 0.0, sizeof(double)*(size)));

	//Preparing r0 and p0
	double *r0, *p0, *z0;
	checkCudaErrors(cudaMalloc((void **)&r0, sizeof(double)*(size)));
	checkCudaErrors(cudaMallocManaged((void **)&p0, sizeof(double)*(size)));
	checkCudaErrors(cudaMalloc((void **)&z0, sizeof(double)*(size)));
	checkCudaErrors(cudaMemcpy(r0, right, sizeof(double)*(size), cudaMemcpyHostToDevice));
	vec2vec << <GPU_BLOCKS(size, deviceProp.maxThreadsPerBlock), deviceProp.maxThreadsPerBlock >> > (size, d_revDiag, r0, z0);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(p0, z0, sizeof(double)*(size), cudaMemcpyHostToDevice));

	//Preparing alpha and beta
	double *alpha = new double, *beta = new double, *negalpha = new double;

	//Preparing pK, nuK, s, t, xK, rK, h
	double **vec_temp1 = new double *[gpu],
		**vec_temp5 = new double *[gpu],
		*xK, *rK, *pK, *zK;
#pragma omp parallel num_threads(gpu)
	{
		checkCudaErrors(cudaSetDevice(omp_get_thread_num()));
		checkCudaErrors(cudaMalloc((void **)&vec_temp1[omp_get_thread_num()], sizeof(double)*(size)));
		checkCudaErrors(cudaMalloc((void **)&vec_temp5[omp_get_thread_num()], sizeof(double)*(size)));

	}

	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaMalloc((void **)&xK, sizeof(double)*(size)));
	checkCudaErrors(cudaMalloc((void **)&rK, sizeof(double)*(size)));
	checkCudaErrors(cudaMallocManaged((void **)&pK, sizeof(double)*(size)));
	checkCudaErrors(cudaMalloc((void **)&zK, sizeof(double)*(size)));

	//Preparing temporary variables.
	//There are 2 types of temporary variables: vec_ and num_
	double *num_temp1 = new double,
		*rKnrm = new double,
		*num_temp3 = new double;
	double *vec_temp2, *vec_temp3, *vec_temp4 = new double[size];
	checkCudaErrors(cudaMalloc((void **)&vec_temp2, sizeof(double)*(size)));
	checkCudaErrors(cudaMalloc((void **)&vec_temp3, sizeof(double)*(size)));

#if !RESIDUE
	double *bnrm = new double;
	checkCudaErrors(cublasDnrm2(cublasHandle, size, r0, 1, bnrm));
#endif

	do
	{
		if (!flag)
		{
			p0 = pK;
			z0 = zK;
		}

		//Step 1
		//A = A1+A2+...+An => A*x = A1*x+A2*x+...+An*x
		//We use all GPU's but then we send everything to first one
		//And don't forget to add diag
		checkCudaErrors(cublasDdot(cublasHandle, size, r0, 1, z0, 1, alpha));
#pragma omp parallel num_threads(gpu)
		{
			checkCudaErrors(cudaSetDevice(omp_get_thread_num()));
			checkCudaErrors(cudaMemcpy(vec_temp5[omp_get_thread_num()], p0, sizeof(double)*(size), cudaMemcpyDefault));
			checkCudaErrors(cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
				size, size, sizes[omp_get_thread_num()],
				one, matDescr, d_val[omp_get_thread_num()],
				d_row[omp_get_thread_num()], d_col[omp_get_thread_num()],
				vec_temp5[omp_get_thread_num()], zero, vec_temp1[omp_get_thread_num()]));

		}
		for (int i = 1; i < gpu; i++)
		{
			checkCudaErrors(cudaSetDevice(i));
			checkCudaErrors(cudaMemcpy(vec_temp4, vec_temp1[i], sizeof(double)*(size), cudaMemcpyDeviceToHost));
			checkCudaErrors(cudaSetDevice(0));
			checkCudaErrors(cudaMemcpy(vec_temp3, vec_temp4, sizeof(double)*(size), cudaMemcpyHostToDevice));
			checkCudaErrors(cublasDaxpy(cublasHandle, size, one, vec_temp3, 1, vec_temp1[0], 1));
		}

		//I don't know how match blocks and threads I can launch
		diVec << <GPU_BLOCKS(size, deviceProp.maxThreadsPerBlock), deviceProp.maxThreadsPerBlock >> >(size, d_diag, p0, vec_temp1[0]);
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cublasDdot(cublasHandle, size, vec_temp1[0], 1, p0, 1, num_temp1));
		*alpha /= (*num_temp1);
		*negalpha = -(*alpha);

		//Step 2
		checkCudaErrors(cublasDaxpy(cublasHandle, size, alpha, p0, 1, x0, 1));
		xK = x0;

		//Step 3
		checkCudaErrors(cublasDdot(cublasHandle, size, z0, 1, r0, 1, num_temp3));
		checkCudaErrors(cublasDaxpy(cublasHandle, size, negalpha, vec_temp1[0], 1, r0, 1));
		rK = r0;

		//Step 4 (check)
		checkCudaErrors(cublasDnrm2(cublasHandle, size, rK, 1, rKnrm));
#if RESIDUE
		if ((*rKnrm) < MAXACC)
#else
		if ((*rKnrm)/(*bnrm) < MAXACC)
#endif
		{
			break;
		}

		//Step 5
		vec2vec << <GPU_BLOCKS(size, deviceProp.maxThreadsPerBlock), deviceProp.maxThreadsPerBlock >> > (size, d_revDiag, rK, zK);
		checkCudaErrors(cudaDeviceSynchronize());

		//Step 6
		checkCudaErrors(cublasDdot(cublasHandle, size, rK, 1, zK, 1, beta));
		*beta /= (*num_temp3);

		//Step 7
		checkCudaErrors(cublasDcopy(cublasHandle, size, zK, 1, vec_temp2, 1));
		checkCudaErrors(cublasDaxpy(cublasHandle, size, beta, p0, 1, vec_temp2, 1));
		checkCudaErrors(cublasDcopy(cublasHandle, size, vec_temp2, 1, pK, 1));

		flag = false;
		step++;

	} while (step <= MAXITER);

	//Check || Ax - b ||
	checkCudaErrors(cudaMemcpy(vec_temp4, xK, sizeof(double)*(size), cudaMemcpyDefault));
#pragma omp parallel num_threads(gpu)
	{
		checkCudaErrors(cudaSetDevice(omp_get_thread_num()));
		checkCudaErrors(cudaMemcpy(vec_temp5[omp_get_thread_num()], vec_temp4, sizeof(double)*(size), cudaMemcpyDefault));
		checkCudaErrors(cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
			size, size, sizes[omp_get_thread_num()],
			one, matDescr, d_val[omp_get_thread_num()],
			d_row[omp_get_thread_num()], d_col[omp_get_thread_num()],
			vec_temp5[omp_get_thread_num()], zero, vec_temp1[omp_get_thread_num()]));

	}
	for (int i = 1; i < gpu; i++)
	{
		checkCudaErrors(cudaSetDevice(i));
		checkCudaErrors(cudaMemcpy(vec_temp4, vec_temp1[i], sizeof(double)*(size), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaSetDevice(0));
		checkCudaErrors(cudaMemcpy(vec_temp3, vec_temp4, sizeof(double)*(size), cudaMemcpyHostToDevice));
		checkCudaErrors(cublasDaxpy(cublasHandle, size, one, vec_temp3, 1, vec_temp1[0], 1));
	}
	diVec << <GPU_BLOCKS(size, deviceProp.maxThreadsPerBlock), deviceProp.maxThreadsPerBlock >> >(size, d_diag, xK, vec_temp1[0]);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(vec_temp3, right, sizeof(double)*(size), cudaMemcpyHostToDevice));
	checkCudaErrors(cublasDaxpy(cublasHandle, size, minus, vec_temp3, 1, vec_temp1[0], 1));
	checkCudaErrors(cublasDnrm2(cublasHandle, size, vec_temp1[0], 1, rKnrm));

	cout << "Residue: " << *rKnrm;
	cout << endl << "Steps: " << step << endl;

	double *res = new double[size];
	checkCudaErrors(cudaMemcpy(res, xK, sizeof(double)*size, cudaMemcpyDeviceToHost));

	checkCudaErrors(cublasDestroy(cublasHandle));
	checkCudaErrors(cusparseDestroy(cusparseHandle));
	checkCudaErrors(cusparseDestroyMatDescr(matDescr));

	delete minus, one, zero, alpha, beta, negalpha, rKnrm, num_temp1, num_temp3;

	checkCudaErrors(cudaFree(x0));
	checkCudaErrors(cudaFree(r0));
	checkCudaErrors(cudaFree(p0));
	checkCudaErrors(cudaFree(z0));
	checkCudaErrors(cudaFree(vec_temp2));
	checkCudaErrors(cudaFree(vec_temp3));
	checkCudaErrors(cudaFree(d_diag));
	checkCudaErrors(cudaFree(d_revDiag));

#pragma omp parallel num_threads(gpu)
	{
		checkCudaErrors(cudaSetDevice(omp_get_thread_num()));
		checkCudaErrors(cudaFree(vec_temp1[omp_get_thread_num()]));
		checkCudaErrors(cudaFree(vec_temp5[omp_get_thread_num()]));
		checkCudaErrors(cudaFree(d_val[omp_get_thread_num()]));
		checkCudaErrors(cudaFree(d_col[omp_get_thread_num()]));
		checkCudaErrors(cudaFree(d_row[omp_get_thread_num()]));

	}

	delete[] sizes, d_val, d_row, d_col, vec_temp4, vec_temp1, vec_temp5;

	return res;
}

double* gpu_solver::GPU_BiCGSTAB(double *val, int *col, int *row, double *right, double *diag, int nnz, int size)
{
	//Count amount of devices
	int gpu;
	checkCudaErrors(cudaGetDeviceCount(&gpu));

	//Arrays for devices
	double ** d_val = new  double *[gpu];
	int ** d_col = new int *[gpu];
	int ** d_row = new int *[gpu];

	//Array with device array's sizes
	int *sizes = new int[gpu];
	sizes = split(gpu, val, col, row, size, nnz, d_val, d_col, d_row);

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
	checkCudaErrors(cudaSetDevice(0));

	//Preparing CUBLAS handle
	cublasHandle_t cublasHandle = NULL;
	checkCudaErrors(cublasCreate(&cublasHandle));

	//Preparing CUSPARSE handle
	cusparseHandle_t cusparseHandle = NULL;
	checkCudaErrors(cusparseCreate(&cusparseHandle));

	//Preparing matrix descriptor
	cusparseMatDescr_t matDescr = NULL;
	checkCudaErrors(cusparseCreateMatDescr(&matDescr));

	//Set base for matrix
	if (row[0]) {
		checkCudaErrors(cusparseSetMatIndexBase(matDescr, CUSPARSE_INDEX_BASE_ONE));
	}
	else {
		checkCudaErrors(cusparseSetMatIndexBase(matDescr, CUSPARSE_INDEX_BASE_ZERO));
	}

	//Prepating device information
	cudaDeviceProp deviceProp;
	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, 0));
	//deviceProp.maxThreadsPerBlock
	//deviceProp.maxGridSize[0]
	//blocks, threads

	//Preparing x0
	double *x0;
	checkCudaErrors(cudaMalloc((void **)&x0, sizeof(double)*(size)));
	checkCudaErrors(cudaMemset(x0, 0.0, sizeof(double)*(size)));

	//Preparing r0 and rT
	double *r0, *rT;

	checkCudaErrors(cudaMallocManaged((void **)&r0, sizeof(double)*(size)));
	checkCudaErrors(cudaMalloc((void **)&rT, sizeof(double)*(size)));
	checkCudaErrors(cudaMemcpy(r0, right, sizeof(double)*(size), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rT, right, sizeof(double)*(size), cudaMemcpyHostToDevice));

	//Preparing rho0, alpha, omega0
	double *rho0 = new double;
	double *alpha = new double;
	double *omega0 = new double;
	*rho0 = 1.0;
	*alpha = 1.0;
	*omega0 = 1.0;

	//Praparing nu0,p0
	double *nu0, *p0;
	checkCudaErrors(cudaMalloc((void **)&nu0, sizeof(double)*(size)));
	checkCudaErrors(cudaMalloc((void **)&p0, sizeof(double)*(size)));
	checkCudaErrors(cudaMemset(nu0, 0.0, sizeof(double)*(size)));
	checkCudaErrors(cudaMemset(p0, 0.0, sizeof(double)*(size)));

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
		checkCudaErrors(cudaSetDevice(omp_get_thread_num()));
		checkCudaErrors(cudaMalloc((void **)&nuK[omp_get_thread_num()], sizeof(double)*(size)));
		checkCudaErrors(cudaMalloc((void **)&t[omp_get_thread_num()], sizeof(double)*(size)));
		checkCudaErrors(cudaMalloc((void **)&vec_temp5[omp_get_thread_num()], sizeof(double)*(size)));
	}

	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaMalloc((void **)&xK, sizeof(double)*(size)));
	checkCudaErrors(cudaMalloc((void **)&rK, sizeof(double)*(size)));
	checkCudaErrors(cudaMalloc((void **)&h, sizeof(double)*(size)));
	checkCudaErrors(cudaMallocManaged((void **)&s, sizeof(double)*(size)));
	checkCudaErrors(cudaMallocManaged((void **)&pK, sizeof(double)*(size)));

	
	//Preparing diag for devices
	double *d_diag;
	checkCudaErrors(cudaMalloc((void **)&d_diag, sizeof(double)*(size)));
	checkCudaErrors(cudaMemcpy(d_diag, diag, sizeof(double)*(size), cudaMemcpyHostToDevice));

	//Preparing temporary variables.
	//There are 2 types of temporary variables: vec_ and num_
	double *num_temp1 = new double,
		*num_temp2 = new double,
		*num_temp6 = new double,
		*rKnrm = new double;
	double *vec_temp1, *vec_temp2, *vec_temp3, *vec_temp4 = new double[size];
	checkCudaErrors(cudaMalloc((void **)&vec_temp1, sizeof(double)*(size)));
	checkCudaErrors(cudaMalloc((void **)&vec_temp2, sizeof(double)*(size)));
	checkCudaErrors(cudaMalloc((void **)&vec_temp3, sizeof(double)*(size)));

	//Others
	double *negalpha = new double;
	double *negomega = new double;
	*negomega = -(*omega0);

#if !RESIDUE
	double *bnrm = new double;
	checkCudaErrors(cublasDnrm2(cublasHandle, size, r0, 1, bnrm));
#endif

	double *temp = new double[size];

	do
	{
		if (!flag)
		{
			nu0 = nuK[0];
			p0 = pK;
			omega0 = omegaK;
			*negomega = -(*omega0);
			*rho0 = *rhoK;
		}

		//Step 1
		checkCudaErrors(cublasDdot(cublasHandle, size, rT, 1, r0, 1, rhoK));

		//Step 2
#if STAB
		if ((abs(*rho0) < MAXACC) || (abs(*omega0) < MAXACC))
		{
			*rho0 = sgn(*rho0) * 1e-7;
			*omega0 = sgn(*omega0) * 1e-7;
		}
#endif
		*beta = ((*rhoK) * (*alpha)) / ((*rho0) * (*omega0));

		//Step 3
		//Here p0 changes while counting axpy, but it's not used further, so we don't care.
		//But r0 is used and we need to copy it to temp
		checkCudaErrors(cublasDaxpy(cublasHandle, size, negomega, nu0, 1, p0, 1));
		checkCudaErrors(cublasDcopy(cublasHandle, size, r0, 1, vec_temp1, 1));
		checkCudaErrors(cublasDaxpy(cublasHandle, size, beta, p0, 1, vec_temp1, 1));
		checkCudaErrors(cublasDcopy(cublasHandle, size, vec_temp1, 1, pK, 1));

		checkCudaErrors(cudaMemcpy(temp, pK, sizeof(double)*(size), cudaMemcpyDefault));

		//Step 4
		//A = A1+A2+...+An => A*x = A1*x+A2*x+...+An*x
		//We use all GPU's but then we send everything to first one
		//And don't forget to add diag

#pragma omp parallel num_threads(gpu)
		{
			checkCudaErrors(cudaSetDevice(omp_get_thread_num()));
			checkCudaErrors(cudaMemcpy(vec_temp5[omp_get_thread_num()], pK, sizeof(double)*(size), cudaMemcpyDefault));
			checkCudaErrors(cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
				size, size, sizes[omp_get_thread_num()],
				one, matDescr, d_val[omp_get_thread_num()],
				d_row[omp_get_thread_num()], d_col[omp_get_thread_num()],
				vec_temp5[omp_get_thread_num()], zero, nuK[omp_get_thread_num()]));
		}
		for (int i = 1; i < gpu; i++)
		{
			checkCudaErrors(cudaSetDevice(i));
			checkCudaErrors(cudaMemcpy(vec_temp4, nuK[i], sizeof(double)*(size), cudaMemcpyDeviceToHost));
			checkCudaErrors(cudaSetDevice(0));
			checkCudaErrors(cudaMemcpy(vec_temp3, vec_temp4, sizeof(double)*(size), cudaMemcpyHostToDevice));
			checkCudaErrors(cublasDaxpy(cublasHandle, size, one, vec_temp3, 1, nuK[0], 1));
		}

		//I don't know how match blocks and threads I can launch
		diVec << <GPU_BLOCKS(size, deviceProp.maxThreadsPerBlock), deviceProp.maxThreadsPerBlock >> >(size, d_diag, pK, nuK[0]);
		checkCudaErrors(cudaDeviceSynchronize());

		checkCudaErrors(cudaMemcpy(temp, nuK[0], sizeof(double)*(size), cudaMemcpyDefault));

		//Step 5
		checkCudaErrors(cublasDdot(cublasHandle, size, rT, 1, nuK[0], 1, num_temp1));
#if STAB
		if (abs(*num_temp1) < MAXACC)
			*num_temp1 = sgn(*num_temp1) * 1e-7;
#endif
		*alpha = (*rhoK) / (*num_temp1);
		*negalpha = -(*alpha);

		//Step 6
		checkCudaErrors(cublasDaxpy(cublasHandle, size, alpha, pK, 1, x0, 1));
		h = x0;
		checkCudaErrors(cudaMemcpy(temp, h, sizeof(double)*(size), cudaMemcpyDefault));

		//Step 8
		//Here r0 would be changed, but we don't care because it would be used further
		checkCudaErrors(cublasDaxpy(cublasHandle, size, negalpha, nuK[0], 1, r0, 1));
		s = r0;
		checkCudaErrors(cudaMemcpy(temp, s, sizeof(double)*(size), cudaMemcpyDefault));

		//Step 9
		//A = A1+A2+...+An => A*x = A1*x+A2*x+...+An*x
		//We use all GPU's but then we send everything to first one
		//And don't forget to add diag
#pragma omp parallel num_threads(gpu)
		{
			checkCudaErrors(cudaSetDevice(omp_get_thread_num()));
			checkCudaErrors(cudaMemcpy(vec_temp5[omp_get_thread_num()], s, sizeof(double)*(size), cudaMemcpyDefault));
			checkCudaErrors(cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
				size, size, sizes[omp_get_thread_num()],
				one, matDescr, d_val[omp_get_thread_num()],
				d_row[omp_get_thread_num()], d_col[omp_get_thread_num()],
				vec_temp5[omp_get_thread_num()], zero, t[omp_get_thread_num()]));
		}
		for (int i = 1; i < gpu; i++)
		{
			checkCudaErrors(cudaSetDevice(i));
			checkCudaErrors(cudaMemcpy(vec_temp4, t[i], sizeof(double)*(size), cudaMemcpyDeviceToHost));
			checkCudaErrors(cudaSetDevice(0));
			checkCudaErrors(cudaMemcpy(vec_temp3, vec_temp4, sizeof(double)*(size), cudaMemcpyHostToDevice));
			checkCudaErrors(cublasDaxpy(cublasHandle, size, one, vec_temp3, 1, t[0], 1));
		}

		//I don't know how match blocks and threads I can launch
		diVec << <GPU_BLOCKS(size, deviceProp.maxThreadsPerBlock), deviceProp.maxThreadsPerBlock >> >(size, d_diag, s, t[0]);
		checkCudaErrors(cudaDeviceSynchronize());

		checkCudaErrors(cudaMemcpy(temp, t[0], sizeof(double)*(size), cudaMemcpyDefault));

		//Step 10
		checkCudaErrors(cublasDdot(cublasHandle, size, t[0], 1, s, 1, num_temp6));
		checkCudaErrors(cublasDdot(cublasHandle, size, t[0], 1, t[0], 1, num_temp2));
#if STAB
		if (abs(*num_temp2) < MAXACC)
			*num_temp2 = sgn(*num_temp2) * 1e-7;
#endif
		*omegaK = (*num_temp6) / (*num_temp2);
		*negomega = -(*omegaK);

		//Step 11
		//Here h would be changed, but it's not used after that, so we don't care
		checkCudaErrors(cublasDaxpy(cublasHandle, size, omegaK, s, 1, h, 1));
		xK = h;

		checkCudaErrors(cudaMemcpy(temp, xK, sizeof(double)*(size), cudaMemcpyDefault));

		//Step 13
		//Here s is changed, but we don't care because it wouldn't be used further
		checkCudaErrors(cublasDaxpy(cublasHandle, size, negomega, t[0], 1, s, 1));
		rK = s;

		checkCudaErrors(cudaMemcpy(temp, rK, sizeof(double)*(size), cudaMemcpyDefault));

		//Step 12 (check)
		//If rK is sufficiently small
		checkCudaErrors(cublasDnrm2(cublasHandle, size, rK, 1, rKnrm));

		if (step % 20 == 0)
		{
			cout << "res: " << *rKnrm << endl;
		}

#if RESIDUE
#if OMEGA
		if (((*rKnrm) < MAXACC) || (abs(*omegaK) <= MAXACC))
#else
		if ((*rKnrm) < MAXACC)
#endif
#else
#if OMEGA
		if (((*rKnrm) / (*bnrm) < MAXACC) || (abs(*omegaK) <= MAXACC))
#else
		if ((*rKnrm) / (*bnrm) < MAXACC)
#endif
#endif
		{
			break;
		}

		flag = false;
		step++;

	} while (step <= MAXITER);

	//Check || Ax - b ||
	checkCudaErrors(cudaMemcpy(vec_temp4, xK, sizeof(double)*(size), cudaMemcpyDefault));
#pragma omp parallel num_threads(gpu)
	{
		checkCudaErrors(cudaSetDevice(omp_get_thread_num()));
		checkCudaErrors(cudaMemcpy(vec_temp5[omp_get_thread_num()], vec_temp4, sizeof(double)*(size), cudaMemcpyDefault));
		checkCudaErrors(cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
			size, size, sizes[omp_get_thread_num()],
			one, matDescr, d_val[omp_get_thread_num()],
			d_row[omp_get_thread_num()], d_col[omp_get_thread_num()],
			vec_temp5[omp_get_thread_num()], zero, t[omp_get_thread_num()]));
	}
	for (int i = 1; i < gpu; i++)
	{
		checkCudaErrors(cudaSetDevice(i));
		checkCudaErrors(cudaMemcpy(vec_temp4, t[i], sizeof(double)*(size), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaSetDevice(0));
		checkCudaErrors(cudaMemcpy(vec_temp3, vec_temp4, sizeof(double)*(size), cudaMemcpyHostToDevice));
		checkCudaErrors(cublasDaxpy(cublasHandle, size, one, vec_temp3, 1, t[0], 1));
	}
	diVec << <GPU_BLOCKS(size, deviceProp.maxThreadsPerBlock), deviceProp.maxThreadsPerBlock >> >(size, d_diag, xK, t[0]);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(vec_temp3, right, sizeof(double)*(size), cudaMemcpyHostToDevice));
	checkCudaErrors(cublasDaxpy(cublasHandle, size, minus, vec_temp3, 1, t[0], 1));
	checkCudaErrors(cublasDnrm2(cublasHandle, size, t[0], 1, rKnrm));

	cout << "Residue: " << *rKnrm;
	cout << endl << "Steps: " << step << endl;

	double *res = new double[size];
	checkCudaErrors(cudaMemcpy(res, xK, sizeof(double)*size, cudaMemcpyDeviceToHost));

	checkCudaErrors(cublasDestroy(cublasHandle));
	checkCudaErrors(cusparseDestroy(cusparseHandle));
	checkCudaErrors(cusparseDestroyMatDescr(matDescr));

	checkCudaErrors(cudaFree(x0));//It deletes h and xK
	checkCudaErrors(cudaFree(r0));//It deletes s and rK
	checkCudaErrors(cudaFree(rT));
	checkCudaErrors(cudaFree(nu0));//It deletes nuK[0] but not others
	checkCudaErrors(cudaFree(pK)); //It deletes p0
	checkCudaErrors(cudaFree(d_diag));
	checkCudaErrors(cudaFree(vec_temp1));
	checkCudaErrors(cudaFree(vec_temp2));
	checkCudaErrors(cudaFree(vec_temp3));

	delete minus, one, zero, rhoK,
		beta, alpha, omega0, omegaK,
		negalpha, negomega, rho0, num_temp1,
		num_temp2, num_temp6, rKnrm;

#pragma omp parallel num_threads(gpu)
	{
		checkCudaErrors(cudaSetDevice(omp_get_thread_num()));
		checkCudaErrors(cudaFree(t[omp_get_thread_num()]));
		checkCudaErrors(cudaFree(d_val[omp_get_thread_num()]));
		checkCudaErrors(cudaFree(d_col[omp_get_thread_num()]));
		checkCudaErrors(cudaFree(d_row[omp_get_thread_num()]));
		checkCudaErrors(cudaFree(vec_temp5[omp_get_thread_num()]));
	}

	if (gpu > 1)
	{
#pragma omp parallel num_threads(gpu - 1)
		{
			checkCudaErrors(cudaSetDevice(omp_get_thread_num() + 1));
			checkCudaErrors(cudaFree(nuK[omp_get_thread_num() + 1]));
		}
	}

	delete[] sizes, d_val, d_row, d_col, t, nuK, vec_temp4, vec_temp5;

	return res;
}

double* gpu_solver::GPU_PBiCGSTAB(double *val, int *col, int *row, double *right, double *diag, int nnz, int size)
{
	//Count amount of devices
	int gpu;
	checkCudaErrors(cudaGetDeviceCount(&gpu));

	//Arrays for devices
	double ** d_val = new  double *[gpu];
	int ** d_col = new int *[gpu];
	int ** d_row = new int *[gpu];

	//Array with device array's sizes
	int *sizes = new int[gpu];
	sizes = split(gpu, val, col, row, size, nnz, d_val, d_col, d_row);

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
	checkCudaErrors(cudaSetDevice(0));

	//Preparing CUBLAS handle
	cublasHandle_t cublasHandle = NULL;
	checkCudaErrors(cublasCreate(&cublasHandle));

	//Preparing CUSPARSE handle
	cusparseHandle_t cusparseHandle = NULL;
	checkCudaErrors(cusparseCreate(&cusparseHandle));

	//Preparing matrix descriptor
	cusparseMatDescr_t matDescr = NULL;
	checkCudaErrors(cusparseCreateMatDescr(&matDescr));

	//Set base for matrix
	if (row[0]) {
		checkCudaErrors(cusparseSetMatIndexBase(matDescr, CUSPARSE_INDEX_BASE_ONE));
	}
	else {
		checkCudaErrors(cusparseSetMatIndexBase(matDescr, CUSPARSE_INDEX_BASE_ZERO));
	}

	//Prepating device information
	cudaDeviceProp deviceProp;
	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, 0));
	//deviceProp.maxThreadsPerBlock
	//deviceProp.maxGridSize[0]
	//blocks, threads

	//Preparing x0
	double *x0;
	checkCudaErrors(cudaMalloc((void **)&x0, sizeof(double)*(size)));
	checkCudaErrors(cudaMemset(x0, 0.0, sizeof(double)*(size)));

	//Preparing r0 and rT
	double *r0, *rT;
	checkCudaErrors(cudaMalloc((void **)&r0, sizeof(double)*(size)));
	checkCudaErrors(cudaMalloc((void **)&rT, sizeof(double)*(size)));
	checkCudaErrors(cudaMemcpy(r0, right, sizeof(double)*(size), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(rT, right, sizeof(double)*(size), cudaMemcpyHostToDevice));

	//Preparing rho0, alpha, omega0
	double *rho0 = new double;
	double *alpha = new double;
	double *omega0 = new double;
	*rho0 = 1.0;
	*alpha = 1.0;
	*omega0 = 1.0;

	//Praparing nu0,p0
	double *nu0, *p0;
	checkCudaErrors(cudaMalloc((void **)&nu0, sizeof(double)*(size)));
	checkCudaErrors(cudaMalloc((void **)&p0, sizeof(double)*(size)));
	checkCudaErrors(cudaMemset(nu0, 0.0, sizeof(double)*(size)));
	checkCudaErrors(cudaMemset(p0, 0.0, sizeof(double)*(size)));

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
		checkCudaErrors(cudaSetDevice(omp_get_thread_num()));
		checkCudaErrors(cudaMalloc((void **)&nuK[omp_get_thread_num()], sizeof(double)*(size)));
		checkCudaErrors(cudaMalloc((void **)&t[omp_get_thread_num()], sizeof(double)*(size)));
		checkCudaErrors(cudaMalloc((void **)&vec_temp6[omp_get_thread_num()], sizeof(double)*(size)));
	}

	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaMalloc((void **)&s, sizeof(double)*(size)));
	checkCudaErrors(cudaMalloc((void **)&pK, sizeof(double)*(size)));
	checkCudaErrors(cudaMalloc((void **)&xK, sizeof(double)*(size)));
	checkCudaErrors(cudaMalloc((void **)&rK, sizeof(double)*(size)));
	checkCudaErrors(cudaMalloc((void **)&h, sizeof(double)*(size)));
	checkCudaErrors(cudaMallocManaged((void **)&y, sizeof(double)*(size)));
	checkCudaErrors(cudaMallocManaged((void **)&z, sizeof(double)*(size)));

	//Preparing diag and "reversed" diag
	double *d_diag, *d_revDiag;
	checkCudaErrors(cudaMalloc((void **)&d_diag, sizeof(double)*(size)));
	checkCudaErrors(cudaMalloc((void **)&d_revDiag, sizeof(double)*(size)));
	checkCudaErrors(cudaMemcpy(d_diag, diag, sizeof(double)*(size), cudaMemcpyHostToDevice));
	diRev << <GPU_BLOCKS(size, deviceProp.maxThreadsPerBlock), deviceProp.maxThreadsPerBlock >> > (size, d_diag, d_revDiag);
	checkCudaErrors(cudaDeviceSynchronize());

	//Preparing temporary variables.
	//There are 2 types of temporary variables: vec_ and num_
	double *num_temp1 = new double,
		*num_temp2 = new double,
		*num_temp6 = new double,
		*rKnrm = new double;
	double *vec_temp1, *vec_temp2, *vec_temp3, *vec_temp4, *vec_temp5 = new double[size];
	checkCudaErrors(cudaMalloc((void **)&vec_temp1, sizeof(double)*(size)));
	checkCudaErrors(cudaMalloc((void **)&vec_temp2, sizeof(double)*(size)));
	checkCudaErrors(cudaMalloc((void **)&vec_temp3, sizeof(double)*(size)));
	checkCudaErrors(cudaMalloc((void **)&vec_temp4, sizeof(double)*(size)));

	//Others
	double *negalpha = new double;
	double *negomega = new double;
	*negomega = -(*omega0);

#if !RESIDUE
	double *bnrm = new double;
	checkCudaErrors(cublasDnrm2(cublasHandle, size, r0, 1, bnrm));
#endif
	
	do
	{
		if (!flag)
		{
			nu0 = nuK[0];
			p0 = pK;
			omega0 = omegaK;
			*negomega = -(*omega0);
			*rho0 = *rhoK;
		}

		//Step 1
		checkCudaErrors(cublasDdot(cublasHandle, size, rT, 1, r0, 1, rhoK));

		//Step 2
#if STAB
		if ((abs(*rho0) < MAXACC) || (abs(*omega0) < MAXACC))
		{
			*rho0 = sgn(*rho0) * 1e-7;
			*omega0 = sgn(*omega0) * 1e-7;
		}
#endif
		*beta = ((*rhoK) * (*alpha)) / ((*rho0) * (*omega0));

		//Step 3
		checkCudaErrors(cublasDaxpy(cublasHandle, size, negomega, nu0, 1, p0, 1));
		checkCudaErrors(cublasDcopy(cublasHandle, size, r0, 1, vec_temp1, 1));
		checkCudaErrors(cublasDaxpy(cublasHandle, size, beta, p0, 1, vec_temp1, 1));
		checkCudaErrors(cublasDcopy(cublasHandle, size, vec_temp1, 1, pK, 1));

		//Step 4
		vec2vec << <GPU_BLOCKS(size, deviceProp.maxThreadsPerBlock), deviceProp.maxThreadsPerBlock >> > (size, d_revDiag, pK, y);
		checkCudaErrors(cudaDeviceSynchronize());
#if STAB
		if (*y < MAXACC)
			*y = sgn(*y) * 1e-7;
#endif

		//Step 5
		//A = A1+A2+...+An => A*x = A1*x+A2*x+...+An*x
		//We use all GPU's but then we send everything to first one
		//And don't forget to add diag
#pragma omp parallel num_threads(gpu)
		{
			checkCudaErrors(cudaSetDevice(omp_get_thread_num()));
			checkCudaErrors(cudaMemcpy(vec_temp6[omp_get_thread_num()], y, sizeof(double)*(size), cudaMemcpyDefault));
			checkCudaErrors(cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
				size, size, sizes[omp_get_thread_num()],
				one, matDescr, d_val[omp_get_thread_num()],
				d_row[omp_get_thread_num()], d_col[omp_get_thread_num()],
				vec_temp6[omp_get_thread_num()], zero, nuK[omp_get_thread_num()]));
		}

		for (int i = 1; i < gpu; i++)
		{
			checkCudaErrors(cudaSetDevice(i));
			checkCudaErrors(cudaMemcpy(vec_temp5, nuK[i], sizeof(double)*size, cudaMemcpyDeviceToHost));
			checkCudaErrors(cudaSetDevice(0));
			checkCudaErrors(cudaMemcpy(vec_temp4, vec_temp5, sizeof(double)*size, cudaMemcpyHostToDevice));
			checkCudaErrors(cublasDaxpy(cublasHandle, size, one, vec_temp4, 1, nuK[0], 1));
		}
		
		//I don't know how match blocks and threads I can launch
		diVec << <GPU_BLOCKS(size, deviceProp.maxThreadsPerBlock), deviceProp.maxThreadsPerBlock >> >(size, d_diag, y, nuK[0]);
		checkCudaErrors(cudaDeviceSynchronize());

		//Step 6
		checkCudaErrors(cublasDdot(cublasHandle, size, rT, 1, nuK[0], 1, num_temp1));
#if STAB
		if (abs(*num_temp1) < MAXACC)
			*num_temp1 = sgn(*num_temp1) * 1e-7;
#endif
		*alpha = (*rhoK) / (*num_temp1);
		*negalpha = -(*alpha);

		//Step 7
		//checkCudaErrors(cublasDnrm2(cublasHandle, size, x0, 1, x0nrm));
		checkCudaErrors(cublasDaxpy(cublasHandle, size, alpha, y, 1, x0, 1));
		h = x0;

		//Step 9
		checkCudaErrors(cublasDaxpy(cublasHandle, size, negalpha, nuK[0], 1, r0, 1));
		s = r0;

		//Step 10
		vec2vec << <GPU_BLOCKS(size, deviceProp.maxThreadsPerBlock), deviceProp.maxThreadsPerBlock >> > (size, d_revDiag, s, z);
		checkCudaErrors(cudaDeviceSynchronize());

		//Step 11
		//A = A1+A2+...+An => A*x = A1*x+A2*x+...+An*x
		//We use all GPU's but then we send everything to first one
		//And don't forget to add diag
#pragma omp parallel num_threads(gpu)
		{
			checkCudaErrors(cudaSetDevice(omp_get_thread_num()));
			checkCudaErrors(cudaMemcpy(vec_temp6[omp_get_thread_num()], z, sizeof(double)*(size), cudaMemcpyDefault));
			checkCudaErrors(cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
				size, size, sizes[omp_get_thread_num()],
				one, matDescr, d_val[omp_get_thread_num()],
				d_row[omp_get_thread_num()], d_col[omp_get_thread_num()],
				vec_temp6[omp_get_thread_num()], zero, t[omp_get_thread_num()]));
		}

		for (int i = 1; i < gpu; i++)
		{
			checkCudaErrors(cudaSetDevice(i));
			checkCudaErrors(cudaMemcpy(vec_temp5, t[i], sizeof(double)*size, cudaMemcpyDeviceToHost));
			checkCudaErrors(cudaSetDevice(0));
			checkCudaErrors(cudaMemcpy(vec_temp4, vec_temp5, sizeof(double)*size, cudaMemcpyHostToDevice));
			checkCudaErrors(cublasDaxpy(cublasHandle, size, one, vec_temp4, 1, t[0], 1));
		}
		
		//I don't know how match blocks and threads I can launch
		diVec << <GPU_BLOCKS(size, deviceProp.maxThreadsPerBlock), deviceProp.maxThreadsPerBlock >> >(size, d_diag, z, t[0]);
		checkCudaErrors(cudaDeviceSynchronize());
#if STAB
		if (*z < MAXACC)
			*z = sgn(*z) * 1e-7;
#endif

		//Step 12
		vec2vec << <GPU_BLOCKS(size, deviceProp.maxThreadsPerBlock), deviceProp.maxThreadsPerBlock >> > (size, d_revDiag, t[0], vec_temp3);
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cublasDdot(cublasHandle, size, vec_temp3, 1, z, 1, num_temp6));
		checkCudaErrors(cublasDdot(cublasHandle, size, vec_temp3, 1, vec_temp3, 1, num_temp2));
#if STAB
		if (abs(*num_temp2) < MAXACC)
			*num_temp2 = sgn(*num_temp2) * 1e-7;
#endif
		*omegaK = (*num_temp6) / (*num_temp2);
		*negomega = -(*omegaK);

		//Step 13
		checkCudaErrors(cublasDaxpy(cublasHandle, size, omegaK, z, 1, h, 1));
		xK = h;

		//Step 15
		checkCudaErrors(cublasDaxpy(cublasHandle, size, negomega, t[0], 1, s, 1));
		rK = s;

		//Step 14 (check)
		checkCudaErrors(cublasDnrm2(cublasHandle, size, rK, 1, rKnrm));
		if (step % 20 == 0)
		{
			cout << "res: " << *rKnrm << endl;
		}
#if RESIDUE
#if OMEGA
		if (((*rKnrm) < MAXACC) || (abs(*omegaK) <= MAXACC))
#else
		if ((*rKnrm) < MAXACC)
#endif
#else
#if OMEGA
		if (((*rKnrm) / (*bnrm) < MAXACC) || (abs(*omegaK) <= MAXACC))
#else
		if ((*rKnrm) / (*bnrm) < MAXACC)
#endif
#endif
		{
			break;
		}

		flag = false;
		step++;

	} while (step <= MAXITER);

	//Check || Ax - b ||
	checkCudaErrors(cudaMemcpy(vec_temp5, xK, sizeof(double)*(size), cudaMemcpyDefault));
#pragma omp parallel num_threads(gpu)
	{
		checkCudaErrors(cudaSetDevice(omp_get_thread_num()));
		checkCudaErrors(cudaMemcpy(vec_temp6[omp_get_thread_num()], vec_temp5, sizeof(double)*(size), cudaMemcpyDefault));
		checkCudaErrors(cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
			size, size, sizes[omp_get_thread_num()],
			one, matDescr, d_val[omp_get_thread_num()],
			d_row[omp_get_thread_num()], d_col[omp_get_thread_num()],
			vec_temp6[omp_get_thread_num()], zero, t[omp_get_thread_num()]));
	}
	for (int i = 1; i < gpu; i++)
	{
		checkCudaErrors(cudaSetDevice(i));
		checkCudaErrors(cudaMemcpy(vec_temp4, t[i], sizeof(double)*(size), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaSetDevice(0));
		checkCudaErrors(cudaMemcpy(vec_temp3, vec_temp4, sizeof(double)*(size), cudaMemcpyHostToDevice));
		checkCudaErrors(cublasDaxpy(cublasHandle, size, one, vec_temp3, 1, t[0], 1));
	}
	diVec << <GPU_BLOCKS(size, deviceProp.maxThreadsPerBlock), deviceProp.maxThreadsPerBlock >> >(size, d_diag, xK, t[0]);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(vec_temp3, right, sizeof(double)*(size), cudaMemcpyHostToDevice));
	checkCudaErrors(cublasDaxpy(cublasHandle, size, minus, vec_temp3, 1, t[0], 1));
	checkCudaErrors(cublasDnrm2(cublasHandle, size, t[0], 1, rKnrm));

	cout << "Residue: " << *rKnrm;
	cout << endl << "Steps: " << step << endl;

	double *res = new double[size];
	checkCudaErrors(cudaMemcpy(res, xK, sizeof(double)*size, cudaMemcpyDeviceToHost));

	checkCudaErrors(cublasDestroy(cublasHandle));
	checkCudaErrors(cusparseDestroy(cusparseHandle));
	checkCudaErrors(cusparseDestroyMatDescr(matDescr));

	checkCudaErrors(cudaFree(x0));//It deletes h and xK
	checkCudaErrors(cudaFree(r0));//It deletes s and rK
	checkCudaErrors(cudaFree(rT));
	checkCudaErrors(cudaFree(nu0));//It deletes nuK[0] but not others
	checkCudaErrors(cudaFree(pK)); //It deletes p0
	checkCudaErrors(cudaFree(d_diag));
	checkCudaErrors(cudaFree(d_revDiag));
	checkCudaErrors(cudaFree(vec_temp1));
	checkCudaErrors(cudaFree(vec_temp2));
	checkCudaErrors(cudaFree(vec_temp3));
	checkCudaErrors(cudaFree(vec_temp4));
	checkCudaErrors(cudaFree(z));
	checkCudaErrors(cudaFree(y));

	delete minus, one, zero, rhoK,
		beta, alpha, omega0, omegaK,
		negalpha, negomega, rho0, num_temp1,
		num_temp2, num_temp6, rKnrm;

#pragma omp parallel num_threads(gpu)
	{
		checkCudaErrors(cudaSetDevice(omp_get_thread_num()));
		checkCudaErrors(cudaFree(t[omp_get_thread_num()]));
		checkCudaErrors(cudaFree(d_val[omp_get_thread_num()]));
		checkCudaErrors(cudaFree(d_col[omp_get_thread_num()]));
		checkCudaErrors(cudaFree(d_row[omp_get_thread_num()]));
		checkCudaErrors(cudaFree(vec_temp6[omp_get_thread_num()]));
	}

	if (gpu > 1)
	{
#pragma omp parallel num_threads(gpu - 1)
		{
			checkCudaErrors(cudaSetDevice(omp_get_thread_num() + 1));
			checkCudaErrors(cudaFree(nuK[omp_get_thread_num() + 1]));
		}
	}

	delete[] sizes, d_val, d_row, d_col, t, nuK, vec_temp5, vec_temp6;

	return res;
}