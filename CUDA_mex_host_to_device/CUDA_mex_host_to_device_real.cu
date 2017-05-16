#include <stdio.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "mex.h"

#include "Utilities.cuh"

#define BLOCKSIZE	512

/*******************/
/* SQUARING KERNEL */
/*******************/
__global__ void squareKernel(double * __restrict__ d_vec, const int N) {

	const int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= N) return;

	d_vec[tid] = d_vec[tid] * d_vec[tid];
}

/****************/
/* MEX FUNCTION */
/****************/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	// --- Recovering the pointer to the input host variable
	double *h_input = mxGetPr(prhs[0]);

	// --- Recovering the number of elements of the input variable (the input variable can be also a matrix)
	int numElements = mxGetN(prhs[0]) * mxGetM(prhs[0]);

	// --- Allocating space for the input/output device variable
	double *d_vec; gpuErrchk(cudaMalloc(&d_vec, numElements * sizeof(double)));
	
	// --- Moving the input from host to device
	gpuErrchk(cudaMemcpy(d_vec, h_input, numElements * sizeof(double), cudaMemcpyHostToDevice));

	squareKernel<<<iDivUp(numElements, BLOCKSIZE), BLOCKSIZE>>>(d_vec, numElements);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	// --- Allocating space for the output output host variable
	plhs[0] = mxCreateDoubleMatrix(1, numElements, mxREAL);
	
	// --- Recovering the pointer to the output host variable
	double *h_output = mxGetPr(plhs[0]);

	gpuErrchk(cudaMemcpy(h_output, d_vec, numElements * sizeof(double), cudaMemcpyDeviceToHost));

	gpuErrchk(cudaFree(d_vec));

}
