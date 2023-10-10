#include "kernel.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cstdlib>

//Matrix A: input features/attention scores/gradients of enwik8/text8, 4 dimension: d1, d2, d3 and d4 
//Matrix B: it can be input features/values/gradients, 4 dimensions
//Matrix C: Output result : attention scores/weighted sum/gradients
//Matrix D: Dilation information: 1 dimension = vector of heads

//https://github.com/allenai/longformer/blob/master/longformer/diagonaled_mm_tvm.py
/* line 156 _diagonaled_mm():  t1 = A, t2 = B, d = D, and r = C
        this should call line 16 = Omid's cuda implementation


 */

// mode 1 : d4c == d4b && transposeT1 == 0
// mode 2 : d4c == d4b = d4a && transposeT1 == 1 && d4a = (Window + WindowUpper)
// mode 3 : d4c != d4b && d4a = d4b

__global__ void mm4d_gpu_mode1(float* a, float* b, float* c, int* dilation, int* params, int* size) {
	int d1 = size[0], d2 = size[1], d3 = size[2];
        int d4a = size[3], d4b = size[4], d4c = size[5];
        int aSize = size[6], bSize = size[7], cSize = size[8];
        int Window = params[0], WindowUpper = params[1], Padding = params[2];
	int idx_a, idx_b, idx;

	int bx, by, tx, ty, B;
	bx = blockIdx.x;
	by = blockIdx.y;
	tx = threadIdx.x;
	ty = threadIdx.y;
	B = blockDim.x*blockDim.y;
	idx = bx*gridDim.y*B + by*B + tx*blockDim.y + ty;

	int l = idx / (d2 * d3 * d4c);
	int i = (idx / (d3 * d4c)) % d2;
	int q = (idx / d4c) % d3;
	int j = idx % d4c;
	int D = dilation[q];

	if (idx < cSize) {
	c[idx] = 0.0f;
	for (int k = 0; k < d4c; k++) {
		int condition = i + D * (k - Window);
		if (condition >= 0 && condition < d2) {
			idx_a = (((l * d2) + i) * d3 + q) *  d4a + k;
			idx_b = (((l * d2) + i + D * (k - Window)) * d3 + q) *  d4b + j;
			if (idx_a < aSize && idx_b < bSize)	c[idx] += a[idx_a] * b[idx_b];
		}
		else {
			c[idx] += Padding;
		}
	}
	}
}

__global__ void mm4d_gpu_mode2(float* a, float* b, float* c, int* dilation, int* params, int* size) {
	int d1 = size[0], d2 = size[1], d3 = size[2];
        int d4a = size[3], d4b = size[4], d4c = size[5];
        int aSize = size[6], bSize = size[7], cSize = size[8];
        int Window = params[0], WindowUpper = params[1], Padding = params[2];
	int idx_a, idx_b, idx;
	int bx, by, tx, ty, B;

	bx = blockIdx.x;
	by = blockIdx.y;
	tx = threadIdx.x;
	ty = threadIdx.y;
	B = blockDim.x*blockDim.y;
	idx = bx*gridDim.y*B + by*B + tx*blockDim.y + ty;

	int l = idx / (d2 * d3 * d4c);
	int i = (idx / (d3 * d4c)) % d2;
	int q = (idx / d4c) % d3;
	int j = idx % d4c;
	int D = dilation[q];

	if (idx < cSize) {
	c[idx] = 0.0f;
	for (int k = 0; k < d4c; k++) {
		int condition = i + D * (k - WindowUpper);
		if (condition >= 0 && condition < d2) {
			idx_a = (((l * d2) + i + D * (k - WindowUpper)) * d3 + q) *  d4a + WindowUpper + Window - k;
			idx_b = (((l * d2) + i + D * (k - WindowUpper)) * d3 + q) *  d4b + j;
			if (idx_a < aSize && idx_b < bSize)	c[idx] += a[idx_a] * b[idx_b];
		}
		else {
			c[idx] += Padding;
		}
	}
	}
}

__global__ void mm4d_gpu_mode3(float* a, float* b, float* c, int* dilation, int* params, int* size) {
	int d1 = size[0], d2 = size[1], d3 = size[2];
	int d4a = size[3], d4b = size[4], d4c = size[5];
	int aSize = size[6], bSize = size[7], cSize = size[8];
	int Window = params[0], WindowUpper = params[1], Padding = params[2];
	int idx_a, idx_b, idx;
	int bx, by, tx, ty, B;

	bx = blockIdx.x;
	by = blockIdx.y;
	tx = threadIdx.x;
	ty = threadIdx.y;
	B = blockDim.x*blockDim.y;
	idx = bx*gridDim.y*B + by*B + tx*blockDim.y + ty;

	int l = idx / (d2 * d3 * d4c);
	int i = (idx / (d3 * d4c)) % d2;
	int q = (idx / d4c) % d3;
	int j = idx % d4c;
	int D = dilation[q];

	if (idx < cSize) {
	c[0] = 0.0f;
	for (int k = 0; k < d4c; k++) {
		int condition = i + D * (j - Window);
		if (condition >= 0 && condition < d2) {
			idx_a = (((l * d2) + i) * d3 + q) * d4a + k;
			idx_b = (((l * d2) + i + D * (j - Window)) * d3 + q) *  d4b + k;
			if (idx_a < aSize && idx_b < bSize)	c[idx] += a[idx_a] * b[idx_b];
		}
		else {
			c[idx] += Padding;
		}
	}
	}
}


void mm4d_cpu_mode1(float* a, float* b, float* c, int* dilation, int* params, int* size) {
	int d1 = size[0], d2 = size[1], d3 = size[2];
        int d4a = size[3], d4b = size[4], d4c = size[5];
        int aSize = size[6], bSize = size[7], cSize = size[8];
        int Window = params[0], WindowUpper = params[1], Padding = params[2];
	int idx_a, idx_b, idx;

	for (idx = 0; idx < cSize; idx++) {
		int l = idx / (d2 * d3 * d4c);
		int i = (idx / (d3 * d4c)) % d2;
		int q = (idx / d4c) % d3;
		int j = idx % d4c;
		int D = dilation[q];
		c[idx] = 0.0f;

		for (int k = 0; k < d4c; k++) {
			int condition = i + D * (k - Window);
			if (condition >= 0 && condition < d2) {
				idx_a = (((l * d2) + i) * d3 + q) *  d4a + k;
				idx_b = (((l * d2) + i + D * (k - Window)) * d3 + q) *  d4b + j;
				c[idx] += a[idx_a] * b[idx_b];
			}
			else {
				c[idx] += Padding;
			}

		}
	}
}

void mm4d_cpu_mode2(float* a, float* b, float* c, int* dilation, int* params, int* size) {
	int d1 = size[0], d2 = size[1], d3 = size[2];
        int d4a = size[3], d4b = size[4], d4c = size[5];
        int aSize = size[6], bSize = size[7], cSize = size[8];
        int Window = params[0], WindowUpper = params[1], Padding = params[2];
	int idx_a, idx_b, idx;

	for (idx = 0; idx < cSize; idx++) {
		int l = idx / (d2 * d3 * d4c);
		int i = (idx / (d3 * d4c)) % d2;
		int q = (idx / d4c) % d3;
		int j = idx % d4c;
		int D = dilation[q];
		c[idx] = 0.0f;

		for (int k = 0; k < d4c; k++) {
			int condition = i + D * (k - WindowUpper);
			if (condition >= 0 && condition < d2) {
				idx_a = (((l * d2) + i + D * (k - WindowUpper)) * d3 + q) *  d4a + WindowUpper + Window - k;
				idx_b = (((l * d2) + i + D * (k - WindowUpper)) * d3 + q) *  d4b + j;
				c[idx] += a[idx_a] * b[idx_b];
			}
			else {
				c[idx] += Padding;
			}
		}
	}
}

void mm4d_cpu_mode3(float* a, float* b, float* c, int* dilation, int* params, int* size) {
	int d1 = size[0], d2 = size[1], d3 = size[2];
        int d4a = size[3], d4b = size[4], d4c = size[5];
        int aSize = size[6], bSize = size[7], cSize = size[8];
        int Window = params[0], WindowUpper = params[1], Padding = params[2];
	int idx_a, idx_b, idx;

	for (idx = 0; idx < cSize; idx++) {
		int l = idx / (d2 * d3 * d4c);
		int i = (idx / (d3 * d4c)) % d2;
		int q = (idx / d4c) % d3;
		int j = idx % d4c;
		int D = dilation[q];
		c[idx] = 0.0f;

		for (int k = 0; k < d4c; k++) {
			int condition = i + D * (j - Window);
			if (condition >= 0 && condition < d2) {
				idx_a = (((l * d2) + i) * d3 + q) * d4a + k;
				idx_b = (((l * d2) + i + D * (j - Window)) * d3 + q) *  d4b + k;
				c[idx] += a[idx_a] * b[idx_b];
			}
			else {
				c[idx] += Padding;
			}
		}
	}
}

void lformerMM(array4d_t<float>& input1, array4d_t<float>& input2, array4d_t<float>& output1, array1d_t<int>& dilation, array1d_t<int>& params, bool GPU){
	int* d = dilation.data_ptr;
	float* a = input1.data_ptr, *b = input2.data_ptr, *c = output1.data_ptr;
	int d1 = output1.last_count, d2 = output1.matrix_count, d3 = output1.row_count;
	int d4a = input1.col_count, d4b = input2.col_count, d4c = output1.col_count;
	int Window = params.data_ptr[0], WindowUpper = params.data_ptr[1], Padding = params.data_ptr[2], transposeT1 = params.data_ptr[3];

	int *p, *p_dev;
        cudaMallocHost(&p, 3 * sizeof(int));
        p[0] = Window, p[1] = WindowUpper, p[2] = Padding;
        cudaMalloc(&p_dev, 3 * sizeof(int));
        cudaMemcpy(p_dev, p, 3 * sizeof(int), cudaMemcpyHostToDevice);

	int* s, *s_dev;
	cudaMallocHost(&s, 9 * sizeof(int));
	s[0] = d1, s[1] = d2, s[2] = d3, s[3] = d4a, s[4] = d4b, s[5] = d4c;
	s[6] = d1*d2*d3*d4a, s[7] = d1*d2*d3*d4b, s[8] = d1*d2*d3*d4c;
	cudaMalloc(&s_dev, 9 * sizeof(int));
	cudaMemcpy(s_dev, s, 9 * sizeof(int), cudaMemcpyHostToDevice);

	dim3 blockSize(16, 16);
	dim3 gridSize((d1 * d2 + blockSize.x - 1) / blockSize.x, (d3 * d4c + blockSize.y - 1) / blockSize.y);

        //mode 1 and mode 3 are for forward
        //mode 2 and mode 3 are for backward
	if (d4c == d4b) {//mode 1 or mode 2
		if (transposeT1 == 0) {//mode 1: can be called in forward and backward
		        if (!GPU) mm4d_cpu_mode1(a, b, c, d, p, s_dev);
			else mm4d_gpu_mode1 <<<gridSize, blockSize >>>(a, b, c, d, p_dev, s_dev);
		}
                else {// mode 2: called during gradient back-propagation
			if (!GPU) mm4d_cpu_mode2(a, b, c, d, p, s_dev);
			else mm4d_gpu_mode2 <<<gridSize, blockSize >>>(a, b, c, d, p_dev, s_dev);
		}
	}
	else {//mode 3: can be called in forward and backward
		if (!GPU) mm4d_cpu_mode3(a, b, c, d, p, s_dev);
		else mm4d_gpu_mode3 <<<gridSize, blockSize>>>(a, b, c, d, p_dev, s_dev);
	}
	cudaDeviceSynchronize();
        cudaFree(s_dev); cudaFreeHost(s); cudaFree(p_dev); cudaFreeHost(p);
}
