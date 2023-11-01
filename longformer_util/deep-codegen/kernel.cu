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

// mode 1 : d4c == d4b != d4a && d4a = (Window + WindowUpper + 1) && transposeT1 == 0
// mode 2 : d4c == d4b != d4a && d4a = (Window + WindowUpper + 1) && transposeT1 == 1
// mode 3 : d4c != d4b && d4a = d4b && dfc == (Window + WindowUpper + 1)

__global__ void mm4d_gpu_mode1_c_padz(float* a, float* b, float* c, int* dilation, int Window, int Padding, int d2, int d3, int d4a, int d4b, int d4c, int aSize, int bSize, int cSize) {
	int idx_a, idx_b, idx;
	int bx, by, tx, ty, B;

	bx = blockIdx.x;
	by = blockIdx.y;
	tx = threadIdx.x;
	ty = threadIdx.y;
	B = blockDim.x*blockDim.y;
	idx = by*gridDim.x*B + bx*B + ty*blockDim.x + tx;

	int l = idx / (d2 * d3 * d4c);
	int i = (idx / (d3 * d4c)) % d2;
	int q = (idx / d4c) % d3;
	int j = idx % d4c;
	int D = dilation[q];

	if (idx < cSize) {
		c[idx] = 0.0f;
		for (int k = 0; k < d4a; k++) {
			int condition = i + D * (k - Window);
			if (condition >= 0 && condition < d2) {
				idx_a = (((l * d2) + i) * d4a + k) * d3 + q;
				idx_b = (((l * d2) + i + D * (k - Window)) * d3 + q) *  d4b + j;
				if (idx_a < aSize && idx_b < bSize)	c[idx] += a[idx_a] * b[idx_b];
			}
		}
	}
}

__global__ void mm4d_gpu_mode3_c_padz(float* a, float* b, float* c, int* dilation, int Window, int Padding, int d2, int d3, int d4a, int d4b, int d4c, int aSize, int bSize, int cSize) {
	int idx_a, idx_b, idx;
	int bx, by, tx, ty, B;

	bx = blockIdx.x;
	by = blockIdx.y;
	tx = threadIdx.x;
	ty = threadIdx.y;
	B = blockDim.x*blockDim.y;
	idx = by*gridDim.x*B + bx*B + ty*blockDim.x + tx;

	int l = idx / (d2 * d3 * d4c);
	int i = (idx / (d3 * d4c)) % d2;
	int q = (idx / d4c) % d3;
	int j = idx % d4c;
	int D = dilation[q];

	if (idx < cSize) {
		c[0] = 0.0f;
		int condition = i + D * (j - Window);
		if (condition >= 0 && condition < d2) {
			for (int k = 0; k < d4a; k++) {
				idx_a = (((l * d2) + i) * d4a + k) * d3 + q;
				idx_b = (((l * d2) + i + D * (j - Window)) * d4b + k) * d3 + q;
				if (idx_a < aSize && idx_b < bSize)	c[idx] += a[idx_a] * b[idx_b];
			}
		}
	}
}

__global__ void mm4d_gpu_mode1_c(float* a, float* b, float* c, int* dilation, int Window, int Padding, int d2, int d3, int d4a, int d4b, int d4c, int aSize, int bSize, int cSize) {
	int idx_a, idx_b, idx;
	int bx, by, tx, ty, B;

	bx = blockIdx.x;
	by = blockIdx.y;
	tx = threadIdx.x;
	ty = threadIdx.y;
	B = blockDim.x*blockDim.y;
	idx = by*gridDim.x*B + bx*B + ty*blockDim.x + tx;

	int l = idx / (d2 * d3 * d4c);
	int i = (idx / (d3 * d4c)) % d2;
	int q = (idx / d4c) % d3;
	int j = idx % d4c;
	int D = dilation[q];

	if (idx < cSize) {
		c[idx] = 0.0f;
		for (int k = 0; k < d4a; k++) {
			int condition = i + D * (k - Window);
			if (condition >= 0 && condition < d2) {
				idx_a = (((l * d2) + i) * d4a + k) * d3 + q;
				idx_b = (((l * d2) + i + D * (k - Window)) * d3 + q) *  d4b + j;
				if (idx_a < aSize && idx_b < bSize)	c[idx] += a[idx_a] * b[idx_b];
			}
			else {
				c[idx] += Padding;
			}
		}
	}
}

__global__ void mm4d_gpu_mode2_c(float* a, float* b, float* c, int* dilation, int Window, int WindowUpper, int d2, int d3, int d4a, int d4b, int d4c, int aSize, int bSize, int cSize) {
	int idx_a, idx_b, idx;
	int bx, by, tx, ty, B;

	bx = blockIdx.x;
	by = blockIdx.y;
	tx = threadIdx.x;
	ty = threadIdx.y;
	B = blockDim.x*blockDim.y;
	idx = by*gridDim.x*B + bx*B + ty*blockDim.x + tx;

	int l = idx / (d2 * d3 * d4c);
	int i = (idx / (d3 * d4c)) % d2;
	int q = (idx / d4c) % d3;
	int j = idx % d4c;
	int D = dilation[q];

	if (idx < cSize) {
		c[idx] = 0.0f;
		for (int k = 0; k < d4a; k++) {
			int condition = i + D * (k - WindowUpper);
			if (condition >= 0 && condition < d2) {
				idx_a = (((l * d2) + i + D * (k - WindowUpper)) * d4a + WindowUpper + Window - k) * d3 + q;
				idx_b = (((l * d2) + i + D * (k - WindowUpper)) * d3 + q) *  d4b + j;
				if (idx_a < aSize && idx_b < bSize)	c[idx] += a[idx_a] * b[idx_b];
			}
		}
	}
}

__global__ void mm4d_gpu_mode3_c(float* a, float* b, float* c, int* dilation, int Window, int Padding, int d2, int d3, int d4a, int d4b, int d4c, int aSize, int bSize, int cSize) {
	int idx_a, idx_b, idx;
	int bx, by, tx, ty, B;

	bx = blockIdx.x;
	by = blockIdx.y;
	tx = threadIdx.x;
	ty = threadIdx.y;
	B = blockDim.x*blockDim.y;
	idx = by*gridDim.x*B + bx*B + ty*blockDim.x + tx;

	int l = idx / (d2 * d3 * d4c);
	int i = (idx / (d3 * d4c)) % d2;
	int q = (idx / d4c) % d3;
	int j = idx % d4c;
	int D = dilation[q];

	if (idx < cSize) {
		c[0] = 0.0f;
		int condition = i + D * (j - Window);
		for (int k = 0; k < d4a; k++) {
			if (condition >= 0 && condition < d2) {
				idx_a = (((l * d2) + i) * d4a + k) * d3 + q;
				idx_b = (((l * d2) + i + D * (j - Window)) * d4b + k) * d3 + q;
				if (idx_a < aSize && idx_b < bSize)	c[idx] += a[idx_a] * b[idx_b];
			}
			else {
				c[idx] += Padding;
			}
		}
	}
}

__global__ void mm4d_gpu_mode1_padz(float* a, float* b, float* c, int* dilation, int Window, int Padding, int d2, int d3, int d4a, int d4b, int d4c, int aSize, int bSize, int cSize) {
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
		for (int k = 0; k < d4a; k++) {
			int condition = i + D * (k - Window);
			if (condition >= 0 && condition < d2) {
				idx_a = (((l * d2) + i) * d3 + q) *  d4a + k;
				idx_b = (((l * d2) + i + D * (k - Window)) * d3 + q) *  d4b + j;
				if (idx_a < aSize && idx_b < bSize)	c[idx] += a[idx_a] * b[idx_b];
			}
		}
	}
}

__global__ void mm4d_gpu_mode3_padz(float* a, float* b, float* c, int* dilation, int Window, int Padding, int d2, int d3, int d4a, int d4b, int d4c, int aSize, int bSize, int cSize) {
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
		int condition = i + D * (j - Window);
		for (int k = 0; k < d4a; k++) {
			if (condition >= 0 && condition < d2) {
				idx_a = (((l * d2) + i) * d3 + q) * d4a + k;
				idx_b = (((l * d2) + i + D * (j - Window)) * d3 + q) *  d4b + k;
				if (idx_a < aSize && idx_b < bSize)	c[idx] += a[idx_a] * b[idx_b];
			}
		}
	}
}

__global__ void mm4d_gpu_mode1(float* a, float* b, float* c, int* dilation, int Window, int Padding, int d2, int d3, int d4a, int d4b, int d4c, int aSize, int bSize, int cSize) {
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
		for (int k = 0; k < d4a; k++) {
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

__global__ void mm4d_gpu_mode2(float* a, float* b, float* c, int* dilation, int Window, int WindowUpper, int d2, int d3, int d4a, int d4b, int d4c, int aSize, int bSize, int cSize) {
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
		for (int k = 0; k < d4a; k++) {
			int condition = i + D * (k - WindowUpper);
			if (condition >= 0 && condition < d2) {
				idx_a = (((l * d2) + i + D * (k - WindowUpper)) * d3 + q) *  d4a + WindowUpper + Window - k;
				idx_b = (((l * d2) + i + D * (k - WindowUpper)) * d3 + q) *  d4b + j;
				if (idx_a < aSize && idx_b < bSize)	c[idx] += a[idx_a] * b[idx_b];
			}
		}
	}
}

__global__ void mm4d_gpu_mode3(float* a, float* b, float* c, int* dilation, int Window, int Padding, int d2, int d3, int d4a, int d4b, int d4c, int aSize, int bSize, int cSize) {
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
		int condition = i + D * (j - Window);
		for (int k = 0; k < d4a; k++) {
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

void mm4d_cpu_mode1(float* a, float* b, float* c, int* dilation, int Window, int Padding, int d2, int d3, int d4a, int d4b, int d4c, int aSize, int bSize, int cSize) {
	int idx_a, idx_b, idx;

	for (idx = 0; idx < cSize; idx++) {
		int l = idx / (d2 * d3 * d4c);
		int i = (idx / (d3 * d4c)) % d2;
		int q = (idx / d4c) % d3;
		int j = idx % d4c;
		int D = dilation[q];
		c[idx] = 0.0f;

		for (int k = 0; k < d4a; k++) {
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

void mm4d_cpu_mode2(float* a, float* b, float* c, int* dilation, int Window, int WindowUpper, int d2, int d3, int d4a, int d4b, int d4c, int aSize, int bSize, int cSize) {
	int idx_a, idx_b, idx;

	for (idx = 0; idx < cSize; idx++) {
		int l = idx / (d2 * d3 * d4c);
		int i = (idx / (d3 * d4c)) % d2;
		int q = (idx / d4c) % d3;
		int j = idx % d4c;
		int D = dilation[q];
		c[idx] = 0.0f;

		for (int k = 0; k < d4a; k++) {
			int condition = i + D * (k - WindowUpper);
			if (condition >= 0 && condition < d2) {
				idx_a = (((l * d2) + i + D * (k - WindowUpper)) * d3 + q) *  d4a + WindowUpper + Window - k;
				idx_b = (((l * d2) + i + D * (k - WindowUpper)) * d3 + q) *  d4b + j;
				if (idx_a < aSize && idx_b < bSize)	c[idx] += a[idx_a] * b[idx_b];
			}
		}
	}
}

void mm4d_cpu_mode3(float* a, float* b, float* c, int* dilation, int Window, int Padding, int d2, int d3, int d4a, int d4b, int d4c, int aSize, int bSize, int cSize) {
	int idx_a, idx_b, idx;

	for (idx = 0; idx < cSize; idx++) {
		int l = idx / (d2 * d3 * d4c);
		int i = (idx / (d3 * d4c)) % d2;
		int q = (idx / d4c) % d3;
		int j = idx % d4c;
		int D = dilation[q];
		c[idx] = 0.0f;

		int condition = i + D * (j - Window);
		for (int k = 0; k < d4a; k++) {
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

void lformerMM(array4d_t<float>& input1, array4d_t<float>& input2, array4d_t<float>& output1, array1d_t<int>& dilation, array1d_t<int>& params, bool GPU){
	int* d = dilation.data_ptr;
	float* a = input1.data_ptr, *b = input2.data_ptr, *c = output1.data_ptr;
	int d1 = output1.last_count, d2 = output1.matrix_count, d3 = output1.row_count;
	int d4a = input1.col_count, d4b = input2.col_count, d4c = output1.col_count;
	int aSize = d1*d2*d3*d4a, bSize = d1*d2*d3*d4b, cSize = d1*d2*d3*d4c;
	int Window = params.data_ptr[0], WindowUpper = params.data_ptr[1], Padding = params.data_ptr[2], transposeT1 = params.data_ptr[3], coalesced = params.data_ptr[4];
	if (coalesced == 1) d4a = input1.row_count;

	dim3 blockSize(16, 16);
	dim3 gridSize((d1 * d2 + blockSize.x - 1) / blockSize.x, (d3 * d4c + blockSize.y - 1) / blockSize.y);

	//mode 1 and mode 3 are for forward
	//mode 2 and mode 3 are for backward
	if (d4c == d4b) {//mode 1 or mode 2
		if (transposeT1 == 0) {//mode 1: can be called in forward and backward
			if (!GPU) mm4d_cpu_mode1(a, b, c, d, Window, Padding, d2, d3, d4a, d4b, d4c, aSize, bSize, cSize);
			else if (coalesced == 1 && Padding == 0) mm4d_gpu_mode1_c_padz <<<gridSize, blockSize >>>(a, b, c, d, Window, Padding, d2, d3, d4a, d4b, d4c, aSize, bSize, cSize);
			else if (coalesced == 1) mm4d_gpu_mode1_c <<<gridSize, blockSize >>>(a, b, c, d, Window, Padding, d2, d3, d4a, d4b, d4c, aSize, bSize, cSize);
			else if (Padding == 0) mm4d_gpu_mode1_padz <<<gridSize, blockSize >>>(a, b, c, d, Window, Padding, d2, d3, d4a, d4b, d4c, aSize, bSize, cSize);
			else mm4d_gpu_mode1 <<<gridSize, blockSize >>>(a, b, c, d, Window, Padding, d2, d3, d4a, d4b, d4c, aSize, bSize, cSize);
		}
		else {// mode 2: called during gradient back-propagation
			if (!GPU) mm4d_cpu_mode2(a, b, c, d, Window, WindowUpper, d2, d3, d4a, d4b, d4c, aSize, bSize, cSize);
			else if (coalesced == 1) mm4d_gpu_mode2_c <<<gridSize, blockSize >>>(a, b, c, d, Window, WindowUpper, d2, d3, d4a, d4b, d4c, aSize, bSize, cSize);
			else mm4d_gpu_mode2 <<<gridSize, blockSize >>>(a, b, c, d, Window, WindowUpper, d2, d3, d4a, d4b, d4c, aSize, bSize, cSize);
		}
	}
	else {//mode 3: can be called in forward and backward
		if (coalesced == 1) d4b = input2.row_count;
		if (!GPU) mm4d_cpu_mode3(a, b, c, d, Window, Padding, d2, d3, d4a, d4b, d4c, aSize, bSize, cSize);
		else if (coalesced == 1 && Padding == 0) mm4d_gpu_mode3_c_padz <<<gridSize, blockSize>>>(a, b, c, d, Window, Padding, d2, d3, d4a, d4b, d4c, aSize, bSize, cSize);
		else if (coalesced == 1) mm4d_gpu_mode3_c <<<gridSize, blockSize>>>(a, b, c, d, Window, Padding, d2, d3, d4a, d4b, d4c, aSize, bSize, cSize);
		else if (Padding == 0) mm4d_gpu_mode3_padz <<<gridSize, blockSize>>>(a, b, c, d, Window, Padding, d2, d3, d4a, d4b, d4c, aSize, bSize, cSize);
		else mm4d_gpu_mode3 <<<gridSize, blockSize>>>(a, b, c, d, Window, Padding, d2, d3, d4a, d4b, d4c, aSize, bSize, cSize);
	}
}
