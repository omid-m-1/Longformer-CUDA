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
#include <stdexcept>

//Matrix A: input features/attention scores/gradients of enwik8/text8, 4 dimension: d1, d2, d3 and d4
#include <stdio.h>
#include "wtime.h"

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

const int d1 = 2; //batch size
const int d2 = 4096; //sequence length
const int d3 = 12; //attention head count
const int d4a = 64; //Fourth dimension of matrix A: hidden dimension that means feature length of one token
const int d4b = 64; //Fourth dimension of matrix B: --
const int d4c = 513;
const int aSize = d1*d2*d3*d4a;
const int bSize = d1*d2*d3*d4b;
const int cSize = d1*d2*d3*d4c;

const int d2d3d4c = d2*d3*d4c;
const int d3d4c = d3*d4c;
const int d3d4b = d3*d4b;
const int d4c_half = (d4c+1)/2;
//const int part = 16; // power of 2
#define  part 16 // power of 2
const int part_1 = part - 1;
const int d4c_part = (d4c + part_1)/part;

__device__ inline float warp_reduce(float val){
    for(int offset = 16; offset > 0; offset /= 2)
        val+= __shfl_down_sync (FULL_WARP_MASK,val,offset);
    return val;
}

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
				idx_a = (((l * d2) + i) * d3 + q) *  d4a + k;
				idx_b = (((l * d2) + i + D * (k - Window)) * d3 + q) *  d4b + j;
				if (idx_a < aSize && idx_b < bSize)	c[idx] += a[idx_a] * b[idx_b];
			}
		}
	}
}

// __global__ void mm4d_gpu_mode3_c_padz_old(float* a, float* b, float* c, int* dilation, int Window, int Padding, int d2, int d3, int d4a, int d4b, int d4c, int aSize, int bSize, int cSize) {
// 	int idx_a, idx_b, idx;
// 	int bx, by, tx, ty, B;
//
// 	bx = blockIdx.x;
// 	by = blockIdx.y;
// 	tx = threadIdx.x;
// 	ty = threadIdx.y;
// 	B = blockDim.x*blockDim.y;
// 	idx = by*gridDim.x*B + bx*B + ty*blockDim.x + tx;
//
// 	int l = idx / (d2 * d3 * d4c);
// 	int i = (idx / (d3 * d4c)) % d2;
// 	int q = (idx / d4c) % d3;
// 	int j = idx % d4c;
// 	int D = dilation[q];
//
// 	if (idx < cSize) {
// 		c[0] = 0.0f;
// 		int condition = i + D * (j - Window);
// 		if (condition >= 0 && condition < d2) {
// 			for (int k = 0; k < d4a; k++) {
// 				idx_a = (((l * d2) + i) * d3 + q) *  d4a + k;
// 				idx_b = (((l * d2) + i + D * (j - Window)) * d3 + q) *  d4b + k;
// 				if (idx_a < aSize && idx_b < bSize)	c[idx] += a[idx_a] * b[idx_b];
// 			}
// 		}
// 	}
// }

__global__ void mm4d_gpu_mode3_c_padz_old(float* a, float* b, float* c, int* dilation, int Window, int Padding) {
	int idx_a, idx_b, idx;
	int l, i, q, j, D;
	int condition, k, ld2, idx_a_base;
	float sum = 0.0f;

	idx = ((blockIdx.y*gridDim.x + blockIdx.x)*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x);
	if (idx >= cSize) return;

	i = (idx / d3d4c) % d2;
	q = (idx / d4c) % d3;
	j = idx % d4c;
	D = dilation[q];
	condition = i + D * (j - Window);
	if (condition < 0 || condition >= d2) return;
	l = idx / d2d3d4c;
	ld2 = l * d2;
	idx_a_base = ((ld2 + i) * d3 + q) * d4a;

	for (k = 0; k < d4a; k++) {
		idx_a = idx_a_base + k;
		idx_b = ((ld2 + condition) * d3 + q) * d4b + k;
		sum += a[idx_a] * b[idx_b];
	}
	c[idx] = sum;
}

//Newly written, very slow
__global__ void mm4d_gpu_mode3_pr(float* a, float* b, float* c, int* dilation, int Window, int Padding) {
	int idx_a, idx_b;
	int l, i, q, j, D;
	int condition, k, ld2, idx_a_base;
	float sum = 0.0f;
    
    int tid = threadIdx.x;
	int warp_id =  blockIdx.x * 4 + threadIdx.y; 
	if (warp_id >= cSize) return;

	i = (warp_id / d3d4c) % d2;
	q = (warp_id / d4c) % d3;
	j = warp_id % d4c;
	D = dilation[q];
	condition = i + D * (j - Window);
	if (condition < 0 || condition >= d2) return;
	l = warp_id / d2d3d4c;
	ld2 = l * d2;
	idx_a_base = ((ld2 + i) * d3 + q) * d4a;
    
	for (k = tid; k < d4a; k += 32) {
		idx_a = idx_a_base + k;
		idx_b = ((ld2 + condition) * d3 + q) * d4b + k;
		sum += a[idx_a] * b[idx_b];
	}
    sum = warp_reduce(sum);
	if (tid == 0) c[warp_id] = sum;
    
}

template <int valid_idx>
inline __device__ void small_exceptions(int idx_a_base, int idx, int warp_id, int d4a, float *a, float *b, float *sum, int idx_b_base) {
	float aa, bb;
	int idx_b;
	unsigned mask;
	for (int k = warp_id; k < d4a ; k += 32) {
		aa = __ldg(&a[idx_a_base + k]);
		for (int p = valid_idx; p < part; p++) {
			idx_b = idx_b_base + p*768 + k;
			bb = __ldg(&b[idx_b]);
			sum[p] += aa * bb;
		}
	}

	for (int offset = 16; offset > 0; offset /= 2) {
		mask = (1 << 2*offset) - 1;
		for (int p = valid_idx; p < part; p++) {
			sum[p] += __shfl_xor_sync(mask, sum[p], offset, 2*offset);
		}
	}
}

template <int valid_idx>
inline __device__ void big_exceptions(int idx_a_base, int idx, int warp_id, int d4a, float *a, float *b, float *sum, int idx_b_base) {
	float aa, bb;
	int idx_b;
	unsigned mask;
	for (int k = warp_id; k < d4a ; k += 32) {
		aa = __ldg(&a[idx_a_base + k]);
		for (int p = 0; p < valid_idx; p++) {
			idx_b = idx_b_base + p*768 + k;
			bb = __ldg(&b[idx_b]);
			sum[p] += aa * bb;
		}
	}

	for (int offset = 16; offset > 0; offset /= 2) {
		mask = (1 << 2*offset) - 1;
		for (int p = 0; p < valid_idx; p++) {
			sum[p] += __shfl_xor_sync(mask, sum[p], offset, 2*offset);
		}
	}
}

__global__ void mm4d_gpu_mode3_c_padz_new(float* a, float* b, float* c, int* dilation, int Window, int Padding) {
	int idx, idx_warp, d4c_idx;
	int i, q, j_first, j_last, D;
	int d4c_label;
	int condition_first, condition_last;
	idx_warp = ((blockIdx.y*gridDim.x + blockIdx.x)*blockDim.x*blockDim.y + threadIdx.y*blockDim.x + threadIdx.x);
	idx = idx_warp/32;
	d4c_idx = idx/d4c_part;
	d4c_idx = d4c_idx*d4c;
	idx = d4c_idx + (idx % d4c_part)*part;
	if (idx >= cSize) return;


	int warp_id = idx_warp%32;
	i = (idx / d3d4c) % d2;
	q = (idx / d4c) % d3;
	D = dilation[q];
	j_first = idx % d4c;
	condition_first = i + D * (j_first - Window);
	condition_last = condition_first + part_1 * D;
    if (condition_last < 0 || condition_first >= d2) return;
	j_last = j_first + part_1;
	int valid_idx;

	if (j_last < d4c) {
		if (condition_first >= 0 && condition_last < d2) {d4c_label = 3;}
		else if (condition_last < d2) {d4c_label = 1; valid_idx = -1*((condition_first + 1 - D)/D);}
		else {d4c_label = 2, valid_idx = (d2 - condition_first + D - 1)/D;}
	} else {
        d4c_label = 2; valid_idx = d4c - j_first;
    }
	int l = idx / d2d3d4c;
	int ld2 = l * d2;
	int idx_a_base = ((ld2 + i) * d3 + q) * d4a;
	int idx_b_base = ((ld2 + condition_first) * d3 + q) * d4b;
	int idx_diff = D * d3d4b;
	int p;
    //if (idx < 513 && idx > 513 + 256) printf("%d %d %d %d %d %d\n", idx_warp, j_first, condition_first, condition_last, idx_diff, Window);

	if (d4c_label == 3) {
		float sum[part] = {0.0f};
		float aa, bb;
		unsigned mask;
		for (int k = warp_id; k < d4a ; k += 32) {
			aa = __ldg(&a[idx_a_base + k]);
			for (p = 0; p < part; p++) {
				bb = __ldg(&b[idx_b_base + p*idx_diff + k]);
				sum[p] += aa * bb;
			}
		}

		for (int offset = 16; offset > 0; offset /= 2) {
			mask = (1 << 2*offset) - 1;
			for (int p = 0; p < part; p++) {
				sum[p] += __shfl_xor_sync(mask, sum[p], offset, 2*offset);
			}
		}
		if (warp_id == 0) {
			for (int p = 0; p < part; p++) {
				c[idx + p] = sum[p];
			}
		}
	}
	else if (d4c_label == 2) {
		float c_idx[part] = {0.0f};
		switch (valid_idx) {
			case 1: big_exceptions<1> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 2: big_exceptions<2> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 3: big_exceptions<3> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 4: big_exceptions<4> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 5: big_exceptions<5> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 6: big_exceptions<6> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 7: big_exceptions<7> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 8: big_exceptions<8> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 9: big_exceptions<9> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 10: big_exceptions<10> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 11: big_exceptions<11> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 12: big_exceptions<12> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 13: big_exceptions<13> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 14: big_exceptions<14> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 15: big_exceptions<15> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 16: big_exceptions<16> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 17: big_exceptions<17> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 18: big_exceptions<18> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 19: big_exceptions<19> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 20: big_exceptions<20> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 21: big_exceptions<21> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 22: big_exceptions<22> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 23: big_exceptions<23> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 24: big_exceptions<24> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 25: big_exceptions<25> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 26: big_exceptions<26> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 27: big_exceptions<27> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 28: big_exceptions<28> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 29: big_exceptions<29> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 30: big_exceptions<30> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 31: big_exceptions<31> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			default: break;
		}
		if (warp_id == 0) {
			for (p = 0; p < valid_idx; p++) {
				c[idx + p] = c_idx[p];
			}
		}
	}
	else if (d4c_label == 1) {
		float c_idx[part] = {0.0f};
		switch (valid_idx) {
			case 1: small_exceptions<1> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 2: small_exceptions<2> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 3: small_exceptions<3> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 4: small_exceptions<4> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 5: small_exceptions<5> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 6: small_exceptions<6> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 7: small_exceptions<7> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 8: small_exceptions<8> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 9: small_exceptions<9> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 10: small_exceptions<10> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 11: small_exceptions<11> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 12: small_exceptions<12> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 13: small_exceptions<13> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 14: small_exceptions<14> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 15: small_exceptions<15> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 16: small_exceptions<16> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 17: small_exceptions<17> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 18: small_exceptions<18> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 19: small_exceptions<19> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 20: small_exceptions<20> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 21: small_exceptions<21> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 22: small_exceptions<22> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 23: small_exceptions<23> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 24: small_exceptions<24> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 25: small_exceptions<25> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 26: small_exceptions<26> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 27: small_exceptions<27> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 28: small_exceptions<28> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 29: small_exceptions<29> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 30: small_exceptions<30> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			case 31: small_exceptions<31> (idx_a_base, idx, warp_id, d4a, a, b, c_idx, idx_b_base); break;
			default: break;
		}
		if (warp_id == 0) {
			for (p = valid_idx; p < part; p++) {
				c[idx + p] = c_idx[p];
			}
		}
	}
}

__global__ void mm4d_gpu_mode3_c_padz(float* a, float* b, float* c, int* dilation, int Window, int Padding, int d2, int d3, int d4a, int d4b, int d4c, int aSize, int bSize, int cSize) {
	int idx_a, idx_b, idx;
	int bx, by, tx, ty, B;
	float aa, bb;

	bx = blockIdx.x;
	by = blockIdx.y;
	tx = threadIdx.x;
	ty = threadIdx.y;
	B = blockDim.x*blockDim.y;
	idx = (by*gridDim.x*B + bx*B + ty*blockDim.x + tx)/32;
	int l = idx / (d2 * d3 * d4c);
	int i = (idx / (d3 * d4c)) % d2;
	int q = (idx / d4c) % d3;
	int j = idx % d4c;
	int D = dilation[q];
	int warp_id = (by*gridDim.x*B + bx*B + ty*blockDim.x + tx)%32;
	float sum = 0.0f;
	if (idx >= cSize) return;
	c[idx] = 0.0f;
	int condition = i + D * (j - Window);
	if (condition < 0 || condition >= d2) return;

	for (int ii = 0; ii < (d4a + abs(32 - d4a) % 32); ii += 32) {
		if (ii + warp_id < aSize) {idx_a = (((l * d2) + i) * d3 + q) *  d4a + ii + warp_id; aa = a[idx_a];} else {aa = 0.0f;}
		if (ii + warp_id < bSize) {idx_b = (((l * d2) + i + D * (j - Window)) * d3 + q) *  d4b + ii + warp_id; bb = b[idx_b];} else {bb = 0.0f;}
		sum += aa * bb;
	}
	__syncwarp();
	for (int offset = 16; offset > 0; offset /= 2) {
		sum += __shfl_xor_sync(0xffffffff, sum, offset, 32);
	}
	if (warp_id == 0) {
		c[idx] = sum;
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
				idx_a = (((l * d2) + i + D * (k - WindowUpper)) * d3 + q) *  d4a + WindowUpper + Window - k;
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
				idx_a = (((l * d2) + i) * d3 + q) *  d4a + k;
				idx_b = (((l * d2) + i + D * (j - Window)) * d3 + q) *  d4b + k;
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

void lformerMM_original(array4d_t<float>& input1, array4d_t<float>& input2, array4d_t<float>& output1, array1d_t<int>& dilation, array1d_t<int>& params, bool GPU){
	int* d = dilation.data_ptr;
	float* a = input1.data_ptr, *b = input2.data_ptr, *c = output1.data_ptr;
	int d1 = output1.last_count, d2 = output1.matrix_count, d3 = output1.row_count;
	int d4a = input1.col_count, d4b = input2.col_count, d4c = output1.col_count;
	int aSize = d1*d2*d3*d4a, bSize = d1*d2*d3*d4b, cSize = d1*d2*d3*d4c;
	int Window = params.data_ptr[0], WindowUpper = params.data_ptr[1], Padding = params.data_ptr[2], transposeT1 = params.data_ptr[3], coalesced = params.data_ptr[4];
	printf("params: %d %d %d %d %d\n", Window, WindowUpper, Padding, transposeT1, coalesced);

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
		if (!GPU) mm4d_cpu_mode3(a, b, c, d, Window, Padding, d2, d3, d4a, d4b, d4c, aSize, bSize, cSize);
		else if (coalesced == 1 && Padding == 0) mm4d_gpu_mode3_c_padz <<<gridSize, blockSize>>>(a, b, c, d, Window, Padding, d2, d3, d4a, d4b, d4c, aSize, bSize, cSize);
		else if (coalesced == 1) mm4d_gpu_mode3_c <<<gridSize, blockSize>>>(a, b, c, d, Window, Padding, d2, d3, d4a, d4b, d4c, aSize, bSize, cSize);
		else if (Padding == 0) mm4d_gpu_mode3_padz <<<gridSize, blockSize>>>(a, b, c, d, Window, Padding, d2, d3, d4a, d4b, d4c, aSize, bSize, cSize);
		else mm4d_gpu_mode3 <<<gridSize, blockSize>>>(a, b, c, d, Window, Padding, d2, d3, d4a, d4b, d4c, aSize, bSize, cSize);
	}
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(cudaStatus));
}

void lformerMM(array4d_t<float>& input1, array4d_t<float>& input2, array4d_t<float>& output1, array1d_t<int>& dilation, array1d_t<int>& params, bool GPU){
	int* d = dilation.data_ptr;
	float* a = input1.data_ptr, *b = input2.data_ptr, *c = output1.data_ptr;
	int Window = params.data_ptr[0], WindowUpper = params.data_ptr[1], Padding = params.data_ptr[2], transposeT1 = params.data_ptr[3], coalesced = params.data_ptr[4];
	//printf("params: %d %d %d %d %d\n", Window, WindowUpper, Padding, transposeT1, coalesced);

	dim3 blockSize(8, 8);
	dim3 gridSize_c((((32 + part_1)/part)*(d1 * d2 + blockSize.x - 1) / blockSize.x) + 100, (d3 * d4c + blockSize.y - 1) / blockSize.y);
	dim3 gridSize((d1 * d2 + blockSize.x - 1) / blockSize.x, (d3 * d4c + blockSize.y - 1) / blockSize.y);
    dim3 blocks(32, 4);
    dim3 grids ((d1*d2*d3*d4c + 3) /4, 1);
    //printf("%d, %d %d, %d, %d %d %d %d %d\n", d1, d2, d3, d4a, d4b, d4c, grids.x, gridSize_c.x, gridSize_c.y);
    //double start = mywtime();

	if (d4c != d4b) { //mode3
		if (coalesced == 1) 
            mm4d_gpu_mode3_c_padz_new<<<gridSize_c, blockSize>>>(a, b, c, d, Window, Padding);
            //mm4d_gpu_mode3_pr<<<grids, blocks>>>(a, b, c, d, Window, Padding);
		else 
            mm4d_gpu_mode3_c_padz_old<<<gridSize, blockSize>>>(a, b, c, d, Window, Padding);
	}
	else {
		throw std::invalid_argument("coalesced kernel for mode 1 and 2 is not implemented.");
	}

    /*
    cudaDeviceSynchronize();
    double end = mywtime();
    printf("cuda time = %f\n", end - start);
    */
}
