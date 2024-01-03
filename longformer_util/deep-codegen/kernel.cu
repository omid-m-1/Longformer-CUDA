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
#include "helper_math.h"

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

#define  part 64
#define  m3_part 64

template <int dim_wcount>
__device__ inline float subwarp_reduce(float val){
    for(int offset = 16/dim_wcount; offset > 0; offset /= 2)
        val+= __shfl_down_sync (FULL_WARP_MASK,val,offset, 32/dim_wcount);
    return val;
}

__global__ void mm4d_mode1_pr4(float* a, float* b, float* c, int* dilation, int d4a, int d4b, int d4c, int d2, int d3, int Window) {
  int l, i, q, j, D;
  int ld2;

  /*
  threadIdx.y = [0, 4)     => 4 i
  blockIdx.x = [0, 513)    => j
  blockIdx.y = [0, 4096/4) => 4 rows (i).
  blockIdx.z = [0, 12)     => heads
  */

  int tid = threadIdx.x;
  int abs_i = blockIdx.y* blockDim.y + threadIdx.y;

  i = abs_i % d2; //which token sequence
  l = abs_i/d2; //which mini-batch
  ld2 = l * d2;
  j = blockIdx.x*part; //which attention result
  q = blockIdx.z; //which head

  D = dilation[q];

  int k_lower = max(Window - i / D, 0);
  int k_upper = min(Window + (d2 - i) / D, d2);
  k_upper = min(k_upper, d4a);

  int idx_a_base = ((ld2 + i) * d3 + q) * d4a + k_lower;
  int idx_b_base = ((((l*d4b) + j)*d3) + q) * d2 + i + D * (k_lower - Window);
  int index = ((ld2 + i)*d3 + q)*d4c + j;

  float a_value = 0.0f;
  //float b_value = 0.0f;
  float sum[part] = {0.0f};

    for (int kk = 0; kk < k_upper - k_lower - tid; kk += 32) {
      a_value = a[idx_a_base + kk + tid];
      for (int jj = 0; jj < part; jj++) {
        //b_value = b[idx_b_base + jj*d3*d2 + (kk + tid)*D];
        sum[jj] += a_value * b[idx_b_base + jj*d3*d2 + (kk + tid)*D];
      }
    }
    for (int jj = 0; jj < part; jj++) sum[jj] = subwarp_reduce<1>(sum[jj]);
    for (int jj = 0; jj < part; jj++) {if(tid == 0) c[index + jj] = sum[jj];}
}

__global__ void mm4d_mode1_pr4_noMerge(float* a, float* b, float* c, int* dilation, int d4a, int d4b, int d4c, int d2, int d3, int Window) {
  int l, i, q, j, D;
  int ld2;

  /*
  threadIdx.y = [0, 4)     => 4 i
  blockIdx.x = [0, 513)    => j
  blockIdx.y = [0, 4096/4) => 4 rows (i).
  blockIdx.z = [0, 12)     => heads
  */

  int tid = threadIdx.x;
  int abs_i = blockIdx.y* blockDim.y + threadIdx.y;

  i = abs_i % d2; //which token sequence
  l = abs_i/d2; //which mini-batch
  ld2 = l * d2;
  j = blockIdx.x*part; //which attention result
  q = blockIdx.z; //which head

  D = dilation[q];

  int k_lower = max(Window - i / D, 0);
  int k_upper = min(Window + (d2 - i) / D, d2);
  k_upper = min(k_upper, d4a);

  int idx_a_base = ((ld2 + i) * d3 + q) * d4a + k_lower;
  int idx_b_base = ((((l*d4b) + j)*d3) + q) * d2 + i + D * (k_lower - Window);
  int index = ((ld2 + i)*d3 + q)*d4c + j;

  float a_value[17] = {0.0f};
  float b_value[part] = {0.0f};
  float sum[part] = {0.0f};

    for (int kk = 0; kk < k_upper - k_lower - tid; kk += 32) a_value[kk/32] = a[idx_a_base + kk + tid];
    for (int kk = 0; kk < k_upper - k_lower - tid; kk += 32) {
      for (int jj = 0; jj < part; jj++) b_value[jj] = b[idx_b_base + jj*d3*d2 + (kk + tid)*D];
      for (int jj = 0; jj < part; jj++) sum[jj] += a_value[kk/32] * b_value[jj];
    }
    for (int jj = 0; jj < part; jj++) sum[jj] = subwarp_reduce<1>(sum[jj]);
    for (int jj = 0; jj < part; jj++) {if(tid == 0) c[index + jj] = sum[jj];}
}

__global__ void mm4d_mode1_pr4_noDilation(float* a, float* b, float* c, int d4a, int d4b, int d4c, int d2, int d3, int Window) {
  int l, i, q, j;
  int ld2;

  /*
  threadIdx.y = [0, 4)     => 4 i
  blockIdx.x = [0, 513)    => j
  blockIdx.y = [0, 4096/4) => 4 rows (i).
  blockIdx.z = [0, 12)     => heads
  */

  int tid = threadIdx.x;
  int abs_i = blockIdx.y* blockDim.y + threadIdx.y;

  i = abs_i % d2; //which token sequence
  l = abs_i/d2; //which mini-batch
  ld2 = l * d2;
  j = blockIdx.x*part; //which attention result
  q = blockIdx.z; //which head

  int k_lower = max(Window - i, 0);
  int k_upper = min(Window + d2 - i, d2);
  k_upper = min(k_upper, d4a);

  int idx_a_base = ((ld2 + i) * d3 + q) * d4a + k_lower;
  int idx_b_base = ((((l*d4b) + j)*d3) + q) * d2 + i + k_lower - Window;
  int index = ((ld2 + i)*d3 + q)*d4c + j;

  float a_value[17] = {0.0f};
  float b_value[part] = {0.0f};
  float sum[part] = {0.0f};

    for (int kk = 0; kk < k_upper - k_lower - tid; kk += 32) a_value[kk/32] = a[idx_a_base + kk + tid];
    for (int kk = 0; kk < k_upper - k_lower - tid; kk += 32) {
      for (int jj = 0; jj < part; jj++) b_value[jj] = b[idx_b_base + jj*d3*d2 + kk + tid];
      for (int jj = 0; jj < part; jj++) sum[jj] += a_value[kk/32] * b_value[jj];
    }
    for (int jj = 0; jj < part; jj++) sum[jj] = subwarp_reduce<1>(sum[jj]);
    for (int jj = 0; jj < part; jj++) {if(tid == 0) c[index + jj] = sum[jj];}
}

__global__ void mm4d_mode2_pr4(float* a, float* b, float* c, int* dilation, int d4a, int d4b, int d4c, int d2, int d3, int Window, int WindowUpper) {
  int l, i, q, j, D;
  int ld2;

  /*
  threadIdx.y = [0, 4)     => 4 i
  blockIdx.x = [0, 513)    => j
  blockIdx.y = [0, 4096/4) => 4 rows (i).
  blockIdx.z = [0, 12)     => heads
  */

  int tid = threadIdx.x;
  int abs_i = blockIdx.y* blockDim.y + threadIdx.y;

  i = abs_i % d2; //which token sequence
  l = abs_i/d2; //which mini-batch
  ld2 = l * d2;
  j = blockIdx.x*part; //which attention result
  q = blockIdx.z; //which head

  D = dilation[q];

  int k_lower = max(WindowUpper - i / D, 0);
  int k_upper = min(WindowUpper + (d2 - i) / D, d2);
  k_upper = min(k_upper, d4a);

  int idx_a_base = ((ld2 + i + D * (k_lower - WindowUpper)) * d3 + q) * d4a + WindowUpper + Window - k_lower;
  int idx_b_base = ((((l*d4b) + j)*d3) + q) * d2 + i + D * (k_lower - WindowUpper);
  int index = ((ld2 + i)*d3 + q)*d4c + j;

  float a_value = 0.0f;
  float sum[part] = {0.0f};

  for (int kk = 0; kk < k_upper - k_lower - tid; kk += 32) {
    a_value = a[idx_a_base + (kk + tid)*(D*d3*d4a - 1)];
    for (int jj = 0; jj < part; jj++) sum[jj] += a_value * b[idx_b_base + jj*d3*d2 + (kk + tid)*D];
  }
  for (int jj = 0; jj < part; jj++) {
    sum[jj] = subwarp_reduce<1>(sum[jj]);
    if(tid == 0) c[index + jj] = sum[jj];
  }
}

__global__ void mm4d_mode2_pr4_noMerge(float* a, float* b, float* c, int* dilation, int d4a, int d4b, int d4c, int d2, int d3, int Window, int WindowUpper) {
  int l, i, q, j, D;
  int ld2;

  /*
  threadIdx.y = [0, 4)     => 4 i
  blockIdx.x = [0, 513)    => j
  blockIdx.y = [0, 4096/4) => 4 rows (i).
  blockIdx.z = [0, 12)     => heads
  */

  int tid = threadIdx.x;
  int abs_i = blockIdx.y* blockDim.y + threadIdx.y;

  i = abs_i % d2; //which token sequence
  l = abs_i/d2; //which mini-batch
  ld2 = l * d2;
  j = blockIdx.x*part; //which attention result
  q = blockIdx.z; //which head

  D = dilation[q];

  int k_lower = max(WindowUpper - i / D, 0);
  int k_upper = min(WindowUpper + (d2 - i) / D, d2);
  k_upper = min(k_upper, d4a);

  int idx_a_base = ((ld2 + i + D * (k_lower - WindowUpper)) * d3 + q) * d4a + WindowUpper + Window - k_lower;
  int idx_b_base = ((((l*d4b) + j)*d3) + q) * d2 + i + D * (k_lower - WindowUpper);
  int index = ((ld2 + i)*d3 + q)*d4c + j;

  float a_value[17] = {0.0f};
  float b_value[part] = {0.0f};
  float sum[part] = {0.0f};

  for (int kk = 0; kk < k_upper - k_lower - tid; kk += 32) a_value[kk/32] = a[idx_a_base + (kk + tid)*(D*d3*d4a - 1)];
  for (int kk = 0; kk < k_upper - k_lower - tid; kk += 32) {
    for (int jj = 0; jj < part; jj++) b_value[jj] = b[idx_b_base + jj*d3*d2 + (kk + tid)*D];
    for (int jj = 0; jj < part; jj++) sum[jj] += a_value[kk/32] * b_value[jj];
  }
  for (int jj = 0; jj < part; jj++) {
    sum[jj] = subwarp_reduce<1>(sum[jj]);
    if(tid == 0) c[index + jj] = sum[jj];
  }
}

__global__ void mm4d_mode2_pr4_noDilation(float* a, float* b, float* c, int d4a, int d4b, int d4c, int d2, int d3, int Window, int WindowUpper) {
  int l, i, q, j;
  int ld2;

  /*
  threadIdx.y = [0, 4)     => 4 i
  blockIdx.x = [0, 513)    => j
  blockIdx.y = [0, 4096/4) => 4 rows (i).
  blockIdx.z = [0, 12)     => heads
  */

  int tid = threadIdx.x;
  int abs_i = blockIdx.y* blockDim.y + threadIdx.y;

  i = abs_i % d2; //which token sequence
  l = abs_i/d2; //which mini-batch
  ld2 = l * d2;
  j = blockIdx.x*part; //which attention result
  q = blockIdx.z; //which head

  int k_lower = max(WindowUpper - i, 0);
  int k_upper = min(WindowUpper + d2 - i, d2);
  k_upper = min(k_upper, d4a);

  int idx_a_base = ((ld2 + i + k_lower - WindowUpper) * d3 + q) * d4a + WindowUpper + Window - k_lower;
  int idx_b_base = ((((l*d4b) + j)*d3) + q) * d2 + i + k_lower - WindowUpper;
  int index = ((ld2 + i)*d3 + q)*d4c + j;

  float a_value[17] = {0.0f};
  float b_value[part] = {0.0f};
  float sum[part] = {0.0f};

  for (int kk = 0; kk < k_upper - k_lower - tid; kk += 32) a_value[kk/32] = a[idx_a_base + (kk + tid)*(d3*d4a - 1)];
  for (int kk = 0; kk < k_upper - k_lower - tid; kk += 32) {
    for (int jj = 0; jj < part; jj++) b_value[jj] = b[idx_b_base + jj*d3*d2 + kk + tid];
    for (int jj = 0; jj < part; jj++) sum[jj] += a_value[kk/32] * b_value[jj];
  }
  for (int jj = 0; jj < part; jj++) {
    sum[jj] = subwarp_reduce<1>(sum[jj]);
    if(tid == 0) c[index + jj] = sum[jj];
  }
}

//data-reuse across i dimension
__global__ void mm4d_mode3_pr4(float* a, float* b, float* c, int* dilation, int m3_d4a, int m3_d4b, int m3_d4c, int d2, int d3, int Window) {
	int l, i, q, j, D;
	int ld2;

    /*
    threadIdx.y = [0, 4)     => 4 i
    blockIdx.x = [0, 513)    => j
    blockIdx.y = [0, 4096/4) => 4 rows (i).
    blockIdx.z = [0, 12)     => heads
    */

    int tid = threadIdx.x;
    int abs_i = blockIdx.y* blockDim.y + threadIdx.y;

    i = abs_i % d2; //which token sequence
	l = abs_i/d2; //which mini-batch
	ld2 = l * d2;
    j = blockIdx.x*m3_part; //which attention result
    q = blockIdx.z; //which head

	D = dilation[q];

    int j_upper = min(j+m3_part, m3_d4c);
    int idx_a = ((ld2 + i) * d3 + q) * m3_d4a;

    int dim_wid = tid / 16;
    int dim_tid = tid % 16;
    float4 a_value = ((float4*)(a + idx_a))[dim_tid];

    for (int jj = j + dim_wid; jj < j_upper; jj += 2) {
	    int condition = (i + D * (jj - Window));
	    float4 b_value;
        if (condition >= 0 && condition < d2) {
            int idx_b = ((ld2 + condition) * d3 + q) * m3_d4b;
            b_value = ((float4*)(b + idx_b))[dim_tid];
        }

        float4 sum  = a_value * b_value;
        float  dot  = sum.x + sum.y + sum.z + sum.w;
        dot         = subwarp_reduce<2>(dot);

	    if (condition >= 0 && condition < d2) {
            int index = ((ld2 + i)*d3 + q)*m3_d4c + jj;
            if (dim_tid == 0) c[index] = dot;
        }
    }
}

//data-reuse across i dimension
__global__ void mm4d_mode3_pr4_noDilation(float* a, float* b, float* c, int m3_d4a, int m3_d4b, int m3_d4c, int d2, int d3, int Window) {
  int l, i, q, j;
	int ld2;

    /*
    threadIdx.y = [0, 4)     => 4 i
    blockIdx.x = [0, 513)    => j
    blockIdx.y = [0, 4096/4) => 4 rows (i).
    blockIdx.z = [0, 12)     => heads
    */

    int tid = threadIdx.x;
    int abs_i = blockIdx.y* blockDim.y + threadIdx.y;

    i = abs_i % d2; //which token sequence
	l = abs_i/d2; //which mini-batch
	ld2 = l * d2;
    j = blockIdx.x*m3_part; //which attention result
    q = blockIdx.z; //which head

    int j_upper = min(j+m3_part, m3_d4c);
    int idx_a = ((ld2 + i) * d3 + q) * m3_d4a;

    int dim_wid = tid / 16;
    int dim_tid = tid % 16;
    float4 a_value = ((float4*)(a + idx_a))[dim_tid];

    for (int jj = j + dim_wid; jj < j_upper; jj += 2) {
	    int condition = (i + jj - Window);
	    float4 b_value;
        if (condition >= 0 && condition < d2) {
            int idx_b = ((ld2 + condition) * d3 + q) * m3_d4b;
            b_value = ((float4*)(b + idx_b))[dim_tid];
        }

        float4 sum  = a_value * b_value;
        float  dot  = sum.x + sum.y + sum.z + sum.w;
        dot         = subwarp_reduce<2>(dot);

	    if (condition >= 0 && condition < d2) {
            int index = ((ld2 + i)*d3 + q)*m3_d4c + jj;
            if (dim_tid == 0) c[index] = dot;
        }
    }
}

//data-reuse across i and j dimensions
__global__ void mm4d_gpu_mode3_pr5(float* a, float* b, float* c, int* dilation, int m3_d4a, int m3_d4b, int m3_d4c, int d2, int d3, int Window) {
	__shared__ float4 s[128];//8 i = 8*64 = 512 float = 128 float4
    int l, i, q, j, D;
	int ld2;

    /*
    threadIdx.y = [0, 4)     => 4 i
    blockIdx.x = [0, 513/part)    => j
    blockIdx.y = [0, 4096/8) => 8 rows (i).
    blockIdx.z = [0, 12)     => heads
    2 ==> means 16 threads of a warp works on 64 dim, creating two dim_warp.
    */

    int abs_i = (blockIdx.y* blockDim.y)*2;//

    i = abs_i % d2; //which token sequence
	l = abs_i/d2; //which mini-batch
	ld2 = l * d2;
    j = blockIdx.x*m3_part; //which attention result
    q = blockIdx.z; //which head

	D = dilation[q];

    int j_upper = min(j+m3_part, m3_d4c);

    int tid = threadIdx.x;
    int dim_tid = tid % 16;
    int dim_wid = tid / 16; //
    int local_wid = threadIdx.y*2 + dim_wid;
    int dim_wcount = blockDim.y*2;

    int idx_a = ((ld2 + (i + local_wid)) * d3 + q) * m3_d4a;
    float4 a_value = ((float4*)(a + idx_a))[dim_tid];
    s[(local_wid)*16 + dim_tid] = a_value;

    __syncthreads();

    float4 b_value;
    float dot = 0.0f;
    int condition;

    for (int jj = j + local_wid; jj < j_upper; jj += dim_wcount) {
        for (int ii = 0; ii < dim_wcount; ++ii) {
            condition = (i + ii + D * (jj - Window));
            a_value     = s[ii*16 + dim_tid];
            if (condition >= 0 && condition < d2) {
                int idx_b = ((ld2 + condition) * d3 + q) * m3_d4b;
                b_value = ((float4*)(b + idx_b))[dim_tid];
                b_value = a_value * b_value;
                dot     = b_value.x + b_value.y + b_value.z + b_value.w;
            }
            dot         = subwarp_reduce<2>(dot);

            if (condition >= 0 && condition < d2) {
                if (dim_tid == 0) c[((ld2 + i + ii)*d3 + q)*m3_d4c + jj] = dot;
            }
        }
    }
}

void mm4d_cpu_mode1(float* a, float* b, float* c, int* dilation, int d4a, int d4b, int d4c, int d2, int d3, int cSize, int Window) {
	int idx_a, idx_b, idx;
  float sum = 0.0f;

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
				sum += a[idx_a] * b[idx_b];
      }
		}
    c[idx] = sum;
	}
}

void mm4d_cpu_mode2(float* a, float* b, float* c, int* dilation, int d4a, int d4b, int d4c, int d2, int d3, int cSize, int Window, int WindowUpper) {
	int idx_a, idx_b, idx;
  float sum = 0.0f;

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
				sum += a[idx_a] * b[idx_b];
      }
		}
    c[idx] = sum;
	}
}

void mm4d_cpu_mode3(float* a, float* b, float* c, int* dilation, int d4a, int d4b, int d4c, int d2, int d3, int cSize, int Window) {
	int idx_a, idx_b, idx;
  float sum = 0.0f;

	for (idx = 0; idx < cSize; idx++) {
		int l = idx / (d2 * d3 * d4c);
		int i = (idx / (d3 * d4c)) % d2;
		int q = (idx / d4c) % d3;
		int j = idx % d4c;
		int D = dilation[q];
		c[idx] = 0.0f;

		int condition = i + D * (j - Window);
		if (condition >= 0 && condition < d2) {
			for (int k = 0; k < d4a; k++) {
				idx_a = (((l * d2) + i) * d3 + q) * d4a + k;
				idx_b = (((l * d2) + condition) * d3 + q) * d4b + k;
				sum += a[idx_a] * b[idx_b];
			}
		}
    c[idx] = sum;
	}
}

void lformerMM(array4d_t<float>& input1, array4d_t<float>& input2, array4d_t<float>& output1, array1d_t<int>& dilation, bool no_dilation, int Window, int WindowUpper, bool transposeT1, bool transposeT2, bool GPU) {
  int* d = dilation.data_ptr; float* a = input1.data_ptr; float* b = input2.data_ptr; float* c = output1.data_ptr;
  int d1 = output1.last_count, d2 = output1.matrix_count, d3 = output1.row_count;
  int d4a = input1.col_count, d4c = output1.col_count; int d4b;
  if (transposeT2) d4b = input2.matrix_count;
  else d4b = input2.col_count;
  int aSize = d1*d2*d3*d4a, bSize = d1*d2*d3*d4b, cSize = d1*d2*d3*d4c;

  //double start = mywtime();
  dim3 blocks(32, 4);
  dim3 blockSize(8, 8);

  if (d4c == d4b) { //mode 1 and mode2
    dim3 grids_m((d4c + part - 1)/part, d1*((d2 + 3)/4), d3);
    if (transposeT1 == 0) {
      if (no_dilation) mm4d_mode1_pr4_noDilation<<<grids_m, blocks>>>(a, b, c, d4a, d4b, d4c, d2, d3, Window);
      else mm4d_mode1_pr4<<<grids_m, blocks>>>(a, b, c, d, d4a, d4b, d4c, d2, d3, Window);
    }
    else if (no_dilation) mm4d_mode2_pr4_noDilation<<<grids_m, blocks>>>(a, b, c, d4a, d4b, d4c, d2, d3, Window, WindowUpper);
    else mm4d_mode2_pr4<<<grids_m, blocks>>>(a, b, c, d, d4a, d4b, d4c, d2, d3, Window, WindowUpper);
  }
  else { //mode3
    //dim3 grids ((d4c + part - 1)/part, d1*((d2 + 7)/8), d3); //pr5
    dim3 grids ((d4c + m3_part - 1)/m3_part, d1*((d2 + 3)/4), d3); //pr4
    if (no_dilation) mm4d_mode3_pr4_noDilation<<<grids, blocks>>>(a, b, c, d4a, d4b, d4c, d2, d3, Window);
    else mm4d_mode3_pr4<<<grids, blocks>>>(a, b, c, d, d4a, d4b, d4c, d2, d3, Window);
  }

  /*
   cudaDeviceSynchronize();
   double end = mywtime();
   printf("cuda time = %f\n", (end - start)/20.0);
  */
}
