#include "kernels.hpp"
#include "check.hpp"

typedef ulong2 Pack128;

inline __device__ void Fetch128(Pack128& v, const Pack128* p) {
  asm volatile("ld.volatile.global.v2.u64 {%0,%1}, [%2];" : "=l"(v.x), "=l"(v.y) : "l"(p) : "memory");
}

inline __device__ void Store128(Pack128* p, Pack128& v) {
  asm volatile("st.volatile.global.v2.u64 [%0], {%1,%2};" :: "l"(p), "l"(v.x), "l"(v.y) : "memory");
}

__global__ void gpu_copy(float* dst, float* src, size_t nbytes) {
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = idx; i < nbytes / sizeof(Pack128); i += stride) {
    Pack128 v;
    Fetch128(v, ((Pack128*)src) + i); // host memory -> register
    Store128(((Pack128*)dst) + i, v); // register -> device global memory
  }
}

void gpu_copy_wrapper(float* dst, float* src, size_t nbytes) {
  dim3 grid = {4, 1, 1};
  dim3 block = {1024, 1, 1};
  gpu_copy<<<grid, block>>>(dst, src, nbytes);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

// Run with single block
__global__ void gpu_reduce(float* d_a, size_t nbytes, double *res) {
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  size_t stride = blockDim.x * gridDim.x;
  __shared__ double s_total[1024];
  s_total[threadIdx.x] = 0;
  for (size_t i = idx; i < nbytes / sizeof(float); i += stride) {
    s_total[threadIdx.x] += d_a[i];
  }
  __syncthreads();
  for (int i = blockDim.x / 2; i > 0; i >>= 1) {
    if (threadIdx.x < i) {
      s_total[threadIdx.x] += s_total[threadIdx.x + i];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    *res = s_total[0];
  }
}

double gpu_reduce_wrapper(float* d_a, size_t nbytes) {
  dim3 grid = {1, 1, 1};
  dim3 block = {1024, 1, 1};
  double* d_res;
  CHECK_CUDA(cudaMalloc(&d_res, sizeof(double)));
  gpu_reduce<<<grid, block>>>(d_a, nbytes, d_res);
  double h_res;
  CHECK_CUDA(cudaMemcpy((void*)&h_res, d_res, sizeof(double), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaFree(d_res));
  return h_res;
}