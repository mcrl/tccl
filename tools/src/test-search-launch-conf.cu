#include <cstdio>
#include <chrono>
#include <cassert>
#include <vector>
#include <random>
#include <algorithm>
#include <numa.h>
#include <numaif.h>
#include <thread>
#include <spdlog/spdlog.h>

#define CHECK_CUDA(call)                                              \
  do {                                                                \
    cudaError_t status_ = call;                                       \
    if (status_ != cudaSuccess) {                                     \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(status_));                           \
      assert(0);                                                  \
    }                                                                 \
  } while (0)

typedef ulong2 Pack128;

inline __device__ void Fetch128(Pack128& v, const Pack128* p) {
  asm volatile("ld.volatile.global.v2.u64 {%0,%1}, [%2];" : "=l"(v.x), "=l"(v.y) : "l"(p) : "memory");
}

inline __device__ void Store128(Pack128* p, Pack128& v) {
  asm volatile("st.volatile.global.v2.u64 [%0], {%1,%2};" :: "l"(p), "l"(v.x), "l"(v.y) : "memory");
}

__device__ uint get_smid(void) {

  uint ret;

  asm("mov.u32 %0, %smid;" : "=r"(ret) );

  return ret;

}

__global__ void gpu_fill_buffer(float* dst, size_t nbytes) {
  size_t idx = threadIdx.x;
  size_t stride = blockDim.x;
  size_t nelem = nbytes / sizeof(float);
  for (size_t i = idx; i < nelem; i += stride) {
    dst[i] = i;
  }
}

static std::chrono::steady_clock cpu_clock;
typedef std::chrono::time_point<std::chrono::steady_clock> tp;

std::chrono::time_point<std::chrono::steady_clock> get_time() {
  return cpu_clock.now();
}

size_t get_duration_us(
    std::chrono::time_point<std::chrono::steady_clock> start,
    std::chrono::time_point<std::chrono::steady_clock> end
    ) {
  return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

void AllocGPUMem(void** ptr, size_t nbytes, int gpu_idx) {
  int org_idx;
  CHECK_CUDA(cudaGetDevice(&org_idx));
  CHECK_CUDA(cudaSetDevice(gpu_idx));
  CHECK_CUDA(cudaMalloc(ptr, nbytes));
  CHECK_CUDA(cudaSetDevice(org_idx));
}

void AllocCPUMemAccessibleFromGPU(void** dptr, void** hptr, size_t nbytes, int numa_idx) {
  *hptr = (float*)numa_alloc_onnode(nbytes, numa_idx);
  CHECK_CUDA(cudaHostRegister(*hptr, nbytes, cudaHostRegisterMapped));
  CHECK_CUDA(cudaHostGetDevicePointer(dptr, *hptr, 0));
}

void fill_random_float(float* buf, size_t numel) {
  std::default_random_engine eng(std::chrono::system_clock::now().time_since_epoch().count());
  std::uniform_real_distribution<float> dis(-1, 1);
  for (size_t i = 0; i < numel; ++i) {
    buf[i] = dis(eng);
  }
}

void fill_random_float_gpu(float* buf, size_t numel) {
  size_t nbytes = numel * sizeof(float);
  float *h_tmp = (float*)malloc(nbytes);
  fill_random_float(h_tmp, numel);
  CHECK_CUDA(cudaMemcpy(buf, h_tmp, nbytes, cudaMemcpyHostToDevice));
  free(h_tmp);
}

__global__ void gpu_copy_single_channel(float* dst, float* src, size_t nbytes) {
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  size_t stride = blockDim.x * gridDim.x;
  size_t nelem = nbytes / sizeof(Pack128);
  for (size_t i = idx; i < nelem; i += stride) {
    Pack128 v;
    Fetch128(v, ((Pack128*)src) + i);
    Store128(((Pack128*)dst) + i, v);
  }
}

__global__ void gpu_copy_full_v2(float* dst, float* src, size_t nbytes) {
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  size_t stride = blockDim.x * gridDim.x;
  size_t nelem = nbytes / sizeof(Pack128);
  #define UNROLL 4
  for (size_t i = idx; i < nelem; i += stride * UNROLL) {
    Pack128 v[UNROLL];
    for (int j = 0; j < UNROLL; ++j) {
      if (i + stride * j < nelem)
        v[j] = ((Pack128*)src)[i + stride * j];
    }
    for (int j = 0; j < UNROLL; ++j) {
      if (i + stride * j < nelem)
        ((Pack128*)dst)[i + stride * j] = v[j];
    }
  }
  #undef UNROLL
}

int main(int argc, char ** argv) {
  int num_gpus = 4;
  int src_gpu = 0;
  int dst_gpu = 2;

  for (int i = 0; i < num_gpus; ++i) {
    for (int j = 0; j < num_gpus; ++j) {
      if (i != j) {
        int can_access;
        CHECK_CUDA(cudaDeviceCanAccessPeer(&can_access, i, j));
        if (can_access) {
          CHECK_CUDA(cudaSetDevice(i));
          CHECK_CUDA(cudaDeviceEnablePeerAccess(j, 0));
        }
      }
    }
  }

  size_t nbytes = 64 * 1024 * 1024; // 64MB

  float *srcbuf, *dstbuf;
  AllocGPUMem((void**)&srcbuf, nbytes, src_gpu);
  fill_random_float_gpu(srcbuf, nbytes / sizeof(float));
  AllocGPUMem((void**)&dstbuf, nbytes, dst_gpu);
  fill_random_float_gpu(dstbuf, nbytes / sizeof(float));

  int niter = 10;
  int th_stride = 32;
  for (int num_th = th_stride; num_th <= 1024; num_th += th_stride) {
    std::vector<size_t> us_db;
    for (int iter = 1; iter <= niter; ++iter) {
      CHECK_CUDA(cudaSetDevice(src_gpu));

      CHECK_CUDA(cudaDeviceSynchronize());
      auto st = get_time();

      //gpu_copy_single_channel<<<1, num_th>>>(dstbuf, srcbuf, nbytes);
      gpu_copy_full_v2<<<1, num_th>>>(dstbuf, srcbuf, nbytes);

      CHECK_CUDA(cudaDeviceSynchronize());
      auto et = get_time();

      CHECK_CUDA(cudaGetLastError());
      auto us = std::chrono::duration_cast<std::chrono::microseconds>(et - st).count();
      us_db.push_back(us);
    }
    std::sort(us_db.begin(), us_db.end());
    size_t us_med = us_db[us_db.size() / 2];
    double gbps = (double)nbytes / us_med / 1e3;
    spdlog::info("Thr,{},gbps,{},us,{}", num_th, gbps, us_med);
  }

  return 0;
}