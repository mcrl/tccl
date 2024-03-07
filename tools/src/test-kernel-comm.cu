#include <cstdio>
#include <chrono>
#include <cassert>
#include <numa.h>
#include <numaif.h>
#include <thread>

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

__global__ void gpu_copy_triple(float* dst0, float* src0, float* dst1, float* src1, float* dst2, float* src2, size_t nbytes) {
  int channel = blockIdx.x;
  size_t idx = threadIdx.x;
  size_t stride = blockDim.x;
  float* dst = channel == 0 ? dst0 : channel == 1 ? dst1 : dst2;
  float* src = channel == 0 ? src0 : channel == 1 ? src1 : src2;
  size_t nelem = nbytes / sizeof(Pack128);
  //if (idx == 0) {
  //  printf("%u %d %d\n", get_smid(), channel, (int)idx);
  //}
  for (size_t i = idx; i < nelem; i += stride) {
    Pack128 v;
    Fetch128(v, ((Pack128*)src) + i);
    Store128(((Pack128*)dst) + i, v);
  }
}

__global__ void gpu_copy_dual_nopack(float* dst0, float* src0, float* dst1, float* src1, size_t nbytes) {
  int channel = blockIdx.x;
  size_t idx = threadIdx.x;
  size_t stride = blockDim.x;
  float* dst = channel == 0 ? dst0 : dst1;
  float* src = channel == 0 ? src0 : src1;
  size_t nelem = nbytes / sizeof(float4);
  //printf("%u %d %d\n", get_smid(), channel, (int)idx);
  for (size_t i = idx; i < nelem; i += stride) {
    float4 v;
    v = ((float4*)src)[i];
    ((float4*)dst)[i] = v;
  }
}

__global__ void gpu_copy_dual(float* dst0, float* src0, float* dst1, float* src1, size_t nbytes) {
  int channel = blockIdx.x;
  size_t idx = threadIdx.x;
  size_t stride = blockDim.x;
  float* dst = channel == 0 ? dst0 : dst1;
  float* src = channel == 0 ? src0 : src1;
  size_t nelem = nbytes / sizeof(Pack128);
  //printf("%u %d %d\n", get_smid(), channel, (int)idx);
  for (size_t i = idx; i < nelem; i += stride) {
    Pack128 v;
    Fetch128(v, ((Pack128*)src) + i);
    Store128(((Pack128*)dst) + i, v);
  }
}

__global__ void gpu_copy_single_nopack(float* dst, float* src, size_t nbytes) {
  size_t idx = threadIdx.x;
  size_t stride = blockDim.x;
  size_t nelem = nbytes / sizeof(float4);
  //printf("%u %d %d\n", get_smid(), channel, (int)idx);
  for (size_t i = idx; i < nelem; i += stride) {
    float4 v;
    v = ((float4*)src)[i];
    ((float4*)dst)[i] = v;
  }
}

__global__ void gpu_copy_single(float* dst, float* src, size_t nbytes) {
  size_t idx = threadIdx.x;
  size_t stride = blockDim.x;
  size_t nelem = nbytes / sizeof(Pack128);
  //printf("%u %d %d\n", get_smid(), channel, (int)idx);
  for (size_t i = idx; i < nelem; i += stride) {
    Pack128 v;
    Fetch128(v, ((Pack128*)src) + i);
    Store128(((Pack128*)dst) + i, v);
  }
}

__global__ void gpu_copy_full(float* dst, float* src, size_t nbytes) {
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  size_t stride = blockDim.x * gridDim.x;
  size_t nelem = nbytes / sizeof(Pack128);
  //printf("%u %d %d\n", get_smid(), channel, (int)idx);
  for (size_t i = idx; i < nelem; i += stride) {
    Pack128 v;
    Fetch128(v, ((Pack128*)src) + i);
    Store128(((Pack128*)dst) + i, v);
  }
}

__global__ void gpu_copy_full_v2(float* dst, float* src, size_t nbytes) {
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  size_t stride = blockDim.x * gridDim.x;
  size_t nelem = nbytes / sizeof(float4);
  #define UNROLL 4
  for (size_t i = idx; i < nelem; i += stride * UNROLL) {
    float4 v[UNROLL];
    for (int j = 0; j < UNROLL; ++j) {
      if (i + stride * j < nelem)
        v[j] = ((float4*)src)[i + stride * j];
    }
    for (int j = 0; j < UNROLL; ++j) {
      if (i + stride * j < nelem)
        ((float4*)dst)[i + stride * j] = v[j];
    }
  }
  #undef UNROLL
}

__global__ void gpu_copy_full_v3(float* dst, float* src, size_t nbytes) {
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  size_t stride = blockDim.x * gridDim.x;
  size_t nelem = nbytes / sizeof(float4);
  #define UNROLL 2
  for (size_t i = idx; i < nelem; i += stride * UNROLL) {
    float4 v[UNROLL];
    for (int j = 0; j < UNROLL; ++j) {
      if (i + stride * j < nelem)
        v[j] = ((float4*)src)[i + stride * j];
    }
    for (int j = 0; j < UNROLL; ++j) {
      if (i + stride * j < nelem)
        ((float4*)dst)[i + stride * j] = v[j];
    }
  }
  #undef UNROLL
  __threadfence_system();
}

__global__ void gpu_copy_dual_full(float* dst0, float* src0, float* dst1, float* src1, size_t nbytes, int channel_thr) {
  // [0, channel_thr) channels copy src0->dst0
  // Remaining channels copy src1->dst1
  size_t idx, stride;
  float *dst, *src;
  if (blockIdx.x < channel_thr) {
    idx = threadIdx.x + (blockIdx.x - 0) * blockDim.x;
    stride = blockDim.x * channel_thr;
    dst = dst0;
    src = src0;
  } else {
    idx = threadIdx.x + (blockIdx.x - channel_thr) * blockDim.x;
    stride = blockDim.x * (gridDim.x - channel_thr);
    dst = dst1;
    src = src1;
  }
  size_t nelem = nbytes / sizeof(Pack128);
  for (size_t i = idx; i < nelem; i += stride) {
    Pack128 v;
    Fetch128(v, ((Pack128*)src) + i);
    Store128(((Pack128*)dst) + i, v);
  }
}

__global__ void gpu_copy_dual_full_v2(float* dst0, float* src0, float* dst1, float* src1, size_t nbytes, int channel_thr) {
  // [0, channel_thr) channels copy src0->dst0
  // Remaining channels copy src1->dst1
  size_t idx, stride;
  float *dst, *src;
  if (blockIdx.x < channel_thr) {
    idx = threadIdx.x + (blockIdx.x - 0) * blockDim.x;
    stride = blockDim.x * channel_thr;
    dst = dst0;
    src = src0;
  } else {
    idx = threadIdx.x + (blockIdx.x - channel_thr) * blockDim.x;
    stride = blockDim.x * (gridDim.x - channel_thr);
    dst = dst1;
    src = src1;
  }
  size_t nelem = nbytes / sizeof(Pack128);
  #define UNROLL 32
  for (size_t i = idx; i < nelem; i += stride * UNROLL) {
    float4 v[UNROLL];
    for (int j = 0; j < UNROLL; ++j) {
      if (i + stride * j < nelem)
        v[j] = ((float4*)src)[i + stride * j];
    }
    for (int j = 0; j < UNROLL; ++j) {
      if (i + stride * j < nelem)
        ((float4*)dst)[i + stride * j] = v[j];
    }
  }
  #undef UNROLL
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
  CHECK_CUDA(cudaSetDevice(gpu_idx));
  CHECK_CUDA(cudaMalloc(ptr, nbytes));
}

void AllocCPUMemAccessibleFromGPU(void** dptr, void** hptr, size_t nbytes, int numa_idx) {
  *hptr = (float*)numa_alloc_onnode(nbytes, numa_idx);
  CHECK_CUDA(cudaHostRegister(*hptr, nbytes, cudaHostRegisterMapped));
  CHECK_CUDA(cudaHostGetDevicePointer(dptr, *hptr, 0));
}

int main(int argc, char ** argv) {
  int sw = -1;
  if (argc >= 2) {
    sw = atoi(argv[1]);
  }

  int NUMA_IDX = 1;

  const int num_gpus = 4;

  int p2p_available = 1;
  for (int i = 0; i < num_gpus; ++i) {
    for (int j = 0; j < num_gpus; ++j) {
      if (i != j) {
        int can_access;
        CHECK_CUDA(cudaDeviceCanAccessPeer(&can_access, i, j));
        if (can_access == 0) p2p_available = 0;
      }
    }
  }
  if (p2p_available) {
    for (int i = 0; i < num_gpus; ++i) {
      for (int j = 0; j < num_gpus; ++j) {
        if (i != j) {
          CHECK_CUDA(cudaSetDevice(i));
          CHECK_CUDA(cudaDeviceEnablePeerAccess(j, 0));
        }
      }
    }
  }

  size_t nbytes = 1024 * 1024 * 1024;

  float *h1a, *h1a_dev;
  AllocCPUMemAccessibleFromGPU((void**)&h1a_dev, (void**)&h1a, nbytes, NUMA_IDX);
  memset(h1a, 0, nbytes);

  float *h1b, *h1b_dev;
  AllocCPUMemAccessibleFromGPU((void**)&h1b_dev, (void**)&h1b, nbytes, NUMA_IDX);
  memset(h1b, 0, nbytes);

  float *h1c, *h1c_dev;
  AllocCPUMemAccessibleFromGPU((void**)&h1c_dev, (void**)&h1c, nbytes, NUMA_IDX);
  memset(h1c, 0, nbytes);

  float *h1d, *h1d_dev;
  AllocCPUMemAccessibleFromGPU((void**)&h1d_dev, (void**)&h1d, nbytes, NUMA_IDX);
  memset(h1d, 0, nbytes);

  float *h1e, *h1e_dev;
  AllocCPUMemAccessibleFromGPU((void**)&h1e_dev, (void**)&h1e, nbytes, NUMA_IDX);
  memset(h1e, 0, nbytes);

  float *h1f, *h1f_dev;
  AllocCPUMemAccessibleFromGPU((void**)&h1f_dev, (void**)&h1f, nbytes, NUMA_IDX);
  memset(h1f, 0, nbytes);

  float *d1a;
  AllocGPUMem((void**)&d1a, nbytes, 1);
  CHECK_CUDA(cudaMemset(d1a, 0, nbytes));

  float *d1b;
  AllocGPUMem((void**)&d1b, nbytes, 1);
  CHECK_CUDA(cudaMemset(d1b, 0, nbytes));

  float *d1c;
  AllocGPUMem((void**)&d1c, nbytes, 1);
  CHECK_CUDA(cudaMemset(d1c, 0, nbytes));

  float *d1d;
  AllocGPUMem((void**)&d1d, nbytes, 1);
  CHECK_CUDA(cudaMemset(d1d, 0, nbytes));

  float *d1e;
  AllocGPUMem((void**)&d1e, nbytes, 1);
  CHECK_CUDA(cudaMemset(d1e, 0, nbytes));

  float *d1f;
  AllocGPUMem((void**)&d1f, nbytes, 1);
  CHECK_CUDA(cudaMemset(d1f, 0, nbytes));

  float *d2a;
  AllocGPUMem((void**)&d2a, nbytes, 2);
  CHECK_CUDA(cudaMemset(d2a, 0, nbytes));

  float *d2b;
  AllocGPUMem((void**)&d2b, nbytes, 2);
  CHECK_CUDA(cudaMemset(d2b, 0, nbytes));

  float *d3a;
  AllocGPUMem((void**)&d3a, nbytes, 3);
  CHECK_CUDA(cudaMemset(d3a, 0, nbytes));

  int leastPriority, greatestPriority;
  CHECK_CUDA(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));

  cudaStream_t stream1a, stream1b;
  CHECK_CUDA(cudaSetDevice(1));
  //CHECK_CUDA(cudaStreamCreate(&stream1a));
  //CHECK_CUDA(cudaStreamCreate(&stream1b));
  CHECK_CUDA(cudaStreamCreateWithPriority(&stream1a, cudaStreamNonBlocking, greatestPriority));
  CHECK_CUDA(cudaStreamCreateWithPriority(&stream1b, cudaStreamNonBlocking, leastPriority));

  cudaStream_t stream2a;
  CHECK_CUDA(cudaSetDevice(2));
  CHECK_CUDA(cudaStreamCreate(&stream2a));

  for (int iter = 1; iter <= 82; ++iter) {
  //for (int iter = 1; iter <= 1; ++iter) {
    for (int i = 0; i < num_gpus; ++i) {
      CHECK_CUDA(cudaSetDevice(i));
      CHECK_CUDA(cudaDeviceSynchronize());
    }
    auto st = get_time();

    CHECK_CUDA(cudaSetDevice(1));

    if (sw == 0) {
      gpu_copy_single<<<1, 1024, 0, stream1a>>>(h1a_dev, d1a, nbytes);
      gpu_copy_single<<<1, 1024, 0, stream1b>>>(d1b, h1b_dev, nbytes);
    } else if (sw == 1) {
      gpu_copy_single<<<1, 1024, 0, stream1a>>>(h1a_dev, d1a, nbytes);
      CHECK_CUDA(cudaMemcpyAsync(d1b, h1b_dev, nbytes, cudaMemcpyDefault, stream1b));
    } else if (sw == 2) {
      CHECK_CUDA(cudaMemcpyAsync(h1a_dev, d1a, nbytes, cudaMemcpyDefault, stream1a));
      gpu_copy_single<<<1, 1024, 0, stream1b>>>(d1b, h1b_dev, nbytes);
    } else if (sw == 3) {
      CHECK_CUDA(cudaMemcpyAsync(h1a_dev, d1a, nbytes, cudaMemcpyDefault, stream1a));
      CHECK_CUDA(cudaMemcpyAsync(d1b, h1b_dev, nbytes, cudaMemcpyDefault, stream1b));
    } else if (sw == 4) {
      gpu_copy_single<<<1, 1024, 0, stream1b>>>(h1a_dev, d1a, nbytes);
      gpu_copy_single<<<1, 1024, 0, stream1a>>>(d1b, h1b_dev, nbytes);
    } else if (sw == 5) {
      gpu_copy_single<<<1, 1024, 0, stream1b>>>(h1a_dev, d1a, nbytes);
      CHECK_CUDA(cudaMemcpyAsync(d1b, h1b_dev, nbytes, cudaMemcpyDefault, stream1a));
    } else if (sw == 6) {
      CHECK_CUDA(cudaMemcpyAsync(h1a_dev, d1a, nbytes, cudaMemcpyDefault, stream1b));
      gpu_copy_single<<<1, 1024, 0, stream1a>>>(d1b, h1b_dev, nbytes);
    } else if (sw == 7) {
      CHECK_CUDA(cudaMemcpyAsync(h1a_dev, d1a, nbytes, cudaMemcpyDefault, stream1b));
      CHECK_CUDA(cudaMemcpyAsync(d1b, h1b_dev, nbytes, cudaMemcpyDefault, stream1a));
    } else if (sw == 8) {
      gpu_copy_dual<<<2, 1024, 0, stream1a>>>(h1a_dev, d1a, d1b, h1b_dev, nbytes);
    } else if (sw == 9) {
      gpu_copy_single<<<1, 1024, 0, stream1a>>>(h1a_dev, d1a, nbytes);
    } else if (sw == 10) {
      gpu_copy_single<<<1, 1024, 0, stream1b>>>(d1b, h1b_dev, nbytes);
    } else if (sw == 11) {
      gpu_fill_buffer<<<1, 1024, 0, stream1a>>>(h1a_dev, nbytes);
    } else if (sw == 12) {
      gpu_copy_triple<<<3, 1024, 0, stream1a>>>(h1a_dev, d1a, d1b, h1b_dev, h1c_dev, d1c, nbytes);
    } else if (sw == 13) {
      gpu_copy_dual<<<2, 1024, 0, stream1a>>>(h1a_dev, d1a, d1b, d1c, nbytes);
    } else if (sw == 14) {
      gpu_copy_single<<<1, 1024, 0, stream1b>>>(d1b, h1b_dev, nbytes);
      std::this_thread::sleep_for(std::chrono::milliseconds(80 + iter));
      gpu_copy_single<<<1, 1024, 0, stream1a>>>(h1a_dev, d1a, nbytes);
    } else if (sw == 15) {
      gpu_copy_single_nopack<<<1, 1024, 0, stream1a>>>(h1a_dev, d1a, nbytes);
      gpu_copy_single_nopack<<<1, 1024, 0, stream1b>>>(d1b, h1b_dev, nbytes);
    } else if (sw == 16) {
      gpu_copy_dual_nopack<<<2, 1024, 0, stream1a>>>(h1a_dev, d1a, d1b, h1b_dev, nbytes);
    } else if (sw == 17) {
      gpu_copy_triple<<<3, 1024, 0, stream1a>>>(h1a_dev, d1a, h1b_dev, d1b, h1c_dev, d1c, nbytes);
      gpu_copy_triple<<<3, 1024, 0, stream1b>>>(d1d, h1d_dev, d1e, h1e_dev, d1f, h1f_dev, nbytes);
    } else if (sw == 18) {
      CHECK_CUDA(cudaMemcpyAsync(h1a_dev, d1a, nbytes, cudaMemcpyDefault, stream1a));
    } else if (sw == 19) {
      CHECK_CUDA(cudaMemcpyAsync(d1b, h1b_dev, nbytes, cudaMemcpyDefault, stream1b));
    } else if (sw == 20) {
      gpu_copy_full<<<iter, 1024, 0, stream1a>>>(h1a_dev, d1a, nbytes);
      gpu_copy_full<<<iter, 1024, 0, stream1b>>>(d1b, h1b_dev, nbytes);
    } else if (sw == 21) {
      gpu_copy_full<<<1, 1024, 0, stream1a>>>(h1a_dev, d1a, nbytes);
      gpu_copy_full<<<iter, 1024, 0, stream1b>>>(d1b, h1b_dev, nbytes);
    } else if (sw == 22) {
      gpu_copy_full<<<iter, 1024, 0, stream1a>>>(h1a_dev, d1a, nbytes);
      gpu_copy_full<<<1, 1024, 0, stream1b>>>(d1b, h1b_dev, nbytes);
    } else if (sw == 23) {
      gpu_copy_dual_full<<<40, 1024, 0, stream1a>>>(h1a_dev, d1a, d1b, h1b_dev, nbytes, 1);
    } else if (sw == 24) {
      gpu_copy_full<<<1, 1024, 0, stream1a>>>(h1a_dev, d1a, nbytes);
      gpu_copy_full_v2<<<iter, 1024, 0, stream1b>>>(d1b, h1b_dev, nbytes);
    } else if (sw == 25) {
      gpu_copy_dual_full_v2<<<2, 256, 0, stream1a>>>(h1a_dev, d1a, d1b, h1b_dev, nbytes, 1);
    } else if (sw == 26) {
      // write to CPU, read from peer GPU
      gpu_copy_full<<<1, 1024, 0, stream1a>>>(h1a_dev, d1a, nbytes);
      gpu_copy_full<<<iter, 1024, 0, stream1b>>>(d1b, d2a, nbytes);
    } else if (sw == 27) {
      gpu_copy_dual_full_v2<<<iter, 1024, 0, stream1a>>>(h1a_dev, d1a, d1b, d3a, nbytes, 1);
    } else if (sw == 28) {
      gpu_copy_full<<<1, 1024, 0, stream1a>>>(h1a_dev, d1a, nbytes);
      //CHECK_CUDA(cudaMemcpyAsync(h1a_dev, d1a, nbytes, cudaMemcpyDefault, stream1a));
      gpu_copy_full<<<iter, 1024, 0, stream1b>>>(d1b, h1b_dev, nbytes);
      //CHECK_CUDA(cudaMemcpyAsync(d1b, h1b_dev, nbytes, cudaMemcpyDefault, stream1b));
    } else if (sw == 29) {
      CHECK_CUDA(cudaSetDevice(1));
      gpu_copy_full_v3<<<1, 1024, 0, stream1a>>>(d2a, d1a, nbytes);
      //CHECK_CUDA(cudaMemcpyAsync(d2a, d1a, nbytes, cudaMemcpyDefault, stream1a));
      CHECK_CUDA(cudaSetDevice(2));
      gpu_copy_full_v3<<<iter, 1024, 0, stream2a>>>(d1b, d2b, nbytes);
      //CHECK_CUDA(cudaSetDevice(1));
      //gpu_copy_full_v3<<<iter, 1024, 0, stream1b>>>(d1b, d2b, nbytes);
    } else if (sw == 30) {
      gpu_copy_full_v2<<<iter, 1024, 0, stream1a>>>(d2a, h1a_dev, nbytes);
      // below achieves full BW 11.3GB/s
      //CHECK_CUDA(cudaMemcpyAsync(d1a, h1a_dev, nbytes, cudaMemcpyDefault, stream1a));
      //CHECK_CUDA(cudaMemcpyAsync(d2a, d1b, nbytes, cudaMemcpyDefault, stream1b));
    }

    for (int i = 0; i < num_gpus; ++i) {
      CHECK_CUDA(cudaSetDevice(i));
      CHECK_CUDA(cudaDeviceSynchronize());
      CHECK_CUDA(cudaGetLastError());
    }
    auto et = get_time();
    size_t us = get_duration_us(st, et);
    double gbps = nbytes / (us / 1e6) / 1e9;
    printf("iter %d, time %zu, gbps %f\n", iter, us, gbps);
  }




  return 0;
}