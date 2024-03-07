#include <cstdio>
#include <spdlog/spdlog.h>
#include <cuda.h>
#include <chrono>

#include "check.hpp"

int num_gpus = 4;
cudaStream_t stream1, stream2;
size_t nbytes = 1024 * 1024 * 1024;
float *da, *db, *d1a;
float *ha, *hb;
cudaIpcMemHandle_t d1a_handle;

std::mutex mtx;
std::condition_variable cv;
int flag = 0;

void thread1_func() {
  //CUcontext ctx_proxy;
  //CHECK_CU(cuCtxCreate(&ctx_proxy, CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST, 0));
  //CHECK_CU(cuCtxSetCurrent(ctx_proxy));
  //CHECK_CUDA(cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking));
  CHECK_CUDA(cudaMalloc(&da, nbytes));
  //CHECK_CUDA(cudaMallocHost(&ha, nbytes));
  ha = (float*)malloc(nbytes);

  {
    std::unique_lock<std::mutex> lk(mtx);
    flag += 1;
    cv.notify_all();
  }

  {
    std::unique_lock<std::mutex> lk(mtx);
    cv.wait(lk, []{ return flag == 2; });
  }

  for (int i = 0; i < 100; ++i) {
    spdlog::info("thread1: {}", i);
    CHECK_CUDA(cudaMemcpy(ha, da, nbytes, cudaMemcpyDefault));
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

void thread2_func() {
  CUcontext ctx_proxy;
  CHECK_CU(cuCtxCreate(&ctx_proxy, CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST, 0));
  CHECK_CU(cuCtxSetCurrent(ctx_proxy));

  //float *d1a_0;
  //CHECK_CUDA(cudaIpcOpenMemHandle((void**)&d1a_0, d1a_handle, cudaIpcMemLazyEnablePeerAccess));

  CHECK_CUDA(cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking));
  CHECK_CUDA(cudaMallocHost(&hb, nbytes));
  CHECK_CUDA(cudaMalloc(&db, nbytes));

  {
    std::unique_lock<std::mutex> lk(mtx);
    flag += 1;
    cv.notify_all();
  }

  {
    std::unique_lock<std::mutex> lk(mtx);
    cv.wait(lk, []{ return flag == 2; });
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(1000));

  for (int i = 0; i < 100; ++i) {
    spdlog::info("thread2: {}", i);
    void* tmp;
    CHECK_CUDA(cudaMallocAsync(&tmp, 1024, stream2));
    //CHECK_CUDA(cudaMemcpyAsync(db, hb, nbytes, cudaMemcpyDefault, stream2));
    //if (i % 2 == 0) {
    //  CHECK_CUDA(cudaMemcpyAsync(db, hb, nbytes, cudaMemcpyDefault, cudaStreamPerThread));
    //} else {
    //  CHECK_CUDA(cudaMemcpyAsync(db, hb, nbytes, cudaMemcpyDefault, cudaStreamLegacy));
    //}
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

int main() {
  CHECK_CUDA(cudaSetDevice(1));
  CHECK_CUDA(cudaMalloc(&d1a, nbytes));
  CHECK_CUDA(cudaIpcGetMemHandle(&d1a_handle, d1a));

  //CHECK_CUDA(cudaSetDevice(0));
  //float *d1a_0;
  //CHECK_CUDA(cudaIpcOpenMemHandle((void**)&d1a_0, d1a_handle, cudaIpcMemLazyEnablePeerAccess));

  CHECK_CUDA(cudaSetDevice(0));

  auto thread1 = std::thread(thread1_func);
  auto thread2 = std::thread(thread2_func);

  //{
  //  std::unique_lock<std::mutex> lk(mtx);
  //  flag = true;
  //  cv.notify_all();
  //}

  thread1.join();
  thread2.join();

  return 0;
}