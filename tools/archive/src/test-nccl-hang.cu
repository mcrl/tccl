#include <cstdio>
#include <spdlog/spdlog.h>
#include <cuda.h>

#include "check.hpp"

cudaStream_t stream1, stream2, stream3;
size_t nbytes = 1024 * 1024 * 1024;
float *da, *db;
float *ha, *hb;

std::mutex mtx;
std::condition_variable cv;
bool flag = false;

__global__ void infinite_loop_kernel() {
  while (true);
}

void proxy_func() {
  //CHECK_CUDA(cudaSetDevice(0));

  CUcontext ctx_proxy;

  spdlog::info("proxy_func: before cuCtxCreate");
  CHECK_CU(cuCtxCreate(&ctx_proxy, CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST, 0));
  spdlog::info("proxy_func: after cuCtxCreate");

  spdlog::info("proxy_func: before cuCtxSetCurrent");
  CHECK_CU(cuCtxSetCurrent(ctx_proxy));
  spdlog::info("proxy_func: after cuCtxSetCurrent");

  spdlog::info("proxy_func: now waiting...");
  {
    std::unique_lock<std::mutex> lk(mtx);
    cv.wait(lk, []{ return flag; });
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(1000));

  spdlog::info("proxy_func: before cudaMemcpyAsync");
  //CHECK_CUDA(cudaMemcpyAsync(db, hb, nbytes, cudaMemcpyDefault, stream3));
  float *tmp;
  CHECK_CUDA(cudaMalloc(&tmp, nbytes));
  spdlog::info("proxy_func: after cudaMemcpyAsync");
}

//void test_ctx() {
//  int signature = 0xdeadbeef;
//  CHECK_CUDA(cudaMalloc(&da, nbytes));
//  CHECK_CUDA(cudaMemcpy(da, &signature, sizeof(int), cudaMemcpyDefault));
//  cuCtxCreate(&ctx_proxy, CU_CTX_SCHED_SPIN, 0);
//  cuCtxSetCurrent(ctx_proxy);
//  CHECK_CUDA(cudaMalloc(&db, nbytes));
//  CHECK_CUDA(cudaMemcpy(db, da, nbytes, cudaMemcpyDefault));
//  int new_signature;
//  CHECK_CUDA(cudaMemcpy(&new_signature, db, sizeof(int), cudaMemcpyDefault));
//  printf("%x\n", new_signature);
//}

int main() {
  //test_ctx();

  CHECK_CUDA(cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking));
  //CHECK_CUDA(cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking));
  stream2 = 0;
  CHECK_CUDA(cudaStreamCreateWithFlags(&stream3, cudaStreamNonBlocking));

  auto proxy_thread = std::thread(proxy_func);

  cudaEvent_t ev;
  CHECK_CUDA(cudaEventCreate(&ev));

  CHECK_CUDA(cudaMalloc(&da, nbytes));
  CHECK_CUDA(cudaMallocHost(&ha, nbytes));
  CHECK_CUDA(cudaMalloc(&db, nbytes));
  CHECK_CUDA(cudaMallocHost(&hb, nbytes));

  infinite_loop_kernel<<<1, 1, 0, stream1>>>();
  CHECK_CUDA(cudaEventRecord(ev, stream1));
  CHECK_CUDA(cudaStreamWaitEvent(stream2, ev, 0));

  {
    std::unique_lock<std::mutex> lk(mtx);
    flag = true;
    cv.notify_one();
  }

  spdlog::info("main: before cudaMemcpyAsync");
  CHECK_CUDA(cudaMemcpy(ha, da, nbytes, cudaMemcpyDefault));
  spdlog::info("main: after cudaMemcpyAsync");

  proxy_thread.join();
  
  return 0;
}