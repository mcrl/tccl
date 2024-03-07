#include <cstdio>
#include <spdlog/spdlog.h>
#include <cuda.h>
#include <chrono>
#include <mpi.h>

#include "check.hpp"

int num_gpus = 4;
cudaStream_t stream1, stream2;
size_t nbytes = 1024 * 1024 * 1024;
float *da, *db, *dc;
float *ha, *hb;
cudaIpcMemHandle_t da_handle;

std::mutex mtx;
std::condition_variable cv;
int flag = 0;

void thread1_func() {
  CHECK_CUDA(cudaMalloc(&db, nbytes));
  hb = (float*)malloc(nbytes);

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
    CHECK_CUDA(cudaMemcpy(hb, db, nbytes, cudaMemcpyDefault));
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

void thread2_func() {
  CUcontext ctx_proxy;
  CHECK_CU(cuCtxCreate(&ctx_proxy, CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST, 0));
  CHECK_CU(cuCtxSetCurrent(ctx_proxy));

  CHECK_CUDA(cudaIpcOpenMemHandle((void**)&da, da_handle, cudaIpcMemLazyEnablePeerAccess));

  CHECK_CUDA(cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking));
  CHECK_CUDA(cudaMalloc(&dc, nbytes));

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
    CHECK_CUDA(cudaMemcpyAsync(dc, da, nbytes, cudaMemcpyDefault, stream2));
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
}

int main() {
  CHECK_MPI(MPI_Init(nullptr, nullptr));

  int rank, size;
  CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &size));

  if (rank == 0) {
    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaMalloc(&da, nbytes));
    CHECK_CUDA(cudaIpcGetMemHandle(&da_handle, da));
    CHECK_MPI(MPI_Send(&da_handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 1, 0, MPI_COMM_WORLD));
  } else if (rank == 1) {
    CHECK_MPI(MPI_Recv(&da_handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));

    auto thread1 = std::thread(thread1_func);
    auto thread2 = std::thread(thread2_func);

    thread1.join();
    thread2.join();
  }

  CHECK_MPI(MPI_Finalize());

  return 0;
}