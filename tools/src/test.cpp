#include <cstdio>
#include <tinyxml2.h>
#include <unistd.h>
#include <numa.h>
#include <numaif.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "benchmark_v2.hpp"
#include "conf_v2.hpp"
#include "util.hpp"
#include "check.hpp"


void tcclSetNuma(void* cpumem, int size, int numaIdx) {
  numa_tonode_memory(cpumem, size, numaIdx);

  // Check numa applied correctly
  int mode;
  int ret = get_mempolicy(&mode, NULL, 0, cpumem, MPOL_F_NODE | MPOL_F_ADDR);
  if (ret != 0) {
    printf("get_mempolicy failed: %d\n", ret);
    return;
  }
  if (numaIdx != mode) {
    printf("NUMA policy not applied correctly. Expected %d, got %d\n", numaIdx, mode);
    return;
  }
  return;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  conf.nbytes = 32 * 1024 * 1024;
  conf.bw_threshold = 0; 
  conf.niters = 10;
  conf.warmup_iters = conf.niters / 10;
  conf.gpu_numa = {3, 1, 1, 0};
  conf.numa_gpu = {{3}, {1, 2}, {}, {0}};
  conf.nic_numa = {3};
  conf.numa_nic = {{}, {}, {}, {0}};
  conf.cpumem_numa = {0, 1, 2, 3};
  conf.numa_cpumem = {{0}, {1}, {2}, {3}};
  conf.debug = true;
  if (conf.debug) {
    spdlog::set_level(spdlog::level::debug);
  }

  ValidateLaunchConf();
  ValidateCuda();
  ValidateNuma();
  CHECK_CUDA(cudaGetDeviceCount(&conf.num_gpus));

  LOG_RANK_ANY("Validation complete");
  cache_t cache;

  std::vector<Transfer> transfers1 = {
    {Transfer::TransferType::GPU_WRITE_CPUMEM_KERNEL, 0, 0, 0, 1, 1024},
    {Transfer::TransferType::GPU_READ_CPUMEM_MEMCPY, 0, 0, 1, 3, 1024},
    {Transfer::TransferType::GPU_WRITE_CPUMEM_KERNEL, 0, 0, 3, 0, 1024},
    {Transfer::TransferType::GPU_READ_CPUMEM_KERNEL, 0, 0, 0, 1, 1024},
    {Transfer::TransferType::GPU_WRITE_CPUMEM_KERNEL, 0, 0, 1, 3, 1024},
    {Transfer::TransferType::GPU_READ_CPUMEM_KERNEL, 0, 0, 3, 2, 1024}
  };
  PopulateCache(cache, transfers1, 9.0);

  std::vector<Transfer> transfers2 = {
    {Transfer::TransferType::GPU_WRITE_CPUMEM_KERNEL, 0, 0, 3, 1, 1024},
    {Transfer::TransferType::GPU_READ_CPUMEM_MEMCPY, 0, 0, 1, 0, 1024},
    {Transfer::TransferType::GPU_WRITE_CPUMEM_KERNEL, 0, 0, 0, 3, 1024},
    {Transfer::TransferType::GPU_READ_CPUMEM_KERNEL, 0, 0, 3, 1, 1024},
    {Transfer::TransferType::GPU_WRITE_CPUMEM_KERNEL, 0, 0, 1, 0, 1024},
    {Transfer::TransferType::GPU_READ_CPUMEM_KERNEL, 0, 0, 0, 2, 1024}
  };
  double bw = CheckCache(cache, transfers2);
  printf("bw=%f\n", bw);

  MPI_Finalize();
  return 0;
}