#include <cstdio>

#include <nccl.h>
#include <mpi.h>
#include "checks.hpp"
#include "utils.hpp"

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int num_gpu;
  CHECK_CUDA(cudaGetDeviceCount(&num_gpu));

  int gpu_idx = rank % num_gpu;
  CHECK_CUDA(cudaSetDevice(gpu_idx));

  ncclUniqueId nid;
  if (rank == 0) {
    LOG_RANK_ANY("Before ncclGetUniqueId");
    CHECK_NCCL(ncclGetUniqueId(&nid));
    LOG_RANK_ANY("After ncclGetUniqueId");
  }
  MPI_Bcast(&nid, sizeof(nid), MPI_BYTE, 0, MPI_COMM_WORLD);

  ncclComm_t comm;
  LOG_RANK_ANY("Before ncclCommInitRank");
  CHECK_NCCL(ncclCommInitRank(&comm, size, nid, rank));
  LOG_RANK_ANY("After ncclCommInitRank");

  LOG_RANK_ANY("Before ncclCommDestroy");
  CHECK_NCCL(ncclCommDestroy(comm));
  LOG_RANK_ANY("After ncclCommDestroy");

  MPI_Finalize();
  return 0;
}