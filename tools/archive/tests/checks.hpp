#pragma once

#include <assert.h>
#include <cstdio>

#define CHECK_CUDA(call) \
  do { \
    cudaError_t status_ = call; \
    if (status_ != cudaSuccess) { \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(status_)); \
      assert(0); \
    } \
  } while (0)

#define CHECK_NCCL(call) \
  do { \
    ncclResult_t status_ = call; \
    if (status_ != ncclSuccess && status_ != ncclInProgress) { \
      fprintf(stderr, "NCCL error (%s:%d): %s\n", __FILE__, __LINE__, \
              ncclGetErrorString(status_)); \
      assert(0); \
    } \
  } while (0)

#define CHECK_MPI(call) \
  do { \
    int code = call; \
    if (code != MPI_SUCCESS) { \
      char estr[MPI_MAX_ERROR_STRING]; \
      int elen; \
      MPI_Error_string(code, estr, &elen); \
      fprintf(stderr, "MPI error (%s:%d): %s\n", __FILE__, __LINE__, estr); \
      assert(0); \
    } \
  } while (0)
