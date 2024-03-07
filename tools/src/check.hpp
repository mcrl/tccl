#pragma once

#include <assert.h>
#include <cstdio>

#define CHECK_CUDA(call)                                              \
  do {                                                                \
    cudaError_t status_ = call;                                       \
    if (status_ != cudaSuccess) {                                     \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(status_));                           \
      assert(0);                                                  \
    }                                                                 \
  } while (0)

#define CHECK_CU(call)                                                \
  do {                                                                \
    CUresult status_ = call;                                          \
    if (status_ != CUDA_SUCCESS) {                                    \
      const char* errstr;                                             \
      cuGetErrorName(status_, &errstr);                               \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, \
              errstr);                                                \
      assert(0);                                                  \
    }                                                                 \
  } while (0)

#define CHECK_NCCL(call)                                               \
  do {                                                                 \
    ncclResult_t status_ = call;                                       \
    if (status_ != ncclSuccess && status_ != ncclInProgress) {                                      \
      fprintf(stderr, "NCCL error (%s:%d): %s\n", __FILE__, __LINE__,  \
              ncclGetErrorString(status_));                            \
      assert(0);                                                   \
    }                                                                  \
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

#define CHECK_ERRNO(call) \
  do { \
    int code = call; \
    if (code != 0) { \
      fprintf(stderr, "ERRNO error (%s:%d): %s(%d)\n", __FILE__, __LINE__, strerror(errno), errno); \
      assert(false); \
    } \
  } while (0)

#define CHECK_ERRNO_PTR(call) \
  do { \
    const void* ptr = call; \
    if (ptr == nullptr) { \
      fprintf(stderr, "ERRNO error (%s:%d): %s(%d)\n", __FILE__, __LINE__, strerror(errno), errno); \
      assert(false); \
    } \
  } while (0)

#define CHECK_PTR(call) \
  do { \
    const void* ptr = call; \
    if (ptr == nullptr) { \
      fprintf(stderr, "NULL returned (%s:%d)\n", __FILE__, __LINE__); \
      assert(false); \
    } \
  } while (0)

#define CHECK_SHARP(call) \
  do { \
    int res = call; \
    if (res != SHARP_COLL_SUCCESS) { \
      fprintf(stderr, "SHARP error (%s:%d): %d(%s)\n", __FILE__, __LINE__, res, sharp_coll_strerror(res)); \
      assert(false); \
    } \
  } while (0)
