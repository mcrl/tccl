#include "util.hpp"

#include <mpi.h>
#include <cstdlib>
#include <random>
#include <chrono>
#include <cuda_runtime.h>

#include "check.hpp"

void mpi_finalize_and_exit(int status) {
  MPI_Finalize();
  exit(status);
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

double sum_float(float* buf, size_t numel) {
  double s_ans = 0;
  for (size_t i = 0; i < numel; ++i) {
    s_ans += buf[i];
  }
  return s_ans;
}

double sum_float_gpu(float* buf, size_t numel) {
  size_t nbytes = numel * sizeof(float);
  float *h_tmp = (float*)malloc(nbytes);
  CHECK_CUDA(cudaMemcpy(h_tmp, buf, nbytes, cudaMemcpyDeviceToHost));
  double s_ans = sum_float(h_tmp, numel);
  free(h_tmp);
  return s_ans;
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

MPI_Comm create_subset_comm(MPI_Comm comm, std::vector<int> ranks) {
  MPI_Group world_group, subset_group;
  MPI_Comm_group(comm, &world_group);
  MPI_Group_incl(world_group, ranks.size(), ranks.data(), &subset_group);
  MPI_Comm subset_comm;
  MPI_Comm_create_group(comm, subset_group, 0, &subset_comm);
  MPI_Group_free(&world_group);
  MPI_Group_free(&subset_group);
  return subset_comm;
}