#pragma once

#include <chrono>
#include <spdlog/spdlog.h>
#include <mpi.h>

#define LOG_RANK_0(...) \
  do { \
    int rank; \
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); \
    if (rank == 0) { \
      std::string _s = fmt::format(__VA_ARGS__); \
      spdlog::info("Rank {}: {}", rank, _s); \
    } \
  } while (0)

#define LOG_RANK_ANY(...) \
  do { \
    int rank; \
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); \
    std::string _s = fmt::format(__VA_ARGS__); \
    spdlog::info("Rank {}: {}", rank, _s); \
  } while (0)

#define DEBUG_RANK_0(...) \
  do { \
    int rank; \
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); \
    if (rank == 0) { \
      std::string _s = fmt::format(__VA_ARGS__); \
      spdlog::debug("Rank {}: {}", rank, _s); \
    } \
  } while (0)

#define DEBUG_RANK_ANY(...) \
  do { \
    int rank; \
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); \
    std::string _s = fmt::format(__VA_ARGS__); \
    spdlog::debug("Rank {}: {}", rank, _s); \
  } while (0)

void mpi_finalize_and_exit(int status);
MPI_Comm create_subset_comm(MPI_Comm comm, std::vector<int> ranks);

void fill_random_float(float* buf, size_t numel);
void fill_random_float_gpu(float* buf, size_t numel);
double sum_float(float* buf, size_t numel);
double sum_float_gpu(float* buf, size_t numel);

std::chrono::time_point<std::chrono::steady_clock> get_time();
size_t get_duration_us(
    std::chrono::time_point<std::chrono::steady_clock> start,
    std::chrono::time_point<std::chrono::steady_clock> end
    );
