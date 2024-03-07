#pragma once

#include <vector>
#include <chrono>

typedef std::chrono::time_point<std::chrono::steady_clock> mytime_t;

struct Conf {
  int p2p_available;
  int num_numa;
  std::vector<int> numa_available;
  size_t nbytes;
  double bw_threshold;
  std::vector<int> node_idx;
  int rank;
  int size;
  int niters;
  int warmup_iters;
  int num_gpus;
  bool debug = false;
  std::vector<int> gpu_numa;
  std::vector<std::vector<int>> numa_gpu;
  std::vector<int> nic_numa;
  std::vector<std::vector<int>> numa_nic;
  std::vector<int> cpumem_numa;
  std::vector<std::vector<int>> numa_cpumem;
  // statistics
  mytime_t start_time;
  int cache_miss;
  int cache_hit;
  bool debug_cache = false;
  bool disable_kernel = false;
  bool disable_memcpy_read = false;
};

extern Conf conf;