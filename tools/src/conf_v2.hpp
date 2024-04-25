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
  int num_bits_idx; // number of bits used to encode a index = ceil(log2(max(num_gpus, num_numa)))
  int num_bits_inter_tf; // number of bits used to encode a inter-node transfer = 2 (four inter-node transfer types) + 2 * num_bits_idx
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
  bool disable_memcpy = false;
  bool disable_memcpy_read = false;
};

extern Conf conf;