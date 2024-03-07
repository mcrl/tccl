#pragma once

#include <cstddef>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <tinyxml2.h>

struct Transfer {
  enum struct TransferType : int {
    GPU_READ_CPUMEM_KERNEL,
    GPU_WRITE_CPUMEM_KERNEL,
    GPU_READ_GPUMEM_KERNEL,
    GPU_WRITE_GPUMEM_KERNEL,
    GPU_READ_CPUMEM_MEMCPY,
    GPU_WRITE_CPUMEM_MEMCPY,
    GPU_READ_GPUMEM_MEMCPY,
    GPU_WRITE_GPUMEM_MEMCPY,
    GPU_GPU_INTER,
    GPU_CPU_INTER,
    CPU_GPU_INTER,
    CPU_CPU_INTER,
    DUMMY
  };
  enum struct DeviceType : int {
    GPU,
    CPUMEM,
    NIC
  };
  TransferType type;
  int src_node, dst_node;
  int src_idx, dst_idx; // device idx if gpu, numa idx if cpumem
  size_t nbytes;

  static std::string TransferTypeToString(Transfer::TransferType type);
  static Transfer::TransferType TransferTypeFromString(std::string);
  std::string SerToString() const;
  static Transfer DesFromString(std::string str);
  static Transfer DesFromXmlElem(tinyxml2::XMLElement *elem);
  std::string ToString();
  int GetGPUIdxKernelLaunchedOn() const;
  bool IsRead() const;
  bool IsWrite() const;
  bool IsMemcpy() const;
  bool IsInter() const;
  bool operator<(const Transfer &that) const;
  int Encode() const;
  static Transfer Decode(int encoded);
  static Transfer DecodeInter(int encoded, int src_node, int dst_node);
};

struct BenchmarkResult {
  std::vector<size_t> us_db;
  std::vector<std::vector<size_t>> us_local_db;
  size_t us_med;
  double us_avg;

  void FillStat();
};

void ValidateLaunchConf();
void ValidateCuda();
void ValidateNuma();
typedef std::map<std::set<int>, double> cache_t;
void PopulateCache(cache_t& cache, std::vector<Transfer>& transfers, double bw);
double CheckCache(cache_t& cache, std::vector<Transfer> transfers);
void RunDijkstra(cache_t& cache, Transfer head, Transfer tail, int gpu_mask, const char* fn, bool intra);
void ProcessPoolLoop();
void CommandExit();
BenchmarkResult Benchmark(std::vector<Transfer> transfers);
std::vector<Transfer> DeserializeTransfers(std::string str);