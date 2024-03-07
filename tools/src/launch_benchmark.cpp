#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <map>
#include <set>
#include "conf_v2.hpp"
#include "benchmark_v2.hpp"
#include "util.hpp"
#include "check.hpp"

static void print_help(const char *prog_name) {
  printf("Usage: %s [-d]", prog_name);
  printf("Options:\n");
  printf("     -d : print debug. (default: off)\n");
  printf("     -t : BW threshold in GB/s. (default: 0)\n");
  printf("     -o : output xml directory.\n");
  printf("     -n : node type (A or B or D).\n");
}

static char dir_name[1024] = "";
static char node_type[1024] = "";
static void parse_opt(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "dt:o:n:")) != -1) {
    switch (c) {
      case 'd': conf.debug = true; break;
      case 't': conf.bw_threshold = atof(optarg) * 1e9; break;
      case 'o': strcpy(dir_name, optarg); break;
      case 'n': strcpy(node_type, optarg); break;
      default: print_help(argv[0]); exit(0);
    }
  }
  if (strlen(dir_name) == 0
      || strlen(node_type) == 0) {
    print_help(argv[0]);
    exit(0);
  }
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  
  // conf defaults
  conf.start_time = get_time();
  conf.nbytes = 32 * 1024 * 1024;
  conf.bw_threshold = 0; 
  conf.niters = 10;
  conf.warmup_iters = conf.niters / 10;
  conf.disable_kernel = true;
  conf.disable_memcpy_read = true;

  parse_opt(argc, argv);
  if (strcmp(node_type, "A") == 0) {
    conf.gpu_numa = {3, 1, 1, 0};
    conf.numa_gpu = {{3}, {1, 2}, {}, {0}};
    conf.nic_numa = {3};
    conf.numa_nic = {{}, {}, {}, {0}};
    conf.cpumem_numa = {0, 1, 2, 3};
    conf.numa_cpumem = {{0}, {1}, {2}, {3}};
  } else if (strcmp(node_type, "B") == 0) {
    conf.gpu_numa = {3, 1, 1, 0};
    conf.numa_gpu = {{3}, {1, 2}, {}, {0}};
    conf.nic_numa = {3};
    conf.numa_nic = {{}, {}, {}, {0}};
    conf.cpumem_numa = {0, 1, 2, 3};
    conf.numa_cpumem = {{0}, {1}, {2}, {3}};
  } else if (strcmp(node_type, "D") == 0) {
    conf.gpu_numa = {0, 0, 2, 2};
    conf.numa_gpu = {{0, 1}, {}, {2, 3}, {}};
    conf.nic_numa = {1}; // Check with lstopo file.png
    conf.numa_nic = {{}, {0}, {}, {}};
    conf.cpumem_numa = {0, 1, 2, 3};
    conf.numa_cpumem = {{0}, {1}, {2}, {3}};
  } else {
    printf("Unknown node type %s\n", node_type);
    return 0;
  }

  ValidateLaunchConf();
  ValidateCuda();
  ValidateNuma();
  if (conf.debug) {
    spdlog::set_level(spdlog::level::debug);
  }
  MPI_Comm_rank(MPI_COMM_WORLD, &conf.rank);
  MPI_Comm_size(MPI_COMM_WORLD, &conf.size);
  CHECK_CUDA(cudaGetDeviceCount(&conf.num_gpus));

  LOG_RANK_ANY("Validation complete");

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    std::map<std::set<int>, double> cache;

    // Intra
    for (int gpu_mask = 0; gpu_mask < (1 << conf.num_gpus); ++gpu_mask) {
      if (__builtin_popcount(gpu_mask) < 2) continue;
      char xmlfn[2048];
      sprintf(xmlfn, "%s/intra_%d.xml", dir_name, gpu_mask);
      LOG_RANK_0("Progress: intra, gpu_mask = {:b}, xmlfn = {}", gpu_mask, xmlfn);
      Transfer dummy = Transfer::DecodeInter(0, 0, 0);
      RunDijkstra(cache, dummy, dummy, gpu_mask, xmlfn, true);
    }
    
    // Inter
    for (int head = 0; head < 64; ++head) {
      for (int tail = 0; tail < 64; ++tail) {
        for (int gpu_mask = 0; gpu_mask < (1 << conf.num_gpus); ++gpu_mask) {
          if (__builtin_popcount(gpu_mask) < 1) continue;
          Transfer head_transfer = Transfer::DecodeInter(head, 2, 0);
          Transfer tail_transfer = Transfer::DecodeInter(tail, 0, 1);
          char xmlfn[2048];
          sprintf(xmlfn, "%s/inter_%d_%d_%d.xml", dir_name, head, tail, gpu_mask);
          LOG_RANK_0("Progress: inter, head = {}, tail = {}, gpu_mask = {:b}, xmlfn = {}", head, tail, gpu_mask, xmlfn);
          RunDijkstra(cache, head_transfer, tail_transfer, gpu_mask, xmlfn, false);
        }
      }
    }

    CommandExit();
  } else {
    ProcessPoolLoop();
  }

  MPI_Finalize();

  return 0;
}

    // Intra setup - iterate through gpu subsets
    //std::vector<std::vector<std::pair<std::vector<Transfer>, BenchmarkResult>>> result_db;
    //result_db.resize(1 << num_gpus);
    //for (int gpu_mask = 0; gpu_mask < (1 << num_gpus); ++gpu_mask) {
    //  std::vector<int> gpu_idxs;
    //  for (int i = 0; i < num_gpus; ++i) {
    //    if (gpu_mask & (1 << i)) {
    //      gpu_idxs.push_back(i);
    //    }
    //  }
    //  if (gpu_idxs.size() <= 1) continue;

    //  std::vector<Transfer> transfers;
    //  // since it is a ring, we only need to start from any GPU; here its 0
    //  VisitGPUIntra(transfers, gpu_idxs[0], gpu_idxs[0], RemoveElem(gpu_idxs, gpu_idxs[0]), result_db[gpu_mask]);
    //  LOG_RANK_0("Intra GPU subset {:b} done: # of final paths = {}", gpu_mask, result_db[gpu_mask].size());
    //  PrintResultDBByRank(result_db[gpu_mask], 10);
    //}

    // Inter setup - TODO need to start from all GPUs
    //std::vector<std::vector<std::pair<std::vector<Transfer>, BenchmarkResult>>> result_db_inter;
    //result_db_inter.resize(1 << num_gpus);
    //for (int gpu_mask = 0; gpu_mask < (1 << num_gpus); ++gpu_mask) {
    //  std::vector<int> gpu_idxs;
    //  for (int i = 0; i < num_gpus; ++i) {
    //    if (gpu_mask & (1 << i)) {
    //      gpu_idxs.push_back(i);
    //    }
    //  }
    //  if (gpu_idxs.size() <= 0) continue;
    //  // TODO
    //  //if (gpu_idxs.size() > 1) break;

    //  // since it is a linear chain, we need to start from all GPUs
    //  for (int gpu_idx: gpu_idxs) {
    //    std::vector<Transfer> transfers;
    //    VisitGPUInter(transfers, gpu_idx, gpu_idx, RemoveElem(gpu_idxs, gpu_idx), result_db_inter[gpu_mask]);
    //  }
    //  LOG_RANK_0("Inter GPU subset {:b} done: # of final paths = {}", gpu_mask, result_db_inter[gpu_mask].size());
    //  //PrintResultDBByRank(result_db_inter[gpu_mask], 10);
    //  //TODO
    //  PrintResultDBByRank(result_db_inter[gpu_mask]);
    //}