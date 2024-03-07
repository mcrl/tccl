#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>
#include "conf_v2.hpp"
#include "benchmark_v2.hpp"
#include "util.hpp"
#include "check.hpp"

static std::vector<Transfer> transfer_to_launch;

static void print_help(const char *prog_name) {
  printf("Usage: %s [-d]", prog_name);
  printf("Options:\n");
  printf("     -d : print debug. (default: off)\n");
  printf("     -x : XML string to launch.\n");
  printf("     -b : Burteforce all bitmasks.\n");
}

static bool bruteforce = false;

static void parse_opt(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "dbx:")) != -1) {
    switch (c) {
      case 'd': conf.debug = true; break;
      case 'b': bruteforce = true; break;
      case 'x': {
        printf("Optarg: %s\n", optarg);
        transfer_to_launch = DeserializeTransfers(optarg); break;
      }
      default: print_help(argv[0]); exit(0);
    }
  }
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  
  // conf defaults
  conf.bw_threshold = 0; 
  conf.niters = 100;
  conf.warmup_iters = conf.niters / 10;

  parse_opt(argc, argv);
  conf.nbytes = transfer_to_launch[0].nbytes;
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
  if (rank == 0) {
    size_t tsize = transfer_to_launch.size();
    for (int bitmask = bruteforce ? 0 : (1 << tsize) - 1; bitmask < (1 << tsize); ++bitmask) {
      std::vector<Transfer> transfers;
      for (int i = 0; i < (int)tsize; ++i) {
        if (bitmask & (1 << i)) {
          transfers.push_back(transfer_to_launch[i]);
        }
      }
      auto res = Benchmark(transfers);
      double bw = conf.nbytes / (res.us_avg / 1e6); // bytes per second
      LOG_RANK_0("bitmask {:08b} BW = {} GB/s {} us", bitmask, bw / 1e9, res.us_avg);
    }
    CommandExit();
  } else {
    ProcessPoolLoop();
  }

  MPI_Finalize();

  return 0;
}