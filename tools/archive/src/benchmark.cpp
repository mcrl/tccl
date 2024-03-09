#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <string>
#include <omp.h>
#include <numa.h>
#include <numaif.h>
#include <random>
#include <mpi.h>
#include <immintrin.h>
#include <fstream>
#include <spdlog/spdlog.h>
#include <cuda_runtime.h>
#include "check.hpp"
#include "conf.hpp"
#include "util.hpp"
#include "job.hpp"

#include <nlohmann/json.hpp>
using json = nlohmann::json;

static const char* conf_fn = nullptr;
static const char* output_fn = nullptr;

static void print_help(const char* prog_name) {
  LOG_RANK_0("Usage: {} [-h] <configuration filename> <output filename>", prog_name);
}

static void parse_opt(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "h")) != -1) {
    switch (c) {
      case 'h':
        print_help(argv[0]);
        mpi_finalize_and_exit(0);
        break;
      default:
        print_help(argv[0]);
        mpi_finalize_and_exit(1);
    }
  }

  for (int i = optind, j = 0; i < argc; ++i, ++j) {
    switch (j) {
      case 0: conf_fn = argv[i]; break;
      case 1: output_fn = argv[i]; break;
      default: break;
    }
  }

  if (conf_fn == nullptr || output_fn == nullptr) {
    print_help(argv[0]);
    mpi_finalize_and_exit(1);
  }
}

static void parse_conf() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::ifstream f(conf_fn);
  json data = json::parse(f);
  conf.niters = data["niters"].get<int>();
  conf.validation = data["validation"].get<bool>();
  conf.output_fn = output_fn;
  for (auto& job : data["jobs"]) {
    int job_rank = job["rank"].get<int>();
    if (rank == job_rank) {
      conf.local_conf = job;
    }
  }
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  parse_opt(argc, argv);
  parse_conf();

  // Double check GPU order
  {
    int count;
    CHECK_CUDA(cudaGetDeviceCount(&count));
    for (int i = 0; i < count; ++i) {
      int bus, dev;
      CHECK_CUDA(cudaDeviceGetAttribute(&bus, cudaDevAttrPciBusId, i));
      CHECK_CUDA(cudaDeviceGetAttribute(&dev, cudaDevAttrPciDeviceId, i));
      LOG_RANK_ANY("GPU {} PCI {:02X}:{:02X}", i, bus, dev);
    }
  }

  auto job_type = conf.local_conf["type"].get<std::string>();
  LOG_RANK_ANY("Job type: {}", job_type);
  if (job_type == "CPU_READ_CPUMEM") {
    cpu_read_cpumem();
  } else if (job_type == "CPU_WRITE_CPUMEM") {
    cpu_write_cpumem();
  } else if (job_type == "GPU_READ_CPUMEM_KERNEL") {
    gpu_read_cpumem_kernel();
  } else if (job_type == "GPU_WRITE_CPUMEM_KERNEL") {
    gpu_write_cpumem_kernel();
  } else if (job_type == "GPU_READ_GPUMEM_KERNEL") {
    gpu_read_gpumem_kernel();
  } else if (job_type == "GPU_WRITE_GPUMEM_KERNEL") {
    gpu_write_gpumem_kernel();
  } else if (job_type == "GPU_READ_CPUMEM_MEMCPY") {
    gpu_read_cpumem_memcpy();
  } else if (job_type == "GPU_WRITE_CPUMEM_MEMCPY") {
    gpu_write_cpumem_memcpy();
  } else if (job_type == "GPU_READ_GPUMEM_MEMCPY") {
    gpu_read_gpumem_memcpy();
  } else if (job_type == "GPU_WRITE_GPUMEM_MEMCPY") {
    gpu_write_gpumem_memcpy();
  } else if (job_type == "NIC_READ_CPUMEM") {
    nic_read_cpumem();
  } else if (job_type == "NIC_WRITE_CPUMEM") {
    nic_write_cpumem();
  } else if (job_type == "NIC_READ_GPUMEM") {
    nic_read_gpumem();
  } else if (job_type == "NIC_WRITE_GPUMEM") {
    nic_write_gpumem();
  } else if (job_type == "NIC_SHARP_ALLREDUCE") {
    nic_sharp_allreduce();
  } else if (job_type == "DUMMY") {
    dummy();
  } else {
    LOG_RANK_ANY("Unknown job type: {}", job_type);
    mpi_finalize_and_exit(1);
  }
  
  MPI_Finalize();

  return 0;
}