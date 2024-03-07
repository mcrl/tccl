#include <cstdio>
#include <random>
#include <chrono>
#include <mpi.h>
#include <nccl.h>
#include <spdlog/spdlog.h>
#include <unistd.h>

#include "check.hpp"

int src = 0, dst = 0;
bool validation = false;

static void print_help(const char* prog_name) {
  printf("Usage: %s [-v]\n", prog_name);
}

static void parse_opt(int argc, char **argv) {
  int c;
  //while ((c = getopt(argc, argv, "pvht:n:")) != -1) {
  while ((c = getopt(argc, argv, "v")) != -1) {
    switch (c) {
      case 'v':
        validation = true;
        break;
      default:
        print_help(argv[0]);
        exit(0);
    }
  }
  for (int i = optind, j = 0; i < argc; ++i, ++j) {
    switch (j) {
      case 0: src = atoi(argv[i]); break;
      case 1: dst = atoi(argv[i]); break;
      default: break;
    }
  }
}

std::chrono::steady_clock cpu_clock;

const int ngpu = 1;
int niter = 50;
const size_t nparam = 16 * 1024 * 1024;
const size_t nbytes = nparam * sizeof(float);
int devs[ngpu];
ncclComm_t comms[ngpu];
ncclComm_t comms2[ngpu];
float* ha[ngpu];
float* hb[ngpu];
float* da[ngpu];
float* db[ngpu];
float* dc[ngpu];
float* dd[ngpu];
cudaStream_t streams[ngpu];
cudaStream_t streams2[ngpu];
int gpu_idx[100] = {};
int size, rank;

std::chrono::steady_clock::time_point syncgpu_gettime() {
  CHECK_CUDA(cudaSetDevice(gpu_idx[rank]));
  CHECK_CUDA(cudaDeviceSynchronize());
  for (int i = 0; i < ngpu; ++i) {
    CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    CHECK_CUDA(cudaStreamSynchronize(streams2[i]));
  }
  MPI_Barrier(MPI_COMM_WORLD);
  return cpu_clock.now();
}

void test_AllReduce() {
  auto st = syncgpu_gettime();
  CHECK_NCCL(ncclGroupStart());
  for (int i = 0; i < ngpu; ++i) {
    CHECK_NCCL(ncclAllReduce(da[i], db[i], nparam, ncclFloat, ncclSum, comms[i], streams[i]));
  }
  CHECK_NCCL(ncclGroupEnd());
  auto et = syncgpu_gettime();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(et - st).count();

  auto algbw = nparam * sizeof(float) / us / 1e3;
  auto busbw = algbw * (2.0 * (ngpu - 1) / ngpu);

  spdlog::info("AllReduce {} us, algbw {} GB/s, busbw {} GB/s", us, algbw, busbw);
}

void test_Broadcast() {
  auto st = syncgpu_gettime();
  CHECK_NCCL(ncclGroupStart());
  for (int i = 0; i < ngpu; ++i) {
    const int root = 0;
    CHECK_NCCL(ncclBroadcast(da[i], db[i], nparam, ncclFloat, root, comms[i], streams[i]));
  }
  CHECK_NCCL(ncclGroupEnd());
  auto et = syncgpu_gettime();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(et - st).count();

  auto algbw = nparam * sizeof(float) / us / 1e3;
  auto busbw = algbw;

  spdlog::info("Broadcast {} us, algbw {} GB/s, busbw {} GB/s", us, algbw, busbw);
}

void test_Reduce() {
  auto st = syncgpu_gettime();
  CHECK_NCCL(ncclGroupStart());
  for (int i = 0; i < ngpu; ++i) {
    const int root = 0;
    CHECK_NCCL(ncclReduce(da[i], db[i], nparam, ncclFloat, ncclSum, root, comms[i], streams[i]));
  }
  CHECK_NCCL(ncclGroupEnd());
  auto et = syncgpu_gettime();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(et - st).count();

  auto algbw = nparam * sizeof(float) / us / 1e3;
  auto busbw = algbw;

  spdlog::info("Reduce {} us, algbw {} GB/s, busbw {} GB/s", us, algbw, busbw);
}

void test_AllGather() {
  auto st = syncgpu_gettime();
  CHECK_NCCL(ncclGroupStart());
  for (int i = 0; i < ngpu; ++i) {
    CHECK_NCCL(ncclAllGather(da[i], db[i], nparam / ngpu, ncclFloat, comms[i], streams[i]));
  }
  CHECK_NCCL(ncclGroupEnd());
  auto et = syncgpu_gettime();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(et - st).count();

  auto algbw = nparam * sizeof(float) / us / 1e3;
  auto busbw = algbw * ((ngpu - 1.0) / ngpu);

  spdlog::info("AllGather {} us, algbw {} GB/s, busbw {} GB/s", us, algbw, busbw);
}

void test_ReduceScatter() {
  auto st = syncgpu_gettime();
  CHECK_NCCL(ncclGroupStart());
  for (int i = 0; i < ngpu; ++i) {
    CHECK_NCCL(ncclReduceScatter(da[i], db[i], nparam / ngpu, ncclFloat, ncclSum, comms[i], streams[i]));
  }
  CHECK_NCCL(ncclGroupEnd());
  auto et = syncgpu_gettime();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(et - st).count();

  auto algbw = nparam * sizeof(float) / us / 1e3;
  auto busbw = algbw * ((ngpu - 1.0) / ngpu);

  spdlog::info("ReduceScatter {} us, algbw {} GB/s, busbw {} GB/s", us, algbw, busbw);
}

void test_CPUGPUAllGather() {
  auto st = syncgpu_gettime();
  for (int i = 0; i < ngpu; ++i) {
    CHECK_CUDA(cudaMemcpyAsync(da[i], ha[i], nparam / ngpu * sizeof(float), cudaMemcpyHostToDevice, streams[i]));
  }
  CHECK_NCCL(ncclGroupStart());
  for (int i = 0; i < ngpu; ++i) {
    CHECK_NCCL(ncclAllGather(da[i], db[i], nparam / ngpu, ncclFloat, comms[i], streams[i]));
  }
  CHECK_NCCL(ncclGroupEnd());
  auto et = syncgpu_gettime();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(et - st).count();

  auto algbw = nparam * sizeof(float) / us / 1e3;

  spdlog::info("CPUGPUAllGather {} us, algbw {} GB/s", us, algbw);
}

void test_CPUGPUAllGather_v2() {
  auto st = syncgpu_gettime();
  for (int i = 0; i < ngpu; ++i) {
    for (int j = 0; j < ngpu; ++j) {
      CHECK_CUDA(cudaMemcpyAsync(db[i] + j * nparam / ngpu, ha[j], nparam / ngpu * sizeof(float), cudaMemcpyHostToDevice, streams[i]));
    }
  }
  auto et = syncgpu_gettime();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(et - st).count();

  auto algbw = nparam * sizeof(float) / us / 1e3;

  spdlog::info("CPUGPUAllGather_v2 {} us, algbw {} GB/s", us, algbw);
}

void test_AGRS() {
  // DEADLOCK, since NCCL does not support multiple operations in multiple streams 
  auto st = syncgpu_gettime();
  CHECK_NCCL(ncclGroupStart());
  for (int i = 0; i < ngpu; ++i) {
    CHECK_NCCL(ncclAllGather(da[i], db[i], nparam / ngpu, ncclFloat, comms[i], streams[i]));
    CHECK_NCCL(ncclReduceScatter(dc[i], dd[i], nparam / ngpu, ncclFloat, ncclSum, comms[i], streams2[i]));
  }
  CHECK_NCCL(ncclGroupEnd());
  auto et = syncgpu_gettime();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(et - st).count();

  auto algbw = nparam * sizeof(float) / us / 1e3;

  spdlog::info("AGRS {} us, algbw {} GB/s", us, algbw);
}

void test_GPUCPUReduceScatter() {
  auto st = syncgpu_gettime();
  CHECK_NCCL(ncclGroupStart());
  for (int i = 0; i < ngpu; ++i) {
    CHECK_NCCL(ncclReduceScatter(da[i], db[i], nparam / ngpu, ncclFloat, ncclSum, comms[i], streams[i]));
  }
  CHECK_NCCL(ncclGroupEnd());
  for (int i = 0; i < ngpu; ++i) {
    CHECK_CUDA(cudaMemcpyAsync(hb[i], db[i], nparam / ngpu * sizeof(float), cudaMemcpyDeviceToHost, streams[i]));
  }
  auto et = syncgpu_gettime();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(et - st).count();

  auto algbw = nparam * sizeof(float) / us / 1e3;

  spdlog::info("GPUCPUReduceScatter {} us, algbw {} GB/s", us, algbw);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  char hostname[MPI_MAX_PROCESSOR_NAME]; int hostnamelen;
  MPI_Get_processor_name(hostname, &hostnamelen);

  ncclUniqueId nid;
  if (rank == 0) {
    CHECK_NCCL(ncclGetUniqueId(&nid));
  }
  MPI_Bcast(&nid, sizeof(nid), MPI_BYTE, 0, MPI_COMM_WORLD);

  std::default_random_engine eng;
  std::uniform_real_distribution<float> dis(0, 1);

  for (int i = 1; i < argc; ++i) {
    gpu_idx[i - 1] = atoi(argv[i]);
  }

  CHECK_CUDA(cudaSetDevice(gpu_idx[rank]));
  spdlog::info("[{}] rank {} / size {} / gpu_idx {}", hostname, rank, size, gpu_idx[rank]);

  //MPI_Finalize();
  //return 0;

  for (int i = 0; i < ngpu; ++i) {
    CHECK_NCCL(ncclCommInitRank(&comms[i], size, nid, rank));
    CHECK_CUDA(cudaMalloc(&da[i], nparam * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&db[i], nparam * sizeof(float)));
    CHECK_CUDA(cudaStreamCreate(&streams[i]));
    CHECK_CUDA(cudaStreamCreate(&streams2[i]));
  }

  // 2^n search
  if (rank == 0) {
    spdlog::info("Forward (0->1->...)");
  }
  for (int bitmask = 0; bitmask < (1 << size); ++bitmask) {
    std::vector<size_t> us_db;
    for (int iter = 0; iter < niter; ++iter) {
      auto st = syncgpu_gettime();
      CHECK_NCCL(ncclGroupStart());
      for (int i = 0; i < ngpu; ++i) {
        if ((bitmask & (1 << rank)) != 0) {
          CHECK_NCCL(ncclSend(da[i], nparam, ncclFloat, (rank + 1) % size, comms[i], streams[i]));
        }
        if ((bitmask & (1 << ((rank + size - 1) % size))) != 0) {
          CHECK_NCCL(ncclRecv(db[i], nparam, ncclFloat, (rank + size - 1) % size, comms[i], streams[i]));
        }
      }
      CHECK_NCCL(ncclGroupEnd());
      auto et = syncgpu_gettime();
      auto us = std::chrono::duration_cast<std::chrono::microseconds>(et - st).count();
      //spdlog::info("us={}", us);
      //std::this_thread::sleep_for(std::chrono::seconds(1));
      us_db.push_back(us);
    }
    std::sort(us_db.begin(), us_db.end());
    size_t us_med = us_db[us_db.size() / 2];
    double gbps = (double)nbytes / us_med / 1e3;
    if (rank == 0) {
      spdlog::info("bitmask {:08b} {} GB/s {} us", bitmask, gbps, us_med);
    }
  }

  //if (rank == 0) {
  //  spdlog::info("Backard (0->n-1->n-2->...)");
  //}
  //for (int bitmask = 0; bitmask < (1 << size); ++bitmask) {
  //  std::vector<size_t> us_db;
  //  for (int iter = 0; iter < niter; ++iter) {
  //    auto st = syncgpu_gettime();
  //    CHECK_NCCL(ncclGroupStart());
  //    for (int i = 0; i < ngpu; ++i) {
  //      if ((bitmask & (1 << ((rank + size - 1) % size))) != 0) {
  //        CHECK_NCCL(ncclSend(da[i], nparam, ncclFloat, (rank + size - 1) % size, comms[i], streams[i]));
  //      }
  //      if ((bitmask & (1 << rank)) != 0) {
  //        CHECK_NCCL(ncclRecv(db[i], nparam, ncclFloat, (rank + 1) % size, comms[i], streams2[i]));
  //      }
  //    }
  //    CHECK_NCCL(ncclGroupEnd());
  //    auto et = syncgpu_gettime();
  //    auto us = std::chrono::duration_cast<std::chrono::microseconds>(et - st).count();
  //    spdlog::info("us={}", us);
  //    us_db.push_back(us);
  //  }
  //  std::sort(us_db.begin(), us_db.end());
  //  size_t us_med = us_db[us_db.size() / 2];
  //  double gbps = (double)nbytes / us_med / 1e3;
  //  if (rank == 0) {
  //    spdlog::info("bitmask {:08b} {} GB/s {} us", bitmask, gbps, us_med);
  //  }
  //}

  //CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
  //spdlog::info("Unidir (forward)");
  //CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
  //for (int iter = 0; iter < niter; ++iter) {
  //  auto st = syncgpu_gettime();
  //  CHECK_NCCL(ncclGroupStart());
  //  for (int i = 0; i < ngpu; ++i) {
  //      CHECK_NCCL(ncclSend(da[i], nparam, ncclFloat, (rank + 1) % size, comms[i], streams[i]));
  //      CHECK_NCCL(ncclRecv(db[i], nparam, ncclFloat, (rank + size - 1) % size, comms[i], streams[i]));
  //  }
  //  CHECK_NCCL(ncclGroupEnd());
  //  auto et = syncgpu_gettime();

  //  auto us = std::chrono::duration_cast<std::chrono::microseconds>(et - st).count();
  //  double gbps = (double)nbytes / us / 1e3;
  //  if (iter == niter - 1 && rank == size - 1)
  //  spdlog::info("iter {} - {} GB/s {} us", iter, gbps, us);
  //}

  //CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
  //spdlog::info("Unidir (reverse)");
  //CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
  //for (int iter = 0; iter < niter; ++iter) {
  //  auto st = syncgpu_gettime();
  //  CHECK_NCCL(ncclGroupStart());
  //  for (int i = 0; i < ngpu; ++i) {
  //    CHECK_NCCL(ncclSend(da[i], nparam, ncclFloat, (rank + size - 1) % size, comms[i], streams[i]));
  //    CHECK_NCCL(ncclRecv(db[i], nparam, ncclFloat, (rank + 1) % size, comms[i], streams[i]));
  //  }
  //  CHECK_NCCL(ncclGroupEnd());
  //  auto et = syncgpu_gettime();

  //  auto us = std::chrono::duration_cast<std::chrono::microseconds>(et - st).count();
  //  double gbps = (double)nbytes / us / 1e3;
  //  if (iter == niter - 1 && rank == size - 1)
  //  spdlog::info("iter {} - {} GB/s {} us", iter, gbps, us);
  //}

  //CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
  //spdlog::info("Unidir (forward) without last->first");
  //CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
  //for (int iter = 0; iter < niter; ++iter) {
  //  auto st = syncgpu_gettime();
  //  CHECK_NCCL(ncclGroupStart());
  //  for (int i = 0; i < ngpu; ++i) {
  //      if (rank != size - 1)
  //        CHECK_NCCL(ncclSend(da[i], nparam, ncclFloat, (rank + 1) % size, comms[i], streams[i]));
  //      if (rank != 0)
  //        CHECK_NCCL(ncclRecv(db[i], nparam, ncclFloat, (rank + size - 1) % size, comms[i], streams[i]));
  //  }
  //  CHECK_NCCL(ncclGroupEnd());
  //  auto et = syncgpu_gettime();

  //  auto us = std::chrono::duration_cast<std::chrono::microseconds>(et - st).count();
  //  double gbps = (double)nbytes / us / 1e3;
  //  if (iter == niter - 1 && rank == size - 1)
  //  spdlog::info("iter {} - {} GB/s {} us", iter, gbps, us);
  //}

  //CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
  //spdlog::info("Unidir (reverse) without first->last");
  //CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
  //for (int iter = 0; iter < niter; ++iter) {
  //  auto st = syncgpu_gettime();
  //  CHECK_NCCL(ncclGroupStart());
  //  for (int i = 0; i < ngpu; ++i) {
  //    if (rank != 0)
  //      CHECK_NCCL(ncclSend(da[i], nparam, ncclFloat, (rank + size - 1) % size, comms[i], streams[i]));
  //    if (rank != size - 1)
  //      CHECK_NCCL(ncclRecv(db[i], nparam, ncclFloat, (rank + 1) % size, comms[i], streams[i]));
  //  }
  //  CHECK_NCCL(ncclGroupEnd());
  //  auto et = syncgpu_gettime();

  //  auto us = std::chrono::duration_cast<std::chrono::microseconds>(et - st).count();
  //  double gbps = (double)nbytes / us / 1e3;
  //  if (iter == niter - 1 && rank == size - 1)
  //  spdlog::info("iter {} - {} GB/s {} us", iter, gbps, us);
  //}

  //CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
  //spdlog::info("Unidir (forward) even->odd only");
  //CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
  //for (int iter = 0; iter < niter; ++iter) {
  //  auto st = syncgpu_gettime();
  //  CHECK_NCCL(ncclGroupStart());
  //  for (int i = 0; i < ngpu; ++i) {
  //      if (rank % 2 == 0)
  //        CHECK_NCCL(ncclSend(da[i], nparam, ncclFloat, (rank + 1) % size, comms[i], streams[i]));
  //      if (rank % 2 == 1)
  //        CHECK_NCCL(ncclRecv(db[i], nparam, ncclFloat, (rank + size - 1) % size, comms[i], streams[i]));
  //  }
  //  CHECK_NCCL(ncclGroupEnd());
  //  auto et = syncgpu_gettime();

  //  auto us = std::chrono::duration_cast<std::chrono::microseconds>(et - st).count();
  //  double gbps = (double)nbytes / us / 1e3;
  //  if (iter == niter - 1 && rank == size - 1)
  //  spdlog::info("iter {} - {} GB/s {} us", iter, gbps, us);
  //}

  //CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
  //spdlog::info("Unidir (reverse) even<-odd only");
  //CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
  //for (int iter = 0; iter < niter; ++iter) {
  //  auto st = syncgpu_gettime();
  //  CHECK_NCCL(ncclGroupStart());
  //  for (int i = 0; i < ngpu; ++i) {
  //    if (rank % 2 == 1)
  //      CHECK_NCCL(ncclSend(da[i], nparam, ncclFloat, (rank + size - 1) % size, comms[i], streams[i]));
  //    if (rank % 2 == 0)
  //      CHECK_NCCL(ncclRecv(db[i], nparam, ncclFloat, (rank + 1) % size, comms[i], streams[i]));
  //  }
  //  CHECK_NCCL(ncclGroupEnd());
  //  auto et = syncgpu_gettime();

  //  auto us = std::chrono::duration_cast<std::chrono::microseconds>(et - st).count();
  //  double gbps = (double)nbytes / us / 1e3;
  //  if (iter == niter - 1 && rank == size - 1)
  //  spdlog::info("iter {} - {} GB/s {} us", iter, gbps, us);
  //}

  //spdlog::info("Bidir (one-by-one)");
  //for (int iter = 0; iter < niter; ++iter) {
  //  auto st = syncgpu_gettime();
  //  CHECK_NCCL(ncclGroupStart());
  //  for (int i = 0; i < ngpu; ++i) {
  //      CHECK_NCCL(ncclSend(da[i], nparam, ncclFloat, 1, comms[i], streams[i]));
  //      CHECK_NCCL(ncclRecv(db[i], nparam, ncclFloat, 0, comms[i], streams[i]));
  //  }
  //  CHECK_NCCL(ncclGroupEnd());
  //  CHECK_NCCL(ncclGroupStart());
  //  for (int i = 0; i < ngpu; ++i) {
  //    if (rank == 0) {
  //      CHECK_NCCL(ncclRecv(db[i], nparam, ncclFloat, 1, comms[i], streams[i]));
  //    }
  //    if (rank == 1) {
  //      CHECK_NCCL(ncclSend(da[i], nparam, ncclFloat, 0, comms[i], streams[i]));
  //    }
  //  }
  //  CHECK_NCCL(ncclGroupEnd());
  //  auto et = syncgpu_gettime();

  //  auto us = std::chrono::duration_cast<std::chrono::microseconds>(et - st).count();
  //  double gbps = (double)nbytes / us / 1e3;
  //  if (iter == niter - 1 && rank == size - 1)
  //  spdlog::info("iter {} - {} GB/s {} us", iter, gbps, us);
  //}

  //spdlog::info("Bidir (Simul)");
  //for (int iter = 0; iter < niter; ++iter) {
  //  auto st = syncgpu_gettime();
  //  CHECK_NCCL(ncclGroupStart());
  //  for (int i = 0; i < ngpu; ++i) {
  //    if (rank == 0) {
  //      CHECK_NCCL(ncclSend(da[i], nparam, ncclFloat, 1, comms[i], streams[i]));
  //      CHECK_NCCL(ncclRecv(db[i], nparam, ncclFloat, 1, comms[i], streams[i]));
  //    }
  //    if (rank == 1) {
  //      CHECK_NCCL(ncclSend(da[i], nparam, ncclFloat, 0, comms[i], streams[i]));
  //      CHECK_NCCL(ncclRecv(db[i], nparam, ncclFloat, 0, comms[i], streams[i]));
  //    }
  //  }
  //  CHECK_NCCL(ncclGroupEnd());
  //  auto et = syncgpu_gettime();

  //  auto us = std::chrono::duration_cast<std::chrono::microseconds>(et - st).count();
  //  double gbps = (double)nbytes / us / 1e3;
  //  if (iter == niter - 1 && rank == size - 1)
  //  spdlog::info("iter {} - {} GB/s {} us", iter, gbps, us);
  //}

  for (int i = 0; i < ngpu; ++i) {
    CHECK_NCCL(ncclCommDestroy(comms[i]));
  }

  MPI_Finalize();

  return 0;
}