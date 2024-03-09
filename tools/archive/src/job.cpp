#include "job.hpp"

#include <omp.h>
#include <numa.h>
#include <numaif.h>
#include <cassert>
#include <mpi.h>
#include <immintrin.h>
#include <fstream>
#include <algorithm>
#include <cuda_runtime.h>
#include "conf.hpp"
#include "util.hpp"
#include "check.hpp"
#include "kernels.hpp"
#include "ibv_helper.hpp"
#include "sharp_helper.hpp"

static int get_node_running() {
  int cpu = sched_getcpu();
  int node = numa_node_of_cpu(cpu);
  return node;
}

static int get_node_allocated(void* addr) {
  int mode;
  CHECK_ERRNO(get_mempolicy(&mode, NULL, 0, addr, MPOL_F_NODE | MPOL_F_ADDR));
  return mode;
}

static json stats = json::array();
static std::vector<size_t> us_db;
static std::vector<std::vector<size_t>> us_local_db;

static void collect_stat(int iter, size_t us, size_t us_local, double expected, double actual) {
  /*
    json structure for stat:
    "stats": [
    {
      // iter 0
      "us": 1234,
      "us_local": [1,2,3,4],
      "expected": [1,2,3,4],
      "actual": [1,2,3,4]
    },
    {
      // iter 1
    }
    ]
  */
  LOG_RANK_0("collect_stat iter: {}", iter);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  size_t us_locals[size];
  double expecteds[size];
  double actuals[size];
  MPI_Gather(&us_local, 1, MPI_UNSIGNED_LONG, us_locals, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  MPI_Gather(&expected, 1, MPI_DOUBLE, expecteds, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gather(&actual, 1, MPI_DOUBLE, actuals, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    json stat;
    stat["us"] = us;
    stat["us_local"] = std::vector<size_t>(us_locals, us_locals + size);
    if (conf.validation) {
      stat["expected"] = std::vector<double>(expecteds, expecteds + size);
      stat["actual"] = std::vector<double>(actuals, actuals + size);
    }
    stats.push_back(stat);

    // Save raw data for later analysis (min, max, ...)
    us_db.push_back(us);
    if (iter == 0) {
      us_local_db.resize(size);
    }
    for (int i = 0; i < size; ++i) {
      us_local_db[i].push_back(us_locals[i]);
    }
  }
}

static void print_stat() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    json output;

    {
      json min_stat;
      min_stat["us"] = *std::min_element(us_db.begin(), us_db.end());
      std::vector<size_t> us_local;
      for (int i = 0; i < size; ++i) {
        us_local.push_back(*std::min_element(us_local_db[i].begin(), us_local_db[i].end()));
      }
      min_stat["us_local"] = us_local;
      output["min_stat"] = min_stat;
    }

    {
      json max_stat;
      max_stat["us"] = *std::max_element(us_db.begin(), us_db.end());
      std::vector<size_t> us_local;
      for (int i = 0; i < size; ++i) {
        us_local.push_back(*std::max_element(us_local_db[i].begin(), us_local_db[i].end()));
      }
      max_stat["us_local"] = us_local;
      output["max_stat"] = max_stat;
    }

    {
      json med_stat;
      std::sort(us_db.begin(), us_db.end());
      med_stat["us"] = us_db[us_db.size() / 2];
      std::vector<size_t> us_local;
      for (int i = 0; i < size; ++i) {
        std::sort(us_local_db[i].begin(), us_local_db[i].end());
        us_local.push_back(us_local_db[i][us_local_db[i].size() / 2]);
      }
      med_stat["us_local"] = us_local;
      output["med_stat"] = med_stat;
    }

    {
      json avg_stat;
      avg_stat["us"] = std::accumulate(us_db.begin(), us_db.end(), 0) / us_db.size();
      std::vector<size_t> us_local;
      for (int i = 0; i < size; ++i) {
        us_local.push_back(std::accumulate(us_local_db[i].begin(), us_local_db[i].end(), 0) / us_local_db[i].size());
      }
      avg_stat["us_local"] = us_local;
      output["avg_stat"] = avg_stat;
    }
  
    output["stats"] = stats;
    std::ofstream o(conf.output_fn);
    o << std::setw(4) << output << std::endl;
  }
}

void cpu_read_cpumem() {
  size_t nbytes = conf.local_conf["nbytes"].get<size_t>();
  int core_numa = conf.local_conf["cpu_numa_idx"].get<int>();
  int mem_numa = conf.local_conf["cpumem_numa_idx"].get<int>();

  size_t nparams = nbytes / sizeof(float);
  #pragma omp parallel
  {
    numa_run_on_node(core_numa);
    assert(get_node_running() == core_numa);
  }

  float *buf = (float*)numa_alloc_onnode(nbytes, mem_numa);
  assert(get_node_allocated(buf) == mem_numa);

  for (int it = 0; it < conf.niters; ++it) {
    // Fill buf with random numbers and save sum in s_ans
    double s_ans = 0;
    if (conf.validation) {
      fill_random_float(buf, nparams);
      s_ans = sum_float(buf, nparams);
    }

    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    auto st = get_time();
    
    __m256 s_total = _mm256_setzero_ps();
    #pragma omp parallel
    { 
      __m256 s_private = _mm256_setzero_ps();
      #pragma omp for nowait
      for (size_t i = 0; i < nparams; i += sizeof(__m256) / sizeof(float)) {
        s_private = _mm256_add_ps(s_private, _mm256_load_ps(buf + i));
      }
      #pragma omp critical
      s_total = _mm256_add_ps(s_total, s_private);
    }

    auto et_local = get_time();
    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    auto et = get_time();

    // sum elements in s_total into sink
    volatile double sink = 0;
    if (conf.validation) {
      sink = sum_float((float*)&s_total, sizeof(__m256) / sizeof(float));
    }

    auto us = get_duration_us(st, et);
    auto us_local = get_duration_us(st, et_local);

    collect_stat(it, us, us_local, s_ans, sink);
  }
  print_stat();

  numa_free(buf, nbytes);
}

void cpu_write_cpumem() {
  size_t nbytes = conf.local_conf["nbytes"].get<size_t>();
  int core_numa = conf.local_conf["cpu_numa_idx"].get<int>();
  int mem_numa = conf.local_conf["cpumem_numa_idx"].get<int>();

  size_t nparams = nbytes / sizeof(float);
  #pragma omp parallel
  {
    numa_run_on_node(core_numa);
    assert(get_node_running() == core_numa);
  }

  float *buf = (float*)numa_alloc_onnode(nbytes, mem_numa);
  assert(get_node_allocated(buf) == mem_numa);

  for (int it = 0; it < conf.niters; ++it) {
    // It's hard to generate random numbers on the fly
    // so we will generate random __m256 and repeat it
    // sum is saved in s_ans
    double s_ans = 0;
    __m256 s_vec;
    if (conf.validation) {
      fill_random_float((float*)&s_vec, sizeof(__m256) / sizeof(float));
      s_ans = sum_float((float*)&s_vec, sizeof(__m256) / sizeof(float));
      s_ans *= nparams / (sizeof(__m256) / sizeof(float));
    }

    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    auto st = get_time();

    #pragma omp parallel for
    for (size_t i = 0; i < nparams; i += sizeof(__m256) / sizeof(float)) {
      _mm256_stream_ps(buf + i, s_vec);
    }

    auto et_local = get_time();
    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    auto et = get_time();

    volatile double sink = 0;
    if (conf.validation) {
      sink = sum_float(buf, nparams);
    }

    auto us = get_duration_us(st, et);
    auto us_local = get_duration_us(st, et_local);

    collect_stat(it, us, us_local, s_ans, sink);
  }
  print_stat();

  numa_free(buf, nbytes);
}

void gpu_read_cpumem_kernel() {
  size_t nbytes = conf.local_conf["nbytes"].get<size_t>();
  int gpu_idx = conf.local_conf["gpu_idx"].get<int>();
  int mem_numa = conf.local_conf["cpumem_numa_idx"].get<int>();

  size_t nparams = nbytes / sizeof(float);
  CHECK_CUDA(cudaSetDevice(gpu_idx));

  float *h_a = (float*)numa_alloc_onnode(nbytes, mem_numa);
  float *d_a, *d_for_h_a;
  CHECK_CUDA(cudaHostRegister(h_a, nbytes, cudaHostRegisterMapped));
  CHECK_CUDA(cudaHostGetDevicePointer(&d_for_h_a, h_a, 0));
  CHECK_CUDA(cudaMalloc(&d_a, nbytes));
  assert(get_node_allocated(h_a) == mem_numa);

  for (int it = 0; it < conf.niters; ++it) {
    // Fill buf with random numbers and save sum in s_ans
    double s_ans = 0;
    if (conf.validation) {
      fill_random_float(h_a, nparams);
      s_ans = sum_float(h_a, nparams);
    }

    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    auto st = get_time();

    gpu_copy_wrapper(d_a, d_for_h_a, nbytes);

    auto et_local = get_time();
    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    auto et = get_time();

    volatile double sink = 0;
    if (conf.validation) {
      sink = sum_float_gpu(d_a, nparams);
    }

    auto us = get_duration_us(st, et);
    auto us_local = get_duration_us(st, et_local);

    collect_stat(it, us, us_local, s_ans, sink);
  }
  print_stat();

  CHECK_CUDA(cudaHostUnregister(h_a));
  numa_free(h_a, nbytes);
  CHECK_CUDA(cudaFree(d_a));
}

void gpu_write_cpumem_kernel() {
  size_t nbytes = conf.local_conf["nbytes"].get<size_t>();
  int gpu_idx = conf.local_conf["gpu_idx"].get<int>();
  int mem_numa = conf.local_conf["cpumem_numa_idx"].get<int>();

  size_t nparams = nbytes / sizeof(float);
  CHECK_CUDA(cudaSetDevice(gpu_idx));

  float *h_a = (float*)numa_alloc_onnode(nbytes, mem_numa);
  float *d_a, *d_for_h_a;
  CHECK_CUDA(cudaHostRegister(h_a, nbytes, cudaHostRegisterMapped));
  CHECK_CUDA(cudaHostGetDevicePointer(&d_for_h_a, h_a, 0));
  CHECK_CUDA(cudaMalloc(&d_a, nbytes));
  assert(get_node_allocated(h_a) == mem_numa);

  for (int it = 0; it < conf.niters; ++it) {
    // Fill buf with random numbers and save sum in s_ans
    double s_ans = 0;
    if (conf.validation) {
      fill_random_float_gpu(d_a, nparams);
      s_ans = sum_float_gpu(d_a, nparams);
    }

    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    auto st = get_time();

    gpu_copy_wrapper(d_for_h_a, d_a, nbytes);

    auto et_local = get_time();
    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    auto et = get_time();

    volatile double sink = 0;
    if (conf.validation) {
      sink = sum_float(h_a, nparams);
    }

    auto us = get_duration_us(st, et);
    auto us_local = get_duration_us(st, et_local);

    collect_stat(it, us, us_local, s_ans, sink);
  }
  print_stat();

  CHECK_CUDA(cudaHostUnregister(h_a));
  numa_free(h_a, nbytes);
  CHECK_CUDA(cudaFree(d_a));
}

void gpu_read_gpumem_kernel() {
  size_t nbytes = conf.local_conf["nbytes"].get<size_t>();
  int gpu_master_idx = conf.local_conf["gpu_idx"].get<int>();
  int gpu_slave_idx = conf.local_conf["gpumem_idx"].get<int>();

  size_t nparams = nbytes / sizeof(float);
  float *d_slave, *d_master;
  CHECK_CUDA(cudaSetDevice(gpu_slave_idx));
  CHECK_CUDA(cudaMalloc(&d_slave, nbytes));

  CHECK_CUDA(cudaSetDevice(gpu_master_idx));
  CHECK_CUDA(cudaMalloc(&d_master, nbytes));
  if (gpu_master_idx != gpu_slave_idx) {
    CHECK_CUDA(cudaDeviceEnablePeerAccess(gpu_slave_idx, 0));
  }

  for (int it = 0; it < conf.niters; ++it) {
    // Fill buf with random numbers and save sum in s_ans
    CHECK_CUDA(cudaSetDevice(gpu_slave_idx));
    double s_ans = 0;
    if (conf.validation) {
      fill_random_float_gpu(d_slave, nparams);
      s_ans = sum_float_gpu(d_slave, nparams);
    }
    CHECK_CUDA(cudaSetDevice(gpu_master_idx));

    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    auto st = get_time();

    gpu_copy_wrapper(d_master, d_slave, nbytes);

    auto et_local = get_time();
    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    auto et = get_time();

    volatile double sink = 0;
    if (conf.validation) {
      sink = sum_float_gpu(d_master, nparams);
    }

    auto us = get_duration_us(st, et);
    auto us_local = get_duration_us(st, et_local);

    collect_stat(it, us, us_local, s_ans, sink);
  }
  print_stat();

  CHECK_CUDA(cudaFree(d_master));
  CHECK_CUDA(cudaFree(d_slave));
}

void gpu_write_gpumem_kernel() {
  size_t nbytes = conf.local_conf["nbytes"].get<size_t>();
  int gpu_master_idx = conf.local_conf["gpu_idx"].get<int>();
  int gpu_slave_idx = conf.local_conf["gpumem_idx"].get<int>();

  size_t nparams = nbytes / sizeof(float);
  float *d_slave, *d_master;
  CHECK_CUDA(cudaSetDevice(gpu_slave_idx));
  CHECK_CUDA(cudaMalloc(&d_slave, nbytes));

  CHECK_CUDA(cudaSetDevice(gpu_master_idx));
  CHECK_CUDA(cudaMalloc(&d_master, nbytes));
  if (gpu_master_idx != gpu_slave_idx) {
    CHECK_CUDA(cudaDeviceEnablePeerAccess(gpu_slave_idx, 0));
  }

  for (int it = 0; it < conf.niters; ++it) {
    // Fill buf with random numbers and save sum in s_ans
    double s_ans = 0;
    if (conf.validation) {
      fill_random_float_gpu(d_master, nparams);
      s_ans = sum_float_gpu(d_master, nparams);
    }

    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    auto st = get_time();

    gpu_copy_wrapper(d_slave, d_master, nbytes);

    auto et_local = get_time();
    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    auto et = get_time();

    volatile double sink = 0;
    if (conf.validation) {
      sink = sum_float_gpu(d_slave, nparams);
    }

    auto us = get_duration_us(st, et);
    auto us_local = get_duration_us(st, et_local);

    collect_stat(it, us, us_local, s_ans, sink);
  }
  print_stat();

  CHECK_CUDA(cudaFree(d_master));
  CHECK_CUDA(cudaFree(d_slave));
}

void gpu_read_cpumem_memcpy() {
  size_t nbytes = conf.local_conf["nbytes"].get<size_t>();
  int gpu_idx = conf.local_conf["gpu_idx"].get<int>();
  int mem_numa = conf.local_conf["cpumem_numa_idx"].get<int>();

  size_t nparams = nbytes / sizeof(float);
  CHECK_CUDA(cudaSetDevice(gpu_idx));

  float *h_a = (float*)numa_alloc_onnode(nbytes, mem_numa);
  float *d_a;
  CHECK_CUDA(cudaHostRegister(h_a, nbytes, cudaHostRegisterMapped));
  CHECK_CUDA(cudaMalloc(&d_a, nbytes));
  assert(get_node_allocated(h_a) == mem_numa);

  for (int it = 0; it < conf.niters; ++it) {
    // Fill buf with random numbers and save sum in s_ans
    double s_ans = 0;
    if (conf.validation) {
      fill_random_float(h_a, nparams);
      s_ans = sum_float(h_a, nparams);
    }

    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    auto st = get_time();
    
    CHECK_CUDA(cudaMemcpy(d_a, h_a, nbytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());

    auto et_local = get_time();
    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    auto et = get_time();

    volatile double sink = 0;
    if (conf.validation) {
      sink = sum_float_gpu(d_a, nparams);
    }

    auto us = get_duration_us(st, et);
    auto us_local = get_duration_us(st, et_local);

    collect_stat(it, us, us_local, s_ans, sink);
  }
  print_stat();

  CHECK_CUDA(cudaHostUnregister(h_a));
  numa_free(h_a, nbytes);
  CHECK_CUDA(cudaFree(d_a));
}

void gpu_write_cpumem_memcpy() {
  size_t nbytes = conf.local_conf["nbytes"].get<size_t>();
  int gpu_idx = conf.local_conf["gpu_idx"].get<int>();
  int mem_numa = conf.local_conf["cpumem_numa_idx"].get<int>();

  size_t nparams = nbytes / sizeof(float);
  CHECK_CUDA(cudaSetDevice(gpu_idx));

  float *h_a = (float*)numa_alloc_onnode(nbytes, mem_numa);
  float *d_a;
  CHECK_CUDA(cudaHostRegister(h_a, nbytes, cudaHostRegisterMapped));
  CHECK_CUDA(cudaMalloc(&d_a, nbytes));
  assert(get_node_allocated(h_a) == mem_numa);

  for (int it = 0; it < conf.niters; ++it) {
    // Fill buf with random numbers and save sum in s_ans
    double s_ans = 0;
    if (conf.validation) {
      fill_random_float_gpu(d_a, nparams);
      s_ans = sum_float_gpu(d_a, nparams);
    }

    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    auto st = get_time();
    
    CHECK_CUDA(cudaMemcpy(h_a, d_a, nbytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());

    auto et_local = get_time();
    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    auto et = get_time();

    volatile double sink = 0;
    if (conf.validation) {
      sink = sum_float(h_a, nparams);
    }

    auto us = get_duration_us(st, et);
    auto us_local = get_duration_us(st, et_local);

    collect_stat(it, us, us_local, s_ans, sink);
  }
  print_stat();

  assert(get_node_allocated(h_a) == mem_numa);
  CHECK_CUDA(cudaHostUnregister(h_a));
  numa_free(h_a, nbytes);
  CHECK_CUDA(cudaFree(d_a));
}

void gpu_read_gpumem_memcpy() {
  size_t nbytes = conf.local_conf["nbytes"].get<size_t>();
  int gpu_master_idx = conf.local_conf["gpu_idx"].get<int>();
  int gpu_slave_idx = conf.local_conf["gpumem_idx"].get<int>();

  size_t nparams = nbytes / sizeof(float);
  float *d_slave, *d_master;
  CHECK_CUDA(cudaSetDevice(gpu_slave_idx));
  CHECK_CUDA(cudaMalloc(&d_slave, nbytes));
  if (gpu_master_idx != gpu_slave_idx) {
    CHECK_CUDA(cudaDeviceEnablePeerAccess(gpu_master_idx, 0));
  }

  CHECK_CUDA(cudaSetDevice(gpu_master_idx));
  CHECK_CUDA(cudaMalloc(&d_master, nbytes));
  if (gpu_master_idx != gpu_slave_idx) {
    CHECK_CUDA(cudaDeviceEnablePeerAccess(gpu_slave_idx, 0));
  }

  for (int it = 0; it < conf.niters; ++it) {
    // Fill buf with random numbers and save sum in s_ans
    CHECK_CUDA(cudaSetDevice(gpu_slave_idx));
    double s_ans = 0;
    if (conf.validation) {
      fill_random_float_gpu(d_slave, nparams);
      s_ans = sum_float_gpu(d_slave, nparams);
    }
    CHECK_CUDA(cudaSetDevice(gpu_master_idx));

    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    auto st = get_time();
    
    CHECK_CUDA(cudaMemcpy(d_master, d_slave, nbytes, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());

    auto et_local = get_time();
    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    auto et = get_time();

    volatile double sink = 0;
    if (conf.validation) {
      sink = sum_float_gpu(d_master, nparams);
    }

    auto us = get_duration_us(st, et);
    auto us_local = get_duration_us(st, et_local);

    collect_stat(it, us, us_local, s_ans, sink);
  }
  print_stat();

  CHECK_CUDA(cudaFree(d_master));
  CHECK_CUDA(cudaFree(d_slave));
}

void gpu_write_gpumem_memcpy() {
  size_t nbytes = conf.local_conf["nbytes"].get<size_t>();
  int gpu_master_idx = conf.local_conf["gpu_idx"].get<int>();
  int gpu_slave_idx = conf.local_conf["gpumem_idx"].get<int>();

  size_t nparams = nbytes / sizeof(float);
  float *d_slave, *d_master;
  CHECK_CUDA(cudaSetDevice(gpu_slave_idx));
  CHECK_CUDA(cudaMalloc(&d_slave, nbytes));
  if (gpu_master_idx != gpu_slave_idx) {
    CHECK_CUDA(cudaDeviceEnablePeerAccess(gpu_master_idx, 0));
  }

  CHECK_CUDA(cudaSetDevice(gpu_master_idx));
  CHECK_CUDA(cudaMalloc(&d_master, nbytes));
  if (gpu_master_idx != gpu_slave_idx) {
    CHECK_CUDA(cudaDeviceEnablePeerAccess(gpu_slave_idx, 0));
  }

  for (int it = 0; it < conf.niters; ++it) {
    // Fill buf with random numbers and save sum in s_ans
    double s_ans = 0;
    if (conf.validation) {
      fill_random_float_gpu(d_master, nparams);
      s_ans = sum_float_gpu(d_master, nparams);
    }

    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    auto st = get_time();
    
    CHECK_CUDA(cudaMemcpy(d_slave, d_master, nbytes, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());

    auto et_local = get_time();
    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    auto et = get_time();

    volatile double sink = 0;
    if (conf.validation) {
      sink = sum_float_gpu(d_slave, nparams);
    }

    auto us = get_duration_us(st, et);
    auto us_local = get_duration_us(st, et_local);

    collect_stat(it, us, us_local, s_ans, sink);
  }
  print_stat();

  CHECK_CUDA(cudaFree(d_master));
  CHECK_CUDA(cudaFree(d_slave));
}

void nic_read_cpumem() {
  size_t nbytes = conf.local_conf["nbytes"].get<size_t>();
  int nic_idx = conf.local_conf["nic_idx"].get<int>();
  int mem_numa = conf.local_conf["cpumem_numa_idx"].get<int>();
  int peer_rank = conf.local_conf["peer_rank"].get<int>();

  size_t nparams = nbytes / sizeof(float);

  float *h_a = (float*)numa_alloc_onnode(nbytes, mem_numa);
  assert(get_node_allocated(h_a) == mem_numa);

  ib_connection* conn = create_ib_connection(nic_idx, peer_rank);
  ib_memory* mem = create_ib_memory(conn, h_a, nbytes);

  for (int it = 0; it < conf.niters; ++it) {
    // Fill buf with random numbers and save sum in s_ans
    double s_ans = 0;
    if (conf.validation) {
      fill_random_float(h_a, nparams);
      s_ans = sum_float(h_a, nparams);
      CHECK_MPI(MPI_Send((void*)&s_ans, 1, MPI_DOUBLE, peer_rank, 0, MPI_COMM_WORLD));
    }

    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    auto st = get_time();

    ib_post_send(conn, mem, 0, nbytes);
    ib_poll_cq(conn);

    auto et_local = get_time();
    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    auto et = get_time();

    volatile double sink = 0;
    if (conf.validation) {
      CHECK_MPI(MPI_Recv((void*)&sink, 1, MPI_DOUBLE, peer_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    }

    auto us = get_duration_us(st, et);
    auto us_local = get_duration_us(st, et_local);

    collect_stat(it, us, us_local, s_ans, sink);
  }
  print_stat();

  destroy_ib_memory(mem);
  numa_free(h_a, nbytes);
  destroy_ib_connection(conn);
}

void nic_write_cpumem() {
  size_t nbytes = conf.local_conf["nbytes"].get<size_t>();
  int nic_idx = conf.local_conf["nic_idx"].get<int>();
  int mem_numa = conf.local_conf["cpumem_numa_idx"].get<int>();
  int peer_rank = conf.local_conf["peer_rank"].get<int>();

  size_t nparams = nbytes / sizeof(float);

  float *h_a = (float*)numa_alloc_onnode(nbytes, mem_numa);
  assert(get_node_allocated(h_a) == mem_numa);

  ib_connection* conn = create_ib_connection(nic_idx, peer_rank);
  ib_memory* mem = create_ib_memory(conn, h_a, nbytes);

  for (int it = 0; it < conf.niters; ++it) {
    // Fill buf with random numbers and save sum in s_ans
    double s_ans = 0;
    if (conf.validation) {
      CHECK_MPI(MPI_Recv((void*)&s_ans, 1, MPI_DOUBLE, peer_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    }

    // Recv should be posted before send
    ib_post_recv(conn, mem, 0, nbytes);

    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    auto st = get_time();

    ib_poll_cq(conn);

    auto et_local = get_time();
    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    auto et = get_time();

    volatile double sink = 0;
    if (conf.validation) {
      sink = sum_float(h_a, nparams);
      CHECK_MPI(MPI_Send((void*)&sink, 1, MPI_DOUBLE, peer_rank, 0, MPI_COMM_WORLD));
    }

    auto us = get_duration_us(st, et);
    auto us_local = get_duration_us(st, et_local);

    collect_stat(it, us, us_local, s_ans, sink);
  }
  print_stat();

  destroy_ib_memory(mem);
  numa_free(h_a, nbytes);
  destroy_ib_connection(conn);
}

void nic_read_gpumem() {
  size_t nbytes = conf.local_conf["nbytes"].get<size_t>();
  int nic_idx = conf.local_conf["nic_idx"].get<int>();
  int gpumem_idx = conf.local_conf["gpumem_idx"].get<int>();
  int peer_rank = conf.local_conf["peer_rank"].get<int>();

  size_t nparams = nbytes / sizeof(float);
  CHECK_CUDA(cudaSetDevice(gpumem_idx));

  float* d_a;
  CHECK_CUDA(cudaMalloc(&d_a, nbytes));

  ib_connection* conn = create_ib_connection(nic_idx, peer_rank);
  ib_memory* mem = create_ib_memory(conn, d_a, nbytes);

  for (int it = 0; it < conf.niters; ++it) {
    // Fill buf with random numbers and save sum in s_ans
    double s_ans = 0;
    if (conf.validation) {
      fill_random_float_gpu(d_a, nparams);
      s_ans = sum_float_gpu(d_a, nparams);
      CHECK_MPI(MPI_Send((void*)&s_ans, 1, MPI_DOUBLE, peer_rank, 0, MPI_COMM_WORLD));
    }

    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    auto st = get_time();

    ib_post_send(conn, mem, 0, nbytes);
    ib_poll_cq(conn);

    auto et_local = get_time();
    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    auto et = get_time();

    volatile double sink = 0;
    if (conf.validation) {
      CHECK_MPI(MPI_Recv((void*)&sink, 1, MPI_DOUBLE, peer_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    }

    auto us = get_duration_us(st, et);
    auto us_local = get_duration_us(st, et_local);

    collect_stat(it, us, us_local, s_ans, sink);
  }
  print_stat();

  destroy_ib_memory(mem);
  CHECK_CUDA(cudaFree(d_a));
  destroy_ib_connection(conn);
}

void nic_write_gpumem() {
  size_t nbytes = conf.local_conf["nbytes"].get<size_t>();
  int nic_idx = conf.local_conf["nic_idx"].get<int>();
  int gpumem_idx = conf.local_conf["gpumem_idx"].get<int>();
  int peer_rank = conf.local_conf["peer_rank"].get<int>();

  size_t nparams = nbytes / sizeof(float);
  CHECK_CUDA(cudaSetDevice(gpumem_idx));

  float* d_a;
  CHECK_CUDA(cudaMalloc(&d_a, nbytes));

  ib_connection* conn = create_ib_connection(nic_idx, peer_rank);
  ib_memory* mem = create_ib_memory(conn, d_a, nbytes);

  for (int it = 0; it < conf.niters; ++it) {
    // Fill buf with random numbers and save sum in s_ans
    double s_ans = 0;
    if (conf.validation) {
      CHECK_MPI(MPI_Recv((void*)&s_ans, 1, MPI_DOUBLE, peer_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    }

    // Recv should be posted before send
    ib_post_recv(conn, mem, 0, nbytes);

    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    auto st = get_time();

    ib_poll_cq(conn);

    auto et_local = get_time();
    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    auto et = get_time();

    volatile double sink = 0;
    if (conf.validation) {
      sink = sum_float_gpu(d_a, nparams);
      CHECK_MPI(MPI_Send((void*)&sink, 1, MPI_DOUBLE, peer_rank, 0, MPI_COMM_WORLD));
    }

    auto us = get_duration_us(st, et);
    auto us_local = get_duration_us(st, et_local);

    collect_stat(it, us, us_local, s_ans, sink);
  }
  print_stat();

  destroy_ib_memory(mem);
  CHECK_CUDA(cudaFree(d_a));
  destroy_ib_connection(conn);
}

void nic_sharp_allreduce() {
  size_t nbytes = conf.local_conf["nbytes"].get<size_t>();
  std::vector<int> peer_rank = conf.local_conf["peer_rank"].get<std::vector<int>>();
  bool src_is_gpu = conf.local_conf["src_device"].get<std::string>() == "gpumem";
  bool dst_is_gpu = conf.local_conf["dst_device"].get<std::string>() == "gpumem";
  float *buf_src, *buf_dst;
  if (src_is_gpu) {
    CHECK_CUDA(cudaSetDevice(conf.local_conf["src_gpumem_idx"].get<int>()));
    CHECK_CUDA(cudaMalloc(&buf_src, nbytes));
  } else {
    int mem_numa = conf.local_conf["src_cpumem_numa_idx"].get<int>();
    buf_src = (float*)numa_alloc_onnode(nbytes, mem_numa);
    assert(get_node_allocated(buf_src) == mem_numa);
  }
  if (dst_is_gpu) {
    CHECK_CUDA(cudaSetDevice(conf.local_conf["dst_gpumem_idx"].get<int>()));
    CHECK_CUDA(cudaMalloc(&buf_dst, nbytes));
  } else {
    int mem_numa = conf.local_conf["dst_cpumem_numa_idx"].get<int>();
    buf_dst = (float*)numa_alloc_onnode(nbytes, mem_numa);
    assert(get_node_allocated(buf_dst) == mem_numa);
  }
  //int nic_idx = conf.local_conf["nic_idx"].get<int>();

  size_t nparams = nbytes / sizeof(float);

  MPI_Comm comm = create_subset_comm(MPI_COMM_WORLD, peer_rank);
  sharp_connection* conn = create_sharp_connection(peer_rank);
  sharp_memory* mem_src = create_sharp_memory(conn, buf_src, nbytes, src_is_gpu);
  sharp_memory* mem_dst = create_sharp_memory(conn, buf_dst, nbytes, dst_is_gpu);

  for (int it = 0; it < conf.niters; ++it) {
    // Fill buf with random numbers and save sum in s_ans
    double s_ans = 0;
    if (conf.validation) {
      if (src_is_gpu) {
        fill_random_float_gpu(buf_src, nparams);
        s_ans = sum_float_gpu(buf_src, nparams);
      } else {
        fill_random_float(buf_src, nparams);
        s_ans = sum_float(buf_src, nparams);
      }
      CHECK_MPI(MPI_Allreduce(MPI_IN_PLACE, (void*)&s_ans, 1, MPI_DOUBLE, MPI_SUM, comm));
    }

    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    auto st = get_time();

    sharp_allreduce(conn, mem_src, mem_dst, 0, nbytes);

    auto et_local = get_time();
    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    auto et = get_time();

    volatile double sink = 0;
    if (conf.validation) {
      if (dst_is_gpu) {
        sink = sum_float_gpu(buf_dst, nparams);
      } else {
        sink = sum_float(buf_dst, nparams);
      }
    }

    auto us = get_duration_us(st, et);
    auto us_local = get_duration_us(st, et_local);

    collect_stat(it, us, us_local, s_ans, sink);
  }
  print_stat();

  CHECK_MPI(MPI_Comm_free(&comm));
  destroy_sharp_memory(conn, mem_src);
  destroy_sharp_memory(conn, mem_dst);
  if (src_is_gpu) {
    CHECK_CUDA(cudaFree(buf_src));
  } else {
    numa_free(buf_src, nbytes);
  }
  if (dst_is_gpu) {
    CHECK_CUDA(cudaFree(buf_dst));
  } else {
    numa_free(buf_dst, nbytes);
  }
  destroy_sharp_connection(conn);
}

void dummy() {
  for (int it = 0; it < conf.niters; ++it) {
    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    auto st = get_time();

    // do nothing

    auto et_local = get_time();
    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    auto et = get_time();

    auto us = get_duration_us(st, et);
    auto us_local = get_duration_us(st, et_local);

    collect_stat(it, us, us_local, 0, 0);
  }
  print_stat();
}