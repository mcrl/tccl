#pragma once

#include <sharp/api/sharp_coll.h>
#include <vector>
#include <mpi.h>

struct sharp_connection {
  MPI_Comm mpi_comm;
  sharp_coll_context* context;
  sharp_coll_comm* comm;
};

struct sharp_memory {
  void* mr;
  void* buf;
  bool is_gpu;
};

sharp_connection* create_sharp_connection(std::vector<int> peer_rank);
void destroy_sharp_connection(sharp_connection* conn);

sharp_memory* create_sharp_memory(sharp_connection* conn, void* buf, size_t size, bool is_gpu);
void destroy_sharp_memory(sharp_connection* conn, sharp_memory* mem);

void sharp_allreduce(sharp_connection* conn, sharp_memory* src, sharp_memory* dst, size_t offset, size_t size);