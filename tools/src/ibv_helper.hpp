#pragma once

#include <infiniband/verbs.h>

struct ib_connection {
  ibv_context* context;
  ibv_pd* pd;
  ibv_cq* cq;
  ibv_qp* qp;
  uint32_t rkey;
  void* raddr;
};

struct ib_memory {
  ibv_mr* mr;
};

ib_connection* create_ib_connection(int nic_idx, int peer_rank);
void destroy_ib_connection(struct ib_connection* conn);

ib_memory* create_ib_memory(ib_connection* conn, void* buf, size_t size);
void destroy_ib_memory(ib_memory* mem);

void ib_post_send(ib_connection* conn, ib_memory* mem, size_t offset, size_t size);
void ib_post_recv(ib_connection* conn, ib_memory* mem, size_t offset, size_t size);
void ib_poll_cq(ib_connection* conn);
void exchange_peer_memory_info(ib_connection* conn, ib_memory* mem, int peer_rank);
void ib_post_rdma_write(ib_connection* conn, ib_memory* mem, size_t offset, size_t size);
void ib_post_recv_to_rdma_write(ib_connection* conn, ib_memory* mem, size_t offset, size_t size);