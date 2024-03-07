#include "ibv_helper.hpp"

#include <mpi.h>
#include "util.hpp"
#include "check.hpp"

#define IB_PORT_NUM 1
#define MAX_SEND_WR 1024

struct qp_attr {
  uint32_t qp_num;
  uint16_t lid;
};

static uint16_t get_lid(ibv_context* ctx) {
  ibv_port_attr attr;
  CHECK_ERRNO(ibv_query_port(ctx, IB_PORT_NUM, &attr));
  return attr.lid;
}

ib_connection* create_ib_connection(int nic_idx, int peer_rank) {
  int num_devices;
  ibv_device** devices = ibv_get_device_list(&num_devices);
  CHECK_ERRNO_PTR(devices);
  
  for (int i = 0; i < num_devices; ++i) {
    const char* dn = ibv_get_device_name(devices[i]);
    CHECK_PTR(dn);
    //LOG_RANK_ANY("IB device {}: {}", i, std::string(dn));
  }

  ibv_context* context = ibv_open_device(devices[nic_idx]);
  CHECK_PTR(context);

  ibv_pd* pd = ibv_alloc_pd(context);
  CHECK_PTR(pd);

  ibv_cq* cq = ibv_create_cq(context, 32, NULL, NULL, 0);
  CHECK_PTR(cq);

  ibv_qp_init_attr qp_init_attr;
  memset(&qp_init_attr, 0, sizeof(qp_init_attr));
  qp_init_attr.qp_type = IBV_QPT_RC;
  qp_init_attr.sq_sig_all = 0;
  qp_init_attr.send_cq = cq;
  qp_init_attr.recv_cq = cq;
  qp_init_attr.cap.max_send_wr = MAX_SEND_WR;
  qp_init_attr.cap.max_recv_wr = 1;
  qp_init_attr.cap.max_send_sge = 1;
  qp_init_attr.cap.max_recv_sge = 1;
  ibv_qp* qp = ibv_create_qp(pd, &qp_init_attr);
  CHECK_PTR(qp);

  struct ibv_qp_attr attr;

  // RESET -> INIT
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_INIT;
  attr.pkey_index = 0;
  attr.port_num = IB_PORT_NUM;
  attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
  CHECK_ERRNO(ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS));

  ibv_free_device_list(devices);

  // Create the connection structure
  struct ib_connection* conn = (struct ib_connection*)malloc(sizeof(struct ib_connection));
  conn->context = context;
  conn->pd = pd;
  conn->cq = cq;
  conn->qp = qp;

  // Exchange queue pair attributes with the remote process
  qp_attr local_attr;
  local_attr.qp_num = conn->qp->qp_num;
  local_attr.lid = get_lid(conn->context);

  qp_attr remote_attr;
  MPI_Sendrecv(&local_attr, sizeof(local_attr), MPI_BYTE, peer_rank, 0,
               &remote_attr, sizeof(remote_attr), MPI_BYTE, peer_rank, 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  // INIT -> RTR
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTR;
  attr.path_mtu = IBV_MTU_4096;
  attr.dest_qp_num = remote_attr.qp_num;
  attr.rq_psn = 0;
  attr.max_dest_rd_atomic = 1;
  attr.min_rnr_timer = 12;
  attr.ah_attr.is_global = 0;
  attr.ah_attr.dlid = remote_attr.lid;
  attr.ah_attr.sl = 0;
  attr.ah_attr.src_path_bits = 0;
  attr.ah_attr.port_num = IB_PORT_NUM;
  CHECK_ERRNO(ibv_modify_qp(conn->qp, &attr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                    IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC |
                    IBV_QP_MIN_RNR_TIMER));

  // RTR -> RTS
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTS;
  attr.timeout = 14;
  attr.retry_cnt = 7;
  attr.rnr_retry = 7;
  attr.sq_psn = 0;
  attr.max_rd_atomic = 1;
  CHECK_ERRNO(ibv_modify_qp(conn->qp, &attr, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                    IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC));

  return conn;
}

void destroy_ib_connection(struct ib_connection* conn) {
  // Destroy the queue pair
  ibv_destroy_qp(conn->qp);

  // Destroy the completion queue
  ibv_destroy_cq(conn->cq);

  // Deallocate the protection domain
  ibv_dealloc_pd(conn->pd);

  // Close the device context
  ibv_close_device(conn->context);

  // Free the connection structure
  free(conn);
}

ib_memory* create_ib_memory(ib_connection* conn, void* buf, size_t size) {
  ibv_mr* mr = ibv_reg_mr(conn->pd, buf, size, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);

  ib_memory* mem = (ib_memory*)malloc(sizeof(ib_memory));
  mem->mr = mr;
  return mem;
}

void exchange_peer_memory_info(ib_connection* conn, ib_memory* mem, int peer_rank) {
  MPI_Sendrecv(&mem->mr->rkey, sizeof(mem->mr->rkey), MPI_BYTE, peer_rank, 0,
               &conn->rkey, sizeof(conn->rkey), MPI_BYTE, peer_rank, 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(&mem->mr->addr, sizeof(mem->mr->addr), MPI_BYTE, peer_rank, 0,
               &conn->raddr, sizeof(conn->raddr), MPI_BYTE, peer_rank, 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void destroy_ib_memory(ib_memory* mem) {
  ibv_dereg_mr(mem->mr);
  free(mem);
}

void ib_post_send(ib_connection* conn, ib_memory* mem, size_t offset, size_t size) {
  struct ibv_sge sge;
  memset(&sge, 0, sizeof(sge));
  sge.addr = (uintptr_t)mem->mr->addr + offset;
  sge.length = size;
  sge.lkey = mem->mr->lkey;

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = 0;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_SEND;
  wr.send_flags = IBV_SEND_SIGNALED;

  struct ibv_send_wr* bad_wr;
  CHECK_ERRNO(ibv_post_send(conn->qp, &wr, &bad_wr));
}

void ib_post_rdma_write(ib_connection* conn, ib_memory* mem, size_t offset, size_t size) {
  // We found out that sending small chunk of data at the same location (e.g., 128KB) repeatedly
  // is faster than sending large chunk of data
  {
    int num_seg = size / 131072;
    //int num_seg = size / (131072 * 16);
    assert(num_seg + 1 <= MAX_SEND_WR);
    for (int seg_id = 0; seg_id < num_seg; ++seg_id) {
      struct ibv_sge sge;
      memset(&sge, 0, sizeof(sge));
      sge.addr = (uintptr_t)mem->mr->addr + offset;
      sge.length = size / num_seg;
      sge.lkey = mem->mr->lkey;

      struct ibv_send_wr wr;
      memset(&wr, 0, sizeof(wr));
      wr.wr_id = 0;
      wr.sg_list = &sge;
      wr.num_sge = 1;
      wr.opcode = IBV_WR_RDMA_WRITE;
      wr.wr.rdma.remote_addr = (uint64_t)conn->raddr + offset;
      wr.wr.rdma.rkey = conn->rkey;
      wr.send_flags = 0;

      struct ibv_send_wr* bad_wr;
      CHECK_ERRNO(ibv_post_send(conn->qp, &wr, &bad_wr));
    }
  }

  {
    struct ibv_send_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = 0;
    wr.sg_list = nullptr;
    wr.num_sge = 0;
    wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
    wr.send_flags = IBV_SEND_SIGNALED;

    struct ibv_send_wr* bad_wr;
    CHECK_ERRNO(ibv_post_send(conn->qp, &wr, &bad_wr));
  }
}

void ib_post_recv(ib_connection* conn, ib_memory* mem, size_t offset, size_t size) {
  struct ibv_sge sge;
  memset(&sge, 0, sizeof(sge));
  sge.addr = (uintptr_t)mem->mr->addr + offset;
  sge.length = size;
  sge.lkey = mem->mr->lkey;

  struct ibv_recv_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = 0;
  wr.sg_list = &sge;
  wr.num_sge = 1;

  struct ibv_recv_wr* bad_wr;
  CHECK_ERRNO(ibv_post_recv(conn->qp, &wr, &bad_wr));
}

void ib_post_recv_to_rdma_write(ib_connection* conn, ib_memory* mem, size_t offset, size_t size) {
  struct ibv_recv_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = 0;
  wr.sg_list = nullptr;
  wr.num_sge = 0;

  struct ibv_recv_wr* bad_wr;
  CHECK_ERRNO(ibv_post_recv(conn->qp, &wr, &bad_wr));
}

void ib_poll_cq(ib_connection* conn) {
  struct ibv_wc wc;
  while (ibv_poll_cq(conn->cq, 1, &wc) != 1);

  if (wc.status != IBV_WC_SUCCESS) {
    fprintf(stderr, "ibv_poll_cq returned failure status: %s\n", ibv_wc_status_str(wc.status));
    exit(1);
  }
}