#include "sharp_helper.hpp"

#include <cstring>
#include "check.hpp"
#include "util.hpp"


static int oob_bcast(void* context, void* buffer, int len, int root) {
  sharp_connection* conn = (sharp_connection*)context;
  //printf("oob_bcast: %p %p %d %d\n", context, buffer, len, root);
  CHECK_MPI(MPI_Bcast(buffer, len, MPI_BYTE, root, conn->mpi_comm));
  return 0;
}

static int oob_barrier(void* context) {
  sharp_connection* conn = (sharp_connection*)context;
  //printf("oob_barrier: %p\n", context);
  CHECK_MPI(MPI_Barrier(conn->mpi_comm));
  return 0;
}

static int oob_gather(void* context, int root, void* sbuf, void* rbuf, int len) {
  sharp_connection* conn = (sharp_connection*)context;
  //printf("oob_gather: %p %d %p %p %d\n", context, root, sbuf, rbuf, len);
  CHECK_MPI(MPI_Gather(sbuf, len, MPI_BYTE, rbuf, len, MPI_BYTE, root, conn->mpi_comm));
  return 0;
}

sharp_connection* create_sharp_connection(std::vector<int> peer_rank) {
  sharp_connection* conn = new sharp_connection;
  
  conn->mpi_comm = create_subset_comm(MPI_COMM_WORLD, peer_rank);

  int rank, size;
  CHECK_MPI(MPI_Comm_rank(conn->mpi_comm, &rank));
  CHECK_MPI(MPI_Comm_size(conn->mpi_comm, &size));

  /*
   * Initialize SHARP and obtain context
   */
  sharp_coll_init_spec sharp_coll_spec;
  memset(&sharp_coll_spec, 0, sizeof(sharp_coll_spec));
  // Unique job id
  // For example, nccl-rdma-sharp-plugins uses (hostid << 32 | (pid ^ tid ^ rand()))
  sharp_coll_spec.job_id = peer_rank[0];
  // Unique process id
  // For MPI, it is rank
  sharp_coll_spec.world_rank = rank;
  // Number of processes in the job
  // For MPI, it is world size
  sharp_coll_spec.world_size = size;
  // Optional user-defined progress function (nullptr should be fine)
  sharp_coll_spec.progress_func = nullptr;
  // Optional channel index (0 should be fine)
  sharp_coll_spec.group_channel_idx = 0;
  // IDK; 0 should be fine
  sharp_coll_spec.world_local_rank = 0;
  sharp_coll_spec.enable_thread_support = 1;

  // Assign default config before proceed
  sharp_coll_spec.config = sharp_coll_default_config;
  // Comma-separated name:port list
  sharp_coll_spec.config.ib_dev_list = "mlx5_0:1";
  // Assign a big number as we don't use progress function
  sharp_coll_spec.config.user_progress_num_polls = 10000000;
  // Follow default
  //sharp_coll_spec.config.coll_timeout = 0;

  sharp_coll_spec.oob_colls.bcast = oob_bcast;
  sharp_coll_spec.oob_colls.barrier = oob_barrier;
  sharp_coll_spec.oob_colls.gather = oob_gather;
  sharp_coll_spec.oob_ctx = conn;

  CHECK_SHARP(sharp_coll_init(&sharp_coll_spec, &conn->context));

  /*
   * Create communicator (similar to MPI_COMM_WORLD)
   */
  sharp_coll_comm_init_spec spec;
  memset(&spec, 0, sizeof(spec));
  spec.rank = rank;
  spec.size = size;
  spec.oob_ctx = conn;
  // null should be ok
  spec.group_world_ranks = nullptr;

  CHECK_SHARP(sharp_coll_comm_init(conn->context, &spec, &conn->comm));

  return conn;
}

void destroy_sharp_connection(sharp_connection* conn) {
  CHECK_MPI(MPI_Comm_free(&conn->mpi_comm));
  CHECK_SHARP(sharp_coll_comm_destroy(conn->comm));
  CHECK_SHARP(sharp_coll_finalize(conn->context));
  delete conn;
}

sharp_memory* create_sharp_memory(sharp_connection* conn, void* buf, size_t size, bool is_gpu) {
  sharp_memory* mem = new sharp_memory;
  CHECK_SHARP(sharp_coll_reg_mr(conn->context, buf, size, &mem->mr));
  mem->buf = buf;
  mem->is_gpu = is_gpu;
  return mem;
}

void destroy_sharp_memory(sharp_connection* conn, sharp_memory* mem) {
  CHECK_SHARP(sharp_coll_dereg_mr(conn->context, mem->mr));
}

void sharp_allreduce(sharp_connection* conn, sharp_memory* src, sharp_memory* dst, size_t offset, size_t size) {
  sharp_coll_reduce_spec reduce_spec;
  memset(&reduce_spec, 0, sizeof(reduce_spec));
  reduce_spec.root = 0;
  reduce_spec.sbuf_desc.type = SHARP_DATA_BUFFER;
  reduce_spec.sbuf_desc.mem_type = src->is_gpu ? SHARP_MEM_TYPE_CUDA : SHARP_MEM_TYPE_HOST;
  reduce_spec.sbuf_desc.buffer.ptr = (void*)((uintptr_t)src->buf + offset);
  reduce_spec.sbuf_desc.buffer.length = size;
  reduce_spec.sbuf_desc.buffer.mem_handle = src->mr;
  reduce_spec.rbuf_desc.type = SHARP_DATA_BUFFER;
  reduce_spec.rbuf_desc.mem_type = dst->is_gpu ? SHARP_MEM_TYPE_CUDA : SHARP_MEM_TYPE_HOST;
  reduce_spec.rbuf_desc.buffer.ptr = (void*)((uintptr_t)dst->buf + offset);
  reduce_spec.rbuf_desc.buffer.length = size;
  reduce_spec.rbuf_desc.buffer.mem_handle = dst->mr;
  reduce_spec.dtype = SHARP_DTYPE_FLOAT;
  reduce_spec.length = size / sizeof(float);
  reduce_spec.op = SHARP_OP_SUM;
  reduce_spec.aggr_mode = SHARP_AGGREGATION_NONE;

  CHECK_SHARP(sharp_coll_do_allreduce(conn->comm, &reduce_spec));
}