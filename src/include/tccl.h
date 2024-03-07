#ifndef TCCL_H_
#define TCCL_H_

#include "nccl.h"
#include <numa.h>
#include <numaif.h>

#define TCCL_MAX_GPU 4
#define TCCL_MAX_GPU_SUBSET (1 << TCCL_MAX_GPU)
#define TCCL_INTER_TRANSFER_ENC_MAX 64 // (type, src_idx, dst_idx)
#define TCCL_INTRA_TRANSFER_LEN_MAX (TCCL_MAX_GPU * 2)

enum tcclTransportType {
  TCCL_TRANSPORT_TYPE_P2P = 0,
  TCCL_TRANSPORT_TYPE_SHM = 1,
  TCCL_TRANSPORT_TYPE_NET = 2
};

struct tcclTransportOpt {
  int type;
  int p2p_use_memcpy;
  int p2p_use_read;
  int shm_send_use_memcpy;
  int shm_recv_use_memcpy;
  int shm_numa_idx;
  int net_send_use_gdr;
  int net_recv_use_gdr;
  int net_use_memcpy;
  int net_numa_idx;
};

struct tcclCommInfo {
  int nChannels;
  bool enabled; // set by ncclTopoTuneModel
  int channelBegin;
  int channelEnd;
  tcclTransportOpt sendOpt;
  tcclTransportOpt recvOpt;
  // dirty hack to tell *CanConnect function that it's send or recv. set by "transport.cc".
  int tmpIsSend;
};

struct tcclTransfer {
  int type;
  int src_idx;
  int dst_idx;
};

struct tcclTransfers {
  double gbps;
  int num_transfers;
  tcclTransfer transfers[TCCL_INTRA_TRANSFER_LEN_MAX];
};

extern const int tcclP2pTransportSendResourceMemcpyOffset;
extern const int tcclShmTransportSendResourceMemcpyOffset;
extern const int tcclShmTransportRecvResourceMemcpyOffset;

enum tcclTransferType {
  TCCL_TRANSFER_TYPE_GPU_READ_CPUMEM_KERNEL = 0,
  TCCL_TRANSFER_TYPE_GPU_WRITE_CPUMEM_KERNEL = 1,
  TCCL_TRANSFER_TYPE_GPU_READ_GPUMEM_KERNEL = 2,
  TCCL_TRANSFER_TYPE_GPU_WRITE_GPUMEM_KERNEL = 3,
  TCCL_TRANSFER_TYPE_GPU_READ_CPUMEM_MEMCPY = 4,
  TCCL_TRANSFER_TYPE_GPU_WRITE_CPUMEM_MEMCPY = 5,
  TCCL_TRANSFER_TYPE_GPU_READ_GPUMEM_MEMCPY = 6,
  TCCL_TRANSFER_TYPE_GPU_WRITE_GPUMEM_MEMCPY = 7,
  TCCL_TRANSFER_TYPE_GPU_GPU_INTER = 8,
  TCCL_TRANSFER_TYPE_GPU_CPU_INTER = 9,
  TCCL_TRANSFER_TYPE_CPU_GPU_INTER = 10,
  TCCL_TRANSFER_TYPE_CPU_CPU_INTER = 11,
  TCCL_TRANSFER_TYPE_INVALID = 999
};

enum tcclInterTransferType {
};

inline tcclTransfer tcclDecodeInterTransfer(int encoded) {
  int type_int;
  switch (encoded & 0x3) {
    case 0:
      type_int = TCCL_TRANSFER_TYPE_GPU_GPU_INTER; break;
    case 1:
      type_int = TCCL_TRANSFER_TYPE_GPU_CPU_INTER; break;
    case 2:
      type_int = TCCL_TRANSFER_TYPE_CPU_GPU_INTER; break;
    case 3:
      type_int = TCCL_TRANSFER_TYPE_CPU_CPU_INTER; break;
  }
  int src_idx = (encoded >> 2) & 0x3;
  int dst_idx = (encoded >> 4) & 0x3;
  return {type_int, src_idx, dst_idx};
}

inline int tcclEncodeInterTransfer(const char* type, int src_idx, int dst_idx) {
  int type_int = -1;
  if (strcmp(type, "GPU_GPU_INTER") == 0) {
    type_int = 0;
  } else if (strcmp(type, "GPU_CPU_INTER") == 0) {
    type_int = 1;
  } else if (strcmp(type, "CPU_GPU_INTER") == 0) {
    type_int = 2;
  } else if (strcmp(type, "CPU_CPU_INTER") == 0) {
    type_int = 3;
  } else {
    return -1;
  }
  return type_int | (src_idx << 2) | (dst_idx << 4);
}

inline int tcclEncodeIntraTransfer(const char* type) {
  int type_int = -1;
  if (strcmp(type, "GPU_READ_CPUMEM_KERNEL") == 0) {
    type_int = TCCL_TRANSFER_TYPE_GPU_READ_CPUMEM_KERNEL;
  } else if (strcmp(type, "GPU_WRITE_CPUMEM_KERNEL") == 0) {
    type_int = TCCL_TRANSFER_TYPE_GPU_WRITE_CPUMEM_KERNEL;
  } else if (strcmp(type, "GPU_READ_GPUMEM_KERNEL") == 0) {
    type_int = TCCL_TRANSFER_TYPE_GPU_READ_GPUMEM_KERNEL;
  } else if (strcmp(type, "GPU_WRITE_GPUMEM_KERNEL") == 0) {
    type_int = TCCL_TRANSFER_TYPE_GPU_WRITE_GPUMEM_KERNEL;
  } else if (strcmp(type, "GPU_READ_CPUMEM_MEMCPY") == 0) {
    type_int = TCCL_TRANSFER_TYPE_GPU_READ_CPUMEM_MEMCPY;
  } else if (strcmp(type, "GPU_WRITE_CPUMEM_MEMCPY") == 0) {
    type_int = TCCL_TRANSFER_TYPE_GPU_WRITE_CPUMEM_MEMCPY;
  } else if (strcmp(type, "GPU_READ_GPUMEM_MEMCPY") == 0) {
    type_int = TCCL_TRANSFER_TYPE_GPU_READ_GPUMEM_MEMCPY;
  } else if (strcmp(type, "GPU_WRITE_GPUMEM_MEMCPY") == 0) {
    type_int = TCCL_TRANSFER_TYPE_GPU_WRITE_GPUMEM_MEMCPY;
  } else {
    return -1;
  }
  return type_int;
}

struct ncclTopoGraph;

ncclResult_t tcclInit(ncclComm *comm, ncclTopoGraph* graph);
ncclResult_t tcclCheckNuma(void* cpumem, int numaIdx);
ncclResult_t tcclSetNuma(void* cpumem, int size, int numaIdx);
ncclResult_t tcclLimitNuma(int numaIdx, struct bitmask** oldmask, struct bitmask** newmask);
ncclResult_t tcclUnlimitNuma(struct bitmask* oldmask, struct bitmask* newmask);

#endif