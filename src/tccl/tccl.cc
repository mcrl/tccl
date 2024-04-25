/*************************************************************************
 * Copyright (c) 2024 Seoul National University. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "tccl.h"
#include "checks.h"
#include "ezxml.h"
#include "comm.h"
#include "bootstrap.h"
#include "channel.h"

#include <numa.h>
#include <numaif.h>
#include <vector>

#define PARSE_STR(var, xml, tag) \
  const char* var; \
  if ((var = ezxml_attr(xml, tag)) == NULL) { \
    WARN(tag " is not set"); \
    return ncclInternalError; \
  }

#define PARSE_INT(var, xml, tag) \
  int var; \
  { \
    const char* _var; \
    if ((_var = ezxml_attr(xml, tag)) == NULL) { \
      WARN(tag " is not set"); \
      return ncclInternalError; \
    } else { \
      var = atoi(_var); \
    } \
  }

#define PARSE_DOUBLE(var, xml, tag) \
  double var; \
  { \
    const char* _var; \
    if ((_var = ezxml_attr(xml, tag)) == NULL) { \
      WARN(tag " is not set"); \
      return ncclInternalError; \
    } else { \
      var = atof(_var); \
    } \
  }

tcclTransfer tcclDecodeInterTransfer(ncclComm *comm, int encoded) {
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
  encoded >>= 2;
  int src_idx = encoded & ((1 << comm->tcclComm.numBitsIdx) - 1);
  encoded >>= comm->tcclComm.numBitsIdx;
  int dst_idx = encoded & ((1 << comm->tcclComm.numBitsIdx) - 1);
  encoded >>= comm->tcclComm.numBitsIdx;
  return {type_int, src_idx, dst_idx};
}

int tcclEncodeInterTransfer(ncclComm* comm, const char* type, int src_idx, int dst_idx) {
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
  return type_int | (src_idx << 2) | (dst_idx << (2 + comm->tcclComm.numBitsIdx));
}

static ncclResult_t setupTCCLChannel(struct ncclComm* comm, int channelId, int rank, int nranks, int* prevRanks) {
  // prevRanks[r] is the previous rank of rank r in the ring
  TRACE(NCCL_TCCL, "channdlId=%d rank=%d nranks=%d", channelId, rank, nranks);
  NCCLCHECK(initChannel(comm, channelId));
  struct ncclRing* ring = &comm->channels[channelId].ring;
  int ringRanks[nranks];
  int cur_rank = 0;
  for (int r = 0; r < nranks; ++r) {
    ringRanks[(nranks - r) % nranks] = cur_rank;
    cur_rank = prevRanks[cur_rank];
  }
  int ixZero = 0, ixRank = 0;
  for (int i = 0; i < nranks; i++) {
    if (ringRanks[i] == 0) ixZero = i;
    if (ringRanks[i] == rank) ixRank = i;
  }
  ring->index = (ixRank - ixZero + nranks) % nranks;
  for (int i = 0; i < nranks; i++) {
    ring->userRanks[i] = ringRanks[(i + ixRank) % nranks];
  }
  ring->prev = ring->userRanks[nranks - 1];
  ring->next = ring->userRanks[1];
  return ncclSuccess;
}


tcclTransfers* tcclGetTransfersFromInterDb(ncclComm* comm, tcclTransfers* db, int subset, int head, int tail) {
  return &db[subset * comm->tcclComm.interTransferEncMax * comm->tcclComm.interTransferEncMax + head * comm->tcclComm.interTransferEncMax + tail];
}

tcclTransfers* tcclGetTransfersFromIntraDb(tcclTransfers* db, int subset) {
  return &db[subset];
}

static ncclResult_t parseTransfer(ezxml_t transfer, tcclTransfer* out) {
  PARSE_STR(type, transfer, "type");
  int type_int = tcclEncodeIntraTransfer(type);
  if (type_int < 0 || type_int >= TCCL_TRANSFER_TYPE_INVALID) {
    WARN("type=%s is not supported", type);
    return ncclInternalError;
  }
  PARSE_INT(src_idx, transfer, "src_idx");
  PARSE_INT(dst_idx, transfer, "dst_idx");
  TRACE(NCCL_TCCL, "type=%s src_idx=%d dst_idx=%d", type, src_idx, dst_idx);
  out->type = type_int;
  out->src_idx = src_idx;
  out->dst_idx = dst_idx;
  return ncclSuccess;
}

static ncclResult_t parseTransfers(ncclComm* comm, ezxml_t transfers, tcclTransfers* out, bool inter, int* head, int* tail) {
  if (inter) {
    PARSE_STR(head_type, transfers, "head_type");
    PARSE_INT(head_src_idx, transfers, "head_src_idx");
    PARSE_INT(head_dst_idx, transfers, "head_dst_idx");
    *head = tcclEncodeInterTransfer(comm, head_type, head_src_idx, head_dst_idx);
    if (*head < 0 || *head >= comm->tcclComm.interTransferEncMax) {
      WARN("head=%d is out of range", *head);
      return ncclInternalError;
    }

    PARSE_STR(tail_type, transfers, "tail_type");
    PARSE_INT(tail_src_idx, transfers, "tail_src_idx");
    PARSE_INT(tail_dst_idx, transfers, "tail_dst_idx");
    *tail = tcclEncodeInterTransfer(comm, tail_type, tail_src_idx, tail_dst_idx);
    if (*tail < 0 || *tail >= comm->tcclComm.interTransferEncMax) {
      WARN("tail=%d is out of range", *tail);
      return ncclInternalError;
    }
  }
  PARSE_DOUBLE(gbps, transfers, "gbps");
  if (out) {
    out->gbps = gbps;
    int num_transfers = 0;
    for (ezxml_t transfer = ezxml_child(transfers, "transfer"); transfer; transfer = transfer->next) {
      NCCLCHECK(parseTransfer(transfer, &out->transfers[num_transfers]));
      ++num_transfers;
    }
    out->num_transfers = num_transfers;
  }
  return ncclSuccess;
}

ncclResult_t tcclGetDbFromXml(ncclComm *comm, tcclTransfers** outInterDb, tcclTransfers** outIntraDb) {
  char* fn = getenv("TCCL_XML_FILE");
  if (!fn) {
    WARN("TCCL_XML_FILE is not set");
    return ncclInternalError;
  }
  INFO(NCCL_TCCL, "TCCL_XML_FILE set by environment to %s", fn);

  ezxml_t root = ezxml_parse_file(fn);
  ezxml_t metadata = ezxml_child(root, "metadata");
  ezxml_t intra = ezxml_child(root, "intra");
  ezxml_t inter = ezxml_child(root, "inter");

  // Read metadata
  {
    PARSE_INT(num_gpus, metadata, "num_gpus");
    PARSE_INT(num_bits_idx, metadata, "num_bits_idx");
    comm->tcclComm.maxGpu = num_gpus;
    comm->tcclComm.maxGpuSubset = 1 << num_gpus;
    comm->tcclComm.interTransferEncMax = 1 << (2 + 2 * num_bits_idx);
    comm->tcclComm.intraTransferLenMax = num_gpus * 2;
    comm->tcclComm.numBitsIdx = num_bits_idx;
  }

  // Init inter db
  // db[gpus < 16][head < 4 * 4 * 4][tail < 4 * 4 * 4]
  tcclTransfers* interDb = (tcclTransfers*)malloc(comm->tcclComm.maxGpuSubset * comm->tcclComm.interTransferEncMax * comm->tcclComm.interTransferEncMax * sizeof(tcclTransfers));
  for (int i = 0; i < comm->tcclComm.maxGpuSubset; i++) {
    for (int j = 0; j < comm->tcclComm.interTransferEncMax; j++) {
      for (int k = 0; k < comm->tcclComm.interTransferEncMax; k++) {
        tcclTransfers* transfers = tcclGetTransfersFromInterDb(comm, interDb, i, j, k);
        transfers->gbps = 0.0;
        transfers->num_transfers = 0;
        transfers->transfers = (tcclTransfer*)malloc(comm->tcclComm.intraTransferLenMax * sizeof(tcclTransfer));
      }
    }
  }

  // Init intra db
  tcclTransfers* intraDb = (tcclTransfers*)malloc(comm->tcclComm.maxGpuSubset * sizeof(tcclTransfers));
  for (int i = 0; i < comm->tcclComm.maxGpuSubset; i++) {
    tcclTransfers* tfs = tcclGetTransfersFromIntraDb(intraDb, i);
    tfs->gbps = 0.0;
    tfs->num_transfers = 0;
    tfs->transfers = (tcclTransfer*)malloc(comm->tcclComm.intraTransferLenMax * sizeof(tcclTransfer));
  }

  // parse intra db
  for (ezxml_t subset = ezxml_child(intra, "subset"); subset; subset = subset->next) {
    PARSE_INT(gpus, subset, "gpus");
    if (gpus < 0 || gpus >= comm->tcclComm.maxGpuSubset) {
      WARN("gpus=%d is out of range", gpus);
      return ncclInternalError;
    }
    for (ezxml_t transfers = ezxml_child(subset, "transfers"); transfers; transfers = transfers->next) {
      NCCLCHECK(parseTransfers(comm, transfers, tcclGetTransfersFromIntraDb(intraDb, gpus), false, NULL, NULL));
    }
  }

  // parse inter db
  for (ezxml_t subset = ezxml_child(inter, "subset"); subset; subset = subset->next) {
    PARSE_INT(gpus, subset, "gpus");
    if (gpus < 0 || gpus >= comm->tcclComm.maxGpuSubset) {
      WARN("gpus=%d is out of range", gpus);
      return ncclInternalError;
    }
    for (ezxml_t transfers = ezxml_child(subset, "transfers"); transfers; transfers = transfers->next) {
      int head, tail;
      // Get head and tail first
      parseTransfers(comm, transfers, NULL, true, &head, &tail);
      // Now construct transfers
      parseTransfers(comm, transfers, tcclGetTransfersFromInterDb(comm, interDb, gpus, head, tail), true, &head, &tail);
    }
  }

  ezxml_free(root);

  *outInterDb = interDb;
  *outIntraDb = intraDb;
  return ncclSuccess;
}

NCCL_PARAM(TcclForceP2p, "TCCL_FORCE_P2P", 0);
NCCL_PARAM(TcclForceShm, "TCCL_FORCE_SHM", 0);

static ncclResult_t findMyselfInTransfers(ncclComm* comm, tcclTransfers* transfers, tcclTransfer headTransfer, tcclTransfer tailTransfer, int* outNextNvmlIdx) {
  int nextNvmlIdx;
  int send_init = 0, recv_init = 0;
  for (int i = 0; i < transfers->num_transfers; ++i) {
    int type = transfers->transfers[i].type;
    tcclTransfer next_transfer = i + 1 < transfers->num_transfers ? transfers->transfers[i + 1] : tailTransfer;
    tcclTransfer prev_transfer = i > 0 ? transfers->transfers[i - 1] : headTransfer;
    int next_type = next_transfer.type;
    int prev_type = prev_transfer.type;
    int src_idx = transfers->transfers[i].src_idx, dst_idx = transfers->transfers[i].dst_idx;
    // setup send first
    // Case: write to cpumem
    if ((type == TCCL_TRANSFER_TYPE_GPU_WRITE_CPUMEM_KERNEL || type == TCCL_TRANSFER_TYPE_GPU_WRITE_CPUMEM_MEMCPY)
        && src_idx == comm->nvmlDev) {
      if (next_type == TCCL_TRANSFER_TYPE_GPU_READ_CPUMEM_KERNEL) {
        comm->tcclComm.sendOpt.type = TCCL_TRANSPORT_TYPE_SHM;
        comm->tcclComm.sendOpt.shm_send_use_memcpy = type == TCCL_TRANSFER_TYPE_GPU_WRITE_CPUMEM_MEMCPY;
        comm->tcclComm.sendOpt.shm_recv_use_memcpy = 0;
        comm->tcclComm.sendOpt.shm_numa_idx = transfers->transfers[i].dst_idx;
        nextNvmlIdx = next_transfer.dst_idx;
        ++send_init;
      } else if (next_type == TCCL_TRANSFER_TYPE_GPU_READ_CPUMEM_MEMCPY) {
        comm->tcclComm.sendOpt.type = TCCL_TRANSPORT_TYPE_SHM;
        comm->tcclComm.sendOpt.shm_send_use_memcpy = type == TCCL_TRANSFER_TYPE_GPU_WRITE_CPUMEM_MEMCPY;
        comm->tcclComm.sendOpt.shm_recv_use_memcpy = 1;
        comm->tcclComm.sendOpt.shm_numa_idx = transfers->transfers[i].dst_idx;
        nextNvmlIdx = next_transfer.dst_idx;
        ++send_init;
      } else if (i + 1 == transfers->num_transfers && tailTransfer.type == TCCL_TRANSFER_TYPE_CPU_GPU_INTER) {
        comm->tcclComm.sendOpt.type = TCCL_TRANSPORT_TYPE_NET;
        comm->tcclComm.sendOpt.net_send_use_gdr = 0;
        comm->tcclComm.sendOpt.net_recv_use_gdr = 1;
        comm->tcclComm.sendOpt.net_use_memcpy = type == TCCL_TRANSFER_TYPE_GPU_WRITE_CPUMEM_MEMCPY;
        comm->tcclComm.sendOpt.net_numa_idx = transfers->transfers[i].dst_idx;
        nextNvmlIdx = -1;
        ++send_init;
      } else if (i + 1 == transfers->num_transfers && tailTransfer.type == TCCL_TRANSFER_TYPE_CPU_CPU_INTER) {
        comm->tcclComm.sendOpt.type = TCCL_TRANSPORT_TYPE_NET;
        comm->tcclComm.sendOpt.net_send_use_gdr = 0;
        comm->tcclComm.sendOpt.net_recv_use_gdr = 0;
        comm->tcclComm.sendOpt.net_use_memcpy = type == TCCL_TRANSFER_TYPE_GPU_WRITE_CPUMEM_MEMCPY;
        comm->tcclComm.sendOpt.net_numa_idx = transfers->transfers[i].dst_idx;
        nextNvmlIdx = -1;
        ++send_init;
      } else {
        WARN("TCCL: should not reach here");
        return ncclInternalError;
      }
    }
    // Case: write to gpumem
    if ((type == TCCL_TRANSFER_TYPE_GPU_WRITE_GPUMEM_KERNEL || type == TCCL_TRANSFER_TYPE_GPU_WRITE_GPUMEM_MEMCPY)
        && src_idx == comm->nvmlDev) {
      comm->tcclComm.sendOpt.type = TCCL_TRANSPORT_TYPE_P2P;
      comm->tcclComm.sendOpt.p2p_use_memcpy = type == TCCL_TRANSFER_TYPE_GPU_WRITE_GPUMEM_MEMCPY;
      comm->tcclComm.sendOpt.p2p_use_read = 0;
      nextNvmlIdx = dst_idx;
      ++send_init;
    }
    // Case: read by
    if ((type == TCCL_TRANSFER_TYPE_GPU_READ_GPUMEM_KERNEL || type == TCCL_TRANSFER_TYPE_GPU_READ_GPUMEM_MEMCPY)
        && src_idx == comm->nvmlDev) {
      comm->tcclComm.sendOpt.type = TCCL_TRANSPORT_TYPE_P2P;
      comm->tcclComm.sendOpt.p2p_use_memcpy = type == TCCL_TRANSFER_TYPE_GPU_READ_GPUMEM_MEMCPY;
      comm->tcclComm.sendOpt.p2p_use_read = 1;
      nextNvmlIdx = dst_idx;
      ++send_init;
    }
    // Case: network
    if (i + 1 == transfers->num_transfers && tailTransfer.type == TCCL_TRANSFER_TYPE_GPU_GPU_INTER
        && tailTransfer.src_idx == comm->nvmlDev) {
      comm->tcclComm.sendOpt.type = TCCL_TRANSPORT_TYPE_NET;
      comm->tcclComm.sendOpt.net_send_use_gdr = 1;
      comm->tcclComm.sendOpt.net_recv_use_gdr = 1;
      comm->tcclComm.sendOpt.net_use_memcpy = 0;
      comm->tcclComm.sendOpt.net_numa_idx = -1;
      nextNvmlIdx = -1;
      ++send_init;
    }
    if (i + 1 == transfers->num_transfers && tailTransfer.type == TCCL_TRANSFER_TYPE_GPU_CPU_INTER
        && tailTransfer.src_idx == comm->nvmlDev) {
      comm->tcclComm.sendOpt.type = TCCL_TRANSPORT_TYPE_NET;
      comm->tcclComm.sendOpt.net_send_use_gdr = 1;
      comm->tcclComm.sendOpt.net_recv_use_gdr = 0;
      comm->tcclComm.sendOpt.net_use_memcpy = 0;
      comm->tcclComm.sendOpt.net_numa_idx = -1;
      nextNvmlIdx = -1;
      ++send_init;
    }
    // now setup recv
    // Case: read from cpumem
    if ((type == TCCL_TRANSFER_TYPE_GPU_READ_CPUMEM_KERNEL || type == TCCL_TRANSFER_TYPE_GPU_READ_CPUMEM_MEMCPY)
        && dst_idx == comm->nvmlDev) {
      if (prev_type == TCCL_TRANSFER_TYPE_GPU_WRITE_CPUMEM_KERNEL) {
        comm->tcclComm.recvOpt.type = TCCL_TRANSPORT_TYPE_SHM;
        comm->tcclComm.recvOpt.shm_send_use_memcpy = 0;
        comm->tcclComm.recvOpt.shm_recv_use_memcpy = type == TCCL_TRANSFER_TYPE_GPU_READ_CPUMEM_MEMCPY;
        comm->tcclComm.recvOpt.shm_numa_idx = transfers->transfers[i].src_idx;
        ++recv_init;
      } else if (prev_type == TCCL_TRANSFER_TYPE_GPU_WRITE_CPUMEM_MEMCPY) {
        comm->tcclComm.recvOpt.type = TCCL_TRANSPORT_TYPE_SHM;
        comm->tcclComm.recvOpt.shm_send_use_memcpy = 1;
        comm->tcclComm.recvOpt.shm_recv_use_memcpy = type == TCCL_TRANSFER_TYPE_GPU_READ_CPUMEM_MEMCPY;
        comm->tcclComm.recvOpt.shm_numa_idx = transfers->transfers[i].src_idx;
        ++recv_init;
      } else if (i == 0 && headTransfer.type == TCCL_TRANSFER_TYPE_GPU_CPU_INTER) {
        comm->tcclComm.recvOpt.type = TCCL_TRANSPORT_TYPE_NET;
        comm->tcclComm.recvOpt.net_send_use_gdr = 1;
        comm->tcclComm.recvOpt.net_recv_use_gdr = 0;
        comm->tcclComm.recvOpt.net_use_memcpy = type == TCCL_TRANSFER_TYPE_GPU_READ_CPUMEM_MEMCPY;
        comm->tcclComm.recvOpt.net_numa_idx = transfers->transfers[i].src_idx;
        ++recv_init;
      } else if (i == 0 && headTransfer.type == TCCL_TRANSFER_TYPE_CPU_CPU_INTER) {
        comm->tcclComm.recvOpt.type = TCCL_TRANSPORT_TYPE_NET;
        comm->tcclComm.recvOpt.net_send_use_gdr = 0;
        comm->tcclComm.recvOpt.net_recv_use_gdr = 0;
        comm->tcclComm.recvOpt.net_use_memcpy = type == TCCL_TRANSFER_TYPE_GPU_READ_CPUMEM_MEMCPY;
        comm->tcclComm.recvOpt.net_numa_idx = transfers->transfers[i].src_idx;
        ++recv_init;
      } else {
        WARN("TCCL: should not reach here");
        return ncclInternalError;
      }
    }
    // Case: read from gpumem
    if ((type == TCCL_TRANSFER_TYPE_GPU_READ_GPUMEM_KERNEL || type == TCCL_TRANSFER_TYPE_GPU_READ_GPUMEM_MEMCPY)
        && dst_idx == comm->nvmlDev) {
      comm->tcclComm.recvOpt.type = TCCL_TRANSPORT_TYPE_P2P;
      comm->tcclComm.recvOpt.p2p_use_memcpy = type == TCCL_TRANSFER_TYPE_GPU_READ_GPUMEM_MEMCPY;
      comm->tcclComm.recvOpt.p2p_use_read = 1;
      ++recv_init;
    }
    // Case: written to
    if ((type == TCCL_TRANSFER_TYPE_GPU_WRITE_GPUMEM_KERNEL || type == TCCL_TRANSFER_TYPE_GPU_WRITE_GPUMEM_MEMCPY)
        && dst_idx == comm->nvmlDev) {
      comm->tcclComm.recvOpt.type = TCCL_TRANSPORT_TYPE_P2P;
      comm->tcclComm.recvOpt.p2p_use_memcpy = type == TCCL_TRANSFER_TYPE_GPU_WRITE_GPUMEM_MEMCPY;
      comm->tcclComm.recvOpt.p2p_use_read = 0;
      ++recv_init;
    }
    // Case: network
    if (i + 1 == transfers->num_transfers && headTransfer.type == TCCL_TRANSFER_TYPE_GPU_GPU_INTER
        && headTransfer.dst_idx == comm->nvmlDev) {
      comm->tcclComm.recvOpt.type = TCCL_TRANSPORT_TYPE_NET;
      comm->tcclComm.recvOpt.net_send_use_gdr = 1;
      comm->tcclComm.recvOpt.net_recv_use_gdr = 1;
      comm->tcclComm.recvOpt.net_use_memcpy = 0;
      comm->tcclComm.recvOpt.net_numa_idx = -1;
      ++recv_init;
    }
    if (i + 1 == transfers->num_transfers && headTransfer.type == TCCL_TRANSFER_TYPE_CPU_GPU_INTER
        && headTransfer.dst_idx == comm->nvmlDev) {
      comm->tcclComm.recvOpt.type = TCCL_TRANSPORT_TYPE_NET;
      comm->tcclComm.recvOpt.net_send_use_gdr = 0;
      comm->tcclComm.recvOpt.net_recv_use_gdr = 1;
      comm->tcclComm.recvOpt.net_use_memcpy = 0;
      comm->tcclComm.recvOpt.net_numa_idx = -1;
      ++recv_init;
    }
  }
  
  if (send_init != 1 || recv_init != 1) {
    WARN("Cannot determine unique transports for TCCL (send_init=%d, recv_init=%d)", send_init, recv_init);
    return ncclInternalError;
  }

  if (comm->tcclComm.sendOpt.type == TCCL_TRANSPORT_TYPE_SHM && ncclParamTcclForceP2p()) {
      comm->tcclComm.sendOpt.type = TCCL_TRANSPORT_TYPE_P2P;
      comm->tcclComm.sendOpt.p2p_use_memcpy = comm->tcclComm.sendOpt.shm_send_use_memcpy;
      comm->tcclComm.sendOpt.p2p_use_read = 0;
  }
  if (comm->tcclComm.sendOpt.type == TCCL_TRANSPORT_TYPE_P2P && ncclParamTcclForceShm()) {
      comm->tcclComm.sendOpt.type = TCCL_TRANSPORT_TYPE_SHM;
      comm->tcclComm.sendOpt.shm_send_use_memcpy = comm->tcclComm.sendOpt.p2p_use_memcpy;
      comm->tcclComm.sendOpt.shm_recv_use_memcpy = 0;
      comm->tcclComm.sendOpt.shm_numa_idx = 0;
  }

  if (comm->tcclComm.recvOpt.type == TCCL_TRANSPORT_TYPE_SHM && ncclParamTcclForceP2p()) {
      comm->tcclComm.recvOpt.type = TCCL_TRANSPORT_TYPE_P2P;
      comm->tcclComm.recvOpt.p2p_use_memcpy = comm->tcclComm.recvOpt.shm_recv_use_memcpy;
      comm->tcclComm.recvOpt.p2p_use_read = 0;
  }
  if (comm->tcclComm.recvOpt.type == TCCL_TRANSPORT_TYPE_P2P && ncclParamTcclForceShm()) {
      comm->tcclComm.recvOpt.type = TCCL_TRANSPORT_TYPE_SHM;
      comm->tcclComm.recvOpt.shm_send_use_memcpy = comm->tcclComm.recvOpt.p2p_use_memcpy;
      comm->tcclComm.recvOpt.shm_recv_use_memcpy = 0;
      comm->tcclComm.recvOpt.shm_numa_idx = 0;
  }

  INFO(NCCL_TCCL, "Rank %d send opt: type=%d, p2p_use_memcpy=%d, p2p_use_read=%d, net_send_use_gdr=%d, net_recv_use_gdr=%d, net_use_memcpy=%d, net_numa_idx=%d, shm_send_use_memcpy=%d, shm_recv_use_memcpy=%d, shm_numa_idx=%d",
        comm->rank, comm->tcclComm.sendOpt.type, comm->tcclComm.sendOpt.p2p_use_memcpy, comm->tcclComm.sendOpt.p2p_use_read,
        comm->tcclComm.sendOpt.net_send_use_gdr, comm->tcclComm.sendOpt.net_recv_use_gdr, comm->tcclComm.sendOpt.net_use_memcpy, comm->tcclComm.sendOpt.net_numa_idx,
        comm->tcclComm.sendOpt.shm_send_use_memcpy, comm->tcclComm.sendOpt.shm_recv_use_memcpy, comm->tcclComm.sendOpt.shm_numa_idx);
  INFO(NCCL_TCCL, "Rank %d recv opt: type=%d, p2p_use_memcpy=%d, p2p_use_read=%d, net_send_use_gdr=%d, net_recv_use_gdr=%d, net_use_memcpy=%d, net_numa_idx=%d, shm_send_use_memcpy=%d, shm_recv_use_memcpy=%d, shm_numa_idx=%d",
        comm->rank, comm->tcclComm.recvOpt.type, comm->tcclComm.recvOpt.p2p_use_memcpy, comm->tcclComm.recvOpt.p2p_use_read,
        comm->tcclComm.recvOpt.net_send_use_gdr, comm->tcclComm.recvOpt.net_recv_use_gdr, comm->tcclComm.recvOpt.net_use_memcpy, comm->tcclComm.recvOpt.net_numa_idx,
        comm->tcclComm.recvOpt.shm_send_use_memcpy, comm->tcclComm.recvOpt.shm_recv_use_memcpy, comm->tcclComm.recvOpt.shm_numa_idx);

  *outNextNvmlIdx = nextNvmlIdx;
  return ncclSuccess;
}

#define TCCL_DP_IDX(n, h, t) ((n) * comm->tcclComm.interTransferEncMax * comm->tcclComm.interTransferEncMax + (h) * comm->tcclComm.interTransferEncMax + (t))
ncclResult_t tcclInit(ncclComm *comm, ncclTopoGraph* graph) {
  ncclResult_t ret = ncclSuccess;

  if (ncclParamTcclForceP2p() && ncclParamTcclForceShm()) {
    WARN("Cannot force both P2P and SHM transports (TCCL_FORCE_P2P=%ld, TCCL_FORCE_SHM=%ld)", ncclParamTcclForceP2p(), ncclParamTcclForceShm());
    return ncclInternalError;
  }

  // NEED-TO-FREE list
  double* bestBw = NULL;
  int* bestBwIdx = NULL;
  int *allNextNvmlIdx = NULL;
  int *prevRanks = NULL;
  int* gpu_bitmasks = NULL;
  tcclTransfers *interDb = NULL;
  tcclTransfers *intraDb = NULL;
  NCCLCHECKGOTO(tcclGetDbFromXml(comm, &interDb, &intraDb), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&bestBw, comm->nNodes * comm->tcclComm.interTransferEncMax * comm->tcclComm.interTransferEncMax), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&bestBwIdx, comm->nNodes * comm->tcclComm.interTransferEncMax * comm->tcclComm.interTransferEncMax), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&allNextNvmlIdx, comm->nRanks), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&prevRanks, comm->nRanks), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&gpu_bitmasks, comm->nNodes), ret, fail);

  if (comm->nRanks <= 1) {
    WARN("TCCL does not handle single GPU communicator.");
    ret = ncclInternalError;
    goto fail;
  }

  // Find gpu index in each node
  for (int r = 0; r < comm->nRanks; ++r) {
    // rely on nvmlDev instead of cudaDev because cudaDev depends on CUDA_VISIBLE_DEVICES
    gpu_bitmasks[comm->rankToNode[r]] |= (1 << comm->peerInfo[r].nvmlDev);
  }

  int nextNvmlIdx;
  if (comm->nNodes > 1) {
    // Inter DP
    for (int n = 0; n < comm->nNodes; ++n) {
      INFO(NCCL_TCCL, "Node %d: gpu_bitmask=%d", n, gpu_bitmasks[n]);
      if (gpu_bitmasks[n] >= comm->tcclComm.maxGpuSubset) {
        WARN("Too many GPUs in the same node: node=%d gpu_bitmask=%d", n, gpu_bitmasks[n]);
        ret = ncclInternalError;
        goto fail;
      }
    }
    // DP init
    for (int h = 0; h < comm->tcclComm.interTransferEncMax; ++h) {
      for (int t = 0; t < comm->tcclComm.interTransferEncMax; ++t) {
        bestBw[TCCL_DP_IDX(0, h, t)] = tcclGetTransfersFromInterDb(comm, interDb, gpu_bitmasks[0], h, t)->gbps;
        if (bestBw[h * comm->tcclComm.interTransferEncMax + t] > 0) {
          TRACE(NCCL_TCCL, "TCCL DP: h=%d, t=%d, bw=%.2lf", h, t, bestBw[TCCL_DP_IDX(0, h, t)]);
        }
      }
    }
    // DP solve
    for (int n = 1; n < comm->nNodes; ++n) {
      for (int h = 0; h < comm->tcclComm.interTransferEncMax; ++h) {
        for (int t = 0; t < comm->tcclComm.interTransferEncMax; ++t) {
          for (int x = 0; x < comm->tcclComm.interTransferEncMax; ++x) {
            double* curBestBw = &bestBw[TCCL_DP_IDX(n, h, t)];
            double newBestBw = std::min(bestBw[TCCL_DP_IDX(n - 1, h, x)], tcclGetTransfersFromInterDb(comm, interDb, gpu_bitmasks[n], x, t)->gbps);
            if (*curBestBw < newBestBw) {
              TRACE(NCCL_TCCL, "TCCL DP: node=%d, h=%d, t=%d, x=%d, bw=%.2lf -> %.2lf", n, h, t, x, *curBestBw, newBestBw);
              *curBestBw = newBestBw;
              bestBwIdx[TCCL_DP_IDX(n, h, t)] = x;
            }
          }
        }
      }
    }
    double ringBestBw;
    int ringBestBwIdx;
    ringBestBw = 0;
    for (int x = 0; x < comm->tcclComm.interTransferEncMax; ++x) {
      if (ringBestBw < bestBw[TCCL_DP_IDX(comm->nNodes - 1, x, x)]) {
        ringBestBw = bestBw[TCCL_DP_IDX(comm->nNodes - 1, x, x)];
        ringBestBwIdx = x;
      }
    }

    if (ringBestBw == 0) {
      WARN("No ring found for current communicator");
      ret = ncclInternalError;
      goto fail;
    }

    INFO(NCCL_TCCL, "TCCL ring best bandwidth: %lf", ringBestBw);

    int curIdx;
    curIdx = ringBestBwIdx;
    for (int n = comm->nNodes - 1; n >= 0; --n) {
      int head = n == 0 ? ringBestBwIdx : bestBwIdx[TCCL_DP_IDX(n, ringBestBwIdx, curIdx)];
      int tail = curIdx;
      tcclTransfer headTransfer = tcclDecodeInterTransfer(comm, head);
      tcclTransfer tailTransfer = tcclDecodeInterTransfer(comm, tail);
      if (n == comm->rankToNode[comm->rank]) {
        tcclTransfers* transfers = tcclGetTransfersFromInterDb(comm, interDb, gpu_bitmasks[n], head, tail);
        NCCLCHECKGOTO(findMyselfInTransfers(comm, transfers, headTransfer, tailTransfer, &nextNvmlIdx), ret, fail);
        break;
      }
      curIdx = head;
    }
  } else { /* comm->nNodes == 1 */
    // Intra transfers are cycle
    tcclTransfers* transfers = tcclGetTransfersFromIntraDb(intraDb, gpu_bitmasks[0]);
    tcclTransfer headTransfer = transfers->transfers[transfers->num_transfers - 1];
    tcclTransfer tailTransfer = transfers->transfers[0];
    NCCLCHECKGOTO(findMyselfInTransfers(comm, transfers, headTransfer, tailTransfer, &nextNvmlIdx), ret, fail);
  }

  allNextNvmlIdx[comm->rank] = nextNvmlIdx;
  NCCLCHECK(bootstrapAllGather(comm->bootstrap, allNextNvmlIdx, sizeof(int)));
  for (int r = 0; r < comm->nRanks; ++r) prevRanks[r] = -1;
  for (int r = 0; r < comm->nRanks; ++r) {
    // skip last GPU in node for now
    if (allNextNvmlIdx[r] == -1) continue;
    int node_idx = comm->rankToNode[r];
    // Find rank inside the node
    int rank_found = -1;
    for (int lr = 0; lr < comm->nodeRanks[node_idx].localRanks; ++lr) {
      int rank_cand = comm->nodeRanks[node_idx].localRankToRank[lr];
      if (comm->peerInfo[rank_cand].nvmlDev == allNextNvmlIdx[r]) {
        rank_found = rank_cand;
        break;
      }
    }
    if (rank_found == -1) {
      WARN("Cannot find next rank for rank %d", r);
      ret = ncclInternalError;
      goto fail;
    }
    prevRanks[rank_found] = r;
  }
  for (int r = 0; r < comm->nRanks; ++r) {
    // Fill in last GPUs
    if (allNextNvmlIdx[r] != -1) continue;
    // Find rank that dont have prevRanks in the next node
    int node_idx = (comm->rankToNode[r] + 1) % comm->nNodes;
    int rank_found = -1;
    for (int lr = 0; lr < comm->nodeRanks[node_idx].localRanks; ++lr) {
      int rank_cand = comm->nodeRanks[node_idx].localRankToRank[lr];
      if (prevRanks[rank_cand] == -1) {
        rank_found = rank_cand;
        break;
      }
    }
    if (rank_found == -1) {
      WARN("Node %d do not have the first GPU; something very wrong", node_idx);
      ret = ncclInternalError;
      goto fail;
    }
    prevRanks[rank_found] = r;
  }

  // Use separate channels because TCCL may choose transports different from NCCL
  comm->tcclComm.nChannels = comm->nChannels;
  comm->tcclComm.channelBegin = std::max(comm->nChannels, comm->nvlsChannels);
  comm->tcclComm.channelEnd = comm->tcclComm.channelBegin + comm->tcclComm.nChannels;
  if (comm->tcclComm.channelEnd >= MAXCHANNELS) {
    WARN("No channels available for TCCL");
    ret = ncclInternalError;
    goto fail;
  }
  // TCCL will use channel [nChannels, nChannels + tcclComm.nChannels)
  for (int c = comm->tcclComm.channelBegin; c < comm->tcclComm.channelEnd; ++c) {
    struct ncclChannel* channel = comm->channels + c;
    NCCLCHECK(setupTCCLChannel(comm, c, comm->rank, comm->nRanks, prevRanks));
    INFO(NCCL_TCCL, "TCCL channel %d: %d -> %d -> %d", c, channel->ring.prev, comm->rank, channel->ring.next);
    if (comm->nRanks == 1) continue;
    NCCLCHECK(ncclTransportP2pConnect(comm, c, 1, &channel->ring.prev, 1, &channel->ring.next, 0));
  }
  // I'm not sure ringGraph hint is necessary here. Transport implementation may or may not use it.
  NCCLCHECK(ncclTransportP2pSetup(comm, graph, 0));
  INFO(NCCL_TCCL, "TCCL channel setup done: channelBegin=%d, channelEnd=%d", comm->tcclComm.channelBegin, comm->tcclComm.channelEnd);

  return ncclSuccess;
fail:
  free(bestBw);
  free(bestBwIdx);
  free(allNextNvmlIdx);
  free(prevRanks);
  for (int i = 0; i < comm->tcclComm.maxGpuSubset; i++) {
    for (int j = 0; j < comm->tcclComm.interTransferEncMax; j++) {
      for (int k = 0; k < comm->tcclComm.interTransferEncMax; k++) {
        tcclTransfers* transfers = tcclGetTransfersFromInterDb(comm, interDb, i, j, k);
        free(transfers->transfers);
      }
    }
  }
  for (int i = 0; i < comm->tcclComm.maxGpuSubset; i++) {
    tcclTransfers* tfs = tcclGetTransfersFromIntraDb(intraDb, i);
    free(tfs->transfers);
  }
  free(interDb);
  free(intraDb);
  return ret;
}
#undef TCCL_DP_IDX

ncclResult_t tcclCheckNuma(void* cpumem, int numaIdx) {
  // Check numa applied correctly
  int mode;
  NEQCHECK(get_mempolicy(&mode, NULL, 0, cpumem, MPOL_F_NODE | MPOL_F_ADDR), 0);
  if (numaIdx != mode) {
    WARN("NUMA policy not applied correctly. Expected %d, got %d", numaIdx, mode);
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t tcclSetNuma(void* cpumem, int size, int numaIdx) {
  numa_tonode_memory(cpumem, size, numaIdx);
  NCCLCHECK(tcclCheckNuma(cpumem, numaIdx));
  return ncclSuccess;
}

ncclResult_t tcclLimitNuma(int numaIdx, struct bitmask** oldmask, struct bitmask** newmask) {
  *oldmask = numa_get_interleave_mask();
  *newmask = numa_allocate_nodemask();
  numa_bitmask_setbit(*newmask, numaIdx);
  numa_set_interleave_mask(*newmask);
  return ncclSuccess;
}

ncclResult_t tcclUnlimitNuma(struct bitmask* oldmask, struct bitmask* newmask) {
  numa_set_interleave_mask(oldmask);
  numa_free_nodemask(newmask);
  return ncclSuccess;
}