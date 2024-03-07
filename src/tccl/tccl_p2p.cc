/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "graph.h"
#include "utils.h"
#include "shm.h"
#include "p2p.h"

enum p2pType { P2P_DIRECT, P2P_INTERMEDIATE, P2P_IPC, P2P_CUMEM };

struct ncclP2pBuff {
  void* directPtr;
  size_t size;
  ncclIpcDesc ipcDesc;
};

struct p2pConnectInfo {
  int rank;
  int read;
  struct ncclP2pBuff p2pBuff;
  // Used by CE memcpy
  char shmName[7];
  int shmSize;
};
static_assert(sizeof(struct p2pConnectInfo) <= CONNECT_SIZE, "p2pConnectInfo is too large");

struct p2pShm {
  struct ncclSendMem sendMem;
  struct ncclRecvMem recvMem;
};
struct p2pShmProxyInfo {
  // Shared memory between proxy and receiving GPU
  struct p2pShm* shm;
  struct p2pShm* devShm;
  char shmName[7];
  int shmSize;
  ncclShmHandle_t handle;

  // Intermediate step for sender
  struct ncclRecvMem* ceRecvMem;
  char* ceDevBuff;

  // Receiver buffer
  char* recvFifo;

  // Used by CE memcpy progress only
  uint64_t step;
  cudaStream_t stream;
  cudaEvent_t events[NCCL_STEPS];
};
static_assert(sizeof(p2pConnectInfo) <= CONNECT_SIZE, "P2P Connect info is too large");

struct p2pResources {
  enum p2pType type;
  union {
    struct ncclSendMem* sendDevMem;
    struct ncclRecvMem* recvDevMem;
  };
  void* sendMemIpc;
  void* recvMemIpc;
  // CE memcpy support
  int useMemcpy;
  struct p2pShmProxyInfo proxyInfo;
  struct p2pShm* shm;
  struct p2pShm* devShm;
  int shmSize;
  ncclShmHandle_t handle;
};

// cuMem API support
struct p2pCuMemProxyInfo {
  struct ncclP2pBuff p2pBuff;
};

struct p2pProxySetupInfo {
  int useMemcpy;
  int size;
};

struct p2pProxyInfo {
  int useMemcpy;
  union {
    struct p2pShmProxyInfo* shmProxyInfo;
    struct p2pCuMemProxyInfo* cuMemProxyInfo;
    void* directPtr;
  };
};

const int tcclP2pTransportSendResourceMemcpyOffset = offsetof(p2pResources, useMemcpy);

#include <sys/types.h>

/* Convert a PCI busId string into a local cudaDev device index (cf. CUDA_VISIBLE_DEVICES) */
static int busIdToCudaDev(int64_t busId) {
  int ndev;
  if (cudaGetDeviceCount(&ndev) != cudaSuccess)
    return -1;
  for (int i = 0; i < ndev; i++) {
    char devBusIdStr[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
    if (cudaDeviceGetPCIBusId(devBusIdStr, NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE, i) != cudaSuccess)
      return -1;
    int64_t devBusId;
    NCCLCHECK(busIdToInt64(devBusIdStr, &devBusId));
    if (busId == devBusId) return i;
  }
  // BusId was not found in our locally visible CUDA devices
  return -1;
}

/* Determine if two peers can communicate through p2p */
static ncclResult_t tcclP2pCanConnect(int* ret, struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2) {
  // CanConnect interface does not provide any interface to pass useMemcpy
  // so we use comm->tcclComm.tmpIsSend to pass the information.
  // Other functions will utilize p2pResources(for main connection) and p2pProxySetupInfo(for proxy connection) structures.
  struct ncclComm* comm = info1->comm;
  int useMemcpy = 0;
  if (comm->tcclComm.tmpIsSend == 1) {
    useMemcpy = comm->tcclComm.sendOpt.p2p_use_memcpy;
  } else {
    useMemcpy = comm->tcclComm.recvOpt.p2p_use_memcpy;
  }

  *ret = 1;
  return ncclSuccess;
}

#define TRACE_DUMP_IPC(DEVIPC)                                                             \
  do {                                                                                     \
    unsigned long *devIpc = (unsigned long *) (DEVIPC);                                    \
    TRACE(P2P,"IPC: %016lx %016lx %016lx %016lx", devIpc[0], devIpc[1], devIpc[2], devIpc[3]); \
    TRACE(P2P,"IPC: %016lx %016lx %016lx %016lx", devIpc[4], devIpc[5], devIpc[6], devIpc[7]); \
  } while (0)

static ncclResult_t p2pGetInfo(struct ncclTopoSystem* topo, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2, int* intermediateRank) {
  int p2p;
  // Queries the topology to see if the GPUs are Ampere and
  // connected via NVLink, if so we enable P2P Read by default
  NCCLCHECK(ncclTopoCheckP2p(topo, info1->busId, info2->busId, &p2p, NULL, intermediateRank));

  return ncclSuccess;
}

static ncclResult_t p2pMap(struct ncclComm *comm, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclP2pBuff* p2pBuff, void** devMem, void** ipcPtr) {
  if (!ncclCuMemEnable() && myInfo->pidHash == peerInfo->pidHash) {
    if (peerInfo->cudaDev != myInfo->cudaDev) {
      // Same PID different GPUs, enable P2P access
      // Legacy CUDA IPC
      cudaError_t err = cudaDeviceEnablePeerAccess(peerInfo->cudaDev, 0);
      if (err == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
      } else if (err != cudaSuccess) {
        WARN("failed to peer with device %d(=%lx): %d %s",
            peerInfo->cudaDev, peerInfo->busId, err, cudaGetErrorString(err));
        return ncclInternalError;
      }
    }
    *devMem = p2pBuff->directPtr;
    *ipcPtr = NULL;
  } else {
    if ((myInfo->pidHash == peerInfo->pidHash) && (peerInfo->cudaDev == myInfo->cudaDev)) {
      // Same PID and GPU
      *devMem = p2pBuff->directPtr;
      *ipcPtr = NULL;
    } else {
      // Different PID or different GPU
      NCCLCHECK(ncclP2pImportShareableBuffer(comm, comm->topParentRanks[peerInfo->rank], p2pBuff->size, &p2pBuff->ipcDesc, devMem));
      *ipcPtr = *devMem;
    }
  }
  return ncclSuccess;
}

/* Send: Create and return connect structures for this peer to connect to me */
static ncclResult_t tcclP2pSendSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo,
    struct ncclConnect* connectInfo, struct ncclConnector* send, int channelId, int connIndex) {
  int useMemcpy = comm->tcclComm.sendOpt.p2p_use_memcpy;
  int useRead = comm->tcclComm.sendOpt.p2p_use_read;
  struct p2pResources* resources;
  int tpProxyRank;
  NCCLCHECK(ncclCalloc(&resources, 1));
  resources->useMemcpy = useMemcpy;
  send->transportResources = resources;
  int intermediateRank;
  NCCLCHECK(p2pGetInfo(comm->topo, myInfo, peerInfo, &intermediateRank));
  if (useMemcpy) useRead = 0;

  static_assert(sizeof(struct p2pConnectInfo) <= sizeof(struct ncclConnect), "p2p Connect Info is too big");
  struct p2pConnectInfo* info = (struct p2pConnectInfo*)connectInfo;
  info->read = useRead;
  // For CollNet, use write for scatter-reduce (conn 1), read for broadcast-gather (conn 0)
  if (graph && connIndex == 1) info->read = 0;
  const char* useReadStr = info->read ? "/read" : "";

  int sendSize = sizeof(struct ncclSendMem);
  // For P2P Read the SIMPLE buffer is tagged on the end of the ncclSendMem structure
  if (info->read) sendSize += comm->buffSizes[NCCL_PROTO_SIMPLE];
  ALIGN_SIZE(sendSize, CUDA_IPC_MIN);

  if (intermediateRank == -1) {
    info->rank = myInfo->rank;
    if (myInfo->pidHash == peerInfo->pidHash && useMemcpy == 0 && !ncclCuMemEnable()) {
      resources->type = P2P_DIRECT;
      send->conn.flags |= info->read ? NCCL_DIRECT_READ : NCCL_DIRECT_WRITE;
      INFO(NCCL_INIT|NCCL_P2P, "Channel %02d/%01d : %d[%d] -> %d[%d] via P2P/direct pointer%s",
          channelId, connIndex, myInfo->rank, myInfo->nvmlDev, peerInfo->rank, peerInfo->nvmlDev, useReadStr);
    } else {
      // cuMem API support
      if (ncclCuMemEnable()) {
        resources->type = P2P_CUMEM;
        INFO(NCCL_INIT|NCCL_P2P,"Channel %02d/%01d : %d[%d] -> %d[%d] via P2P/CUMEM%s%s",
             channelId, connIndex, myInfo->rank, myInfo->nvmlDev, peerInfo->rank, peerInfo->nvmlDev, useReadStr, useMemcpy ? "/CE" : "");;
      } else {
        // Legacy CUDA IPC
        resources->type = P2P_IPC;
        INFO(NCCL_INIT|NCCL_P2P,"Channel %02d/%01d : %d[%d] -> %d[%d] via P2P/IPC%s%s",
             channelId, connIndex, myInfo->rank, myInfo->nvmlDev, peerInfo->rank, peerInfo->nvmlDev, useReadStr, useMemcpy ? "/CE" : "");
      }
      send->conn.flags |= info->read ? NCCL_IPC_READ : NCCL_IPC_WRITE;
    }
  } else {
    resources->type = P2P_INTERMEDIATE;
    info->rank = intermediateRank;
    INFO(NCCL_INIT|NCCL_P2P, "Channel %02d/%01d : %d[%d] -> %d[%d] via P2P/indirect/%d[%d]%s",
        channelId, connIndex, myInfo->rank, myInfo->nvmlDev, peerInfo->rank, peerInfo->nvmlDev, intermediateRank,
	  comm->peerInfo[intermediateRank].nvmlDev, useReadStr);
  }

  tpProxyRank = comm->topParentRanks[info->rank];
  NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_TCCL_P2P, 1, tpProxyRank, &send->proxyConn));
  if (useMemcpy) {
    struct p2pProxySetupInfo setupInfo;
    setupInfo.useMemcpy = 1;
    NCCLCHECK(ncclProxyCallBlocking(comm, &send->proxyConn, ncclProxyMsgSetup, &setupInfo, sizeof(struct p2pProxySetupInfo), &resources->proxyInfo, sizeof(struct p2pShmProxyInfo)));
    info->shmSize = resources->proxyInfo.shmSize;
    memcpy(info->shmName, resources->proxyInfo.shmName, sizeof(info->shmName));
  } else {
    struct p2pProxySetupInfo setupInfo;
    setupInfo.useMemcpy = 0;
    setupInfo.size = sendSize;
    NCCLCHECK(ncclProxyCallBlocking(comm, &send->proxyConn, ncclProxyMsgSetup, &setupInfo, sizeof(struct p2pProxySetupInfo), &info->p2pBuff, sizeof(struct ncclP2pBuff)));
    NCCLCHECK(p2pMap(comm, myInfo, comm->peerInfo+info->rank, &info->p2pBuff, (void**)&resources->sendDevMem, &resources->sendMemIpc));
  }

  return ncclSuccess;
}

/* Create and return connect structures for this peer to connect to me */
static ncclResult_t tcclP2pRecvSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo,
    struct ncclConnect* connectInfo, struct ncclConnector * recv, int channelId, int connIndex) {
  int useMemcpy = comm->tcclComm.recvOpt.p2p_use_memcpy;
  int useRead = comm->tcclComm.recvOpt.p2p_use_read;
  struct p2pResources* resources;
  int tpProxyRank;
  NCCLCHECK(ncclCalloc(&resources, 1));
  resources->useMemcpy = useMemcpy;
  recv->transportResources = resources;
  int intermediateRank;
  NCCLCHECK(p2pGetInfo(comm->topo, myInfo, peerInfo, &intermediateRank));

  static_assert(sizeof(struct p2pConnectInfo) <= sizeof(struct ncclConnect), "p2p Connect Info is too big");
  struct p2pConnectInfo* info = (struct p2pConnectInfo*)connectInfo;
  info->read = useRead;
  // For CollNet, use write for scatter-reduce (conn 1), read for broadcast-gather (conn 0)
  if (graph && connIndex == 1) info->read = 0;

  int recvSize = sizeof(struct ncclRecvMem);
  // For P2P Read the SIMPLE buffer is tagged on the end of the ncclSendMem structure
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) if (!(info->read && p == NCCL_PROTO_SIMPLE)) recvSize += comm->buffSizes[p];
  ALIGN_SIZE(recvSize, CUDA_IPC_MIN);

  if (intermediateRank == -1) {
    info->rank = myInfo->rank;
    if (myInfo->pidHash == peerInfo->pidHash && useMemcpy == 0 && !ncclCuMemEnable()) {
      resources->type = P2P_DIRECT;
      recv->conn.flags |= info->read ? NCCL_DIRECT_READ : NCCL_DIRECT_WRITE;
    } else {
      if (ncclCuMemEnable()) {
        // cuMem API support
        resources->type = P2P_CUMEM;
        TRACE(NCCL_INIT|NCCL_P2P,"Ring %02d : %d[%d] <- %d[%d] via P2P/CUMEM",
              channelId, myInfo->rank, myInfo->nvmlDev, peerInfo->rank, peerInfo->nvmlDev);
      } else {
        // Legacy CUDA IPC
        resources->type = P2P_IPC;
      }
      recv->conn.flags |= info->read ? NCCL_IPC_READ : NCCL_IPC_WRITE;
    }
  } else {
    resources->type = P2P_INTERMEDIATE;
    info->rank = intermediateRank;
  }

  tpProxyRank = comm->topParentRanks[info->rank];
  NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_TCCL_P2P, 0, tpProxyRank, &recv->proxyConn));
  struct p2pProxySetupInfo setupInfo;
  setupInfo.size = recvSize;
  NCCLCHECK(ncclProxyCallBlocking(comm, &recv->proxyConn, ncclProxyMsgSetup, &setupInfo, sizeof(struct p2pProxySetupInfo), &info->p2pBuff, sizeof(struct ncclP2pBuff)));

  NCCLCHECK(p2pMap(comm, myInfo, comm->peerInfo+info->rank, &info->p2pBuff, (void**)&resources->recvDevMem, &resources->recvMemIpc));
  return ncclSuccess;
}

/* Connect/Send to this peer */
static ncclResult_t tcclP2pSendConnect(struct ncclComm* comm, struct ncclConnect* connectInfo, int nranks, int rank, struct ncclConnector* send) {
  struct p2pResources* resources = (struct p2pResources*)send->transportResources;
  int useMemcpy = resources->useMemcpy;
  struct ncclRecvMem* remDevMem = NULL;
  struct p2pConnectInfo* info = (struct p2pConnectInfo*)connectInfo;

  // TODO
  INFO(NCCL_TCCL, "p2pSendConnect proxyState->cudaCtx %p", comm->proxyState->cudaCtx);
  if (comm->proxyState->cudaCtx != NULL) {
    INFO(NCCL_TCCL, "p2pSendConnect cuCtxSetCurrent %p", comm->proxyState->cudaCtx);
    CUPFN(cuCtxSetCurrent(comm->proxyState->cudaCtx));
  }

  // TODO
  //{
  //  CUcontext ctx;
  //  CUPFN(cuCtxGetCurrent(&ctx));
  //  INFO(NCCL_TCCL, "Proxy thread context before p2pMap remDevMem %p", ctx);
  //}
  NCCLCHECK(p2pMap(comm, comm->peerInfo+rank, comm->peerInfo+info->rank, &info->p2pBuff, (void**)&remDevMem, &resources->recvMemIpc));
  // TODO
  //{
  //  CUcontext ctx;
  //  CUPFN(cuCtxGetCurrent(&ctx));
  //  INFO(NCCL_TCCL, "Proxy thread context after p2pMap remDevMem(%p) %p", remDevMem, ctx);
  //}

  char* buff = (char*)(remDevMem+1);
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    if (info->read && p == NCCL_PROTO_SIMPLE) {
      /* For P2P Read the SIMPLE buffer is local (ncclSendMem) */
      if (resources->sendDevMem == NULL) return ncclInternalError; // We should not use read + memcpy
      send->conn.buffs[p] = (char*)(resources->sendDevMem+1);
    } else {
      send->conn.buffs[p] = buff;
      buff += comm->buffSizes[p];
    }
  }

  if (useMemcpy) {
    send->conn.tail = &resources->proxyInfo.ceRecvMem->tail;
    send->conn.sizesFifo = resources->proxyInfo.ceRecvMem->sizesFifo;
    send->conn.head = &resources->proxyInfo.devShm->sendMem.head;
    // Send SIMPLE buff to proxy, and replace it by local buffer
    INFO(NCCL_TCCL, "Sending send->conn.buffs[NCCL_PROTO_SIMPLE] = %p", send->conn.buffs[NCCL_PROTO_SIMPLE]);
    NCCLCHECK(ncclProxyCallBlocking(comm, &send->proxyConn, ncclProxyMsgConnect, &send->conn.buffs[NCCL_PROTO_SIMPLE], sizeof(void*), NULL, 0));
    send->conn.buffs[NCCL_PROTO_SIMPLE] = resources->proxyInfo.ceDevBuff;
  } else {
    send->conn.tail = &remDevMem->tail;
    send->conn.head = &resources->sendDevMem->head;
    send->conn.ptrExchange = &resources->sendDevMem->ptrExchange;
    send->conn.redOpArgExchange = resources->sendDevMem->redOpArgExchange;
  }
  return ncclSuccess;
}

/* Connect/Recv from this peer */
static ncclResult_t tcclP2pRecvConnect(struct ncclComm* comm, struct ncclConnect* connectInfo, int nranks, int rank, struct ncclConnector* recv) {
  struct p2pResources* resources = (struct p2pResources*)recv->transportResources;
  int useMemcpy = resources->useMemcpy;
  struct p2pConnectInfo* info = (struct p2pConnectInfo*)connectInfo;

  struct ncclSendMem* remDevMem = NULL;

  if (useMemcpy) {
    char shmPath[PATH_MAX];
    sprintf(shmPath, "/dev/shm/nccl-%s", info->shmName);
    TRACE(NCCL_SHM,"Open shmName %s shmSize %d", shmPath, info->shmSize);
    resources->shmSize = info->shmSize;
    // Attach to peer's SHM segment
    NCCLCHECK(ncclShmOpen(shmPath, info->shmSize, (void**)&resources->shm, (void**)&resources->devShm, -1, &resources->handle));

    recv->conn.tail = &resources->devShm->recvMem.tail;
    recv->conn.head = &resources->devShm->sendMem.head;
  } else {
    NCCLCHECK(p2pMap(comm, comm->peerInfo+rank, comm->peerInfo+info->rank, &info->p2pBuff, (void**)&remDevMem, &resources->sendMemIpc));

    struct ncclRecvMem* devMem = resources->recvDevMem;
    recv->conn.tail = &devMem->tail;
    recv->conn.head = &remDevMem->head;
    recv->conn.ptrExchange = &remDevMem->ptrExchange;
    recv->conn.redOpArgExchange = remDevMem->redOpArgExchange;
  }

  char* buff = (char*)(resources->recvDevMem+1);
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    if (info->read && p == NCCL_PROTO_SIMPLE) {
      if (remDevMem == NULL) return ncclInternalError; // We should not use read + memcpy
      /* For P2P Read the SIMPLE buffer is remote (ncclSendMem) */
      recv->conn.buffs[p] = (char*)(remDevMem+1);
    } else {
      recv->conn.buffs[p] = buff;
      buff += comm->buffSizes[p];
    }
  }
  return ncclSuccess;
}

static ncclResult_t tcclP2pSendFree(struct ncclConnector* send) {
  struct p2pResources* resources = (struct p2pResources*)send->transportResources;
  if (resources) {
    if (ncclCuMemEnable()) {
      // cuMem API support
      if (resources->sendMemIpc) NCCLCHECK(ncclCudaFree(resources->sendMemIpc));
      if (resources->recvMemIpc) NCCLCHECK(ncclCudaFree(resources->recvMemIpc));
    }
    else {
      if (resources->sendMemIpc) CUDACHECK(cudaIpcCloseMemHandle(resources->sendMemIpc));
      if (resources->recvMemIpc) CUDACHECK(cudaIpcCloseMemHandle(resources->recvMemIpc));
    }
    free(resources);
  }
  return ncclSuccess;
}

static ncclResult_t tcclP2pRecvFree(struct ncclConnector* recv) {
  struct p2pResources* resources = (struct p2pResources*)recv->transportResources;
  if (resources) {
    if (ncclCuMemEnable()) {
      // cuMem API support
      if (resources->sendMemIpc) NCCLCHECK(ncclCudaFree(resources->sendMemIpc));
      if (resources->recvMemIpc) NCCLCHECK(ncclCudaFree(resources->recvMemIpc));
    }
    else {
      if (resources->sendMemIpc) CUDACHECK(cudaIpcCloseMemHandle(resources->sendMemIpc));
      if (resources->recvMemIpc) CUDACHECK(cudaIpcCloseMemHandle(resources->recvMemIpc));
      if (resources->useMemcpy) {
        NCCLCHECK(ncclShmClose(resources->handle));
      }
    }
    free(resources);
  }
  return ncclSuccess;
}

static ncclResult_t tcclP2pSendProxySetup(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  struct p2pProxySetupInfo *setupInfo = (struct p2pProxySetupInfo*)reqBuff;
  struct p2pProxyInfo *generalProxyInfo;
  NCCLCHECK(ncclCalloc(&generalProxyInfo, 1));
  connection->transportResources = generalProxyInfo;
  generalProxyInfo->useMemcpy = setupInfo->useMemcpy;
  if (setupInfo->useMemcpy) {
    // TODO
    INFO(NCCL_TCCL, "p2pSendProxySetup proxyState->cudaCtx %p", proxyState->cudaCtx);
    if (proxyState->cudaCtx != NULL) {
      CUPFN(cuCtxSetCurrent(proxyState->cudaCtx));
      INFO(NCCL_TCCL, "p2pSendProxySetup cuCtxSetCurrent %p", proxyState->cudaCtx);
    }

    // CE memcpy support
    struct p2pShmProxyInfo* proxyInfo;
    NCCLCHECK(ncclCalloc(&proxyInfo, 1));
    generalProxyInfo->shmProxyInfo = proxyInfo;

    // TODO
    {
      CUcontext ctx;
      CUPFN(cuCtxGetCurrent(&ctx));
      INFO(NCCL_TCCL, "Proxy thread context before ceDevBuff alloc %p", ctx);
    }

    NCCLCHECK(ncclCudaCalloc(&proxyInfo->ceDevBuff, proxyState->buffSizes[NCCL_PROTO_SIMPLE]));

    // TODO
    {
      CUcontext ctx;
      CUPFN(cuCtxGetCurrent(&ctx));
      INFO(NCCL_TCCL, "Proxy thread context after ceDevBuff(%p) alloc %p", proxyInfo->ceDevBuff, ctx);
    }

    char shmPath[PATH_MAX];
    shmPath[0] = '\0';
    proxyInfo->shmSize = sizeof(struct ncclSendMem) + sizeof(struct ncclRecvMem);
    // Create a SHM segment for the peer to attach to
    NCCLCHECK(ncclShmOpen(shmPath, proxyInfo->shmSize, (void**)&proxyInfo->shm, (void**)&proxyInfo->devShm, 1, &proxyInfo->handle));
    TRACE(NCCL_SHM,"Opened shmName %s shmSize %d", shmPath, proxyInfo->shmSize);
    memcpy(proxyInfo->shmName, shmPath+sizeof("/dev/shm/nccl-")-1, sizeof(proxyInfo->shmName));

    NCCLCHECK(ncclCudaHostCalloc(&proxyInfo->ceRecvMem, 1));

    if (respSize != sizeof(struct p2pShmProxyInfo)) return ncclInternalError;
    memcpy(respBuff, proxyInfo, sizeof(struct p2pShmProxyInfo));
  } else {
    if (reqSize != sizeof(struct p2pProxySetupInfo)) return ncclInternalError;
    int size = setupInfo->size;
    if (respSize != sizeof(struct ncclP2pBuff)) return ncclInternalError;
    struct ncclP2pBuff* p2pBuff = (struct ncclP2pBuff*)respBuff;
    NCCLCHECK(ncclP2pAllocateShareableBuffer(size, &p2pBuff->ipcDesc, &p2pBuff->directPtr));
    p2pBuff->size = size;
    if (ncclCuMemEnable()) {
      // cuMem API support
      struct p2pCuMemProxyInfo* proxyInfo;
      NCCLCHECK(ncclCalloc(&proxyInfo, 1));
      generalProxyInfo->cuMemProxyInfo = proxyInfo;
      memcpy(&proxyInfo->p2pBuff, p2pBuff, sizeof(*p2pBuff));
    } else {
      generalProxyInfo->directPtr = p2pBuff->directPtr;
    }
  }
  *done = 1;
  return ncclSuccess;
}

static ncclResult_t tcclP2pRecvProxySetup(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  if (reqSize != sizeof(struct p2pProxySetupInfo)) return ncclInternalError;
  struct p2pProxySetupInfo *setupInfo = (struct p2pProxySetupInfo*)reqBuff;
  struct p2pProxyInfo *generalProxyInfo;
  NCCLCHECK(ncclCalloc(&generalProxyInfo, 1));
  connection->transportResources = generalProxyInfo;
  generalProxyInfo->useMemcpy = setupInfo->useMemcpy;
  int size = setupInfo->size;
  if (respSize != sizeof(struct ncclP2pBuff)) return ncclInternalError;
  struct ncclP2pBuff* p2pBuff = (struct ncclP2pBuff*)respBuff;
  NCCLCHECK(ncclP2pAllocateShareableBuffer(size, &p2pBuff->ipcDesc, &p2pBuff->directPtr));
  p2pBuff->size = size;
  if (ncclCuMemEnable()) {
    // cuMem API support
    struct p2pCuMemProxyInfo* proxyInfo;
    NCCLCHECK(ncclCalloc(&proxyInfo, 1));
    memcpy(&proxyInfo->p2pBuff, p2pBuff, sizeof(*p2pBuff));
    generalProxyInfo->cuMemProxyInfo = proxyInfo;
  } else {
    generalProxyInfo->directPtr = p2pBuff->directPtr;
  }
  *done = 1;
  return ncclSuccess;
}

static ncclResult_t tcclP2pSendProxyConnect(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  // Only called when useMemcpy by p2pSendConnect
  struct p2pShmProxyInfo* proxyInfo = ((struct p2pProxyInfo*)connection->transportResources)->shmProxyInfo;

  if (reqSize != sizeof(void*)) return ncclInternalError;
  proxyInfo->recvFifo = *((char**)reqBuff);

  CUDACHECK(cudaStreamCreateWithFlags(&proxyInfo->stream, cudaStreamNonBlocking));
  for (int i=0; i<NCCL_STEPS; i++) {
    CUDACHECK(cudaEventCreate(proxyInfo->events+i));
  }
  connection->proxyAppendPtr = &connection->proxyAppend;
  return ncclSuccess;
}

static ncclResult_t tcclP2pSendProxyFree(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState) {
  struct p2pProxyInfo *generalProxyInfo = (struct p2pProxyInfo*)connection->transportResources;
  // CE memcpy support
  if (generalProxyInfo->useMemcpy) {
    struct p2pShmProxyInfo* proxyInfo = generalProxyInfo->shmProxyInfo;
    if (proxyInfo) {
      NCCLCHECK(ncclShmClose(proxyInfo->handle));
      NCCLCHECK(ncclCudaHostFree(proxyInfo->ceRecvMem));
      NCCLCHECK(ncclCudaFree(proxyInfo->ceDevBuff));
      CUDACHECK(cudaStreamDestroy(proxyInfo->stream));
      for (int i=0; i<NCCL_STEPS; i++) {
        CUDACHECK(cudaEventDestroy(proxyInfo->events[i]));
      }
      free(proxyInfo);
    }
  } else {
    if (ncclCuMemEnable()) {
      // cuMem API support
      struct p2pCuMemProxyInfo *proxyInfo = generalProxyInfo->cuMemProxyInfo;
      if (proxyInfo) {
        struct ncclP2pBuff *p2pBuff = &proxyInfo->p2pBuff;
        ncclP2pFreeShareableBuffer(&p2pBuff->ipcDesc);
        ncclCudaFree(p2pBuff->directPtr);
        free(proxyInfo);
      }
    } else {
      // Do not check return code as CUDA may have already shut down
      ncclCudaFree(generalProxyInfo->directPtr);
    }
  }
  free(generalProxyInfo);
  return ncclSuccess;
}

static ncclResult_t tcclP2pRecvProxyFree(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState) {
  struct p2pProxyInfo *generalProxyInfo = (struct p2pProxyInfo*)connection->transportResources;
  if (ncclCuMemEnable()) {
    struct p2pCuMemProxyInfo *proxyInfo = generalProxyInfo->cuMemProxyInfo;
    if (proxyInfo) {
      struct ncclP2pBuff *p2pBuff = &proxyInfo->p2pBuff;
      ncclP2pFreeShareableBuffer(&p2pBuff->ipcDesc);
      ncclCudaFree(p2pBuff->directPtr);
      free(proxyInfo);
    }
  } else {
    // Do not check return code as CUDA may have already shut down
    ncclCudaFree(generalProxyInfo->directPtr);
  }
  free(generalProxyInfo);
  return ncclSuccess;
}

// CE memcpy support
static ncclResult_t tcclP2pSendProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args) {
  //INFO(NCCL_TCCL, "P2P/CE send proxy progress %d/%d state %d", args->done, args->nsubs, args->state);
  if (args->state == ncclProxyOpReady) {
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      struct p2pProxyInfo *generalProxyInfo = (struct p2pProxyInfo*)sub->connection->transportResources;
      struct p2pShmProxyInfo* resources = generalProxyInfo->shmProxyInfo;
      // Round to next multiple of sliceSteps
      sub->base = ROUNDUP(resources->step, args->chunkSteps);
      sub->posted = sub->transmitted = sub->done = 0;
    }
    args->state = ncclProxyOpProgress;
  }
  args->idle = 1;
  if (args->state == ncclProxyOpProgress) {
    int p = args->protocol;
    int stepSize = proxyState->buffSizes[p] / NCCL_STEPS;
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      struct p2pProxyInfo *generalProxyInfo = (struct p2pProxyInfo*)sub->connection->transportResources;
      struct p2pShmProxyInfo* resources = generalProxyInfo->shmProxyInfo;
      if (p != NCCL_PROTO_SIMPLE) { // Only Simple uses cudaMemcpy
          resources->step = sub->base + sub->nsteps;
          args->done++;
          continue;
      }
      //INFO(NCCL_TCCL, "P2P/CE send proxy progress %d/%d base %d nsteps %d", s, args->nsubs, sub->base, sub->nsteps);
      //INFO(NCCL_TCCL, "P2P/CE send proxy progress %d/%d posted %d transmitted %d done %d", s, args->nsubs, sub->posted, sub->transmitted, sub->done);
      if (sub->transmitted < sub->done + NCCL_STEPS && sub->transmitted < sub->nsteps) {
        int buffSlot = (sub->base+sub->transmitted)%NCCL_STEPS;
        volatile int* sizesFifo = resources->ceRecvMem->sizesFifo;
        volatile uint64_t* recvTail = &resources->ceRecvMem->tail;
        // Check GPU has sent everything
        //INFO(NCCL_TCCL, "P2P/CE send proxy progress %d/%d recvTail %ld", s, args->nsubs, *recvTail);
        if ((*recvTail > sub->base+sub->transmitted)) {
          //INFO(NCCL_TCCL, "Inside if statement");
          int size = sizesFifo[buffSlot];
          //INFO(NCCL_TCCL, "P2P/CE src %p dst %p size %d stream %d", resources->ceDevBuff+buffSlot*stepSize, resources->recvFifo+buffSlot*stepSize, size, resources->stream);
          //cudaPointerAttributes attr;

          //CUDACHECK(cudaPointerGetAttributes(&attr, resources->recvFifo+buffSlot*stepSize));
          //int dst_device = attr.device;
          //INFO(NCCL_TCCL, "dst type %d device %d deviceptr %p hostptr %p", attr.type, attr.device, attr.devicePointer, attr.hostPointer);

          //CUDACHECK(cudaPointerGetAttributes(&attr, resources->ceDevBuff+buffSlot*stepSize));
          //int src_device = attr.device;
          //INFO(NCCL_TCCL, "src type %d device %d deviceptr %p hostptr %p", attr.type, attr.device, attr.devicePointer, attr.hostPointer);

          //if (CUPFN(cuCtxSetCurrent(proxyState->cudaCtx)) != CUDA_SUCCESS) {
          //  INFO(NCCL_TCCL, "p2pSendProxyProgress Failed to set CUDA context on device %d", proxyState->cudaDev);
          //}

          //INFO(NCCL_TCCL, "current device %d", proxyState->cudaDev);
          //CUDACHECK(cudaSetDevice(proxyState->cudaDev));
          //INFO(NCCL_TCCL, "set device to %d", proxyState->cudaDev);

          // TODO
          //{
          //  CUcontext ctx;
          //  CUPFN(cuCtxGetCurrent(&ctx));
          //  INFO(NCCL_TCCL, "Proxy thread context before cudaMemcpyAsync %p", ctx);
          //}

          // TODO delete v
          //CUDACHECK(cudaMemcpyAsync(heehoon_global_tmp_buf, resources->ceDevBuff+buffSlot*stepSize, size, cudaMemcpyDeviceToDevice, resources->stream));

          // Original Call here v
          CUDACHECK(cudaMemcpyAsync(resources->recvFifo+buffSlot*stepSize, resources->ceDevBuff+buffSlot*stepSize, size, cudaMemcpyDeviceToDevice, resources->stream));

          //CUDACHECK(cudaMemcpyPeerAsync(resources->recvFifo+buffSlot*stepSize, dst_device, resources->ceDevBuff+buffSlot*stepSize, src_device, size, resources->stream));
          //INFO(NCCL_TCCL, "P2P/CE memcpy %p -> %p size %d", resources->ceDevBuff+buffSlot*stepSize, resources->recvFifo+buffSlot*stepSize, size);
          CUDACHECK(cudaEventRecord(resources->events[buffSlot], resources->stream));
          sub->transmitted += args->sliceSteps;
        }
      }
      if (sub->done < sub->transmitted) {
        int buffSlot = (sub->base+sub->done)%NCCL_STEPS;
        cudaError_t res = cudaEventQuery(resources->events[buffSlot]);
        if (res != cudaErrorNotReady) CUDACHECK(res);
        if (res == cudaSuccess) {
          sub->done += args->sliceSteps;
          // Notify SHM
          resources->shm->recvMem.tail = sub->base + sub->done;
        }
        if (sub->done == sub->nsteps) {
          resources->step = sub->base + sub->nsteps;
          args->done++;
        }
      }
    }
    if (args->done == args->nsubs) {
      args->state = ncclProxyOpNone;
    }
  }
  return ncclSuccess;
}

struct ncclTransport tcclP2pTransport = {
  "TCCL_P2P",
  tcclP2pCanConnect,
  { tcclP2pSendSetup, tcclP2pSendConnect, tcclP2pSendFree, NULL, tcclP2pSendProxySetup, tcclP2pSendProxyConnect, tcclP2pSendProxyFree, tcclP2pSendProxyProgress },
  { tcclP2pRecvSetup, tcclP2pRecvConnect, tcclP2pRecvFree, NULL, tcclP2pRecvProxySetup, NULL, tcclP2pRecvProxyFree, NULL }
};