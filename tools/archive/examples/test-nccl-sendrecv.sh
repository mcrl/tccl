#!/bin/bash
hosts=a1:4,a2:4
gpus="3 2 1 0 3 2 1 0"
mpirun -H $hosts -mca btl ^openib -mca pml ucx --bind-to none \
  -x NCCL_DEBUG=INFO \
  -x NCCL_DEBUG_SUBSYS=TCCL \
  -x NCCL_NET_GDR_LEVEL=SYS \
  -x TCCL_XML_FILE=$HOME/tccl-tools/akmu_a_tccl_routing_db.xml \
  -x LD_PRELOAD=$HOME/tccl/build/lib/libnccl.so \
  ./test-nccl-sendrecv $gpus

  #-x LD_LIBRARY_PATH=$HOME/tccl-exp/nccl-2.18.3-1/build/lib:$LD_LIBRARY_PATH \
  #-x LD_LIBRARY_PATH=$HOME/tccl/build/lib:$LD_LIBRARY_PATH \

  #-x NCCL_DEBUG=INFO \
  #-x NCCL_DEBUG_SUBSYS=INIT,GRAPH,TUNING,NET,P2P \
  
  #/usr/local/cuda/bin/nsys profile --nic-metrics=true --gpu-metrics-device=all -w true -t cuda,nvtx,osrt,cudnn,cublas -s none \
  #-o %q{OMPI_COMM_WORLD_RANK}  -f true -x true \

  #$HOME/HPCX-Thunder/nccl-tests/build/all_reduce_perf \
  #-b 32M -e 64M -f 2 --op sum

  #-x NCCL_DEBUG=INFO \
  #-x NCCL_DEBUG_SUBSYS=INIT,GRAPH,TUNING,NET \
  #-x NCCL_TOPO_DUMP_FILE=$HOME/nccl-topo.txt \
  #-x NCCL_DEBUG=TRACE \
  #-x NCCL_DEBUG_SUBSYS=GRAPH,TUNING \
  #-x NCCL_ALGO=CollnetChain \
  # Tree,Ring,CollnetDirect,CollnetChain