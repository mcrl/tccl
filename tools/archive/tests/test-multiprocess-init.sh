#!/bin/bash
hosts=a1,a1,a3,a3
devices=0,3
mpirun -H $hosts -mca btl ^openib -mca pml ucx --bind-to none -mca rmaps seq \
  -x NCCL_DEBUG=INFO \
  -x NCCL_DEBUG_SUBSYS=TCCL \
  -x TCCL_XML_FILE=$HOME/tccl-tools/tests/test-multiprocess-init.xml \
  -x CUDA_VISIBLE_DEVICES=$devices \
  -x NCCL_ALGO=TCCL \
  -x LD_PRELOAD=$HOME/tccl/build/lib/libnccl.so \
  -x NCCL_NET_GDR_LEVEL=SYS \
  -x NCCL_NET_GDR_READ=1 \
  ./test-multiprocess-init