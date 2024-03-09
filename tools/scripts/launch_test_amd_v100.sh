#!/bin/bash
SLURM_PARTITION=asplosA
TCCL_XML_FILE=$TCCL_AEC_ROOT/workspace/amd_v100.xml
salloc -p $SLURM_PARTITION -N 2 --exclusive \
  mpirun -mca btl ^openib -mca pml ucx --bind-to none \
  -npernode 4 \
  -x LD_PRELOAD=$TCCL_ROOT/build/lib/libnccl.so \
  -x NCCL_DEBUG=INFO \
  -x NCCL_DEBUG_SUBSYS=TCCL \
  -x NCCL_ALGO=TCCL,RING \
  -x TCCL_XML_FILE=$TCCL_XML_FILE \
  $TCCL_AEC_ROOT/nccl-tests-2.13.8/build/all_reduce_perf -b 8 -e 128M -f 2