#!/bin/bash
hosts=a2,a2
mpirun -H $hosts -mca btl ^openib -mca pml ucx --bind-to none -mca rmaps seq \
  -x OMP_NUM_THREADS=2 \
  -x LD_LIBRARY_PATH \
  ./ThunderStream $@

  #/usr/local/cuda/bin/nsys profile --nic-metrics=true --gpu-metrics-device=all -w true -t cuda,nvtx,osrt,cudnn,cublas -s none \
  #-o %q{OMPI_COMM_WORLD_RANK} -f true -x true \