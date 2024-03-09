#!/bin/bash
hosts=b0:11,b1:1,b2:1
mpirun -H $hosts -mca btl ^openib -mca pml ucx --bind-to none \
  -x LD_LIBRARY_PATH \
  ./launch_benchmark "$@"