#!/bin/bash
hosts=a1:11,a2:1,a3:1
mpirun -H $hosts -mca btl ^openib -mca pml ucx --bind-to none \
  -x LD_LIBRARY_PATH \
  ./launch_benchmark "$@"