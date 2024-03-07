#!/bin/bash
hosts=d0:11,d2:1,d3:1
mpirun -H $hosts -mca btl ^openib -mca pml ucx --bind-to none \
  -x LD_LIBRARY_PATH \
  ./launch_benchmark "$@"