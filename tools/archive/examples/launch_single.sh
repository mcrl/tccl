#!/bin/bash
hosts=a3:10
mpirun -H $hosts -mca btl ^openib -mca pml ucx --bind-to none \
  -x LD_LIBRARY_PATH \
  ./launch_single "$@"