#!/bin/bash
mpirun -mca btl ^openib -mca pml ucx --bind-to none \
-mca rmaps seq -H a2,a2,a2,a2,a3,a3,a3,a3,a3,a2 \
-x OMP_NUM_THREADS=2 \
-x LD_LIBRARY_PATH \
-x SHARP_COLL_LOG_LEVEL=3 \
-x SHARP_COLL_ENABLE_SAT=1 \
./benchmark $@
