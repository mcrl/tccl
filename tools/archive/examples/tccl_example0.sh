#!/bin/bash
CMD='$HOME/tmp/msccl-test/nccl-tests/build/all_reduce_perf
-b 1K -e 64M -f 2 -g 1 -c 1 -n 10 -w 10 -m 1'
if [[ $OMPI_COMM_WORLD_RANK == 0 ]]
then
  $CMD
else
  $CMD 1>/dev/null 2>/dev/null
fi 