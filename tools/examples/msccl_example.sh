#!/bin/bash
mpirun -H a0:4,a1:4,a2:4,a3:4,a4:4,a5:4 -x LD_LIBRARY_PATH=/home/n1/heehoon/tmp/msccl-test/msccl/build/lib/:$LD_LIBRARY_PATH \
 -x NCCL_DEBUG=INFO \
 -x NCCL_DEBUG_SUBSYS=INIT,ENV \
 -x MSCCL_XML_FILES=../build/Allreduce.n24-Custom-AKMU_P2P_ENABLE_64MB-.n4-steps56.chunks24-gurobisol-1687349990-allreduce-1687350045_i1_scRemote1_IBContig.sccl.xml \
 -x NCCL_ALGO=MSCCL,RING,TREE \
 /home/n1/heehoon/tmp/msccl-test/nccl-tests/build/all_reduce_perf -b 128 -e 64MB -f 2 -g 1 -c 1 -n 10 -w 10 -G 10 -z 0

 #-x NCCL_ALGO=MSCCL,RING,TREE \


# mpirun -np <ngpus> -x LD_LIBRARY_PATH=msccl/build/lib/:$LD_LIBRARY_PATH -x NCCL_DEBUG=INFO -x NCCL_DEBUG_SUBSYS=INIT,ENV 
# -x MSCCL_XML_FILES=<taccl-ef> 
# -x NCCL_ALGO=MSCCL,RING,TREE  nccl-tests/build/<nccl-test-binary> -b 128 -e 32MB -f 2 -g 1 -c 1 -n 100 -w 100 -G 100 -z 0