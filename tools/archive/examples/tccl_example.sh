#!/bin/bash
#hosts=a2:4,a3:4
hosts=d0:4,d2:4,d3:4
#hosts=d0:2
echo $hosts
mpirun -H $hosts -mca btl ^openib -mca pml ucx --bind-to none \
  -x NCCL_DEBUG_SUBSYS=INIT,TCCL,NET \
  -x NCCL_ALGO=TCCL \
  -x TCCL_XML_FILE=$HOME/tccl-tools/akmu_d_tccl_routing_db.xml \
  -x LD_PRELOAD=$HOME/tccl/build/lib/libnccl.so \
  -x NCCL_CREATE_THREAD_CONTEXT=1 \
  $HOME/tccl-exp/nccl-tests/build/all_reduce_perf \
  -b 1M -e 256M -f 2 -n 5 -m 1 -w 0 -c 1 -G 0 --op sum
  #/usr/local/cuda/bin/nsys profile --nic-metrics=true --gpu-metrics-device=all -w true -t cuda,nvtx,osrt,cudnn,cublas -s none \
  #-o a%q{OMPI_COMM_WORLD_RANK}  -f true -x true \
  #-x CUDA_VISIBLE_DEVICE=3,2,1,0 \
  #-x NCCL_NET_GDR_LEVEL=SYS \
  #-x NCCL_NET_GDR_READ=0 \
  #$HOME/tccl-exp/nccl-tests/build/all_reduce_perf \
  #-x TCCL_XML_FILE=$HOME/tccl-tools/tests/test-multiprocess-init.xml \

  #-b 16K -e 128M -f 2 --op sum
  #-x NCCL_BUFFSIZE=8388608 \
  #-x NCCL_BUFFSIZE=4194304 \
  #-x NCCL_BUFFSIZE=2097152 \

#mpirun -H $hosts -mca btl ^openib -mca pml ucx --bind-to none -mca rmaps seq \

 # -x TCCL_XML_FILE=$HOME/tccl-tools/tests/test-multiprocess-init.xml \

  #./TestNcclCtl.sh
  #-x LD_PRELOAD=$HOME/tccl-exp/nccl-2.18.3-1/build/lib/libnccl.so \
  #-x NCCL_DEBUG=INFO \
  #-x NCCL_DEBUG_SUBSYS=INIT,GRAPH,TUNING \

  #-x NCCL_SHM_USE_CUDA_MEMCPY=1 \
  #-x NCCL_DEBUG=TRACE \
  #-x NCCL_DEBUG_SUBSYS=ALL \


  #-x NCCL_COLLNET_ENABLE=0 \
  #-x CUDA_VISIBLE_DEVICES=$devices \
  #-x LD_LIBRARY_PATH=$HOME/tccl-exp/nccl-2.18.3-1/build/lib:$LD_LIBRARY_PATH \
  #-x LD_LIBRARY_PATH=$HOME/tmp/msccl-test/msccl/build/lib/:$LD_LIBRARY_PATH \
  #-x SHARP_COLL_LOG_LEVEL=3 \
  #-x SHARP_COLL_ENABLE_SAT=1  \
  #-x NCCL_NET_GDR_LEVEL=SYS \
  #-x NCCL_NET_GDR_READ=1 \

  #-x NCCL_NET_GDR_LEVEL=SYS \
  #-x NCCL_NET_GDR_READ=0 \
  #-x NCCL_P2P_LEVEL=LOC \
  #-x NCCL_DEBUG=INFO \
  #-x NCCL_DEBUG_SUBSYS=INIT,GRAPH,TUNING \
  #-x NCCL_TOPO_DUMP_FILE=$HOME/nccl-topo.txt \
  #-x NCCL_DEBUG=TRACE \
  #-x NCCL_DEBUG_SUBSYS=GRAPH,TUNING \
  #-x NCCL_ALGO=CollnetChain \
  # Tree,Ring,CollnetDirect,CollnetChain

#mpirun -np 2 -H a0,a3 --map-by node --bind-to none \
#  -x CUDA_VISIBLE_DEVICES=3 \
# -mca btl_openib_warn_default_gid_prefix 0  -mca rmaps_dist_device mlx5_0:1 -mca rmaps_base_mapping_policy dist:span \
#  -mca btl_openib_if_include mlx5_0:1  -x HCOLL_MAIN_IB=mlx5_0:1  -mca pml ucx  -x UCX_NET_DEVICES=mlx5_0:1 \
#   -x HCOLL_SBGP=p2p -x HCOLL_BCOL=ucx_p2p  -x HCOLL_ENABLE_MCAST_ALL=0  -x HCOLL_ALLREDUCE_ZCOPY_TUNE=static \
#   -x LD_LIBRARY_PATH \
#   -x SHARP_COLL_LOG_LEVEL=3  -x HCOLL_ENABLE_SHARP=1  -x SHARP_COLL_ENABLE_MCAST_TARGET=0 \
#    -x SHARP_COLL_JOB_QUOTA_PAYLOAD_PER_OST=1024  -x SHARP_COLL_JOB_QUOTA_OSTS=128  -x SHARP_COLL_ENABLE_SAT=1 \
#     -x SHARP_COLL_SAT_THRESHOLD=1024  -x HCOLL_BCOL_P2P_LARGE_ALLREDUCE_ALG=4  -x NCCL_COLLNET_ENABLE=1 \
#      -x HCOLL_ALLREDUCE_ZCOPY_THRESH_CUDA=0  -x HCOLL_CUDA_SBGP=p2p  -x HCOLL_CUDA_BCOL=nccl  -x HCOLL_GPU_ENABLE=1 \
#       -x SHARP_COLL_JOB_MEMBER_LIST_TYPE=2  -x SHARP_COLL_JOB_REQ_EXCLUSIVE_LOCK_MODE=0 \
#         taskset -c 1 numactl --membind=0 \
#          $HOME/HPCX-Thunder/nccl-tests/build/all_reduce_perf -b 4 -e 512M -f 2 --op sum -t 1 -g 1

# Original command w/ SHARP
#mpirun -np 2 -hostfile /tmp/sharp_benchmark2_logs.3811927/hostfile --map-by node --bind-to none \
# -mca btl_openib_warn_default_gid_prefix 0  -mca rmaps_dist_device mlx5_0:1 -mca rmaps_base_mapping_policy dist:span \
#  -mca btl_openib_if_include mlx5_0:1  -x HCOLL_MAIN_IB=mlx5_0:1  -mca pml ucx  -x UCX_NET_DEVICES=mlx5_0:1 \
#   -x HCOLL_SBGP=p2p -x HCOLL_BCOL=ucx_p2p  -x HCOLL_ENABLE_MCAST_ALL=0  -x HCOLL_ALLREDUCE_ZCOPY_TUNE=static \
#   -x LD_LIBRARY_PATH \
#   -x SHARP_COLL_LOG_LEVEL=3  -x HCOLL_ENABLE_SHARP=1  -x SHARP_COLL_ENABLE_MCAST_TARGET=0 \
#    -x SHARP_COLL_JOB_QUOTA_PAYLOAD_PER_OST=1024  -x SHARP_COLL_JOB_QUOTA_OSTS=128  -x SHARP_COLL_ENABLE_SAT=1 \
#     -x SHARP_COLL_SAT_THRESHOLD=1024  -x HCOLL_BCOL_P2P_LARGE_ALLREDUCE_ALG=4  -x NCCL_COLLNET_ENABLE=1 \
#      -x HCOLL_ALLREDUCE_ZCOPY_THRESH_CUDA=0  -x HCOLL_CUDA_SBGP=p2p  -x HCOLL_CUDA_BCOL=nccl  -x HCOLL_GPU_ENABLE=1 \
#       -x SHARP_COLL_JOB_MEMBER_LIST_TYPE=2  -x SHARP_COLL_JOB_REQ_EXCLUSIVE_LOCK_MODE=0 \
#         taskset -c 1 numactl --membind=0 \
#          $HOME/HPCX-Thunder/nccl-tests/build/all_reduce_perf -b 4 -e 512M -f 2 --op sum -t 2 -g 1

# Original command w/o SHARP
#$HOME/HPCX-Thunder/hpcx/ompi/bin/mpirun -np 2 -hostfile /tmp/sharp_benchmark2_logs.3829613/hostfile --map-by node --bind-to none  -mca btl_openib_warn_default_gid_prefix 0  -mca rmaps_dist_device mlx5_0:1 -mca rmaps_base_mapping_policy dist:span  -mca btl_openib_if_include mlx5_0:1  -x HCOLL_MAIN_IB=mlx5_0:1  -mca pml ucx  -x UCX_NET_DEVICES=mlx5_0:1  -x HCOLL_SBGP=p2p -x HCOLL_BCOL=ucx_p2p  -x HCOLL_ENABLE_MCAST_ALL=0  -x HCOLL_ALLREDUCE_ZCOPY_TUNE=static -x LD_LIBRARY_PATH=$HOME/HPCX-Thunder/numactl/.libs:$HOME/HPCX-Thunder/hpcx/nccl_rdma_sharp_plugin/lib:$HOME/HPCX-Thunder/hpcx/ucc/lib/ucc:$HOME/HPCX-Thunder/hpcx/ucc/lib:$HOME/HPCX-Thunder/hpcx/ucx/lib/ucx:$HOME/HPCX-Thunder/hpcx/ucx/lib:$HOME/HPCX-Thunder/hpcx/sharp/lib:$HOME/HPCX-Thunder/hpcx/hcoll/lib:$HOME/HPCX-Thunder/hpcx/ompi/lib:$HOME/opt/ucx/lib:$HOME/opt/openmpi/lib:$HOME/nccl/build/lib:/usr/local/cuda/lib64: -x LD_PRELOAD=$HOME/HPCX-Thunder/hpcx/sharp/lib/libsharp.so:$HOME/HPCX-Thunder/hpcx/sharp/lib/libsharp_coll.so   taskset -c 1 numactl --membind=0  $HOME/HPCX-Thunder/nccl-tests/build/all_reduce_perf -b 4 -e 512M -f 2 --op sum -t 2 -g 1