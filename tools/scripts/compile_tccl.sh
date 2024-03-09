#!/bin/bash

# compile pathfinder
cd $TCCL_ROOT/tools
mkdir build
cd build
cmake ..
make -j `nproc`

# compile runtime
cd $TCCL_ROOT
make -j `nproc` src.build NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_86,code=sm_86"
