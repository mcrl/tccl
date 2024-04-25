#!/bin/bash
SLURM_PARTITION=asplosA
XMLTMPDIR=$TCCL_AEC_ROOT/workspace/amd_v100_xmls
OUTPUT_FILE=$TCCL_AEC_ROOT/workspace/amd_v100.xml

mkdir $XMLTMPDIR

salloc -p $SLURM_PARTITION -N 3 --exclusive \
  mpirun -mca btl ^openib -mca pml ucx --bind-to none \
  -npernode 11 \
  $TCCL_ROOT/tools/build/pathfinder -o $XMLTMPDIR

python $TCCL_SCRIPTS/preprocess_xml.py --dir $XMLTMPDIR --output $OUTPUT_FILE