#!/bin/bash

python $TCCL_SCRIPTS/eval_dl.py --output-dir $TCCL_AEC_ROOT/workspace/eval_dl_workspace --target-machine a
python $TCCL_SCRIPTS/eval_dl.py --output-dir $TCCL_AEC_ROOT/workspace/eval_dl_workspace --target-machine b
python $TCCL_SCRIPTS/eval_dl.py --output-dir $TCCL_AEC_ROOT/workspace/eval_dl_workspace --target-machine d