#!/bin/bash

python $TCCL_SCRIPTS/eval_cc.py --output-dir $TCCL_AEC_ROOT/workspace/eval_cc_workspace --target-machine a --bind-none
python $TCCL_SCRIPTS/eval_cc.py --output-dir $TCCL_AEC_ROOT/workspace/eval_cc_workspace --target-machine b --bind-none
python $TCCL_SCRIPTS/eval_cc.py --output-dir $TCCL_AEC_ROOT/workspace/eval_cc_workspace --target-machine d --bind-none