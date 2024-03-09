import argparse
import subprocess
import os
import sys
import time

#DEVICES = ['0', '1', '2', '3', '0,1', '0,2', '0,3', '1,2', '1,3', '2,3', '0,1,2', '0,1,3', '0,2,3', '1,2,3', '0,1,2,3']
DEVICES = ['0,1,2,3']


def get_machine_configuration(machine):
  if machine == 'a':
    #num_nodes = list(range(1, 4 + 1))
    num_nodes = [4]
    #tp_pp = []
    #for prod in [1, 2, 4, 8, 16]:
    #  for tp in [1, 2, 4, 8, 16]:
    #    if prod % tp == 0:
    #      pp = prod // tp
    #      tp_pp.append((tp, pp))
    tp_pp = [(1, 1), (2, 1), (4, 1)]
    #nodes = ['a%d' % i for i in range(4)]
    nodes = ['a0', 'a1', 'a2', 'a3']
    partition = 'asplosA'
  elif machine == 'b':
    #num_nodes = list(range(1, 8 + 1))
    num_nodes = [6]
    #tp_pp = []
    #for prod in [1, 2, 4, 8, 16, 24]:
    #  for tp in [1, 2, 4, 8, 16, 24]:
    #    if prod % tp == 0:
    #      pp = prod // tp
    #      tp_pp.append((tp, pp))
    tp_pp = [(1, 1), (2, 1), (4, 1)]
    nodes = ['b%d' % i for i in range(7, -1, -1)]
    partition = 'asplosB'
  elif machine == 'd':
    #num_nodes = list(range(1, 3 + 1))
    num_nodes = [3]
    #tp_pp = []
    #for prod in [1, 2, 4, 8, 12]:
    #  for tp in [1, 2, 4, 8, 12]:
    #    if prod % tp == 0:
    #      pp = prod // tp
    #      tp_pp.append((tp, pp))
    tp_pp = [(1, 1), (2, 1), (4, 1)]
    nodes = ['d0', 'd2', 'd3']
    partition = 'asplosD'
  else:
    assert False, 'Unknown machine: %s' % machine
  return num_nodes, nodes, tp_pp, partition

"""
USAGE: all_gather_perf 
        [-t,--nthreads <num threads>] 
        [-g,--ngpus <gpus per thread>] 
        [-b,--minbytes <min size in bytes>] 
        [-e,--maxbytes <max size in bytes>] 
        [-i,--stepbytes <increment size>] 
        [-f,--stepfactor <increment factor>] 
        [-n,--iters <iteration count>] 
        [-m,--agg_iters <aggregated iteration count>] 
        [-w,--warmup_iters <warmup iteration count>] 
        [-p,--parallel_init <0/1>] 
        [-c,--check <check iteration count>] 
        [-o,--op <sum/prod/min/max/avg/mulsum/all>] 
        [-d,--datatype <nccltype/all>] 
        [-r,--root <root>] 
        [-z,--blocking <0/1>] 
        [-y,--stream_null <0/1>] 
        [-T,--timeout <time in seconds>] 
        [-G,--cudagraph <num graph launches>] 
        [-C,--report_cputime <0/1>] 
        [-a,--average <0/1/2/3> report average iteration time <0=RANK0/1=AVG/2=MIN/3=MAX>] 
        [-h,--help]"""

def gen_script(args, torch_script_fn, nodes_to_use, num_gpu, libname, target_machine, num_node, devices, model, tp, pp, partition, mbs, accum):
  master_host = nodes_to_use[0]
  hosts = ','.join([f'{node}' for node in nodes_to_use])
  #gbs = num_node * num_gpu * mbs // tp // pp
  gbs = num_node * num_gpu * mbs * accum
  mpi_script = f"""#!/bin/bash
EMPTY_PORT=$(comm -23 <(seq 10000 65535 | sort) <(ss -tan | awk '{{print $4}}' | cut -d':' -f2 | grep -v '^\s*$' | sort -u) | shuf -n 1)
TARGET_MACHINE={target_machine}
TARGET_BACKEND={libname}
# should be first of hosts
MASTER_ADDR={master_host}
HOSTS={hosts}
NNODES={num_node}
GPUS_PER_NODE={num_gpu}
MBS={mbs}
GBS={gbs}
salloc -p {partition} -N {num_node} --exclusive \
mpirun -mca btl ^openib -mca pml ucx --bind-to none \
-npernode 1 \
-x EMPTY_PORT=${{EMPTY_PORT}} \
-x TARGET_MACHINE=${{TARGET_MACHINE}} \
-x TARGET_BACKEND=${{TARGET_BACKEND}} \
-x MASTER_ADDR=${{MASTER_ADDR}} \
-x MASTER_PORT=${{EMPTY_PORT}} \
-x NNODES=${{NNODES}} \
-x GPUS_PER_NODE=${{GPUS_PER_NODE}} \
-x MBS=${{MBS}} \
-x GBS=${{GBS}} \
-x CUDA_VISIBLE_DEVICES={devices} \
{torch_script_fn}
"""
  if model == 'gpt':
    vocab_file = '$TCCL_AEC_ROOT/refs/gpt2-vocab.json'
    merge_file = '$TCCL_AEC_ROOT/refs/gpt2-merges.txt'
    data_path = '$TCCL_AEC_ROOT/workspace/datasets/wikitext_text_document'
    if libname == 'msccl':
      script_name = 'pretrain_gpt_msccl.py'
    else:
      script_name = 'pretrain_gpt.py'
    if target_machine == 'b':
      model_args = 'GPT_LARGE_ARGS'
    else:
      model_args = 'GPT_XL_ARGS'
  elif model == 'bert':
    vocab_file = '$TCCL_AEC_ROOT/refs/bert-vocab.txt'
    merge_file = '$TCCL_AEC_ROOT/refs/gpt2-merges.txt'
    data_path = '$TCCL_AEC_ROOT/workspace/datasets/wikitext_text_sentence'
    if libname == 'msccl':
      script_name = 'pretrain_bert_msccl.py'
    else:
      script_name = 'pretrain_bert.py'
    if target_machine == 'b':
      model_args = 'GPT_LARGE_ARGS'
    else:
      model_args = 'GPT_XL_ARGS'
  elif model == 't5':
    vocab_file = '$TCCL_AEC_ROOT/refs/bert-vocab.txt'
    merge_file = '$TCCL_AEC_ROOT/refs/gpt2-merges.txt'
    data_path = '$TCCL_AEC_ROOT/workspace/datasets/wikitext_text_sentence'
    if libname == 'msccl':
      script_name = 'pretrain_t5_msccl.py'
    else:
      script_name = 'pretrain_t5.py'
    if target_machine == 'b':
      model_args = 'T5_LARGE_ARGS'
    else:
      model_args = 'T5_XL_ARGS'
  else:
    assert False

  if target_machine == 'a':
    xml_fn = f'amd_v100.xml'
  elif target_machine == 'b':
    xml_fn = f'amd_3090.xml'
  elif target_machine == 'd':
    xml_fn = f'intel_v100.xml'
  else:
    assert False, 'Unknown machine: %s' % target_machine

  torch_script = f"""#!/bin/bash -l

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tccl-aec

# Runs the "345M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1
SCRIPT_NAME={script_name}

if [ $TARGET_BACKEND = "nccl" ]; then
  export NCCL_DEBUG=INFO
  export NCCL_DEBUG_SUBSYS=INIT,ENV
  export LD_PRELOAD=$TCCL_AEC_ROOT/nccl-2.18.3-1/build/lib/libnccl.so
elif [ $TARGET_BACKEND = "tccl" ]; then
  export LD_PRELOAD=$TCCL_AEC_ROOT/tccl/build/lib/libnccl.so
  export NCCL_DEBUG=INFO
  export NCCL_DEBUG_SUBSYS=INIT,ENV,TCCL
  export NCCL_ALGO=TCCL,RING
  export TCCL_XML_FILE=$TCCL_AEC_ROOT/workspace/{xml_fn}
elif [ $TARGET_BACKEND = "msccl" ]; then
  export LD_PRELOAD=$TCCL_AEC_ROOT/msccl-0.7.4/build/lib/libnccl.so
  export NCCL_DEBUG=INFO
  export NCCL_DEBUG_SUBSYS=INIT,ENV
  export NCCL_ALGO=MSCCL,RING,TREE
  export XML_DIR=$TCCL_AEC_ROOT/refs/msccl-exp-synthesis-plans/{target_machine.upper()}.0.LOC.0
else
  echo "Unknown backend"
  exit 1
fi

GPUS_PER_NODE=$GPUS_PER_NODE
# Change for multinode config
echo "NODELIST="${{SLURM_NODELIST}}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

MASTER_PORT=$MASTER_PORT
NNODES=$NNODES
NODE_RANK=$OMPI_COMM_WORLD_RANK
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

echo `hostname` $NODE_RANK $MASTER_ADDR $MASTER_PORT

VOCAB_FILE={vocab_file}
MERGE_FILE={merge_file}
DATA_PATH={data_path}


DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

PARALLEL_ARGS="
    --sequence-parallel \
    --tensor-model-parallel-size {tp} \
    --pipeline-model-parallel-size {pp} \
    --train-iters 12 \
"

T5_220M_ARGS="
    --encoder-num-layers 12 \
    --decoder-num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --kv-channels 64 \
    --ffn-hidden-size 3072 \
    --encoder-seq-length 512 \
    --decoder-seq-length 128 \
    --max-position-embeddings 512 \
    --vocab-extra-ids 100 \
    --micro-batch-size $MBS \
    --global-batch-size $GBS \
    --lr 0.00015 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16
"

T5_XL_ARGS="
    --sequence-parallel \
    --encoder-num-layers 12 \
    --decoder-num-layers 12 \
    --hidden-size 2064 \
    --num-attention-heads 24 \
    --encoder-seq-length 512 \
    --decoder-seq-length 512 \
    --max-position-embeddings 1024 \
    --vocab-extra-ids 100 \
    --micro-batch-size $MBS \
    --global-batch-size $GBS \
    --lr 0.00015 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16
"

T5_LARGE_ARGS="
    --sequence-parallel \
    --encoder-num-layers 12 \
    --decoder-num-layers 12 \
    --hidden-size 1536 \
    --num-attention-heads 16 \
    --encoder-seq-length 512 \
    --decoder-seq-length 512 \
    --max-position-embeddings 1024 \
    --vocab-extra-ids 100 \
    --micro-batch-size $MBS \
    --global-batch-size $GBS \
    --lr 0.00015 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16
"

GPT_175B_ARGS="
    --sequence-parallel \
    --num-layers 96 \
    --hidden-size 12288 \
    --num-attention-heads 96 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size $MBS \
    --global-batch-size $GBS \
    --lr 0.00015 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16
"

GPT_2_7B_ARGS="
    --sequence-parallel \
    --num-layers 32 \
    --hidden-size 2560 \
    --num-attention-heads 32 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size $MBS \
    --global-batch-size $GBS \
    --lr 0.00015 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16
"

GPT_XL_ARGS="
    --sequence-parallel \
    --num-layers 24 \
    --hidden-size 2064 \
    --num-attention-heads 24 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size $MBS \
    --global-batch-size $GBS \
    --lr 0.00015 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16
"

GPT_LARGE_ARGS="
    --sequence-parallel \
    --num-layers 24 \
    --hidden-size 1536 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size $MBS \
    --global-batch-size $GBS \
    --lr 0.00015 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16
"

GPT_345M_ARGS="
    --sequence-parallel \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size $MBS \
    --global-batch-size $GBS \
    --lr 0.00015 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --split 949,50,1 \
"

OUTPUT_ARGS="
    --log-interval 3 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 0 \
"

MISC_ARGS="
"

LBL_ARGS="
"

cd $TCCL_AEC_ROOT/Megatron-LM-23.05
torchrun $DISTRIBUTED_ARGS $SCRIPT_NAME \
    ${model_args} \
    $PARALLEL_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $MISC_ARGS \
    $LBL_ARGS
"""
  return mpi_script, torch_script

def run(cmd, text=True, returncode=False):
  p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=text)
  output, errput = p.communicate()
  if returncode:
    return (output, errput, p.returncode)
  else:
    return (output, errput)

def main(args):
  # check args.output_dir exits, if not, create it
  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
  target_machine = args.target_machine
  num_nodes, nodes, tp_pp_list, partition = get_machine_configuration(target_machine)
  for num_node in num_nodes:
    for devices in DEVICES:
      num_gpu = len(devices.split(','))
      if num_node == 1 and num_gpu == 1: continue
      nodes_to_use = nodes[:num_node]
      for tp, pp in tp_pp_list:
        for libname in ['tccl', 'nccl', 'msccl']:
          for model in ['gpt', 'bert', 't5']:
            tag_list = [libname, target_machine, num_node, devices, model, tp, pp]
            print(f'Running libname={libname} cluster={target_machine} num_node={num_node} CUDA_VISIBLE_DEVICES={devices} model={model} tp={tp}')
            tag = '.'.join([str(arg) for arg in tag_list])
            mpi_script_fn = f'{args.output_dir}/{tag}_mpi.sh' 
            torch_script_fn = f'{args.output_dir}/{tag}_torch.sh' 

            mpi_script, torch_script = gen_script(args, torch_script_fn, nodes_to_use, num_gpu, libname, target_machine, num_node, devices, model, tp, pp, partition, mbs=2, accum=20)

            with open(mpi_script_fn, 'w') as f:
              f.write(mpi_script)
            run(f'chmod +x {mpi_script_fn}')
            with open(torch_script_fn, 'w') as f:
              f.write(torch_script)
            run(f'chmod +x {torch_script_fn}')

            print(f'Start at {time.ctime(time.time())}')
            output, errput, rc = run(f'{mpi_script_fn}', returncode=True)
            print(f'End at {time.ctime(time.time())}')
            errmsg = ''
            if 'NCCL WARN' in output:
              errmsg = [line.strip() for line in output.split('\n') if 'NCCL WARN' in line][0]
            print(f'returncode={rc} errmsg={errmsg}')
            output_fn = f'{args.output_dir}/{tag}.out' 
            errput_fn = f'{args.output_dir}/{tag}.err' 
            with open(output_fn, 'w') as f:
              f.write(output)
            with open(errput_fn, 'w') as f:
              f.write(errput)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--output-dir', type=str, required=True)
  parser.add_argument('--target-machine', type=str, required=True, choices=['a', 'b', 'd'])
  args = parser.parse_args()
  main(args)