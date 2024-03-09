import argparse
import subprocess
import os

NUM_OP = 5
NCCL_API_OPTS = ['allreduce', 'broadcast', 'reduce', 'allgather', 'reducescatter']
NCCL_TESTS_BINS = ['all_reduce_perf', 'broadcast_perf', 'reduce_perf', 'all_gather_perf', 'reduce_scatter_perf']
DEVICES = ['0', '1', '2', '3', '0,1', '0,2', '0,3', '1,2', '1,3', '2,3', '0,1,2', '0,1,3', '0,2,3', '1,2,3', '0,1,2,3']


def get_machine_configuration(machine):
  if machine == 'a':
    num_nodes = list(range(1, 4 + 1))
    partition = 'asplosA'
  elif machine == 'b':
    num_nodes = list(range(1, 6 + 1))
    partition = 'asplosB'
  elif machine == 'd':
    num_nodes = list(range(1, 3 + 1))
    partition = 'asplosD'
  else:
    assert False, 'Unknown machine: %s' % machine
  return num_nodes, partition

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
def gen_script_nccl(args, partition, num_gpu, libname, target_machine, num_node, devices, op_idx):
  txt = ""
  txt += f"""#!/bin/bash
salloc -p {partition} -N {num_node} --exclusive \
mpirun -mca btl ^openib -mca pml ucx {'--bind-to none' if args.bind_none else ''} \
  -npernode {num_gpu} \
  -x LD_PRELOAD=$TCCL_AEC_ROOT/nccl-2.18.3-1/build/lib/libnccl.so \
  -x NCCL_DEBUG=INFO \
  -x NCCL_DEBUG_SUBSYS=INIT,ENV \
  -x CUDA_VISIBLE_DEVICES={devices} \
  $TCCL_AEC_ROOT/nccl-tests-2.13.8/build/{NCCL_TESTS_BINS[op_idx]} \
  -b 1K -e 1G -f 2 -g 1 -c 0 -n 1 -w 1 -G 0 -z 0
"""
  return txt
def gen_script_msccl(args, partition, num_gpu, libname, target_machine, num_node, devices, op_idx):
  txt = ""
  txt += f"""#!/bin/bash
salloc -p {partition} -N {num_node} --exclusive \
mpirun -mca btl ^openib -mca pml ucx {'--bind-to none' if args.bind_none else ''} \
  -x LD_PRELOAD=$TCCL_AEC_ROOT/msccl-0.7.4/build/lib/libnccl.so \
  -npernode {num_gpu} \
  -x NCCL_DEBUG=INFO \
  -x NCCL_DEBUG_SUBSYS=INIT,ENV \
  -x CUDA_VISIBLE_DEVICES={devices} \
  -x NCCL_ALGO=MSCCL,RING,TREE \
  -x MSCCL_XML_FILES=$TCCL_AEC_ROOT/refs/msccl-exp-synthesis-plans/{target_machine.upper()}.0.LOC.0/{NCCL_API_OPTS[op_idx]}.{target_machine.upper()}.{num_node}.{num_gpu}.i3.xml \
  $TCCL_AEC_ROOT/nccl-tests-2.13.8/build/{NCCL_TESTS_BINS[op_idx]} \
  -b 1K -e 1G -f 2 -g 1 -c 0 -n 1 -w 1 -G 0 -z 0
"""
  return txt
def gen_script_tccl(args, partition, num_gpu, libname, target_machine, num_node, devices, op_idx):
  txt = ""
  hosts = ','.join([f'{node}:{num_gpu}' for node in partition])
  if target_machine == 'a':
    xml_fn = f'amd_v100.xml'
  elif target_machine == 'b':
    xml_fn = f'amd_3090.xml'
  elif target_machine == 'd':
    xml_fn = f'intel_v100.xml'
  else:
    assert False, 'Unknown machine: %s' % target_machine
  txt += f"""#!/bin/bash
salloc -p {partition} -N {num_node} --exclusive \
mpirun -mca btl ^openib -mca pml ucx {'--bind-to none' if args.bind_none else ''} \
  -x LD_PRELOAD=$TCCL_AEC_ROOT/tccl/build/lib/libnccl.so \
  -npernode {num_gpu} \
  -x NCCL_DEBUG=INFO \
  -x NCCL_DEBUG_SUBSYS=INIT,ENV,TCCL \
  -x CUDA_VISIBLE_DEVICES={devices} \
  -x NCCL_ALGO=TCCL,RING \
  -x TCCL_XML_FILE=$TCCL_AEC_ROOT/workspace/{xml_fn} \
  $TCCL_AEC_ROOT/nccl-tests-2.13.8/build/{NCCL_TESTS_BINS[op_idx]} \
  -b 1K -e 1G -f 2 -g 1 -c 1 -n 1 -w 1 -G 0 -z 0
"""
  return txt

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
  num_nodes, partition = get_machine_configuration(target_machine)
  for num_node in num_nodes:
    for devices in DEVICES:
      num_gpu = len(devices.split(','))
      if num_node == 1 and num_gpu == 1: continue
      for op_idx in range(NUM_OP):
        for libname in ['nccl', 'msccl', 'tccl']:
          tag_list = [libname, target_machine, num_node, devices, op_idx]
          if libname == 'nccl':
            script = gen_script_nccl(args, partition, num_gpu, *tag_list)
          elif libname == 'msccl':
            script = gen_script_msccl(args, partition, num_gpu, *tag_list)
          elif libname == 'tccl':
            script = gen_script_tccl(args, partition, num_gpu, *tag_list)
          else:
            assert False, f'Unknown libname: {libname}'
          tag = '.'.join([str(arg) for arg in tag_list])
          script_fn = f'{args.output_dir}/{tag}.sh' 
          with open(script_fn, 'w') as f:
            f.write(script)
          run(f'chmod +x {script_fn}')
          print(f'Running lib={libname} partition={partition} num_node={num_node} CUDA_VISIBLE_DEVICES={devices} op={NCCL_API_OPTS[op_idx]}')
          output, errput, rc = run(f'{script_fn}', returncode=True)
          errmsg = ''
          if 'NCCL WARN' in output:
            errmsg = [line.strip() for line in output.split('\n') if 'NCCL WARN' in line][0]
          print(f'==> returncode={rc} errmsg={errmsg}')
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
  parser.add_argument('--bind-none', action='store_true')
  args = parser.parse_args()
  main(args)