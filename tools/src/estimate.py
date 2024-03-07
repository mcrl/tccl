import itertools
import argparse
import os
import tempfile
import json
import stat
import uuid
import subprocess
import sys
import copy
from conf import annotate_conf, generate_mpirun_script, launch_benchmark_from_conf
from topology import convert_pair_up_to_symmetry

hostname = "a2"
nbytes = 64 * 1024**2

def get_all_sym_pairs():
  num_gpu = 4
  num_numa = 4
  hostname = "a3"
  nbytes = 64 * 1024**2
  niters = 10

  num_ring = 0
  sym_db = set()
  for perm in itertools.permutations(list(range(1, num_gpu))):
    # iterate permutations up to rotation
    gpu_order = (0,) + perm
    for transfer_types in itertools.product(list(range(num_numa + 2)), repeat=num_gpu):
      # 0 ~ num_numa - 1 : GPU A writes to memory in NUMA C, then GPU B reads from the memory
      # num_numa: GPU A writes to GPU B
      # num_numa + 1: GPU B reads from GPU A
      jobs = []
      for tidx, ttype in enumerate(transfer_types):
        if ttype < num_numa:
          jobs.append({
            "type": "GPU_WRITE_CPUMEM_MEMCPY",
            "host": hostname,
            "gpu_idx": gpu_order[tidx],
            "cpumem_numa_idx": ttype,
            "nbytes": nbytes
          })
          jobs.append({
            "type": "GPU_READ_CPUMEM_MEMCPY",
            "host": hostname,
            "gpu_idx": gpu_order[(tidx + 1) % num_gpu],
            "cpumem_numa_idx": ttype,
            "nbytes": nbytes
          })
        elif ttype == num_numa:
          jobs.append({
            "type": "GPU_WRITE_GPUMEM_MEMCPY",
            "host": hostname,
            "gpu_idx": gpu_order[tidx],
            "gpumem_idx": gpu_order[(tidx + 1) % num_gpu],
            "nbytes": nbytes
          })
        elif ttype == num_numa + 1:
          jobs.append({
            "type": "GPU_READ_GPUMEM_MEMCPY",
            "host": hostname,
            "gpu_idx": gpu_order[(tidx + 1) % num_gpu],
            "gpumem_idx": gpu_order[tidx],
            "nbytes": nbytes
          })
        else:
          assert False
      num_ring += 1 
      num_job = len(jobs)
      for i in range(num_job):
        for j in range(i + 1, num_job):
          sym_pair = convert_pair_up_to_symmetry([jobs[i], jobs[j]])
          if not sym_pair in sym_db:
            sym_db.add(sym_pair)
  for sym_pair in sorted(list(sym_db)):
    print(sym_pair)
  print(f'Total number of rings: {num_ring}')
  print(f'Total number of sym pairs: {len(sym_db)}')
  return sym_db

def rematerialize_sym_pair(sym_pair, workspace, dry_run):
  num_numa = 4
  dev_cnt = [{'gpu': -1, 'cpu': -1} for _ in range(num_numa)]
  # max dev idx
  dev_cnt[sym_pair[0][1][0]][sym_pair[0][1][1]] = max(dev_cnt[sym_pair[0][1][0]][sym_pair[0][1][1]], sym_pair[0][1][2]) # first job src dev type
  dev_cnt[sym_pair[0][2][0]][sym_pair[0][2][1]] = max(dev_cnt[sym_pair[0][2][0]][sym_pair[0][2][1]], sym_pair[0][2][2]) # first job dst dev type
  dev_cnt[sym_pair[1][1][0]][sym_pair[1][1][1]] = max(dev_cnt[sym_pair[1][1][0]][sym_pair[1][1][1]], sym_pair[1][1][2]) # second job src dev type
  dev_cnt[sym_pair[1][2][0]][sym_pair[1][2][1]] = max(dev_cnt[sym_pair[1][2][0]][sym_pair[1][2][1]], sym_pair[1][2][2]) # second job dst dev type

  max_dev_cnt = [
    {'gpu': 1, 'cpu': 1},
    {'gpu': 2, 'cpu': 1},
    {'gpu': 0, 'cpu': 1},
    {'gpu': 1, 'cpu': 1},
  ]
  for perm in itertools.permutations(list(range(num_numa))):
    flag = True
    for sym_numa_idx in range(num_numa):
      if dev_cnt[sym_numa_idx]['gpu'] >= max_dev_cnt[perm[sym_numa_idx]]['gpu'] \
          or dev_cnt[sym_numa_idx]['cpu'] >= max_dev_cnt[perm[sym_numa_idx]]['cpu']:
        flag = False
        break
    if flag:
      break
  else:
    print(perm)
    print(dev_cnt)
    print(max_dev_cnt)
    print(sym_pair)
    assert False

  jobs = []
  for sym_job in sym_pair:
    if sym_job[0] == 0:
      jobs.append({
        "type": symbol_to_job_type[sym_job[0]],
        "host": hostname,
        "gpu_idx": symbol_to_gpu_idx[(perm[sym_job[1][0]], sym_job[1][2])],
        "cpumem_numa_idx": perm[sym_job[2][0]],
        "nbytes": nbytes
      })
    if sym_job[0] == 1:
      jobs.append({
        "type": symbol_to_job_type[sym_job[0]],
        "host": hostname,
        "gpu_idx": symbol_to_gpu_idx[(perm[sym_job[1][0]], sym_job[1][2])],
        "cpumem_numa_idx": perm[sym_job[2][0]],
        "nbytes": nbytes
      })
    if sym_job[0] == 2:
      jobs.append({
        "type": symbol_to_job_type[sym_job[0]],
        "host": hostname,
        "gpu_idx": symbol_to_gpu_idx[(perm[sym_job[1][0]], sym_job[1][2])],
        "gpumem_idx": symbol_to_gpu_idx[(perm[sym_job[2][0]], sym_job[2][2])],
        "nbytes": nbytes
      })
    if sym_job[0] == 3:
      jobs.append({
        "type": symbol_to_job_type[sym_job[0]],
        "host": hostname,
        "gpu_idx": symbol_to_gpu_idx[(perm[sym_job[1][0]], sym_job[1][2])],
        "gpumem_idx": symbol_to_gpu_idx[(perm[sym_job[2][0]], sym_job[2][2])],
        "nbytes": nbytes
      })
  config = {
    "niters": 10,
    "validation": False,
    "jobs": jobs
  }
  launch_benchmark_from_conf(config, workspace, dry_run)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--workspace', type=str, required=True, help='Path to the workspace')
  parser.add_argument('--dry-run', action='store_true', help='Does not actually run the benchmark; only generate conf files.')
  args = parser.parse_args()

  workspace = os.path.abspath(args.workspace)

  sym_db = get_all_sym_pairs()
  for sym_pair in sorted(list(sym_db)):
    rematerialize_sym_pair(sym_pair, workspace, args.dry_run)

if __name__ == '__main__':
  main()