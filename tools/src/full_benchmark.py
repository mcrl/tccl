import itertools
import argparse
import os
import tempfile
import json
import stat
import uuid
import subprocess
import sys
from conf import annotate_conf, generate_mpirun_script, launch_benchmark_from_conf

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--workspace', type=str, required=True, help='Path to the workspace')
  parser.add_argument('--dry-run', action='store_true', help='Does not actually run the benchmark; only generate conf files.')
  args = parser.parse_args()

  workspace = os.path.abspath(args.workspace)

  num_gpu = 4
  num_numa = 4
  hostname = "a3"
  nbytes = 64 * 1024**2
  niters = 10
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
      config = {
        "_comment": f'gpu_order: {gpu_order}, transfer_types: {transfer_types}',
        "metadata": {
          "gpu_order": gpu_order,
          "transfer_types": transfer_types
        },
        "niters": 10,
        "validation": False,
        "jobs": jobs
      }
      launch_benchmark_from_conf(config, workspace, args.dry_run)

if __name__ == '__main__':
  main()