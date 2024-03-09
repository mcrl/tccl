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
from tqdm import tqdm
from conf import annotate_conf, generate_mpirun_script, launch_benchmark_from_conf
from topology import convert_pair_up_to_symmetry, symbol_to_gpu_idx, symbol_to_cpu_idx, symbol_to_job_type

nbytes = 64 * 1024**2
num_numa = 4
num_gpu = 4

# here, prev, next
numa_to_hostname = ['a3', 'a4', 'a5']

def get_all_sym_pairs():
  # Enumerate all jobs that can appear
  # TODO Remove GPU self writes/reads to itself
  jobs = []
  for i in range(num_gpu):
    for j in range(num_numa):
      jobs.append({
        "type": "GPU_WRITE_CPUMEM_MEMCPY",
        "gpu_idx": i,
        "cpumem_numa_idx": j,
      })
      jobs.append({
        "type": "GPU_READ_CPUMEM_MEMCPY",
        "gpu_idx": i,
        "cpumem_numa_idx": j,
      })
  for i in range(num_gpu):
    for j in range(num_gpu):
      jobs.append({
        "type": "GPU_WRITE_GPUMEM_MEMCPY",
        "gpu_idx": i,
        "gpumem_idx": j,
      })
      jobs.append({
        "type": "GPU_READ_GPUMEM_MEMCPY",
        "gpu_idx": i,
        "gpumem_idx": j,
      })
  # prev to here
  for i in range(num_gpu + num_numa):
    for j in range(num_gpu + num_numa):
      if i < num_gpu:
        src = {
          "host": "prev",
          "nic_idx": 0,
          "gpumem_idx": i,
        }
      else:
        src = {
          "host": "prev",
          "nic_idx": 0,
          "cpumem_numa_idx": i - num_gpu,
        }
      if j < num_gpu:
        dst = {
          "host": "here",
          "nic_idx": 0,
          "gpumem_idx": j,
        }
      else:
        dst = {
          "host": "here",
          "nic_idx": 0,
          "cpumem_numa_idx": j - num_gpu,
        }
      jobs.append({
        "type": "NIC_SENDRECV",
        "peers": (src, dst)
      })
  # here to next
  for i in range(num_gpu + num_numa):
    for j in range(num_gpu + num_numa):
      if i < num_gpu:
        src = {
          "host": "here",
          "nic_idx": 0,
          "gpumem_idx": i,
        }
      else:
        src = {
          "host": "here",
          "nic_idx": 0,
          "cpumem_numa_idx": i - num_gpu,
        }
      if j < num_gpu:
        dst = {
          "host": "next",
          "nic_idx": 0,
          "gpumem_idx": j,
        }
      else:
        dst = {
          "host": "next",
          "nic_idx": 0,
          "cpumem_numa_idx": j - num_gpu,
        }
      jobs.append({
        "type": "NIC_SENDRECV",
        "peers": (src, dst)
      })
  # here to prev (to support 2-node case)
  for i in range(num_gpu + num_numa):
    for j in range(num_gpu + num_numa):
      if i < num_gpu:
        src = {
          "host": "here",
          "nic_idx": 0,
          "gpumem_idx": i,
        }
      else:
        src = {
          "host": "here",
          "nic_idx": 0,
          "cpumem_numa_idx": i - num_gpu,
        }
      if j < num_gpu:
        dst = {
          "host": "prev",
          "nic_idx": 0,
          "gpumem_idx": j,
        }
      else:
        dst = {
          "host": "prev",
          "nic_idx": 0,
          "cpumem_numa_idx": j - num_gpu,
        }
      jobs.append({
        "type": "NIC_SENDRECV",
        "peers": (src, dst)
      })

  num_job = len(jobs)

  # Enumerate all pairs of jobs
  sym_db = set()
  for i in range(num_job):
    for j in range(i + 1, num_job):
      # Some cutting
      gpu_src1, gpu_dst1, gpu_src2, gpu_dst2 = -1, -1, -1, -1
      if jobs[i]['type'] == 'GPU_WRITE_GPUMEM_MEMCPY':
        gpu_src1 = jobs[i]['gpu_idx']
        gpu_dst1 = jobs[i]['gpumem_idx']
      if jobs[i]['type'] == 'GPU_READ_GPUMEM_MEMCPY':
        gpu_dst1 = jobs[i]['gpu_idx']
        gpu_src1 = jobs[i]['gpumem_idx']
      if jobs[i]['type'] == 'GPU_WRITE_CPUMEM_MEMCPY':
        gpu_src1 = jobs[i]['gpu_idx']
      if jobs[i]['type'] == 'GPU_READ_CPUMEM_MEMCPY':
        gpu_dst1 = jobs[i]['gpu_idx']
      if jobs[j]['type'] == 'GPU_WRITE_GPUMEM_MEMCPY':
        gpu_src2 = jobs[j]['gpu_idx']
        gpu_dst2 = jobs[j]['gpumem_idx']
      if jobs[j]['type'] == 'GPU_READ_GPUMEM_MEMCPY':
        gpu_dst2 = jobs[j]['gpu_idx']
        gpu_src2 = jobs[j]['gpumem_idx']
      if jobs[j]['type'] == 'GPU_WRITE_CPUMEM_MEMCPY':
        gpu_src2 = jobs[j]['gpu_idx']
      if jobs[j]['type'] == 'GPU_READ_CPUMEM_MEMCPY':
        gpu_dst2 = jobs[j]['gpu_idx']
      if gpu_src1 == gpu_src2 and gpu_src1 != -1:
        continue
      if gpu_dst1 == gpu_dst2 and gpu_dst1 != -1:
        continue
      sym_pair = convert_pair_up_to_symmetry((jobs[i], jobs[j]))
      sym_db.add(sym_pair)

  for sym_pair in sorted(list(sym_db)):
    print(sym_pair)
  print("Total number of jobs:", num_job)
  print(f'Total number of sym pairs: {len(sym_db)}')
  return sym_db

def rematerialize_sym_pair(sym_pair, workspace, dry_run):
  dev_cnt = [{'gpu': -1, 'cpu': -1, 'nic': -1} for _ in range(num_numa * 3)]

  # max dev idx
  for job in sym_pair:
    for dev in job[1:]:
      dev_cnt[dev[0]][dev[1]] = max(dev_cnt[dev[0]][dev[1]], dev[2])

  max_dev_cnt = [
    {'gpu': 1, 'cpu': 1, 'nic': 1},
    {'gpu': 2, 'cpu': 1, 'nic': 0},
    {'gpu': 0, 'cpu': 1, 'nic': 0},
    {'gpu': 1, 'cpu': 1, 'nic': 0},
  ] * 3

  # Find NUMA mapping
  # heuristics mapping does not work TT...
  #perm = [-1] * (num_numa * 3)
  #for i in range(num_numa * 3):
  #  # 1. nic should go numa 0
  #  if dev_cnt[i]['nic'] >= 0:
  #    perm[i] = i // num_numa * num_numa + 0
  #  # 2. 2 GPU should go numa 1
  #  if dev_cnt[i]['gpu'] >= 1:
  #    perm[i] = i // num_numa * num_numa + 1
  #for i in range(num_numa * 3):
  #  if perm[i] == -1:
  #    numa_base = i // num_numa * num_numa
  #    for j in range(numa_base, numa_base + num_numa):
  #      if not j in perm:
  #        if not (dev_cnt[i]['gpu'] >= max_dev_cnt[j]['gpu'] \
  #            or dev_cnt[i]['cpu'] >= max_dev_cnt[j]['cpu'] \
  #            or dev_cnt[i]['nic'] >= max_dev_cnt[j]['nic']):
  #          perm[i] = j
  #          break

  for pp in itertools.product(itertools.permutations(range(num_numa)), \
                    itertools.permutations(range(num_numa, num_numa * 2)), \
                    itertools.permutations(range(num_numa * 2, num_numa * 3))):
    perm = list(pp[0]) + list(pp[1]) + list(pp[2])
    flag = True
    for sym_numa_idx in range(num_numa * 3):
      if dev_cnt[sym_numa_idx]['gpu'] >= max_dev_cnt[perm[sym_numa_idx]]['gpu'] \
          or dev_cnt[sym_numa_idx]['cpu'] >= max_dev_cnt[perm[sym_numa_idx]]['cpu'] \
          or dev_cnt[sym_numa_idx]['nic'] >= max_dev_cnt[perm[sym_numa_idx]]['nic']:
        flag = False
        break
    if flag:
      break

  if sorted(set(perm)) != list(range(num_numa * 3)):
    assert False

  # Check perm
  flag = True
  for sym_numa_idx in range(num_numa * 3):
    if dev_cnt[sym_numa_idx]['gpu'] >= max_dev_cnt[perm[sym_numa_idx]]['gpu'] \
        or dev_cnt[sym_numa_idx]['cpu'] >= max_dev_cnt[perm[sym_numa_idx]]['cpu'] \
        or dev_cnt[sym_numa_idx]['nic'] >= max_dev_cnt[perm[sym_numa_idx]]['nic']:
      print(sym_numa_idx)
      flag = False
      break
  if not flag:
    print(perm)
    print(dev_cnt)
    print(max_dev_cnt)
    print(sym_pair)
    assert False

  jobs = []
  for sym_job in sym_pair:
    if sym_job[0] == 0:
      jobs.append({
        "type": symbol_to_job_type(sym_job[0]),
        "host": numa_to_hostname[sym_job[1][0] // num_numa],
        "gpu_idx": symbol_to_gpu_idx(perm[sym_job[1][0]], sym_job[1][2]),
        "cpumem_numa_idx": symbol_to_cpu_idx(perm[sym_job[2][0]]),
        "nbytes": nbytes
      })
    if sym_job[0] == 1:
      jobs.append({
        "type": symbol_to_job_type(sym_job[0]),
        "host": numa_to_hostname[sym_job[1][0] // num_numa],
        "gpu_idx": symbol_to_gpu_idx(perm[sym_job[1][0]], sym_job[1][2]),
        "cpumem_numa_idx": symbol_to_cpu_idx(perm[sym_job[2][0]]),
        "nbytes": nbytes
      })
    if sym_job[0] == 2:
      jobs.append({
        "type": symbol_to_job_type(sym_job[0]),
        "host": numa_to_hostname[sym_job[1][0] // num_numa],
        "gpu_idx": symbol_to_gpu_idx(perm[sym_job[1][0]], sym_job[1][2]),
        "gpumem_idx": symbol_to_gpu_idx(perm[sym_job[2][0]], sym_job[2][2]),
        "nbytes": nbytes
      })
    if sym_job[0] == 3:
      jobs.append({
        "type": symbol_to_job_type(sym_job[0]),
        "host": numa_to_hostname[sym_job[1][0] // num_numa],
        "gpu_idx": symbol_to_gpu_idx(perm[sym_job[1][0]], sym_job[1][2]),
        "gpumem_idx": symbol_to_gpu_idx(perm[sym_job[2][0]], sym_job[2][2]),
        "nbytes": nbytes
      })
    if sym_job[0] == 4:
      if sym_job[1][1] == 'cpu':
        src = {
          "host": numa_to_hostname[perm[sym_job[1][0]] // num_numa],
          "nic_idx": 0,
          "device": "cpumem",
          "cpumem_numa_idx": symbol_to_cpu_idx(perm[sym_job[1][0]]),
        }
      if sym_job[1][1] == 'gpu':
        src = {
          "host": numa_to_hostname[perm[sym_job[1][0]] // num_numa],
          "nic_idx": 0,
          "device": "gpumem",
          "gpumem_idx": symbol_to_gpu_idx(perm[sym_job[1][0]], sym_job[1][2]),
        }
      if sym_job[3][1] == 'cpu':
        dst = {
          "host": numa_to_hostname[perm[sym_job[3][0]] // num_numa],
          "nic_idx": 0,
          "device": "cpumem",
          "cpumem_numa_idx": symbol_to_cpu_idx(perm[sym_job[3][0]]),
        }
      if sym_job[3][1] == 'gpu':
        dst = {
          "host": numa_to_hostname[perm[sym_job[3][0]] // num_numa],
          "nic_idx": 0,
          "device": "gpumem",
          "gpumem_idx": symbol_to_gpu_idx(perm[sym_job[3][0]], sym_job[3][2]),
        }
      jobs.append({
        "type": symbol_to_job_type(sym_job[0]),
        "nbytes": nbytes,
        "peers": [src, dst],
      })
  config = {
    "niters": 10,
    "validation": False,
    "jobs": jobs
  }
  #if sym_pair == eval("((0, (0, 'gpu', 0), (1, 'cpu', 0)), (2, (2, 'gpu', 0), (3, 'gpu', 0)))"):
  #  print("Found!")
  #  print(perm)
  #  print(dev_cnt)
  #  print(max_dev_cnt)
  #  print(sym_pair)
  #  print(config)
  launch_benchmark_from_conf(config, workspace, dry_run, check_dup=True)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--workspace', type=str, required=True, help='Path to the workspace')
  parser.add_argument('--dry-run', action='store_true', help='Does not actually run the benchmark; only generate conf files.')
  args = parser.parse_args()

  workspace = os.path.abspath(args.workspace)

  sym_db = get_all_sym_pairs()
  for sym_pair in tqdm(sorted(list(sym_db))):
    rematerialize_sym_pair(sym_pair, workspace, args.dry_run)

if __name__ == '__main__':
  main()