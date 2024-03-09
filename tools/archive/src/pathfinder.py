import argparse, copy
import os
import itertools
import json
from tqdm import tqdm
from conf import launch_benchmark_from_conf
from topology import convert_pair_up_to_symmetry

num_gpu = 4
num_numa = 4
hostname_to_sym = {
  'a3': 'here',
  'a4': 'prev',
  'a5': 'next',
}
sym_to_hostname = {
  'here': 'a3',
  'prev': 'a4',
  'next': 'a5',
}
nbytes = 64 * 1024**2
niters = 10

#us_threshold = 8000
#us_threshold = 5700
us_threshold = 7000

def synthesize_path():
  # Prepare congestion db
  db = {}
  with open(os.path.expanduser('~/tccl_results/multi_node_pair/sorted_db.json')) as f:
    for congestion in json.load(f):
      jobs = []
      for job_idx, job in enumerate(congestion['jobs']):
        if job['type'].startswith('NIC_READ'):
          src_job = congestion['jobs'][job_idx]
          dst_job = congestion['jobs'][job_idx + 1]
          if src_job['device'] == 'gpumem':
            src = {
              "host": hostname_to_sym[src_job['host']],
              "nic_idx": src_job['nic_idx'],
              "gpumem_idx": src_job['gpumem_idx'],
            }
          else:
            src = {
              "host": hostname_to_sym[src_job['host']],
              "nic_idx": src_job['nic_idx'],
              "cpumem_numa_idx": src_job['cpumem_numa_idx'],
            }
          if dst_job['device'] == 'gpumem':
            dst = {
              "host": hostname_to_sym[dst_job['host']],
              "nic_idx": dst_job['nic_idx'],
              "gpumem_idx": dst_job['gpumem_idx'],
            }
          else:
            dst = {
              "host": hostname_to_sym[dst_job['host']],
              "nic_idx": dst_job['nic_idx'],
              "cpumem_numa_idx": dst_job['cpumem_numa_idx'],
            }
          jobs.append({
            "type": "NIC_SENDRECV",
            "peers": (src, dst)
          })
        elif job['type'].startswith('NIC_WRITE'):
          pass
        else:
          jobs.append(job)
      assert len(jobs) == 2
      sym_pair = convert_pair_up_to_symmetry(jobs)
      if sym_pair in db:
        print(f"Duplicated pair detected")
        print(f"Old {db[sym_pair]['jobs']} {db[sym_pair]['us']} us")
        print(f"New {jobs} {congestion['us']} us")
        #assert False
      else:
        db[sym_pair] = {
          'us': congestion["us"],
          'jobs': jobs,
        }
      if congestion['us'] == 5319:
        print(sym_pair)
      if sym_pair == eval("((0, (0, 'gpu', 0), (1, 'cpu', 0)), (2, (2, 'gpu', 0), (3, 'gpu', 0)))"):
        print(f'Found!')

  path_db = []
  path_candidates = []
  for perm in tqdm(itertools.permutations(list(range(num_gpu))), desc='perm'):
    # iterate permutations up to rotation
    gpu_order = perm
    for transfer_types in itertools.product(list(range(num_numa + 2)), repeat=num_gpu - 1):
      # 0 ~ num_numa - 1 : GPU A writes to memory in NUMA C, then GPU B reads from the memory
      # num_numa: GPU A writes to GPU B
      # num_numa + 1: GPU B reads from GPU A
      # TODO
      if 1 in transfer_types or 2 in transfer_types or 3 in transfer_types:
        continue
      jobs = []
      for tidx, ttype in enumerate(transfer_types):
        if ttype < num_numa:
          jobs.append({
            "type": "GPU_WRITE_CPUMEM_MEMCPY",
            "host": "here",
            "gpu_idx": gpu_order[tidx],
            "cpumem_numa_idx": ttype,
            "nbytes": nbytes
          })
          jobs.append({
            "type": "GPU_READ_CPUMEM_MEMCPY",
            "host": "here",
            "gpu_idx": gpu_order[(tidx + 1) % num_gpu],
            "cpumem_numa_idx": ttype,
            "nbytes": nbytes
          })
        elif ttype == num_numa:
          jobs.append({
            "type": "GPU_WRITE_GPUMEM_MEMCPY",
            "host": "here",
            "gpu_idx": gpu_order[tidx],
            "gpumem_idx": gpu_order[(tidx + 1) % num_gpu],
            "nbytes": nbytes
          })
        elif ttype == num_numa + 1:
          jobs.append({
            "type": "GPU_READ_GPUMEM_MEMCPY",
            "host": "here",
            "gpu_idx": gpu_order[(tidx + 1) % num_gpu],
            "gpumem_idx": gpu_order[tidx],
            "nbytes": nbytes
          })
        else:
          assert False
      num_job = len(jobs)
      us_lower_limit = 0
      for i in range(num_job):
        for j in range(i + 1, num_job):
          pair = (jobs[i], jobs[j])
          sym_pair = convert_pair_up_to_symmetry(pair)
          if sym_pair in db:
            us_lower_limit = max(us_lower_limit, db[sym_pair]['us'])
          else:
            print("Pair not found in DB, something wrong: ", sym_pair)
            assert False
      #print(f'Path {gpu_order} TransferType {transfer_types} LowThr {us_lower_limit} us')
      path_db.append({
        'gpu_order': gpu_order,
        'transfer_types': transfer_types,
        'us_lower_limit': us_lower_limit,
      })
      if us_lower_limit < us_threshold:
        path_candidates.append({
          'gpu_order': gpu_order,
          'transfer_types': transfer_types,
          'us_lower_limit': us_lower_limit,
          'jobs': jobs,
        })
  #path_db = sorted(path_db, key=lambda x: x['us_lower_limit'])
  #for path in path_db:
  #  print(f'Path {path["gpu_order"]} TransferType {path["transfer_types"]} LowThr {path["us_lower_limit"]} us')

  print(f'Number of path candidates: {len(path_candidates)}')

  #for cand in path_candidates:
  #  print(cand)
  #return

  ########
  # Add inter-node prefix (prev -> here)
  ########
  new_path_candidates = []
  for path in tqdm(path_candidates, desc='inter-node prefix'):
    for prev_transfer_type in range(num_numa + num_gpu):
      # 0 ~ num_numa - 1 : NIC reads from CPU A 
      # TODO
      if prev_transfer_type in [1,2,3]:
        continue
      # num_numa ~ num_numa + num_gpu - 1 : NIC reads from GPU A
      for nic_transfer_type in range(num_numa + 1):
        # 0 ~ num_numa - 1 : NIC writes to CPU A 
        # num_numa : NIC writes to GPU
        # TODO
        if nic_transfer_type in [1,2,3]:
          continue
        new_path = copy.deepcopy(path)
        if prev_transfer_type < num_numa:
          src = {
            "host": "prev",
            "nic_idx": 0,
            "device": "cpumem",
            "cpumem_numa_idx": prev_transfer_type,
          }
        else:
          src = {
            "host": "prev",
            "nic_idx": 0,
            "device": "gpumem",
            "gpumem_idx": prev_transfer_type - num_numa,
          }
        if nic_transfer_type < num_numa:
          dst = {
            "host": "here",
            "nic_idx": 0,
            "device": "cpumem",
            "cpumem_numa_idx": nic_transfer_type,
          }
        else:
          dst = {
            "host": "here",
            "nic_idx": 0,
            "device": "gpumem",
            "gpumem_idx": path["gpu_order"][0],
          }
        new_job = {
          "type": "NIC_SENDRECV",
          "nbytes": nbytes,
          "peers": [src, dst],
        }

        # Check conflict with NIC job
        for job in new_path['jobs']:
          pair = (job, new_job)
          sym_pair = convert_pair_up_to_symmetry(pair)
          if sym_pair in db:
            new_path['us_lower_limit'] = max(new_path['us_lower_limit'], db[sym_pair]['us'])
          else:
            print("Pair not found in DB, something wrong: ", sym_pair)
            assert False
        if new_path['us_lower_limit'] >= us_threshold:
          continue
        new_path['jobs'].insert(0, new_job)

        if nic_transfer_type < num_numa:
          # Check conflict with new job
          new_job = {
            "type": "GPU_READ_CPUMEM_MEMCPY",
            "host": "here",
            "gpu_idx": path["gpu_order"][0],
            "cpumem_numa_idx": nic_transfer_type,
            "nbytes": nbytes,
          }
          for job in new_path['jobs']:
            pair = (job, new_job)
            sym_pair = convert_pair_up_to_symmetry(pair)
            if sym_pair in db:
              new_path['us_lower_limit'] = max(new_path['us_lower_limit'], db[sym_pair]['us'])
            else:
              print("Pair not found in DB, something wrong: ", sym_pair)
              assert False
          if new_path['us_lower_limit'] >= us_threshold:
            continue
          new_path['jobs'].insert(1, new_job)

        new_path_candidates.append(new_path)
  path_candidates = new_path_candidates
  print(f'Number of path candidates: {len(path_candidates)}')

  ########
  # Add inter-node suffix (here -> next)
  ########
  new_path_candidates = []
  for path in tqdm(path_candidates, desc='inter-node suffix'):
    for next_transfer_type in range(num_numa + num_gpu):
      # 0 ~ num_numa - 1 : NIC writes to CPU A 
      # num_numa ~ num_numa + num_gpu - 1 : NIC writes to GPU A
      # TODO
      if next_transfer_type in [1,2,3]:
        continue
      for nic_transfer_type in range(num_numa + 1):
        # 0 ~ num_numa - 1 : NIC writes to CPU A 
        # num_numa : NIC writes to GPU
        # TODO
        if nic_transfer_type in [1,2,3]:
          continue
        new_path = copy.deepcopy(path)
        if next_transfer_type < num_numa:
          dst = {
            "host": "next",
            "nic_idx": 0,
            "device": "cpumem",
            "cpumem_numa_idx": next_transfer_type,
          }
        else:
          dst = {
            "host": "next",
            "nic_idx": 0,
            "device": "gpumem",
            "gpumem_idx": next_transfer_type - num_numa,
          }
        if nic_transfer_type < num_numa:
          src = {
            "host": "here",
            "nic_idx": 0,
            "device": "cpumem",
            "cpumem_numa_idx": nic_transfer_type,
          }
        else:
          src = {
            "host": "here",
            "nic_idx": 0,
            "device": "gpumem",
            "gpumem_idx": path["gpu_order"][-1],
          }
        new_job = {
          "type": "NIC_SENDRECV",
          "nbytes": nbytes,
          "peers": [src, dst],
        }

        # Check conflict with NIC job
        for job in new_path['jobs']:
          pair = (job, new_job)
          sym_pair = convert_pair_up_to_symmetry(pair)
          if sym_pair in db:
            new_path['us_lower_limit'] = max(new_path['us_lower_limit'], db[sym_pair]['us'])
          else:
            print("Pair not found in DB, something wrong: ", sym_pair)
            assert False
        if new_path['us_lower_limit'] >= us_threshold:
          continue
        new_path['jobs'].append(new_job)

        if nic_transfer_type < num_numa:
          # Check conflict with new job
          new_job = {
            "type": "GPU_WRITE_CPUMEM_MEMCPY",
            "host": "here",
            "gpu_idx": path["gpu_order"][-1],
            "cpumem_numa_idx": nic_transfer_type,
            "nbytes": nbytes,
          }
          for job in new_path['jobs']:
            pair = (job, new_job)
            sym_pair = convert_pair_up_to_symmetry(pair)
            if sym_pair in db:
              new_path['us_lower_limit'] = max(new_path['us_lower_limit'], db[sym_pair]['us'])
            else:
              print("Pair not found in DB, something wrong: ", sym_pair)
              assert False
          if new_path['us_lower_limit'] >= us_threshold:
            continue
          new_path['jobs'].insert(-1, new_job)

        new_path_candidates.append(new_path)
  path_candidates = new_path_candidates
  print(f'Number of path candidates: {len(path_candidates)}')

  return path_candidates

def rematerialize_path(path, workspace, dry_run):
  config = copy.deepcopy(path)
  for job in config['jobs']:
    if job['type'] == 'NIC_SENDRECV':
      for peer in job['peers']:
        peer['host'] = sym_to_hostname[peer['host']]
    else:
      job['host'] = sym_to_hostname[job['host']]
  config['niters'] = 10
  config['validation'] = False
  launch_benchmark_from_conf(config, workspace, dry_run, check_dup=True)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--workspace', type=str, required=True, help='Path to the workspace')
  parser.add_argument('--dry-run', action='store_true', help='Does not actually run the benchmark; only generate conf files.')
  args = parser.parse_args()

  workspace = os.path.abspath(args.workspace)

  path_candidates = synthesize_path()
  for path in tqdm(path_candidates):
    rematerialize_path(path, workspace, args.dry_run)

if __name__ == '__main__':
  main()