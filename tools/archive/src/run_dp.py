import argparse, copy
import os, json

def run_dp_to_find_ring():
  with open(os.path.expanduser('~/tccl_results/multi_node_full_7000/sorted_db.json')) as f:
    sorted_db = json.load(f)
  
  single_memo = {}
  inter_set = set()
  for path in sorted_db:
    path_sig = []
    for job in path['jobs']:
      if job['type'] == 'NIC_READ_GPUMEM':
        path_sig.append(f'NIC;R;G{job["gpumem_idx"]}')
      if job['type'] == 'NIC_WRITE_GPUMEM':
        path_sig.append(f'NIC;W;G{job["gpumem_idx"]}')
      if job['type'] == 'NIC_READ_CPUMEM':
        path_sig.append(f'NIC;R;C{job["cpumem_numa_idx"]}')
      if job['type'] == 'NIC_WRITE_CPUMEM':
        path_sig.append(f'NIC;W;C{job["cpumem_numa_idx"]}')
      if job['type'] == 'GPU_READ_GPUMEM_MEMCPY':
        path_sig.append(f'G{job["gpu_idx"]};R;G{job["gpumem_idx"]}')
      if job['type'] == 'GPU_WRITE_GPUMEM_MEMCPY':
        path_sig.append(f'G{job["gpu_idx"]};W;G{job["gpumem_idx"]}')
      if job['type'] == 'GPU_READ_CPUMEM_MEMCPY':
        path_sig.append(f'G{job["gpu_idx"]};R;C{job["cpumem_numa_idx"]}')
      if job['type'] == 'GPU_WRITE_CPUMEM_MEMCPY':
        path_sig.append(f'G{job["gpu_idx"]};W;C{job["cpumem_numa_idx"]}')
    path_sig = tuple(path_sig)
    head = path_sig[:2]
    inter_set.add(head)
    tail = path_sig[-2:]
    inter_set.add(tail)
    state = (head, tail)
    if state in single_memo:
      if single_memo[state]['us'] > path['us']:
        single_memo[state] = {
          "us": path['us'],
          "path": path_sig
        }
    else:
      single_memo[state] = {
        "us": path['us'],
        "path": path_sig
      }

  multi_memo = copy.deepcopy(single_memo)
  num_node = 6
  for n in range(2, num_node + 1):
    new_memo = {}
    for head in inter_set:
      for tail in inter_set:
        for mid in inter_set:
          if not (head, mid) in multi_memo:
            continue
          if not (mid, tail) in single_memo:
            continue
          state = (head, tail)
          new_us = max(multi_memo[(head, mid)]['us'], single_memo[(mid, tail)]['us'])
          if not state in new_memo or new_memo[state]['us'] > new_us:
            new_memo[state] = {
              "us": new_us,
              "path": multi_memo[(head, mid)]['path'] + single_memo[(mid, tail)]['path'][2:]
            }
    multi_memo = new_memo

  ring_memo = {}
  for head in inter_set:
    state = (head, head)
    if not state in multi_memo:
      continue
    if not state in ring_memo or ring_memo[state]['us'] > multi_memo[state]['us']:
      ring_memo[state] = multi_memo[state]

  res = sorted(ring_memo.values(), key=lambda x: x['us'])
  for path in res:
    print(path)
  return res

def main():
  parser = argparse.ArgumentParser()
  #parser.add_argument('--workspace', type=str, required=True, help='Path to the workspace')
  parser.add_argument('--dry-run', action='store_true', help='Does not actually run the benchmark; only generate conf files.')
  args = parser.parse_args()

  #workspace = os.path.abspath(args.workspace)

if __name__ == '__main__':
    main()