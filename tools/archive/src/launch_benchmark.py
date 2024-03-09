import json
import argparse
import copy
import os
import stat
from typing import Tuple, List
from .conf import annotate_conf, generate_mpirun_script

def generate_nccl_conf(gdr_level: str, gdr_read: int, devices: List[int], algo: str, hosts: str):
  gpu_to_numa = {0: 3, 1: 1, 2: 1, 3: 0}
  nbytes = 64 * 1024 ** 2
  assert algo == 'Ring'
  jobs = []
  hostlist = []
  new_config = {
    "niters": 10,
    "validation": False,
    "output_fn": "result.json",
    "jobs": jobs
  }
  num_hosts = len(hosts)
  num_devices = len(devices)
  nic_write_gpu = gdr_level == 'SYS'
  nic_read_gpu = gdr_level == 'SYS' and gdr_read == 1
  rank_to_assign = 0
  for host_idx in range(num_hosts):
    for device_idx in range(num_devices):
      # if NIC writes to CPUMEM, the first GPU should read from the CPUMEM
      if device_idx == 0 and not nic_write_gpu:
        new_job = {
          "type": "GPU_READ_CPUMEM_MEMCPY",
          "gpu_idx": devices[device_idx],
          "cpumem_numa_idx": gpu_to_numa[devices[device_idx]],
          "nbytes": nbytes,
          "rank": rank_to_assign
        }
        rank_to_assign += 1
        jobs.append(new_job)
        hostlist.append(hosts[host_idx])
      # if the GPU is not the last GPU, the GPU should write to the next GPU
      if device_idx != num_devices - 1:
        new_job = {
          "type": "GPU_WRITE_GPUMEM_MEMCPY",
          "gpu_idx": devices[device_idx],
          "gpumem_idx": devices[device_idx + 1],
          "nbytes": nbytes,
          "rank": rank_to_assign
        }
        rank_to_assign += 1
        jobs.append(new_job)
        hostlist.append(hosts[host_idx])
      # if NIC reads from CPUMEM, the last GPU should write to the CPUMEM
      if device_idx == num_devices - 1 and not nic_read_gpu:
        new_job = {
          "type": "GPU_WRITE_CPUMEM_MEMCPY",
          "gpu_idx": devices[device_idx],
          "cpumem_numa_idx": gpu_to_numa[devices[device_idx]],
          "nbytes": nbytes,
          "rank": rank_to_assign
        }
        rank_to_assign += 1
        jobs.append(new_job)
        hostlist.append(hosts[host_idx])
    rank = rank_to_assign
    peer_rank = rank_to_assign + 1
    rank_to_assign += 2
    if nic_read_gpu:
      new_job = {
        "type": "NIC_READ_GPUMEM",
        "nic_idx": 0,
        "gpumem_idx": devices[num_devices - 1],
        "nbytes": nbytes,
        "rank": rank,
        "peer_rank": peer_rank
      }
    else:
      new_job = {
        "type": "NIC_READ_CPUMEM",
        "nic_idx": 0,
        "cpumem_numa_idx": gpu_to_numa[devices[num_devices - 1]],
        "nbytes": nbytes,
        "rank": rank,
        "peer_rank": peer_rank
      }
    jobs.append(new_job)
    hostlist.append(hosts[host_idx])
    if nic_write_gpu:
      new_job = {
        "type": "NIC_WRITE_GPUMEM",
        "nic_idx": 0,
        "gpumem_idx": devices[0],
        "nbytes": nbytes,
        "rank": peer_rank,
        "peer_rank": rank
      }
    else:
      new_job = {
        "type": "NIC_WRITE_CPUMEM",
        "nic_idx": 0,
        "cpumem_numa_idx": gpu_to_numa[devices[0]],
        "nbytes": nbytes,
        "rank": peer_rank,
        "peer_rank": rank
      }
    jobs.append(new_job)
    hostlist.append(hosts[(host_idx + 1) % num_hosts])
  return new_config, hostlist


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--conf', type=str, help='Path to the configuration file')
  parser.add_argument('--output-conf', type=str, required=True, help='Path to the output configuration file')
  parser.add_argument('--output-script', type=str, required=True, help='Path to the mpirun launch script')
  parser.add_argument('--result', type=str, required=True, help='Path to the result json')
  parser.add_argument('--nccl', action='store_true', help='If given, generate NCCL-like pattern')
  parser.add_argument('--nccl-net-gdr-level', type=str, help='LOC or SYS')
  parser.add_argument('--nccl-net-gdr-read', type=int, help='0 or 1')
  parser.add_argument('--nccl-devices', type=lambda x: list(map(int, x.split(','))), default=[], help='e.g., --nccl-devices 0,1,2,3')
  parser.add_argument('--nccl-algo', type=str, help='Tree or Ring')
  parser.add_argument('--nccl-hosts', type=lambda x: x.split(','), default=[], help='e.g., --nccl-hosts a0,a1')

  args = parser.parse_args()
  
  if args.nccl:
    assert args.nccl_net_gdr_level in ['LOC', 'SYS']
    if args.nccl_net_gdr_level == 'SYS':
      assert args.nccl_net_gdr_read in [0, 1]
    assert args.nccl_algo in ['Ring']
    assert args.nccl_devices != []
    assert args.nccl_hosts != []
    new_config, hosts = generate_nccl_conf(args.nccl_net_gdr_level, args.nccl_net_gdr_read, args.nccl_devices, args.nccl_algo, args.nccl_hosts)
  else:
    assert args.conf is not None
    with open(args.conf) as f:
      config = json.load(f)
    new_config, hosts = annotate_conf(config)
  new_config['output_fn'] = args.result
  mpirun_script = generate_mpirun_script(hosts)

  with open(args.output_conf, 'w') as f:
    json.dump(new_config, f, indent=2)

  script_name = args.output_script
  with open(script_name, 'w') as f:
    f.write(mpirun_script)
  st = os.stat(script_name)
  os.chmod(script_name, st.st_mode | stat.S_IEXEC)

if __name__ == '__main__':
  main()