from typing import Tuple, List
import copy, glob
import uuid, json, os, stat, subprocess

# Annotate the given configuration with actual rank and peer_rank.
# Return the new configuration and a list of hosts in rank order.
def annotate_conf(config: dict) -> Tuple[dict, list]:
  rank_to_assign = 0
  new_jobs = []
  hosts = []
  for job in config['jobs']:
    if job['type'] == 'NIC_SENDRECV':
      # 2-process job
      send_rank = rank_to_assign
      rank_to_assign += 1
      recv_rank = rank_to_assign
      rank_to_assign += 1
      for idx, peer in enumerate(job['peers']):
        new_job = copy.deepcopy(job)
        new_job.pop('peers')
        if idx == 0:
          new_job['rank'] = send_rank
          new_job['peer_rank'] = recv_rank
          if peer['device'] == 'gpumem':
            new_job['type'] = 'NIC_READ_GPUMEM'
          if peer['device'] == 'cpumem':
            new_job['type'] = 'NIC_READ_CPUMEM'
        if idx == 1:
          new_job['rank'] = recv_rank
          new_job['peer_rank'] = send_rank
          if peer['device'] == 'gpumem':
            new_job['type'] = 'NIC_WRITE_GPUMEM'
          if peer['device'] == 'cpumem':
            new_job['type'] = 'NIC_WRITE_CPUMEM'
        for k, v in peer.items():
          new_job[k] = v
        new_jobs.append(new_job)
        hosts.append(new_job['host'])
    elif job['type'] == 'NIC_SHARP_ALLREDUCE':
      # multi-process job
      base_rank = rank_to_assign
      num_peers = len(job['peers'])
      rank_to_assign += num_peers
      peers = list(range(base_rank, base_rank + num_peers))
      for idx, peer in enumerate(job['peers']):
        new_job = copy.deepcopy(job)
        new_job.pop('peers')
        new_job['rank'] = base_rank + idx
        new_job['peer_rank'] = peers
        for k, v in peer.items():
          new_job[k] = v
        new_jobs.append(new_job)
        hosts.append(new_job['host'])
    else:
      # single-process job
      new_job = copy.deepcopy(job)
      new_job['rank'] = rank_to_assign
      rank_to_assign += 1
      new_jobs.append(new_job)
      hosts.append(new_job['host'])
  new_config = copy.deepcopy(config)
  new_config['jobs'] = new_jobs
  new_config['config'] = config
  return new_config, hosts

def generate_mpirun_script(hosts: List[str]):
  script = ''
  script += '#!/bin/bash\n'
  script += 'mpirun -mca btl ^openib -mca pml ucx --bind-to none \\\n'
  script += '-mca rmaps seq -H {} \\\n'.format(','.join(hosts))
  script += '-x OMP_NUM_THREADS=2 \\\n'
  script += '-x LD_LIBRARY_PATH \\\n'
  script += './benchmark $@\n'
  return script

workspace_info = {}
def launch_benchmark_from_conf(config, workspace, dry_run=False, check_dup=False):
  new_config, hosts = annotate_conf(config)
  mpirun_script = generate_mpirun_script(hosts)

  if check_dup:
    if not workspace in workspace_info:
      db = []
      job_fns = glob.glob(os.path.join(workspace, "????????-????-????-????-????????????.json"))
      for job_fn in job_fns:
        if job_fn.endswith('_result.json'):
          continue
        with open(job_fn, 'r') as job_f:
          job_data = json.load(job_f)
        db.append(job_data['jobs'])
      workspace_info[workspace] = db
    if new_config['jobs'] in workspace_info[workspace]:
      return

  while True:
    fn = str(uuid.uuid4())
    conf_fn = os.path.join(workspace, fn + '.json')
    if not os.path.exists(conf_fn):
      break

  with open(conf_fn, 'w') as f:
    json.dump(new_config, f, indent=2)

  script_fn = os.path.join(workspace, fn + '.sh')
  with open(script_fn, 'w') as f:
    f.write(mpirun_script)
  st = os.stat(script_fn)
  os.chmod(script_fn, st.st_mode | stat.S_IEXEC)

  result_fn = os.path.join(workspace, fn + '_result.json')

  if not dry_run:
    p = subprocess.run([script_fn, conf_fn, result_fn])