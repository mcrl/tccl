import copy

num_numa = 4

job_type_to_symbol = {
  "GPU_WRITE_CPUMEM_MEMCPY": 0,
  "GPU_READ_CPUMEM_MEMCPY": 1,
  "GPU_WRITE_GPUMEM_MEMCPY": 2,
  "GPU_READ_GPUMEM_MEMCPY": 3,
  "NIC_SENDRECV": 4,
}
symbol_to_job_type_db = {
  0: "GPU_WRITE_CPUMEM_MEMCPY",
  1: "GPU_READ_CPUMEM_MEMCPY",
  2: "GPU_WRITE_GPUMEM_MEMCPY",
  3: "GPU_READ_GPUMEM_MEMCPY",
  4: "NIC_SENDRECV",
}
def symbol_to_job_type(symbol):
  return symbol_to_job_type_db[symbol]

gpu_idx_to_symbol_db = {0: [3, 'gpu', 0], 1: [1, 'gpu', 0], 2: [1, 'gpu', 1], 3: [0, 'gpu', 0]}
symbol_to_gpu_idx_db = {
  (0, 0): 3,
  (1, 0): 1,
  (1, 1): 2,
  (3, 0): 0,
}
def gpu_idx_to_symbol(gpu_idx):
  return gpu_idx_to_symbol_db[gpu_idx].copy()
def symbol_to_gpu_idx(numa_idx, gpu_idx):
  return symbol_to_gpu_idx_db[(numa_idx % num_numa, gpu_idx)]

cpu_numa_idx_to_symbol_db = {0: [0, 'cpu', 0], 1: [1, 'cpu', 0], 2: [2, 'cpu', 0], 3: [3, 'cpu', 0]}
def cpu_numa_idx_to_symbol(cpu_numa_idx):
  return cpu_numa_idx_to_symbol_db[cpu_numa_idx].copy()
def symbol_to_cpu_idx(symbol):
  return symbol % num_numa

def nic_target_to_symbol(target):
  if 'gpumem_idx' in target:
    sym = gpu_idx_to_symbol(target['gpumem_idx'])
  if 'cpumem_numa_idx' in target:
    sym = cpu_numa_idx_to_symbol(target['cpumem_numa_idx'])
  nic = [0, 'nic', 0]
  if target['host'] == 'here':
    sym[0] += 0
    nic[0] += 0
  if target['host'] == 'prev':
    sym[0] += num_numa
    nic[0] += num_numa
  if target['host'] == 'next':
    sym[0] += num_numa * 2
    nic[0] += num_numa * 2
  return (sym, nic)

def swap_numa(pair, numa0, numa1):
  for job in pair:
    for dev in job[1:]:
      if dev[0] == numa0:
        dev[0] = numa1
      elif dev[0] == numa1:
        dev[0] = numa0

def swap_dev(pair, numa, dev0, dev1):
  for job in pair:
    for dev in job[1:]:
      if dev[0] == numa and dev[1] == 'gpu' and dev[2] == dev0:
        dev[2] = dev1
      elif dev[0] == numa and dev[1] == 'gpu' and dev[2] == dev1:
        dev[2] = dev0

def convert_to_symbolic_graph(pair):
  # pair of (job_type, src (numa, dev_type, dev_idx), dst (numa, dev_type, dev_idx))
  new_jobs = []
  for job in pair:
    if job['type'] == 'GPU_WRITE_CPUMEM_MEMCPY':
      new_jobs.append((
        job_type_to_symbol[job['type']],
        gpu_idx_to_symbol(job['gpu_idx']),
        cpu_numa_idx_to_symbol(job['cpumem_numa_idx']),
      ))
    if job['type'] == 'GPU_READ_CPUMEM_MEMCPY':
      new_jobs.append((
        job_type_to_symbol[job['type']],
        gpu_idx_to_symbol(job['gpu_idx']),
        cpu_numa_idx_to_symbol(job['cpumem_numa_idx']),
      ))
    if job['type'] == 'GPU_WRITE_GPUMEM_MEMCPY':
      new_jobs.append((
        job_type_to_symbol[job['type']],
        gpu_idx_to_symbol(job['gpu_idx']),
        gpu_idx_to_symbol(job['gpumem_idx']),
      ))
    if job['type'] == 'GPU_READ_GPUMEM_MEMCPY':
      new_jobs.append((
        job_type_to_symbol[job['type']],
        gpu_idx_to_symbol(job['gpu_idx']),
        gpu_idx_to_symbol(job['gpumem_idx']),
      ))
    if job['type'] == 'NIC_SENDRECV':
      new_jobs.append((
        job_type_to_symbol[job['type']],
        *nic_target_to_symbol(job['peers'][0]),
        *nic_target_to_symbol(job['peers'][1]),
      ))
  return new_jobs

def convert_pair_up_to_symmetry(pair):
  input_backup = copy.deepcopy(pair)
  pair = convert_to_symbolic_graph(pair)

  # prevent editing to_symbol tables
  pair = copy.deepcopy(pair)

  # 1. Sort job type
  if pair[0][0] > pair[1][0]:
    pair = (pair[1], pair[0])
  
  # 2. Sort NUMA and dev index
  numa_fixed = [False for _ in range(num_numa * 3)]
  dev_fixed = [[False for _ in range(2)] for _ in range(num_numa * 3)] # max 2 GPUs per NUMA

  def swap_numa_helper(cur_numa):
    numa_base = cur_numa // num_numa * num_numa
    for i in range(numa_base, cur_numa):
      if not numa_fixed[i] and not numa_fixed[cur_numa]:
        swap_numa(pair, i, cur_numa)
        cur_numa = i
        break
    numa_fixed[cur_numa] = True
  
  def swap_dev_helper(cur_numa, cur_dev):
    for i in range(cur_dev):
      if not dev_fixed[cur_numa][i] and not dev_fixed[cur_numa][cur_dev]:
        swap_dev(pair, cur_numa, i, cur_dev)
        cur_dev = i
        break
    dev_fixed[cur_numa][cur_dev] = True

  for job in pair:
    for dev in job[1:]:
      swap_numa_helper(dev[0])
      if dev[1] == 'gpu':
        swap_dev_helper(dev[0], dev[2])

  # 3. Sort if job type is the same
  #if pair[0][0] == pair[1][0]:
  #  pair = sorted(pair)

  # Convert to hashable type
  def convert_to_tuple(l):
    if type(l) is list or type(l) is tuple:
      return tuple(convert_to_tuple(x) for x in l)
    return l
  pair = convert_to_tuple(pair)
  
  return pair