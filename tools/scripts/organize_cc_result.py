import os
import argparse

OP_NAMES = ['AllReduce', 'Broadcast', 'Reduce', 'AllGather', 'ReduceScatter']

def main(args):
  with open(args.output, 'w') as outcsv:
    outcsv.write('system,op,num_node x num_gpu,nccl,msccl,tccl\n')
    for system in ['a', 'b', 'd']:
      for op in range(5):
        if system == 'a':
          gpu_confs = [(1, 2), (1, 4), (2, 1), (2, 2), (2, 4), (4, 1), (4, 2), (4, 4)]
        elif system == 'b':
          gpu_confs = [(1, 2), (1, 4), (2, 1), (2, 2), (2, 4), (4, 1), (4, 2), (4, 4), (8, 1), (8, 2), (8, 4)]
        elif system == 'd':
          gpu_confs = [(1, 2), (1, 4), (2, 1), (2, 2), (2, 4), (3, 1), (3, 2), (3, 4)]
        for gpu_conf in gpu_confs:
          num_node = str(gpu_conf[0])
          num_gpu = gpu_conf[1]
          devices = ','.join(['0', '1', '2', '3'][:num_gpu])
          better = []
          if num_gpu == 1 and num_node == '1': continue
          for size_idx in range(21):
            bws = []
            for backend in ['nccl', 'msccl', 'tccl']:
              path = args.workspace
              fn = f'{backend}.{system}.{num_node}.{devices}.{op}.out'
              if not os.path.exists(f'{path}/{fn}'):
                continue
              with open(f'{path}/{fn}', 'r') as f:
                alllines = f.readlines()
                #b7:171324:171376 [3] NCCL INFO 4 coll channels, 0 nvls channels, 4 p2p channels, 2 p2p channels per peer
                num_coll_channels = [line for line in alllines if 'coll channels' in line][0].split()[4]
                perf_lines = [line for line in alllines if '     float    ' in line]
                if len(perf_lines) != 21:
                  print(f'{path}/{fn}')
                  print(perf_lines)
                  exit()
                #line = perf_lines[20] # 1GB
                #line = perf_lines[10] # 1MB
              line = perf_lines[size_idx]
              tokens = line.split()
              algbw_out, busbw_out, algbw_in, busbw_in = tokens[6], tokens[7], tokens[10], tokens[11]
              bws.append(float(algbw_in))
            if len(bws) != 3:
              continue
            speedup = bws[2] / max(bws[0], bws[1])
            better.append(bws[0] > bws[2])
          if len(better) != 21:
            continue
          for i in range(20, -1, -1):
            if better[i]:
              break
          outcsv.write(f'{system},{OP_NAMES[op]},{num_node}x{num_gpu},{bws[0]},{bws[1]},{bws[2]}\n')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--workspace', type=str, required=True)
  parser.add_argument('--output', type=str, required=True)
  args = parser.parse_args()
  main(args)