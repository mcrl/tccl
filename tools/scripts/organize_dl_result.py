import os
import argparse

OP_NAMES = ['AllReduce', 'Broadcast', 'Reduce', 'AllGather', 'ReduceScatter']

def main(args):
  with open(args.output, 'w') as outcsv:
    outcsv.write('system,model,tp_degree,nccl,msccl,tccl\n')
    for system in ['a', 'b', 'd']:
      for model in ['bert', 'gpt', 't5']:
        if system == 'a':
          num_node = 3
          num_gpu = 4
          gpu_confs = [(1, 1), (2, 1), (4, 1)]
        elif system == 'b':
          num_node = 6
          num_gpu = 4
          gpu_confs = [(1, 1), (2, 1), (4, 1)]
        elif system == 'd':
          num_node = 3
          num_gpu = 4
          gpu_confs = [(1, 1), (2, 1), (4, 1)]
        for gpu_conf in gpu_confs:
          tp = gpu_conf[0]
          pp = gpu_conf[1]
          num_node = str(num_node)
          devices = ','.join(['0', '1', '2', '3'][:num_gpu])
          elist = []
          for backend in ['nccl', 'msccl', 'tccl']:
            path = args.workspace
            fn = f'{backend}.{system}.{num_node}.{devices}.{model}.{tp}.{pp}.out'
            if not os.path.exists(f'{path}/{fn}'):
              print(f'{path}/{fn} does not exist')
              continue
            with open(f'{path}/{fn}', 'r') as f:
              alllines = f.readlines()
            #  iteration        5/      30 | consumed samples:           80 | elapsed time per iteration (ms): 904.3 | learning rate: 0.000E+00 | global batch size:    16 | loss scale: 268435456.0 | number of skipped iterations:   5 | number of nan iterations:   0 |
            times = [line for line in alllines if 'elapsed time per iteration' in line]
            if len(times) == 0:
              print(f'{path}/{fn}')
              print(times)
              continue

            final_idx = -1
            tokens = times[final_idx].split('|')
            elapsed = float(list(filter(lambda x: 'elapsed time per iteration' in x, tokens))[0].split(':')[1])
            skipped = int(list(filter(lambda x: 'skipped' in x, tokens))[0].split(':')[1])
            elist.append(elapsed)

            #all_elapsed = []
            #for i in range(len(times)):
            #  tokens = times[i].split('|')
            #  elapsed = float(list(filter(lambda x: 'elapsed time per iteration' in x, tokens))[0].split(':')[1])
            #  skipped = int(list(filter(lambda x: 'skipped' in x, tokens))[0].split(':')[1])
            #  all_elapsed.append(elapsed)
            #elist.append(min(all_elapsed))

            #times = [{token.split(':')[0].strip() : token.split(':')[1].strip() for token in line.split('|')} for line in times]
            #final_time = times[final_idx]
          if len(elist) != 3: 
            continue
          speedup = elist[1] / elist[2]
          #print(f'{fn} {backend},{system},{num_node}x{num_gpu},{model},{tp},{pp},{elist},{speedup}')
          outcsv.write(f'{system},{model},{tp},{",".join(map(str,elist))}\n')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--workspace', type=str, required=True)
  parser.add_argument('--output', type=str, required=True)
  args = parser.parse_args()
  main(args)