import logging
import subprocess

HOSTS = [f'a{i}' for i in range(4)] + [f'b{i}' for i in range(8)] + [f'd{i}' for i in range(4)]

def run(cmd, text=True, returncode=False):
  p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=text)
  output, _ = p.communicate()
  if returncode:
    return (output, p.returncode)
  else:
    return output

def run_async(cmd, env=None, split=True, stdin=None, stdout=None, stderr=None):
  if split:
    cmd = cmd.split()
  p = subprocess.Popen(cmd, env=env, stdin=stdin, stdout=stdout, stderr=stderr)
  return p

def get_logger(name, host=None):
  fmt = f'[{host}] ' if host else ''
  fmt += '[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s] %(message)s'
  formatter = logging.Formatter(fmt)
  sh = logging.StreamHandler()
  sh.setFormatter(formatter)
  logger = logging.getLogger(name)
  logger.addHandler(sh)
  logger.setLevel(logging.INFO)
  return logger

logger = get_logger(__name__)

def main():
  alives = []
  for host in HOSTS:
    _, rc = run(f'ping -c 1 -W 1 {host}', returncode=True)
    if rc == 0:
      logger.info(f'{host} is alive')
      alives.append(host)
    else:
      logger.info(f'{host} is dead')

  for host in alives:
    # Retrieve NVIDIA GPUs
    res = run(f'ssh {host} lspci -d 10de:*')
    bdfs = [line.split()[0] for line in res.strip().split('\n')]
    bdfs = [bdf for bdf in bdfs if bdf.endswith('.0')]
    for bdf in bdfs:
      res = run(f'ssh {host} lspci -vvv -s {bdf}')
      lnksta = [line for line in res.strip().split('\n') if 'LnkSta:' in line][0] # '\t\tLnkSta:\tSpeed 2.5GT/s (downgraded), Width x16 (ok)'
      lnksta = lnksta.split(':')[1] # '\tSpeed 2.5GT/s (downgraded), Width x16 (ok)'
      speedstr, widthstr = lnksta.split(',')
      speedstr, widthstr = speedstr.strip(), widthstr.strip() # 'Speed 2.5GT/s (downgraded)', 'Width x16 (ok)'
      logger.info(f'{host} GPU {bdf} {speedstr} {widthstr}')

    # Retrieve Mellanox NICs
    res = run(f'ssh {host} lspci -d 15b3:*')
    bdfs = [line.split()[0] for line in res.strip().split('\n')]
    for bdf in bdfs:
      res = run(f'ssh {host} lspci -vvv -s {bdf}')
      lnksta = [line for line in res.strip().split('\n') if 'LnkSta:' in line][0] # '\t\tLnkSta:\tSpeed 2.5GT/s (downgraded), Width x16 (ok)'
      lnksta = lnksta.split(':')[1] # '\tSpeed 2.5GT/s (downgraded), Width x16 (ok)'
      speedstr, widthstr = lnksta.split(',')
      speedstr, widthstr = speedstr.strip(), widthstr.strip() # 'Speed 2.5GT/s (downgraded)', 'Width x16 (ok)'
      logger.info(f'{host} NIC {bdf} {speedstr} {widthstr}')
    
    # Check nvidia_peermem
    res = run(f'ssh {host} lsmod')
    res = [line for line in res.strip().split('\n') if line.startswith('nvidia_peermem')]
    if len(res) == 0:
      # Try to load nvidia_peermem
      res = run(f'ssh {host} modprobe nvidia_peermem')
    res = run(f'ssh {host} lsmod')
    res = [line for line in res.strip().split('\n') if line.startswith('nvidia_peermem')]
    if len(res) == 0:
      logger.info(f'{host} nvidia_peermem not loaded')
    else:
      logger.info(f'{host} {res[0]}')

    # Check ACS is disabled (https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html)
    res = run(f'ssh {host} lspci -vvv')
    res = [line for line in res.strip().split('\n') if 'ACSCtl: SrcValid+' in line]
    if len(res) == 0:
      logger.info(f'{host} ACS is disabled')
    else:
      logger.info(f'{host} !!! ACS is enabled !!!')


    
if __name__ == "__main__":
  main()