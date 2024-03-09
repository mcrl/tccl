import glob, os, json
import argparse

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--workspace', type=str, required=True, help='Path to the workspace')
  args = parser.parse_args()

  workspace = os.path.abspath(args.workspace)

  db = []
  result_fns = glob.glob(os.path.join(workspace, "*_result.json"))
  for result_fn in result_fns:
    job_id = os.path.basename(result_fn)[:36]
    job_fn = os.path.join(workspace, job_id + ".json")

    with open(job_fn, 'r') as job_f, open(result_fn, 'r') as result_f:
      job_data = json.load(job_f)
      result_data = json.load(result_f)

    db.append({
      'jobs': job_data['jobs'],
      'us': result_data['med_stat']['us'],
    })
    #if job_data['jobs'][0]['type'] == 'GPU_WRITE_CPUMEM_MEMCPY' \
    #  and job_data['jobs'][0]['gpu_idx'] == 3 \
    #  and job_data['jobs'][0]['cpumem_numa_idx'] == 1 \
    #  and job_data['jobs'][1]['type'] == 'GPU_WRITE_GPUMEM_MEMCPY' \
    #  and job_data['jobs'][1]['gpu_idx'] == 0 \
    #  and job_data['jobs'][1]['gpumem_idx'] == 3:
    #    print('Found!')
    #    print(job_id)
  sorted_db = sorted(db, key=lambda x: x['us'])
  with open(os.path.join(workspace, "sorted_db.json"), 'w') as f:
    json.dump(sorted_db, f, indent=2)


if __name__ == '__main__':
  main()