import torch
import argparse
from datasets import load_dataset, load_from_disk
import pyarrow as pa
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
from itertools import chain
from transformers import (
  AutoTokenizer,
)
import sys, os
import torch.multiprocessing
import itertools

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--dataset_config", type=str, default=None)
parser.add_argument("--cache_dir", type=str)
parser.add_argument("--block_size", type=int, default=1024)
parser.add_argument("--preprocessing_num_workers", type=int, default=None)
parser.add_argument("--output", type=str)
args = parser.parse_args()

st = time.time()
raw_datasets = load_dataset(args.dataset_name, args.dataset_config, cache_dir=args.cache_dir,
                            num_proc=args.preprocessing_num_workers, ignore_verifications=True)
#raw_datasets = load_dataset(args.dataset_name, args.dataset_config, cache_dir=args.cache_dir, keep_in_memory=True)
elapsed = time.time() - st
print(f'load_dataset Elapsed time: {elapsed} seconds')

raw_datasets = raw_datasets['train']
print(f'first raw_datasets: {raw_datasets[0]}')

raw_datasets.to_json(args.output)