import argparse
import glob
import os
import re
import xml.etree.ElementTree as ET

def main(args):
  root = ET.Element('root')

  # Lets build intra data
  intra = ET.SubElement(root, 'intra')
  flist = glob.glob(os.path.join(args.dir, 'intra_*.xml'))
  for fn in flist:
    m = re.search('intra_(\d*)\.xml', fn)
    gpu_mask = m.group(1)
    subset = ET.SubElement(intra, 'subset', {'gpus': gpu_mask})

    transfers = ET.parse(fn).getroot()
    subset.append(transfers)

  # Lets build inter data
  inter = ET.SubElement(root, 'inter')
  flist = glob.glob(os.path.join(args.dir, 'inter_*_*_*.xml'))
  subsets = {}
  for fn in flist:
    m = re.search('inter_(\d*)_(\d*)_(\d*)\.xml', fn)
    gpu_mask = m.group(3)
    if gpu_mask not in subsets:
      subsets[gpu_mask] = ET.SubElement(inter, 'subset', {'gpus': gpu_mask})
    subset = subsets[gpu_mask]

    transfers = ET.parse(fn).getroot()
    transfer_list = list(transfers.iter('transfer'))
    head, body, tail = transfer_list[0], transfer_list[1:-1], transfer_list[-1]
    new_transfers = ET.SubElement(subset, 'transfers', {
      'gbps': transfers.attrib['gbps'],
      'head_type': head.attrib['type'],
      'head_src_idx': head.attrib['src_idx'],
      'head_dst_idx': head.attrib['dst_idx'],
      'tail_type': tail.attrib['type'],
      'tail_src_idx': tail.attrib['src_idx'],
      'tail_dst_idx': tail.attrib['dst_idx'],
    })
    for transfer in body:
      new_transfers.append(transfer)

  tree = ET.ElementTree(root)
  ET.indent(tree, space="  ")
  tree.write(args.output)
  


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--dir', type=str, required=True)
  parser.add_argument('--output', type=str, required=True)
  args = parser.parse_args()
  main(args)
