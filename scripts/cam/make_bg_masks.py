import argparse
import multiprocessing
import os
import sys

import numpy as np
from PIL import Image, UnidentifiedImageError

import datasets
from tqdm import tqdm

from tools.ai.demo_utils import crf_inference_dlv2_softmax, crf_inference_label
from tools.ai.log_utils import log_config
from tools.general.io_utils import load_cam_file, load_saliency_file, load_dlv2_seg_file, str2bool

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", type=str, required=True)
parser.add_argument("--num_workers", default=48, type=int)
parser.add_argument("--verbose", default=1, type=int)

parser.add_argument("--dataset", default="voc12", choices=datasets.DATASOURCES)
parser.add_argument("--domain", default="train", type=str)
parser.add_argument("--data_dir", default="../VOCtrainval_11-May-2012/", type=str)
parser.add_argument("--output_dir", default=None, type=str)


def worker(dataset: datasets.PathsDataset, start, step):
  steps = range(start, len(dataset), step)

  if start == 0:
    steps = tqdm(steps, mininterval=2.0)

  for idx in steps:
    image_id, image_path, mask_path = dataset[idx]

    y_true = dataset.data_source.get_label(image_id)
    out_file = os.path.join(OUT_DIR, image_id + ".png")

    if image_id == "000000500257":
      _l = np.where(y_true==1)[0]
      print(idx, image_id, _l, dataset.data_source.classification_info.classes[_l])

    if y_true.sum() > 0 or os.path.exists(out_file):
      # Ignored non-bg images and existing masks.
      continue

    print(idx, image_id, "[BG image]")

    image = dataset.data_source.get_image(image_id)
    (W, H) = image.size
    y_pred = np.zeros((H, W))

    try:
      with Image.fromarray(y_pred.astype(np.uint8)) as p:
        p.save(out_file)
    except:
      if os.path.exists(out_file):
        os.remove(out_file)
      raise


def run(args, dataset: datasets.PathsDataset):
  p_list = []
  for i in range(args.num_workers):
    p = multiprocessing.Process(target=worker, args=(dataset, i, args.num_workers))
    p.start()
    p_list.append(p)
  for p in p_list:
    p.join()


if __name__ == "__main__":
  args = parser.parse_args()
  TAG = args.experiment_name
  OUT_DIR = args.output_dir or f"./experiments/predictions/{TAG}/pseudo_labels"

  log_config(vars(args), TAG)

  ds = datasets.custom_data_source(args.dataset, args.data_dir, args.domain)
  dataset = datasets.PathsDataset(ds, ignore_bg_images=False)

  try:
    run(args, dataset)
  except KeyboardInterrupt:
    print("\ninterrupted")
