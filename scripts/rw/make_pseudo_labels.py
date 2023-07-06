# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import argparse
import os
from pickle import UnpicklingError
from typing import List

import imageio
import numpy as np
import torch
from torch import multiprocessing
from torch.utils.data import Subset
from tqdm import tqdm

import datasets
from tools.ai.demo_utils import crf_inference_label
from tools.ai.torch_utils import set_seed
from tools.general.io_utils import create_directory, load_saliency_file

parser = argparse.ArgumentParser()

parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=24, type=int)

parser.add_argument('--dataset', default='voc12', choices=datasets.DATASOURCES)
parser.add_argument('--data_dir', required=True, type=str)
parser.add_argument('--domain', default='train', type=str)
parser.add_argument('--sample_ids', default=None, type=str)

parser.add_argument('--experiment_name', default='', type=str)
parser.add_argument('--sal_dir', default=None, type=str)
parser.add_argument('--pred_dir', default=None, type=str)

parser.add_argument('--threshold', default=0.25, type=float)
parser.add_argument('--crf_t', default=10, type=int)
parser.add_argument('--crf_gt_prob', default=0.7, type=float)


def split_dataset(dataset, n_splits):
  return [Subset(dataset, np.arange(i, len(dataset), n_splits)) for i in range(n_splits)]


def _work(
    process_id: int,
    dataset: List[datasets.PathsDataset],
    args: argparse.Namespace,
    CAM_DIR: str,
    SAL_DIR: str,
    PRED_DIR: str,
):
  subset = dataset[process_id]
  data_source = subset.dataset.data_source
  processed = 0
  missing = []
  errors = []

  if process_id == 0:
    subset = tqdm(subset, mininterval=5.)

  with torch.no_grad():


    for image_id, image_path, mask_path in subset:
      png_path = os.path.join(PRED_DIR, image_id + '.png')
      cam_path = os.path.join(CAM_DIR, image_id + '.npy')
      sal_file = os.path.join(SAL_DIR, image_id + '.png') if SAL_DIR else None

      if os.path.isfile(png_path):
        continue

      image = data_source.get_image(image_id)
      label = data_source.get_label(image_id)

      W, H = image.size

      if label.sum() == 0:  # only background regions
        conf = np.zeros((H, W), dtype=np.uint8)
      else:
        try:
          data = np.load(cam_path, allow_pickle=True).item()
        except UnpicklingError as error:
          errors.append(cam_path)
          print(f"{image_id} skipped (cam error={error})")
          continue
        except FileNotFoundError:
          missing.append(cam_path)
          print(f"{image_id} skipped (cam missing)")
          continue

        keys = data['keys']
        cam = data['rw']
        if sal_file:
          try:
            sal = load_saliency_file(sal_file)
            cam = np.concatenate((sal, cam), axis=0)
          except FileNotFoundError:
            missing.append(sal_file)
            continue
        else:
          cam = np.pad(cam, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.threshold)

        cam = np.argmax(cam, axis=0)

        if args.crf_t > 0:
          x = np.array(image)
          image.close()
          cam = crf_inference_label(x, cam, n_labels=keys.shape[0], t=args.crf_t, gt_prob=args.crf_gt_prob)

        conf = keys[cam]

      imageio.imwrite(png_path, conf.astype(np.uint8))
      processed += 1

  if missing: print(f"{len(missing)} files were missing and were not processed:", *missing, sep='\n  - ', flush=True)
  if errors: print(f"{len(errors)} CAM files could not be read:", *errors, sep="\n  - ", flush=True)

  print(f"{processed} images successfully processed")


if __name__ == '__main__':
  try:
    multiprocessing.set_start_method('spawn')
  except RuntimeError:
    ...

  args = parser.parse_args()

  CAM_DIR = f'./experiments/predictions/{args.experiment_name}/'
  SAL_DIR = args.sal_dir
  PRED_DIR = create_directory(args.pred_dir or f'./experiments/predictions/{args.experiment_name}@crf={args.crf_t}/')

  set_seed(args.seed)

  ds = datasets.custom_data_source(args.dataset, args.data_dir, args.domain, sample_ids=args.sample_ids)
  dataset = datasets.PathsDataset(ds, ignore_bg_images=False)
  dataset = split_dataset(dataset, args.num_workers)

  multiprocessing.spawn(
    _work,
    nprocs=args.num_workers,
    args=(dataset, args, CAM_DIR, SAL_DIR, PRED_DIR),
    join=True
  )
