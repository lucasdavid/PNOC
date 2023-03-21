# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os
import argparse
from torch import multiprocessing
from pickle import UnpicklingError

import imageio
import numpy as np
from PIL import Image

from core.datasets import get_paths_dataset
from tools.ai.demo_utils import crf_inference_label
from tools.ai.log_utils import log_config
from tools.ai.torch_utils import set_seed
from tools.general.io_utils import create_directory, load_saliency_file

parser = argparse.ArgumentParser()

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=24, type=int)
parser.add_argument('--dataset', default='voc12', choices=['voc12', 'coco14'])
parser.add_argument('--data_dir', default='../VOCtrainval_11-May-2012/', type=str)
parser.add_argument('--cams_dir', required=True, type=str)
parser.add_argument('--sal_dir', default=None, type=str)

###############################################################################
# Inference parameters
###############################################################################
parser.add_argument('--tag', type=str, required=True)
parser.add_argument('--domain', default='train', type=str)

parser.add_argument('--fg_threshold', default=0.30, type=float)
parser.add_argument('--bg_threshold', default=0.05, type=float)

parser.add_argument('--crf_t', default=0, type=int)
parser.add_argument('--crf_gt_prob', default=0.7, type=float)

import cv2
cv2.setNumThreads(0)


def _work(process_id, dataset, args, work_dir):
  subset = range(process_id, len(dataset), args.num_workers)
  errors = []
  missing = []
  processed = 0

  for step in subset:
    image_id, image_path, _ = dataset[step]
    png_path = os.path.join(work_dir, image_id + '.png')
    cam_path = os.path.join(args.cams_dir, image_id + '.npy')

    if os.path.isfile(png_path):
      processed += 1
      continue

    print(f"id={image_id}", end=" ", flush=True)

    try:
      with Image.open(image_path) as image:
        image = np.asarray(image.convert("RGB"))
    except FileNotFoundError:
      missing.append(image_path)
      # print(f"{image_id} skipped (image missing)")
      continue

    try:
      data = np.load(cam_path, allow_pickle=True).item()
    except UnpicklingError as error:
      errors.append(cam_path)
      # print(f"{image_id} skipped (cam error)")
      continue
    except FileNotFoundError:
      missing.append(cam_path)
      # print(f"{image_id} skipped (cam missing)")
      continue

    keys = data['keys']
    cam = data['hr_cam']

    if keys.shape[0] < 2:
      processed += 1
      # print(f"{image_id} skipped (bg)")
      continue

    if not args.sal_dir:
      # 1. find confident fg & bg
      fg_cam = np.pad(cam, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.fg_threshold)
      fg_cam = np.argmax(fg_cam, axis=0)
      fg_conf = keys[crf_inference_label(image, fg_cam, n_labels=keys.shape[0])]

      bg_cam = np.pad(cam, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.bg_threshold)
      bg_cam = np.argmax(bg_cam, axis=0)
      bg_conf = keys[crf_inference_label(image, bg_cam, n_labels=keys.shape[0])]
    else:
      # If saliency maps are available, use them to determine the background regions:
      try:
        sal_file = os.path.join(args.sal_dir, image_id + '.png')
        sal = load_saliency_file(sal_file, kind='saliency')
      except FileNotFoundError:
        missing.append(sal_file)
        continue

      fg_mask = (sal > args.bg_threshold).squeeze().astype('int')
      bg_conf = crf_inference_label(image, fg_mask, n_labels=2, t=args.crf_t, gt_prob=args.crf_gt_prob)

      fg_cam = np.pad(cam, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.fg_threshold)
      fg_cam = bg_conf * np.argmax(fg_cam, axis=0)
      fg_conf = keys[crf_inference_label(image, fg_cam, n_labels=keys.shape[0], t=args.crf_t, gt_prob=args.crf_gt_prob)]

    conf = fg_conf.copy()
    conf[fg_conf == 0] = 255
    conf[bg_conf + fg_conf == 0] = 0

    # print(f"crf={conf.shape}", end=" ")

    try:
      imageio.imwrite(png_path, conf.astype(np.uint8))
    except:
      if os.path.exists(png_path):
        os.remove(png_path)
      raise
    processed += 1
    print(f"{image_id} ok")

  if missing: print(f"{len(missing)} files were missing and were not processed:", *missing, sep='\n  - ', flush=True)
  if errors: print(f"{len(errors)} CAM files could not be read:", *errors, sep="\n  - ", flush=True)

  print(f"{processed} images successfully processed")


if __name__ == '__main__':
  args = parser.parse_args()
  TAG = args.tag

  log_config(vars(args), TAG)

  set_seed(args.seed)
  work_dir = create_directory(
    f'./experiments/predictions/{TAG}@aff_fg={args.fg_threshold:.2f}_bg={args.bg_threshold:.2f}/'
  )

  dataset = get_paths_dataset(args.dataset, args.data_dir, args.domain)

  if args.num_workers > 1:
    multiprocessing.spawn(_work, nprocs=args.num_workers, args=(dataset, args, work_dir), join=True)
  else:
    _work(0, dataset, args, work_dir)
