# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import argparse
import os
import sys
from pickle import UnpicklingError

import imageio
import numpy as np
from torch import multiprocessing
from torch.utils.data import Subset

from core.datasets import get_inference_dataset
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


def split_dataset(dataset, n_splits):
  return [Subset(dataset, np.arange(i, len(dataset), n_splits)) for i in range(n_splits)]


def _work(process_id, dataset, args, work_dir):
  subset = dataset[process_id]
  length = len(subset)
  missing = 0

  tag = args.tag
  sal_dir = args.sal_dir

  cams_dir = args.cams_dir
  work_dir = f'./experiments/predictions/{tag}@aff_fg={args.fg_threshold:.2f}_bg={args.bg_threshold:.2f}/'

  for step, (ori_image, image_id, _) in enumerate(subset):
    png_path = os.path.join(work_dir, image_id + '.png')
    cam_path = os.path.join(cams_dir, image_id + '.npy')

    if os.path.isfile(png_path):
      continue

    # load
    image = np.asarray(ori_image)
    try:
      data = np.load(cam_path, allow_pickle=True).item()
    except UnpicklingError as error:
      print(f"Error loading CAM file {cam_path}: {error}")
      continue
    except FileNotFoundError:
      missing += 1
      continue

    keys = data['keys']
    cam = data['hr_cam']

    if not sal_dir:
      # 1. find confident fg & bg
      fg_cam = np.pad(cam, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.fg_threshold)
      fg_cam = np.argmax(fg_cam, axis=0)
      fg_conf = keys[crf_inference_label(image, fg_cam, n_labels=keys.shape[0])]

      bg_cam = np.pad(cam, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.bg_threshold)
      bg_cam = np.argmax(bg_cam, axis=0)
      bg_conf = keys[crf_inference_label(image, bg_cam, n_labels=keys.shape[0])]
    else:
      # If saliency maps are available.
      sal_file = os.path.join(sal_dir, image_id + '.png')
      sal = load_saliency_file(sal_file, kind='saliency')

      fg_mask = (sal > args.bg_threshold).squeeze().astype('int')
      bg_conf = crf_inference_label(image, fg_mask, n_labels=2, t=args.crf_t, gt_prob=args.crf_gt_prob)

      fg_cam = np.pad(cam, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.fg_threshold)
      fg_cam = bg_conf * np.argmax(fg_cam, axis=0)
      fg_conf = keys[crf_inference_label(image, fg_cam, n_labels=keys.shape[0], t=args.crf_t, gt_prob=args.crf_gt_prob)]

    conf = fg_conf.copy()
    conf[fg_conf == 0] = 255
    conf[bg_conf + fg_conf == 0] = 0

    imageio.imwrite(png_path, conf.astype(np.uint8))

    if process_id == args.num_workers - 1 and step % max(1, length // 100) == 0:
      sys.stdout.write(
        '\r# Make affinity labels [{}/{}] = {:.2f}%, ({}, {})'.format(
          step + 1, length, (step + 1) / length * 100, image.shape, cam.shape
        )
      )
      sys.stdout.flush()

  if missing:
    print(f"{missing} files were missing and were not processed.")


if __name__ == '__main__':
  args = parser.parse_args()
  TAG = args.tag

  log_config(vars(args), TAG)

  set_seed(args.seed)
  work_dir = create_directory(
    f'./experiments/predictions/{TAG}@aff_fg={args.fg_threshold:.2f}_bg={args.bg_threshold:.2f}/'
  )

  dataset = get_inference_dataset(args.dataset, args.data_dir, args.domain)
  dataset = split_dataset(dataset, args.num_workers)

  try:
    multiprocessing.spawn(_work, nprocs=args.num_workers, args=(dataset, args, work_dir), join=True)
  except KeyboardInterrupt:
    print('interrupted')
