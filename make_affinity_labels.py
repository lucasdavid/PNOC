# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import argparse
import os
import sys

import imageio
import numpy as np
from torch import multiprocessing
from torch.utils.data import Subset

from core.datasets import VOC_Dataset_For_Making_CAM
from tools.ai.demo_utils import crf_inference_label
from tools.ai.torch_utils import set_seed
from tools.general.io_utils import create_directory, load_saliency_file

parser = argparse.ArgumentParser()

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=24, type=int)
parser.add_argument('--data_dir', default='../VOCtrainval_11-May-2012/', type=str)
parser.add_argument('--sal_dir', default=None, type=str)

###############################################################################
# Inference parameters
###############################################################################
parser.add_argument(
  '--experiment_name', default='resnet50@seed=0@bs=16@ep=5@nesterov@train@scale=0.5,1.0,1.5,2.0', type=str
)
parser.add_argument('--domain', default='train', type=str)

parser.add_argument('--fg_threshold', default=0.30, type=float)
parser.add_argument('--bg_threshold', default=0.05, type=float)


def split_dataset(dataset, n_splits):
  return [Subset(dataset, np.arange(i, len(dataset), n_splits)) for i in range(n_splits)]


def _work(process_id, dataset, args):
  subset = dataset[process_id]
  length = len(subset)

  tag = args.experiment_name
  sal_dir = args.sal_dir

  pred_dir = f'./experiments/predictions/{tag}/'
  aff_dir = f'./experiments/predictions/{tag}@aff_fg={args.fg_threshold:.2f}_bg={args.bg_threshold:.2f}/'

  for step, (ori_image, image_id, _, _) in enumerate(dataset):
    png_path = aff_dir + image_id + '.png'
    if os.path.isfile(png_path):
      continue

    # load
    image = np.asarray(ori_image)
    data = np.load(pred_dir + image_id + '.npy', allow_pickle=True).item()

    keys = data['keys']
    cams = data['hr_cam']

    if sal_dir is None:
      # 1. find confident fg & bg
      fg_cam = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.fg_threshold)
      fg_cam = np.argmax(fg_cam, axis=0)
      fg_conf = keys[crf_inference_label(image, fg_cam, n_labels=keys.shape[0])]

      bg_cam = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.bg_threshold)
      bg_cam = np.argmax(bg_cam, axis=0)
      bg_conf = keys[crf_inference_label(image, bg_cam, n_labels=keys.shape[0])]

      # 2. combine confident fg & bg
      conf = fg_conf.copy()
      conf[fg_conf == 0] = 255
      conf[bg_conf + fg_conf == 0] = 0
    else:
      sal_file = os.path.join(sal_dir, image_id + '.png')
      sal = load_saliency_file(sal_file, kind='saliency')
      cams = np.concatenate((1 - sal, cams), axis=0)

    imageio.imwrite(png_path, conf.astype(np.uint8))

    if process_id == args.num_workers - 1 and step % max(1, length // 20) == 0:
      sys.stdout.write(
        '\r# Make affinity labels [{}/{}] = {:.2f}%, ({}, {})'.format(
          step + 1, length, (step + 1) / length * 100, tuple(reversed(image.size)), cams.shape
        )
      )
      sys.stdout.flush()


if __name__ == '__main__':
  args = parser.parse_args()
  tag = args.experiment_name

  set_seed(args.seed)
  create_directory(f'./experiments/predictions/{tag}@aff_fg={args.fg_threshold:.2f}_bg={args.bg_threshold:.2f}/')

  dataset = VOC_Dataset_For_Making_CAM(args.data_dir, args.domain)
  dataset = split_dataset(dataset, args.num_workers)

  multiprocessing.spawn(_work, nprocs=args.num_workers, args=(dataset, args), join=True)
