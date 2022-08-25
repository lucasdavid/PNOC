# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>
# modified by Sierkinhane <sierkinhane@163.com>

import argparse
import os
import sys

import imageio
import numpy as np
from torch import multiprocessing
from torch.utils.data import DataLoader, Subset

from core.datasets import VOC_Dataset_For_Making_CAM
from tools.ai.demo_utils import *
from tools.ai.torch_utils import set_seed
from tools.general.io_utils import *

parser = argparse.ArgumentParser()

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=24, type=int)
parser.add_argument('--data_dir', default='/data1/xjheng/dataset/VOC2012/', type=str)

###############################################################################
# Inference parameters
###############################################################################
parser.add_argument('--experiment_name', default='', type=str)
parser.add_argument('--domain', default='train', type=str)

parser.add_argument('--threshold', default=0.3, type=float)
parser.add_argument('--crf_iteration', default=10, type=int)
parser.add_argument('--activation', default='relu', type=str, choices=['relu', 'sigmoid'])


def split_dataset(dataset, n_splits):
  return [Subset(dataset, np.arange(i, len(dataset), n_splits)) for i in range(n_splits)]


def _work(process_id, dataset, args):
  subset = dataset[process_id]
  length = len(subset)

  ccam_dir = f'./experiments/predictions/{args.experiment_name}/'
  pred_dir = f'./experiments/predictions/{args.experiment_name}@t={args.threshold}@crf={args.crf_iteration}/'

  for step, (image, _id, _, _) in enumerate(subset):
    png_path = pred_dir + _id + '.png'
    if os.path.isfile(png_path):
      continue

    pack = np.load(ccam_dir + _id + '.npy', allow_pickle=True).item()
    cams = pack['hr_cam']

    if args.activation == 'relu':
      cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.threshold)
      cams = np.argmax(cams, axis=0)

      if args.crf_iteration > 0:
        cams = crf_inference_label(np.asarray(image), cams, n_labels=2, t=args.crf_iteration)
    else:
      cams = np.concatenate((1 - cams, cams))

      if args.crf_iteration > 0:
        cams = np.argmax(crf_inference(np.asarray(image), cams, t=args.crf_iteration, labels=2), axis=0)

    imageio.imwrite(png_path, (cams * 255).clip(0, 255).astype(np.uint8))

    if process_id == args.num_workers - 1 and step % (length // 20) == 0:
      sys.stdout.write(
        '\r# CAMs CRF Inference [{}/{}] = {:.2f}%, ({}, {})'.format(
          step + 1, length, (step + 1) / length * 100, tuple(reversed(image.size)), cams.shape
        )
      )
      sys.stdout.flush()


if __name__ == '__main__':
  args = parser.parse_args()

  set_seed(args.seed)

  create_directory(f'./experiments/predictions/{args.experiment_name}@t={args.threshold}@crf={args.crf_iteration}/')

  dataset = VOC_Dataset_For_Making_CAM(args.data_dir, args.domain)
  dataset = split_dataset(dataset, args.num_workers)

  multiprocessing.spawn(_work, nprocs=args.num_workers, args=(dataset, args), join=True)
