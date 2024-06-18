# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>
# modified by Sierkinhane <sierkinhane@163.com>

import argparse
import os
import sys

import imageio
import numpy as np
from torch import multiprocessing
from torch.utils.data import Subset
from tqdm import tqdm

import datasets
from tools.ai.demo_utils import *
from tools.ai.torch_utils import set_seed
from tools.general.io_utils import *

parser = argparse.ArgumentParser()

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=24, type=int)
parser.add_argument('--dataset', default='voc12', choices=datasets.DATASOURCES)
parser.add_argument('--data_dir', default='/data1/xjheng/dataset/VOC2012/', type=str)
parser.add_argument('--exclude_bg_images', default=False, type=str2bool)

###############################################################################
# Inference parameters
###############################################################################
parser.add_argument('--experiment_name', default='', type=str)
parser.add_argument('--domain', default='train', type=str)

parser.add_argument('--threshold', default=0.3, type=float)
parser.add_argument('--crf_t', default=0, type=int)
parser.add_argument('--crf_gt_prob', default=0.7, type=float)
parser.add_argument('--activation', default='relu', type=str, choices=['relu', 'sigmoid'])


def split_dataset(dataset, n_splits):
  return [Subset(dataset, np.arange(i, len(dataset), n_splits)) for i in range(n_splits)]


def _work(process_id, dataset, args):
  dataset = dataset[process_id]
  data_source = dataset.dataset.data_source

  if process_id == 0:
    dataset = tqdm(dataset, mininterval=2.0)

  ccam_dir = f'./experiments/predictions/{args.experiment_name}/'
  pred_dir = f'./experiments/predictions/{args.experiment_name}@t={args.threshold}@crf={args.crf_t}/'

  for image_id, _, _ in dataset:
    png_path = os.path.join(pred_dir, image_id + '.png')
    if os.path.isfile(png_path):
      continue

    image = data_source.get_image(image_id)
    label = data_source.get_label(image_id)

    if label.sum() == 0:
      W, H = image.size
      cams = np.zeros((H, W))
    else:
      pack = np.load(ccam_dir + image_id + '.npy', allow_pickle=True).item()
      cams = pack['hr_cam']

      if args.activation == 'relu':
        cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.threshold)
        cams = np.argmax(cams, axis=0)

        if args.crf_t > 0:
          cams = crf_inference_label(np.asarray(image), cams, n_labels=2, t=args.crf_t, gt_prob=args.crf_gt_prob)
      else:
        cams = np.concatenate((1 - cams, cams))

        if args.crf_t > 0:
          cams = np.argmax(crf_inference(np.asarray(image), cams, t=args.crf_t, gt_prob=args.crf_gt_prob), axis=0)

    imageio.imwrite(png_path, (cams * 255).clip(0, 255).astype(np.uint8))


if __name__ == '__main__':
  args = parser.parse_args()

  set_seed(args.seed)

  create_directory(f'./experiments/predictions/{args.experiment_name}@t={args.threshold}@crf={args.crf_t}/')

  ds = datasets.custom_data_source(args.dataset, args.data_dir, args.domain)
  dataset = datasets.PathsDataset(ds, ignore_bg_images=args.exclude_bg_images)
  dataset = split_dataset(dataset, args.num_workers)

  if args.num_workers > 1:
    multiprocessing.spawn(_work, nprocs=args.num_workers, args=(dataset, args), join=True)
  else:
    _work(0, dataset, args)
