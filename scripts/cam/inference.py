# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import argparse
import copy
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.datasets import *
from core.networks import *
from tools.ai.augment_utils import *
from tools.ai.demo_utils import *
from tools.ai.evaluate_utils import *
from tools.ai.log_utils import *
from tools.ai.optim_utils import *
from tools.ai.randaugment import *
from tools.ai.torch_utils import *
from tools.general.io_utils import *
from tools.general.json_utils import *
from tools.general.time_utils import *

parser = argparse.ArgumentParser()

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--dataset', default='voc12', choices=['voc12', 'coco14'])
parser.add_argument('--data_dir', default='../VOCtrainval_11-May-2012/', type=str)

###############################################################################
# Network
###############################################################################
parser.add_argument('--architecture', default='resnet50', type=str)
parser.add_argument('--mode', default='normal', type=str)  # fix
parser.add_argument('--regularization', default=None, type=str)  # kernel_usage
parser.add_argument('--trainable-stem', default=True, type=str2bool)
parser.add_argument('--dilated', default=False, type=str2bool)
parser.add_argument('--restore', default=None, type=str)

###############################################################################
# Inference parameters
###############################################################################
parser.add_argument('--tag', default='', type=str)
parser.add_argument('--weights', default='', type=str)
parser.add_argument('--domain', default='train', type=str)

parser.add_argument('--scales', default='0.5,1.0,1.5,2.0', type=str)

try:
  GPUS = os.environ["CUDA_VISIBLE_DEVICES"]
except KeyError:
  GPUS = "0"
GPUS = GPUS.split(",")
GPUS_COUNT = len(GPUS)

normalize_fn = Normalize(*imagenet_stats())


def run(args):
  dataset = get_inference_dataset(args.dataset, args.data_dir, args.domain)
  print(f'{TAG} dataset={args.dataset} num_classes={dataset.info.num_classes}')

  model = Classifier(
    args.architecture,
    dataset.info.num_classes,
    mode=args.mode,
    dilated=args.dilated,
    regularization=args.regularization,
    trainable_stem=args.trainable_stem,
  )
  model = model.to(DEVICE)

  print('[i] Architecture is {}'.format(args.architecture))
  print('[i] Total Params: %.2fM' % (calculate_parameters(model)))
  print()

  if GPUS_COUNT > 1:
    print(f"GPUS={GPUS}")
    model = nn.DataParallel(model)

  load_model(model, WEIGHTS_PATH, parallel=GPUS_COUNT > 1)
  model.eval()

  scales = [float(scale) for scale in args.scales.split(',')]

  with torch.no_grad():
    length = len(dataset)
    for step, (image, image_id, label) in enumerate(dataset):
      W, H = image.size

      npy_path = PREDICTIONS_DIR + image_id + '.npy'
      if os.path.isfile(npy_path):
        continue

      strided_size = get_strided_size((H, W), 4)
      strided_up_size = get_strided_up_size((H, W), 16)

      cams_list = [get_cam(model, image, scale) for scale in scales]

      strided_cams_list = [resize_for_tensors(cams.unsqueeze(0), strided_size)[0] for cams in cams_list]
      strided_cams = torch.sum(torch.stack(strided_cams_list), dim=0)

      hr_cams_list = [resize_for_tensors(cams.unsqueeze(0), strided_up_size)[0] for cams in cams_list]
      hr_cams = torch.sum(torch.stack(hr_cams_list), dim=0)[:, :H, :W]

      keys = torch.nonzero(torch.from_numpy(label))[:, 0]

      strided_cams = strided_cams[keys]
      strided_cams /= F.adaptive_max_pool2d(strided_cams, (1, 1)) + 1e-5

      hr_cams = hr_cams[keys]
      hr_cams /= F.adaptive_max_pool2d(hr_cams, (1, 1)) + 1e-5

      # save cams
      keys = np.pad(keys + 1, (1, 0), mode='constant')
      np.save(npy_path, {
        "keys": keys,
        "cam": strided_cams.cpu(),
        "hr_cam": hr_cams.cpu().numpy()
      })

      sys.stdout.write(
        f'\r# Make CAM [{step + 1}/{length}] = {(step + 1) / length:.2%}%, '
        f'({(H, W)}, {tuple(hr_cams.shape)})'
      )
      sys.stdout.flush()
    print()


def get_cam(model, ori_image, scale):
  W, H = ori_image.size

  # Preprocessing
  x = copy.deepcopy(ori_image)
  x = x.resize((round(W * scale), round(H * scale)), resample=PIL.Image.CUBIC)
  x = normalize_fn(x)
  x = x.transpose((2, 0, 1))
  x = torch.from_numpy(x)
  xf = x.flip(-1)
  images = torch.stack([x, xf])
  images = images.to(DEVICE)

  _, features = model(images, with_cam=True)
  cams = F.relu(features)
  cams = cams[0] + cams[1].flip(-1)

  return cams


if __name__ == '__main__':
  args = parser.parse_args()

  DEVICE = args.device
  SEED = args.seed
  TAG = args.tag
  TAG += '@train' if 'train' in args.domain else '@val'
  TAG += '@scale=%s' % args.scales

  PREDICTIONS_DIR = create_directory(f'./experiments/predictions/{TAG}/')
  WEIGHTS_PATH = './experiments/models/' + f'{args.weights or args.tag}.pth'

  set_seed(SEED)
  run(args)
