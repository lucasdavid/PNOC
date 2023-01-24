# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import argparse
import copy
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import multiprocessing
from torch.utils.data import Subset

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
# parser.add_argument('--num_workers', default=8, type=int)
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
  print(f'Loading weights from {WEIGHTS_PATH}.')
  load_model(model, WEIGHTS_PATH)
  model.eval()

  dataset = [Subset(dataset, np.arange(i, len(dataset), GPUS_COUNT)) for i in range(GPUS_COUNT)]

  scales = [float(scale) for scale in args.scales.split(',')]

  multiprocessing.spawn(_work, nprocs=GPUS_COUNT, args=(model, dataset, scales, PREDICTIONS_DIR, DEVICE), join=True)


def _work(process_id, model, dataset, scales, preds_dir, device):
  dataset = dataset[process_id]
  length = len(dataset)

  with torch.no_grad(), torch.cuda.device(process_id):
    model.cuda()

    for step, (image, image_id, label) in enumerate(dataset):
      W, H = image.size

      npy_path = os.path.join(preds_dir, image_id + '.npy')
      if os.path.isfile(npy_path):
        continue

      strided_size = get_strided_size((H, W), 4)
      strided_up_size = get_strided_up_size((H, W), 16)

      cams = [forward_tta(model, image, scale, device) for scale in scales]

      cams_st = [resize_for_tensors(c.unsqueeze(0), strided_size)[0] for c in cams]
      cams_st = torch.sum(torch.stack(cams_st), dim=0)

      cams_hr = [resize_for_tensors(cams.unsqueeze(0), strided_up_size)[0] for cams in cams]
      cams_hr = torch.sum(torch.stack(cams_hr), dim=0)[:, :H, :W]

      keys = torch.nonzero(torch.from_numpy(label))[:, 0]

      cams_st = cams_st[keys]
      cams_st /= F.adaptive_max_pool2d(cams_st, (1, 1)) + 1e-5

      cams_hr = cams_hr[keys]
      cams_hr /= F.adaptive_max_pool2d(cams_hr, (1, 1)) + 1e-5

      # save cams
      keys = np.pad(keys + 1, (1, 0), mode='constant')
      np.save(npy_path, {"keys": keys, "cam": cams_st.cpu(), "hr_cam": cams_hr.cpu().numpy()})

      if process_id == 0:
        sys.stdout.write(f'\r# Make CAM [{step + 1}/{length}] = {(step + 1) / length:.2%}, ({(H, W)}, {tuple(cams_hr.shape)})')
        sys.stdout.flush()
    if process_id == 0: print()


def forward_tta(model, ori_image, scale, DEVICE):
  W, H = ori_image.size

  # Preprocessing
  x = copy.deepcopy(ori_image)
  x = x.resize((round(W * scale), round(H * scale)), resample=PIL.Image.BICUBIC)
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
