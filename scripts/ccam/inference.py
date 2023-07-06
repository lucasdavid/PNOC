# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>
# modified by Sierkinhane <sierkinhane@163.com>

import argparse
import copy
import os
import sys
from typing import List

import numpy as np
import PIL
import torch
import torch.nn.functional as F
from torch import multiprocessing
from torch.utils.data import Subset
from tqdm import tqdm

import datasets
from core.networks import *
from tools.ai.augment_utils import *
from tools.ai.demo_utils import *
from tools.ai.evaluate_utils import *
from tools.ai.log_utils import *
from tools.ai.optim_utils import *
from tools.ai.torch_utils import *
from tools.general.io_utils import *
from tools.general.json_utils import *
from tools.general.time_utils import *

parser = argparse.ArgumentParser()

# Dataset
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--dataset', default='voc12', choices=datasets.DATASOURCES)
parser.add_argument('--data_dir', required=True, type=str)
parser.add_argument('--exclude_bg_images', default=True, type=str2bool)

# Network
parser.add_argument('--architecture', default='resnet50', type=str)
parser.add_argument('--mode', default='normal', type=str)  # fix
parser.add_argument('--weights', default='imagenet', type=str)
parser.add_argument('--trainable-stem', default=True, type=str2bool)
parser.add_argument('--dilated', default=False, type=str2bool)
parser.add_argument('--pretrained', type=str, required=True)
parser.add_argument('--stage4_out_features', default=1024, type=int)

# Inference parameters
parser.add_argument('--tag', default='', type=str)
parser.add_argument('--domain', default='train', type=str)
parser.add_argument('--scales', default='0.5,1.0,1.5,2.0', type=str)
parser.add_argument('--activation', default='relu', type=str, choices=['relu', 'sigmoid'])

GPUS_VISIBLE = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
GPUS_COUNT = len(GPUS_VISIBLE.split(','))

normalize_fn = Normalize(*datasets.imagenet_stats())


def run(args):
  ds = datasets.custom_data_source(args.dataset, args.data_dir, args.domain)
  dataset = datasets.PathsDataset(ds, ignore_bg_images=args.exclude_bg_images)

  print(f'{TAG} dataset={args.dataset} num_classes={dataset.info.num_classes}')

  model = CCAM(
    args.architecture,
    weights=args.weights,
    mode=args.mode,
    dilated=args.dilated,
    stage4_out_features=args.stage4_out_features
  )
  load_model(model, WEIGHTS_PATH)
  model.eval()

  dataset = [Subset(dataset, np.arange(i, len(dataset), GPUS_COUNT)) for i in range(GPUS_COUNT)]
  scales = [float(scale) for scale in args.scales.split(',')]

  multiprocessing.spawn(
    _work, nprocs=GPUS_COUNT, args=(model, dataset, scales, PREDICTIONS_DIR, DEVICE, args), join=True
  )


def _work(
    process_id: int,
    model: CCAM,
    dataset: List[datasets.PathsDataset],
    scales: List[float],
    preds_dir: str,
    device: str,
    args,
  ):
  dataset = dataset[process_id]
  data_source = dataset.dataset.data_source

  if process_id == 0:
    dataset = tqdm(dataset, mininterval=2.0)

  with torch.no_grad(), torch.cuda.device(process_id):
    model.cuda()

    for image_id, _, _ in dataset:
      npy_path = os.path.join(preds_dir, image_id + '.npy')
      if os.path.isfile(npy_path):
        continue

      image = data_source.get_image(image_id)
      W, H = image.size

      strided_size = get_strided_size((H, W), 4)
      strided_up_size = get_strided_up_size((H, W), 16)

      cams = [forward_tta(model, image, scale, device, activation=args.activation) for scale in scales]

      cams_st = [resize_for_tensors(c.unsqueeze(0), strided_size)[0] for c in cams]
      cams_st = torch.mean(torch.stack(cams_st), dim=0)  # (1, 1, H, W)

      cams_hr = [resize_for_tensors(c.unsqueeze(0), strided_up_size)[0] for c in cams]
      cams_hr = torch.mean(torch.stack(cams_hr), dim=0)[:, :H, :W]  # (1, 1, H, W)

      cams_st = make_cam(cams_st.unsqueeze(1)).squeeze(1)
      cams_hr = make_cam(cams_hr.unsqueeze(1)).squeeze(1)

      try:
        np.save(npy_path, {"keys": [0, 1], "cam": cams_st.cpu(), "hr_cam": cams_hr.cpu().numpy()})
      except:
        if os.path.exists(npy_path):
          os.remove(npy_path)
        raise


def forward_tta(model, image, scale, device, activation: str = "relu"):
  W, H = image.size

  # preprocessing
  x = copy.deepcopy(image)
  x = x.resize((round(W * scale), round(H * scale)), resample=PIL.Image.BICUBIC)
  x = normalize_fn(x)
  x = x.transpose((2, 0, 1))

  x = torch.from_numpy(x)
  flipped_image = x.flip(-1)

  images = torch.stack([x, flipped_image])
  images = images.to(device)

  # inferenece
  _, _, cam = model(images)

  if activation == 'relu':
    cams = F.relu(cam)
  else:
    cams = torch.sigmoid(cam)

  cams = cams[0] + cams[1].flip(-1)

  return cams  # (1, H, W)


if __name__ == '__main__':
  args = parser.parse_args()

  SEED = args.seed
  DEVICE = args.device
  TAG = f'{args.tag}@train' if 'train' in args.domain else f'{args.tag}@val'
  TAG += '@scale=%s' % args.scales

  WEIGHTS_PATH = args.pretrained
  PREDICTIONS_DIR = create_directory(f'./experiments/predictions/{TAG}/')

  set_seed(SEED)
  run(args)
