# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>
# modified by Sierkinhane <sierkinhane@163.com>

import argparse
import copy
import os
import sys

import numpy as np
import PIL
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
from tools.ai.torch_utils import *
from tools.general.io_utils import *
from tools.general.json_utils import *
from tools.general.time_utils import *

parser = argparse.ArgumentParser()

# Dataset
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--dataset', default='voc12', choices=['voc12', 'coco14'])
parser.add_argument('--data_dir', default='../VOCtrainval_11-May-2012/', type=str)

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

normalize_fn = Normalize(*imagenet_stats())


def run(args):
  dataset = get_inference_dataset(args.dataset, args.data_dir, args.domain)
  print(f'{TAG} dataset={args.dataset} num_classes={dataset.info.num_classes}')

  model = CCAM(
    args.architecture,
    weights=args.weights,
    mode=args.mode,
    dilated=args.dilated,
    stage4_out_features=args.stage4_out_features
  )
  print(f'Loading weights from {WEIGHTS_PATH}.')
  load_model(model, WEIGHTS_PATH)
  model.eval()

  dataset = [Subset(dataset, np.arange(i, len(dataset), GPUS_COUNT)) for i in range(GPUS_COUNT)]
  scales = [float(scale) for scale in args.scales.split(',')]

  multiprocessing.spawn(_work, nprocs=GPUS_COUNT, args=(model, dataset, scales, PREDICTIONS_DIR, DEVICE, args), join=True)


def _work(process_id, model, dataset, scales, preds_dir, device, args):
  dataset = dataset[process_id]
  length = len(dataset)

  with torch.no_grad(), torch.cuda.device(process_id):
    model.cuda()

    for step, (ori_image, image_id, _) in enumerate(dataset):
      W, H = ori_image.size
      npy_path = os.path.join(preds_dir, image_id + '.npy')
      if os.path.isfile(npy_path):
        continue
      strided_size = get_strided_size((H, W), 4)
      strided_up_size = get_strided_up_size((H, W), 16)

      cams = [forward_tta(model, ori_image, scale, device, args) for scale in scales]

      cams_st = [resize_for_tensors(c.unsqueeze(0), strided_size)[0] for c in cams]
      cams_st = torch.mean(torch.stack(cams_st), dim=0)  # (1, 1, H, W)

      cams_hr = [resize_for_tensors(c.unsqueeze(0), strided_up_size)[0] for c in cams]
      cams_hr = torch.mean(torch.stack(cams_hr), dim=0)[:, :H, :W]  # (1, 1, H, W)

      cams_st = make_cam(cams_st.unsqueeze(1)).squeeze(1)
      cams_hr = make_cam(cams_hr.unsqueeze(1)).squeeze(1)

      np.save(npy_path, {"keys": [0, 1], "cam": cams_st.cpu(), "hr_cam": cams_hr.cpu().numpy()})

      if process_id == 0:
        sys.stdout.write(f'\r# Make CAM [{step + 1}/{length}] = {(step + 1) / length:.2%}, ({(H, W)}, {tuple(cams_hr.shape)})')
        sys.stdout.flush()

    if process_id == 0: print()


def forward_tta(model, image, scale, device, args):
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

  if args.activation == 'relu':
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

