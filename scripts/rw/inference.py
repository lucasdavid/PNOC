# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from core.aff_utils import propagate_to_edge
from core.datasets import *
from core.networks import *
from core.puzzle_utils import *
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

# Dataset
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--dataset', default='voc12', choices=['voc12', 'coco14'])
parser.add_argument('--data_dir', default='../VOCtrainval_11-May-2012/', type=str)
parser.add_argument('--domain', default='train', type=str)

# Network
parser.add_argument('--architecture', default='resnet50', type=str)
parser.add_argument('--mode', default='fix', type=str)  # normal
parser.add_argument('--regularization', default=None, type=str)  # kernel_usage
parser.add_argument('--trainable-stem', default=False, type=str2bool)  # only false allowed.
parser.add_argument('--dilated', default=False, type=str2bool)
parser.add_argument('--restore', default=None, type=str)

# Hyper-parameters
parser.add_argument('--image_size', default=512, type=int)

parser.add_argument('--model_name', default='', type=str)
parser.add_argument('--cam_dir', default='', type=str)
parser.add_argument('--beta', default=10, type=int)
parser.add_argument('--exp_times', default=8, type=int)

try:
  GPUS = os.environ["CUDA_VISIBLE_DEVICES"]
except KeyError:
  GPUS = "0"
GPUS = GPUS.split(",")
GPUS_COUNT = len(GPUS)


def run(model, dataset):
  skipped = 0

  with torch.no_grad():
    length = len(dataset)
    for step, (ori_image, image_id, _) in enumerate(dataset):
      ori_w, ori_h = ori_image.size

      npy_path = os.path.join(PRED_DIR, image_id + '.npy')
      if os.path.isfile(npy_path):
        skipped += 1
        continue

      # preprocessing
      with ori_image:
        image = np.asarray(ori_image)
      image = normalize_fn(image)
      image = image.transpose((2, 0, 1))
      image = torch.from_numpy(image)
      flipped_image = image.flip(-1)

      images = torch.stack([image, flipped_image])
      images = images.to(DEVICE)

      # inference
      edge = model.get_edge(images)

      # postprocessing
      cam_dict = np.load(os.path.join(CAMS_DIR, image_id + '.npy'), allow_pickle=True).item()
      cams = cam_dict['cam']

      cam_downsized_values = cams.to(DEVICE)
      rw = propagate_to_edge(cam_downsized_values, edge, beta=args.beta, exp_times=args.exp_times, radius=5)

      rw_up = F.interpolate(rw, scale_factor=4, mode='bilinear', align_corners=False)[..., 0, :ori_h, :ori_w]
      rw_up /= torch.max(rw_up)

      np.save(npy_path, {"keys": cam_dict['keys'], "rw": to_numpy(rw_up)})

      sys.stdout.write(
        '\r# Make CAM with Random Walk [{}/{}] = {:.2f}%, ({}, rw_up={}, rw={})'.format(
          step + 1, length, (step + 1) / length * 100, (ori_h, ori_w), rw_up.size(), rw.size()
        )
      )
      sys.stdout.flush()
    print()


if __name__ == '__main__':
  # Arguments
  args = parser.parse_args()

  SEED = args.seed
  DEVICE = args.device
  TAG = args.model_name
  TAG += '@train' if 'train' in args.domain else '@val'
  TAG += f"@beta={args.beta}@exp_times={args.exp_times}@rw"

  log_config(vars(args), TAG)

  CAMS_DIR = os.path.join(f'./experiments/predictions/', args.cam_dir)
  PRED_DIR = create_directory(f'./experiments/predictions/{TAG}/')

  model_path = './experiments/models/' + f'{args.model_name}.pth'

  set_seed(SEED)

  normalize_fn = Normalize(*imagenet_stats())
  path_index = PathIndex(radius=10, default_size=(args.image_size // 4, args.image_size // 4))

  dataset = get_inference_dataset(args.dataset, args.data_dir, args.domain)

  # Network
  model = AffinityNet(
    args.architecture,
    path_index,
    mode=args.mode,
    dilated=args.dilated,
  )
  if args.restore:
    print(f'Restoring weights from {args.restore}')
    model.load_state_dict(torch.load(args.restore), strict=True)
  log_model("AffinityNet", model, args)

  model = model.to(DEVICE)
  model.eval()

  if GPUS_COUNT > 1:
    print(f"GPUs={GPUS_COUNT}")
    model = torch.nn.DataParallel(model)

  print(f'loading weights from {model_path}')
  load_model(model, model_path, parallel=GPUS_COUNT > 1)

  try:
    run(model, dataset)
  except KeyboardInterrupt:
    print("\ninterrupted")
  else:
    print(f"\n{TAG}/{args.domain} done")
