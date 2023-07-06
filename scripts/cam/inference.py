# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import argparse
import copy
import os
import sys
from typing import List

import numpy as np
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
parser.add_argument('--dataset', default='voc12', choices=datasets.DATASOURCES)
parser.add_argument('--data_dir', required=True, type=str)
parser.add_argument('--sample_ids', default=None, type=str)
parser.add_argument('--architecture', default='resnet50', type=str)
parser.add_argument('--mode', default='normal', type=str)  # fix
parser.add_argument('--regularization', default=None, type=str)  # kernel_usage
parser.add_argument('--trainable-stem', default=True, type=str2bool)
parser.add_argument('--dilated', default=False, type=str2bool)
parser.add_argument('--restore', default=None, type=str)
parser.add_argument('--tag', default='', type=str)
parser.add_argument('--weights', default='', type=str)
parser.add_argument('--domain', default='train', type=str)
parser.add_argument('--scales', default='0.5,1.0,1.5,2.0', type=str)
parser.add_argument('--exclude_bg_images', default=True, type=str2bool)

try:
  GPUS = os.environ["CUDA_VISIBLE_DEVICES"]
except KeyError:
  GPUS = "0"
GPUS = GPUS.split(",")
GPUS_COUNT = len(GPUS)

normalize_fn = Normalize(*datasets.imagenet_stats())


def run(args):
  ds = datasets.custom_data_source(args.dataset, args.data_dir, args.domain)
  dataset = datasets.PathsDataset(ds, ignore_bg_images=args.exclude_bg_images)
  print(f'{TAG} dataset={args.dataset} num_classes={dataset.info.num_classes}')

  model = Classifier(
    args.architecture,
    dataset.info.num_classes,
    mode=args.mode,
    dilated=args.dilated,
    regularization=args.regularization,
    trainable_stem=args.trainable_stem,
  )
  load_model(model, WEIGHTS_PATH, map_location=torch.device(DEVICE))
  model.eval()

  dataset = [Subset(dataset, np.arange(i, len(dataset), GPUS_COUNT)) for i in range(GPUS_COUNT)]
  scales = [float(scale) for scale in args.scales.split(',')]

  if GPUS_COUNT > 1:
    multiprocessing.spawn(_work, nprocs=GPUS_COUNT, args=(model, dataset, scales, PREDS_DIR, DEVICE), join=True)
  else:
    _work(0, model, dataset, scales, PREDS_DIR, DEVICE)


def _work(
    process_id: int,
    model: Classifier,
    dataset: List[datasets.PathsDataset],
    scales: List[float],
    preds_dir: str,
    device: str,
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
      label = data_source.get_label(image_id)

      W, H = image.size

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
      keys = np.pad(keys + 1, (1, 0), mode='constant')

      try:
        np.save(npy_path, {"keys": keys, "cam": cams_st.cpu(), "hr_cam": cams_hr.cpu().numpy()})
      except:
        if os.path.exists(npy_path):
          os.remove(npy_path)
        raise


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

  PREDS_DIR = create_directory(f'./experiments/predictions/{TAG}/')
  WEIGHTS_PATH = './experiments/models/' + f'{args.weights or args.tag}.pth'

  set_seed(SEED)
  run(args)
