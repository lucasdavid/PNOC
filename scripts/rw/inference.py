# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import multiprocessing
from torch.utils.data import Subset
from pickle import UnpicklingError

from core.aff_utils import propagate_to_edge
from datasets import *
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
# parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--dataset', default='voc12', choices=['voc12', 'coco14'])
parser.add_argument('--data_dir', default='../VOCtrainval_11-May-2012/', type=str)
parser.add_argument('--domain', default='train', type=str)
parser.add_argument('--exclude_bg_images', default=True, type=str2bool)
parser.add_argument('--sample_ids', default=None, type=str)
parser.add_argument('--mixed_precision', default=False, type=str2bool)

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

parser.add_argument('--verbose', default=0, type=int)


try:
  GPUS = os.environ["CUDA_VISIBLE_DEVICES"]
except KeyError:
  GPUS = "0"
GPUS = GPUS.split(",")
GPUS_COUNT = len(GPUS)


def run(args):
  normalize_fn = Normalize(*imagenet_stats())
  path_index = PathIndex(radius=10, default_size=(args.image_size // 4, args.image_size // 4))

  dataset = get_inference_dataset(args.dataset, args.data_dir, args.domain, ignore_bg_images=args.exclude_bg_images, sample_ids=args.sample_ids)

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

  print(f'loading weights from {model_path}')
  load_model(model, model_path)
  model.eval()

  dataset = [Subset(dataset, np.arange(i, len(dataset), GPUS_COUNT)) for i in range(GPUS_COUNT)]

  if GPUS_COUNT > 1:
    multiprocessing.spawn(_work, nprocs=GPUS_COUNT, args=(model, dataset, normalize_fn, CAMS_DIR, PREDS_DIR, DEVICE, args), join=True)
  else:
    _work(0, model, dataset, normalize_fn, CAMS_DIR, PREDS_DIR, DEVICE, args)

def _work(process_id, model, dataset, normalize_fn, cams_dir, preds_dir, device, args):
  import cv2
  cv2.setNumThreads(0)

  dataset = dataset[process_id]
  length = len(dataset)
  errors = []
  missing = []
  processed = 0
  skipped = 0

  with torch.no_grad(), torch.cuda.device(process_id):
    model.cuda()

    for step, (x, image_id, label) in enumerate(dataset):
      W, H = x.size

      cam_path = os.path.join(cams_dir, image_id + '.npy')
      npy_path = os.path.join(preds_dir, image_id + '.npy')

      if os.path.isfile(npy_path):
        skipped += 1
        continue

      if args.verbose >= 2:
        print(f"id={image_id}", end=" ", flush=True)

      if label.sum() == 0:
        if args.verbose >= 2:
          print(f"{image_id} skipped (bg)")
        continue

      try:
        cam_dict = np.load(cam_path, allow_pickle=True).item()
      except UnpicklingError as error:
        errors.append(cam_path)
        if args.verbose >= 3:
          print(f"{image_id} skipped (cam error={error})")
        continue
      except FileNotFoundError:
        missing.append(cam_path)
        if args.verbose >= 3:
          print(f"{image_id} skipped (cam missing)")
        continue

      if "cam" in cam_dict:
        cams = cam_dict['cam']
        cams = torch.as_tensor(cams)
      else:
        cams = cam_dict['hr_cam']
        cams = torch.as_tensor(cams)
        size = cams.shape[1:]
        cams = resize_for_tensors(cams.unsqueeze(0), get_strided_size(size, 4))[0]
        # print(f"{image_id} hr-cam resized {tuple(size)} -> {tuple(cams.shape)}")

      # preprocessing
      x = normalize_fn(x)
      x = x.transpose((2, 0, 1))
      x = torch.from_numpy(x)
      x = torch.stack([x, x.flip(-1)])
      x = x.to(device)

      with torch.autocast(device_type=device, dtype=torch.float16, enabled=args.mixed_precision):
        # inference
        edge = model.get_edge(x, image_size=args.image_size)

        # postprocessing
        rw = cams.to(device)
        try:
          rw = propagate_to_edge(rw, edge, beta=args.beta, exp_times=args.exp_times, radius=5)
        except RuntimeError as error:  # usually memory issues.
          print(error, file=sys.stderr)
          continue

        rw_up = F.interpolate(rw, scale_factor=4, mode='bilinear', align_corners=False)[..., 0, :H, :W]
        rw_up /= torch.max(rw_up)

      try:
        np.save(npy_path, {"keys": cam_dict['keys'], "rw": to_numpy(rw_up)})
      except:
        if os.path.exists(npy_path):
          os.remove(npy_path)
        raise
      processed += 1

      if args.verbose >= 2:
        print(f"{image_id} ok")

    if missing: print(f"{len(missing)} files were missing and were not processed:", *missing, sep='\n  - ', flush=True)
    if errors: print(f"{len(errors)} CAM files could not be read:", *errors, sep="\n  - ", flush=True)

    print(f"{processed} images successfully processed")


if __name__ == '__main__':
  try:
    multiprocessing.set_start_method('spawn')
  except RuntimeError:
    ...

  # Arguments
  args = parser.parse_args()

  SEED = args.seed
  DEVICE = args.device
  TAG = args.model_name
  TAG += '@train' if 'train' in args.domain else '@val'
  TAG += f"@beta={args.beta}@exp_times={args.exp_times}@rw"

  log_config(vars(args), TAG)

  CAMS_DIR = args.cam_dir
  PREDS_DIR = create_directory(f'./experiments/predictions/{TAG}/')

  model_path = './experiments/models/' + f'{args.model_name}.pth'

  set_seed(SEED)

  try:
    run(args)
  except KeyboardInterrupt:
    print("\ninterrupted")
  else:
    print(f"\n{TAG} ({args.domain}) done")
