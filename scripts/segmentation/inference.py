# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import argparse
import copy

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import multiprocessing
from torch.utils.data import Subset
from tqdm import tqdm

import datasets
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
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=4, type=int)

# Network
parser.add_argument('--backbone', default='resnest269', type=str)
parser.add_argument('--mode', default='fix', type=str)
parser.add_argument('--dilated', default=False, type=str2bool)
parser.add_argument('--use_gn', default=True, type=str2bool)

# Inference parameters
parser.add_argument('--tag', default='', type=str)
parser.add_argument("--pred_dir", default="", type=str)
parser.add_argument('--weights', default=None, type=str)
parser.add_argument('--dataset', default='voc12', choices=datasets.DATASOURCES)
parser.add_argument('--data_dir', required=True, type=str)
parser.add_argument('--domain', default='val', type=str)
parser.add_argument('--scales', default='0.5,1.0,1.5,2.0', type=str)
parser.add_argument('--crf_t', default=0, type=int)
parser.add_argument('--crf_gt_prob', default=0.7, type=float)

try:
  GPUS = os.environ["CUDA_VISIBLE_DEVICES"]
except KeyError:
  GPUS = "0"
GPUS = GPUS.split(",")
GPUS_COUNT = len(GPUS)


def run(args):
  print(TAG)
  print(f"Saving predictions for {args.dataset}/{args.domain} to '{PRED_DIR}'.")

  scales = [float(scale) for scale in args.scales.split(',')]

  ds = datasets.custom_data_source(args.dataset, args.data_dir, args.domain, segmentation=True)
  dataset = datasets.PathsDataset(ds)
  print(f'{TAG} dataset={args.dataset} num_classes={dataset.info.num_classes}')

  normalize_fn = Normalize(*datasets.imagenet_stats())

  # Network
  model = DeepLabV3Plus(
    model_name=args.backbone,
    num_classes=dataset.info.num_classes,
    mode=args.mode,
    use_group_norm=args.use_gn,
  )
  load_model(model, MODEL_PATH, parallel=False)
  model.eval()

  dataset = [Subset(dataset, np.arange(i, len(dataset), GPUS_COUNT)) for i in range(GPUS_COUNT)]
  scales = [float(scale) for scale in args.scales.split(',')]

  if GPUS_COUNT > 1:
    multiprocessing.spawn(_work, nprocs=GPUS_COUNT, args=(model, normalize_fn, dataset, scales, PRED_DIR, DEVICE, args), join=True)
  else:
    _work(0, model, normalize_fn, dataset, scales, PRED_DIR, DEVICE, args)


def forward_tta(model, images, image_size, device):
  images = images.to(device)
  logits = model(images)
  logits = resize_for_tensors(logits, image_size).cpu().float()
  logits = logits[0] + logits[1].flip(-1)

  return logits


def _work(process_id, model, normalize_fn, dataset, scales, preds_dir, device, args):
  dataset = dataset[process_id]
  data_source = dataset.dataset.data_source

  if process_id == 0:
    dataset = tqdm(dataset, mininterval=5.0)

  with torch.no_grad(), torch.cuda.device(process_id):
    model = model.cuda()

    for image_id, _, _ in dataset:
      image = data_source.get_image(image_id)
      W, H = image.size

      ps = []

      for scale in scales:
        x = copy.deepcopy(image)
        x = x.resize((round(W * scale), round(H * scale)), resample=PIL.Image.BICUBIC)

        x = normalize_fn(x)
        x = x.transpose((2, 0, 1))

        x = torch.from_numpy(x)
        x = torch.stack([x, x.flip(-1)])

        p = forward_tta(model, x, (H, W), device)
        ps.append(p)

      p = to_numpy(F.softmax(torch.stack(ps).mean(dim=0), dim=0))

      if args.crf_t > 0:
        x = np.array(image)
        p = crf_inference(x, p, t=args.crf_t, gt_prob=args.crf_gt_prob)
        image.close()

      p = np.argmax(p, axis=0)

      p = Image.fromarray(p.astype(np.uint8))
      p.save(os.path.join(preds_dir, image_id + '.png'))
      p.close()
      image.close()


if __name__ == '__main__':
  args = parser.parse_args()

  TAG = args.tag
  SEED = args.seed
  DEVICE = args.device if torch.cuda.is_available() else "cpu"

  set_seed(SEED)

  PRED_DIR = TAG
  PRED_DIR += '@train' if 'train' in args.domain else f'@{args.domain}'
  PRED_DIR += f'@scale={args.scales}'
  if args.crf_t > 0:
    PRED_DIR += f'@crf_t={args.crf_t}'
  PRED_DIR = create_directory(args.pred_dir or f'./experiments/predictions/{PRED_DIR}/')
  MODELS_DIR = create_directory('./experiments/models/')
  MODEL_PATH = args.weights or os.path.join(MODELS_DIR, f'{TAG}.pth')

  run(args)
