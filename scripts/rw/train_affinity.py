# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import argparse
import os

import torch
from torch.utils.data import DataLoader

import datasets
import wandb
from core.networks import *
from tools.ai.augment_utils import *
from tools.ai.demo_utils import *
from tools.ai.evaluate_utils import *
from tools.ai.log_utils import *
from tools.ai.optim_utils import *
from tools.ai.randaugment import *
from tools.ai.torch_utils import *
from tools.general import wandb_utils
from tools.general.io_utils import *
from tools.general.json_utils import *
from tools.general.time_utils import *

parser = argparse.ArgumentParser()

# Dataset
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--dataset', default='voc12', choices=datasets.DATASOURCES)
parser.add_argument('--data_dir', required=True, type=str)
parser.add_argument('--train_domain', default=None, type=str)

# Network
parser.add_argument('--architecture', default='resnet50', type=str)
parser.add_argument('--mode', default='fix', type=str)  # normal
parser.add_argument('--regularization', default=None, type=str)  # kernel_usage
parser.add_argument('--trainable-stem', default=False, type=str2bool)  # only false allowed.
parser.add_argument('--dilated', default=False, type=str2bool)
parser.add_argument('--restore', default=None, type=str)

# Hyperparameter
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--first_epoch', default=0, type=int)
parser.add_argument('--max_epoch', default=3, type=int)
parser.add_argument('--accumulate_steps', default=1, type=int)

parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--wd', default=1e-4, type=float)
parser.add_argument('--image_size', default=512, type=int)
parser.add_argument('--min_image_size', default=320, type=int)
parser.add_argument('--max_image_size', default=640, type=int)

parser.add_argument('--print_ratio', default=0.1, type=float)

parser.add_argument('--tag', default='', type=str)
parser.add_argument('--label_dir', default='./experiments/predictions/rn50@train_aug@aff', type=str)

import cv2

cv2.setNumThreads(0)

try:
  GPUS = os.environ["CUDA_VISIBLE_DEVICES"]
except KeyError:
  GPUS = "0"
GPUS = GPUS.split(",")
GPUS_COUNT = len(GPUS)

if __name__ == '__main__':
  # Arguments
  args = parser.parse_args()

  TAG = args.tag
  SEED = args.seed
  DEVICE = args.device

  wb_run = wandb_utils.setup(TAG, args)
  log_config(vars(args), TAG)

  data_dir = create_directory(f'./experiments/data/')
  model_dir = create_directory('./experiments/models/')

  data_path = data_dir + f'{args.tag}.json'
  model_path = model_dir + f'{args.tag}.pth'

  set_seed(args.seed)

  path_index = PathIndex(radius=10, default_size=(args.image_size // 4, args.image_size // 4))

  ts = datasets.custom_data_source(args.dataset, args.data_dir, args.train_domain, masks_dir=args.label_dir, split="train")
  tt = datasets.get_affinity_transforms(args.min_image_size, args.max_image_size, args.image_size)
  train_dataset = datasets.AffinityDataset(ts, path_index=path_index, transform=tt)

  train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True)
  train_iterator = datasets.Iterator(train_loader)
  log_dataset(args.dataset, train_dataset, tt, None)

  step_valid = len(train_loader)
  step_log = int(step_valid * args.print_ratio)
  step_init = args.first_epoch * step_valid
  step_max = args.max_epoch * step_valid
  print(f"Iterations: first={step_init} logging={step_log} validation={step_valid} max={step_max}")

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

  param_groups = list(model.edge_layers.parameters())

  model = model.to(DEVICE)
  model.train()

  if GPUS_COUNT > 1:
    print(f"GPUs={GPUS_COUNT}")
    model = torch.nn.DataParallel(model)

  # Loss, Optimizer
  optimizer = PolyOptimizer(
    [{
      'params': param_groups,
      'lr': args.lr,
      'weight_decay': args.wd
    }], lr=args.lr, momentum=0.9, max_step=step_max
  )

  # Train
  train_timer = Timer()
  train_metrics = MetricsContainer(['loss', 'bg_loss', 'fg_loss', 'neg_loss'])

  torch.autograd.set_detect_anomaly(True)

  for step in range(step_init, step_max):
    images, labels = train_iterator.get()

    images = images.to(DEVICE)

    bg_pos_label = labels[0].to(DEVICE, non_blocking=True)
    fg_pos_label = labels[1].to(DEVICE, non_blocking=True)
    neg_label = labels[2].to(DEVICE, non_blocking=True)

    # Affinity Matrix
    edge, aff = model(images, with_affinity=True)

    pos_aff_loss = (-1) * torch.log(aff + 1e-5)
    neg_aff_loss = (-1) * torch.log(1. + 1e-5 - aff)

    bg_pos_aff_loss = torch.sum(bg_pos_label * pos_aff_loss) / (torch.sum(bg_pos_label) + 1e-5)
    fg_pos_aff_loss = torch.sum(fg_pos_label * pos_aff_loss) / (torch.sum(fg_pos_label) + 1e-5)

    pos_aff_loss = bg_pos_aff_loss / 2 + fg_pos_aff_loss / 2
    neg_aff_loss = torch.sum(neg_label * neg_aff_loss) / (torch.sum(neg_label) + 1e-5)

    loss = (pos_aff_loss + neg_aff_loss) / 2

    loss.backward()

    if (step + 1) % args.accumulate_steps == 0:
      optimizer.step()
      optimizer.zero_grad()

    train_metrics.update(
      {
        'loss': loss.item(),
        'bg_loss': bg_pos_aff_loss.item(),
        'fg_loss': fg_pos_aff_loss.item(),
        'neg_loss': neg_aff_loss.item(),
      }
    )

    epoch = step // step_valid
    do_logging = (step + 1) % step_log == 0
    do_validation = (step + 1) % step_valid == 0

    if do_logging:
      loss, bg_loss, fg_loss, neg_loss = train_metrics.get(clear=True)
      learning_rate = float(get_learning_rate_from_optimizer(optimizer))

      data = {
        'iteration': step + 1,
        'learning_rate': learning_rate,
        'loss': loss,
        'bg_loss': bg_loss,
        'fg_loss': fg_loss,
        'neg_loss': neg_loss,
        'time': train_timer.tok(clear=True),
      }

      wb_logs = {f"train/{k}": v for k, v in data.items()}
      wb_logs["train/epoch"] = epoch
      wandb.log(wb_logs)

      print(
        'iteration={iteration:,} '
        'learning_rate={learning_rate:.4f} '
        'loss={loss:.4f} '
        'bg_loss={bg_loss:.4f} '
        'fg_loss={fg_loss:.4f} '
        'neg_loss={neg_loss:.4f} '
        'time={time:.0f}sec'.format(**data)
      )

    if do_validation:
      save_model(model, model_path, parallel=GPUS_COUNT > 1)

  print(f'saving weights `{model_path}`')
  save_model(model, model_path, parallel=GPUS_COUNT > 1)

  print(TAG)
  wb_run.finish()
