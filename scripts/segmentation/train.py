# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import wandb
from core.networks import *
from core.training import segmentation_validation_step
from tools.ai.augment_utils import *
from tools.ai.demo_utils import *
from tools.ai.evaluate_utils import *
from tools.ai.log_utils import *
from tools.ai.optim_utils import *
from tools.ai.randaugment import *
from tools.ai.torch_utils import *
from tools.general import wandb_utils
from tools.general.io_utils import *
from tools.general.time_utils import *

parser = argparse.ArgumentParser()

# Dataset
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--dataset', default='voc12', choices=datasets.DATASOURCES)
parser.add_argument('--data_dir', required=True, type=str)
parser.add_argument('--masks_dir', default='../VOCtrainval_11-May-2012/SegmentationMasks/', type=str)
parser.add_argument('--domain_train', default=None, type=str)
parser.add_argument('--domain_valid', default=None, type=str)

# Network
parser.add_argument('--architecture', default='resnest269', type=str)
parser.add_argument('--mode', default='normal', type=str)
parser.add_argument('--dilated', default=False, type=str2bool)
parser.add_argument('--use_gn', default=True, type=str2bool)
parser.add_argument('--restore', default=None, type=str)
parser.add_argument('--restore_strict', default=True, type=str2bool)

# Hyperparameter
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--first_epoch', default=0, type=int)
parser.add_argument('--max_epoch', default=50, type=int)
parser.add_argument('--accumulate_steps', default=1, type=int)
parser.add_argument('--mixed_precision', default=False, type=str2bool)

parser.add_argument('--lr', default=0.007, type=float)
parser.add_argument('--wd', default=4e-5, type=float)
parser.add_argument('--label_smoothing', default=0, type=float)

parser.add_argument('--image_size', default=512, type=int)
parser.add_argument('--min_image_size', default=256, type=int)
parser.add_argument('--max_image_size', default=1024, type=int)

parser.add_argument('--print_ratio', default=1.0, type=float)

parser.add_argument('--tag', default='', type=str)
parser.add_argument('--augment', default='', type=str)
parser.add_argument('--cutmix_prob', default=1.0, type=float)
parser.add_argument('--mixup_prob', default=1.0, type=float)

try:
  GPUS = os.environ['CUDA_VISIBLE_DEVICES']
except KeyError:
  GPUS = '0'
GPUS = GPUS.split(',')
GPUS_COUNT = len(GPUS)


if __name__ == '__main__':
  args = parser.parse_args()

  TAG = args.tag
  SEED = args.seed
  DEVICE = args.device if torch.cuda.is_available() else "cpu"

  wb_run = wandb_utils.setup(TAG, args)
  log_config(vars(args), TAG)

  log_dir = create_directory(f'./experiments/logs/')
  data_dir = create_directory(f'./experiments/data/')
  model_dir = create_directory('./experiments/models/')

  log_path = log_dir + f'{args.tag}.txt'
  data_path = data_dir + f'{args.tag}.json'
  model_path = model_dir + f'{args.tag}.pth'

  set_seed(SEED)

  ts = datasets.custom_data_source(args.dataset, args.data_dir, args.domain_train, masks_dir=args.masks_dir, split="train", segmentation=True)
  vs = datasets.custom_data_source(args.dataset, args.data_dir, args.domain_valid, masks_dir=args.masks_dir, split="valid", segmentation=True)
  tt, tv = datasets.get_segmentation_transforms(args.min_image_size, args.max_image_size, args.image_size, args.augment)
  train_dataset = datasets.SegmentationDataset(ts, transform=tt)
  valid_dataset = datasets.SegmentationDataset(vs, transform=tv)
  train_dataset = datasets.apply_augmentation(train_dataset, args.augment, args.image_size, args.cutmix_prob, args.mixup_prob)

  train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True)
  valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True)
  train_iterator = datasets.Iterator(train_loader)
  log_dataset(args.dataset, train_dataset, tt, tv)

  step_val = len(train_loader)
  step_log = int(step_val * args.print_ratio)
  step_init = args.first_epoch * step_val
  step_max = args.max_epoch * step_val
  print(f'Iterations: first={step_init} logging={step_log} validation={step_val} max={step_max}')

  # Network
  model = DeepLabV3Plus(
    model_name=args.architecture,
    num_classes=train_dataset.info.num_classes,
    mode=args.mode,
    dilated=args.dilated,
    use_group_norm=args.use_gn,
  )
  if args.restore:
    print(f'Restoring weights from {args.restore}')
    model.load_state_dict(torch.load(args.restore), strict=args.restore_strict)
  log_model('DeepLabV3+', model, args)

  param_groups = model.get_parameter_groups()

  model = model.to(DEVICE)
  model.train()

  if GPUS_COUNT > 1:
    print(f'GPUs={GPUS_COUNT}')
    model = torch.nn.DataParallel(model)
    # for sync bn
    # patch_replication_callback(model)

  # Loss, Optimizer
  class_loss_fn = nn.CrossEntropyLoss(ignore_index=255, label_smoothing=args.label_smoothing).to(DEVICE)

  opt = get_optimizer(args.lr, args.wd, int(step_max // args.accumulate_steps), param_groups)
  log_opt_params('DeepLabV3+', param_groups)

  scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)

  # Train
  train_timer = Timer()
  train_meter = MetricsContainer(['loss'])
  miou_best_ = -1

  # torch.autograd.set_detect_anomaly(True)

  for step in tqdm(range(step_init, step_max), 'Training', mininterval=2.0):
    _, images, _, targets = train_iterator.get()
    images, targets = images.to(DEVICE), targets.to(DEVICE)

    with torch.autocast(device_type=DEVICE, enabled=args.mixed_precision):
      logits = model(images)
      loss = class_loss_fn(logits, targets)

    scaler.scale(loss).backward()

    if (step + 1) % args.accumulate_steps == 0:
      scaler.step(opt)
      scaler.update()
      opt.zero_grad()

    train_meter.update({'loss': loss.item()})

    epoch = step // step_val
    do_logging = (step + 1) % step_log == 0
    do_validation = (step + 1) % step_val == 0

    if do_logging:
      loss = train_meter.get(clear=True)
      lr = float(get_learning_rate_from_optimizer(opt))

      data = {
        'iteration': step + 1,
        'learning_rate': lr,
        'loss': loss,
        'time': train_timer.tok(clear=True),
      }

      wb_logs = {f"train/{k}": v for k, v in data.items()}
      wb_logs["train/epoch"] = epoch
      wandb.log(wb_logs, commit=not do_validation)

      print('step={iteration:,} lr={learning_rate:.4f} loss={loss:.4f} time={time:.0f} sec'.format(**data))

    if do_validation:
      model.eval()
      with torch.autocast(device_type=DEVICE, enabled=args.mixed_precision):
        miou, iou, val_time = segmentation_validation_step(model, valid_loader, train_dataset.info.classes, DEVICE)
      model.train()

      if miou_best_ < miou:
        miou_best_ = miou
        wandb.run.summary['val/best_miou'] = miou
        wandb.run.summary['val/best_iou'] = iou

        save_model(model, model_path, parallel=GPUS_COUNT > 1)

      data = {
        'iteration': step + 1,
        'mIoU': miou,
        'iou': iou,
        'best_valid_mIoU': miou_best_,
        'time': val_time,
      }
      wandb.log({f'val/{k}': v for k, v in data.items()})

      print(
        'step={iteration:,} mIoU={mIoU:.2f}% best_valid_mIoU={best_valid_mIoU:.2f}% time={time:.0f}sec'.format(**data)
      )

  print(TAG)
  wb_run.finish()
