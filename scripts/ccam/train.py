# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>
# modified by Sierkinhane <sierkinhane@163.com>

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

import datasets
import wandb
from core.ccam import SimMaxLoss, SimMinLoss
from core.networks import *
from core.training import saliency_validation_step
from tools.ai.augment_utils import *
from tools.ai.demo_utils import *
from tools.ai.evaluate_utils import *
from tools.ai.log_utils import *
from tools.ai.optim_utils import *
from tools.ai.randaugment import *
from tools.ai.torch_utils import *
from tools.general import wandb_utils
from tools.general.cam_utils import *
from tools.general.io_utils import *
from tools.general.time_utils import *

parser = argparse.ArgumentParser()

# Dataset
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--dataset', default='voc12', choices=datasets.DATASOURCES)
parser.add_argument('--data_dir', required=True, type=str)
parser.add_argument('--domain_train', default=None, type=str)
parser.add_argument('--domain_valid', default=None, type=str)

# Network
parser.add_argument('--architecture', default='resnet50', type=str)
parser.add_argument('--weights', default='imagenet', type=str)
parser.add_argument('--mode', default='normal', type=str)  # fix
parser.add_argument('--regularization', default=None, type=str)  # kernel_usage
parser.add_argument('--trainable-stem', default=True, type=str2bool)
parser.add_argument('--dilated', default=False, type=str2bool)
parser.add_argument('--stage4_out_features', default=1024, type=int)
parser.add_argument('--restore', default=None, type=str)

# Hyperparameter
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--batch_size_val', default=32, type=int)
parser.add_argument('--first_epoch', default=0, type=int)
parser.add_argument('--max_epoch', default=10, type=int)
parser.add_argument('--depth', default=50, type=int)
parser.add_argument('--accumulate_steps', default=1, type=int)
parser.add_argument("--mixed_precision", default=False, type=str2bool)

parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--wd', default=1e-4, type=float)

parser.add_argument('--image_size', default=448, type=int)
parser.add_argument('--print_ratio', default=0.25, type=float)

parser.add_argument('--tag', default='', type=str)
parser.add_argument('--augment', default='', type=str)
parser.add_argument('--cutmix_prob', default=1.0, type=float)
parser.add_argument('--mixup_prob', default=1.0, type=float)

parser.add_argument('--alpha', type=float, default=0.25)

import cv2

cv2.setNumThreads(0)

try:
  GPUS = os.environ["CUDA_VISIBLE_DEVICES"]
except KeyError:
  GPUS = "0"
GPUS = GPUS.split(",")
GPUS_COUNT = len(GPUS)

IS_POSITIVE = True
THRESHOLDS = np.arange(0., 0.50, 0.05).astype(float).tolist()

if __name__ == '__main__':
  args = parser.parse_args()

  TAG = args.tag
  SEED = args.seed
  DEVICE = args.device if torch.cuda.is_available() else "cpu"
  SIZE = args.image_size

  wb_run = wandb_utils.setup(TAG, args)
  log_config(vars(args), TAG)

  data_dir = create_directory('./experiments/data/')
  model_dir = create_directory('./experiments/models/')
  cam_path = create_directory(f'./experiments/images/{TAG}')
  create_directory(cam_path + '/train')
  create_directory(cam_path + '/test')
  create_directory(cam_path + '/train/colormaps')
  create_directory(cam_path + '/test/colormaps')

  data_path = data_dir + f'{TAG}.json'
  model_path = model_dir + f'{TAG}.pth'

  set_seed(SEED)

  ts = datasets.custom_data_source(args.dataset, args.data_dir, args.domain_train, split="train")
  vs = datasets.custom_data_source(args.dataset, args.data_dir, args.domain_valid, split="valid")
  tt, tv = datasets.get_classification_transforms(512, 512, args.image_size, args.augment)
  train_dataset = datasets.ClassificationDataset(ts, transform=tt)
  valid_dataset = datasets.SegmentationDataset(vs, transform=tv)
  train_dataset = datasets.apply_augmentation(train_dataset, args.augment, args.image_size, args.cutmix_prob, args.mixup_prob)
  train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True)
  valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size_val, num_workers=args.num_workers, drop_last=True)
  log_dataset(args.dataset, train_dataset, tt, tv)

  step_valid = len(train_loader)
  step_log = int(step_valid * args.print_ratio)
  step_init = args.first_epoch * step_valid
  step_max = args.max_epoch * step_valid
  print(f"Iterations: first={step_init} logging={step_log} validation={step_valid} max={step_max}")

  # Network
  model = CCAM(
    args.architecture,
    weights=args.weights,
    mode=args.mode,
    dilated=args.dilated,
    stage4_out_features=args.stage4_out_features,
  )
  if args.restore:
    print(f'Restoring weights from {args.restore}')
    model.load_state_dict(torch.load(args.restore), strict=True)
  log_model("CCAM", model, args)

  ccam_param_groups = model.get_parameter_groups()
  log_opt_params("CCAM", ccam_param_groups)

  model = model.to(DEVICE)
  model.train()

  if GPUS_COUNT > 1:
    print(f"GPUs={GPUS_COUNT}")
    model = torch.nn.DataParallel(model)

  # Loss, Optimizer
  criterion = [
    SimMaxLoss(metric='cos', alpha=args.alpha).to(DEVICE),
    SimMinLoss(metric='cos').to(DEVICE),
    SimMaxLoss(metric='cos', alpha=args.alpha).to(DEVICE)
  ]

  optimizer = get_optimizer(args.lr, args.wd, int(step_max // args.accumulate_steps), ccam_param_groups)
  scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)

  # Train
  train_timer = Timer()
  train_metrics = MetricsContainer(['loss', 'positive_loss', 'negative_loss'])

  best_train_mIoU = -1

  for epoch in range(args.max_epoch):
    model.train()

    for step, (_, images, labels) in enumerate(train_loader):
      with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=args.mixed_precision):
        fg_feats, bg_feats, ccams = model(images.to(DEVICE))

        loss1 = criterion[0](fg_feats)
        loss2 = criterion[1](bg_feats, fg_feats)
        loss3 = criterion[2](bg_feats)

        loss = loss1 + loss2 + loss3

      scaler.scale(loss).backward()

      if (step + 1) % args.accumulate_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

      if epoch == 0 and step == 600:
        IS_POSITIVE = check_positive(ccams)
        print(f"Is Negative: {IS_POSITIVE}")
      if IS_POSITIVE:
        ccams = 1 - ccams

      # region logging
      do_logging = (step + 1) % step_log == 0
      do_validation = (step + 1) % step_valid == 0

      train_metrics.update(
        {
          'loss': loss.item(),
          'positive_loss': loss1.item() + loss3.item(),
          'negative_loss': loss2.item(),
        }
      )

      if do_logging:
        ccams = torch.sigmoid(ccams)
        visualize_heatmap(TAG, images.clone().detach(), ccams, 0, step)
        loss, positive_loss, negative_loss = train_metrics.get(clear=True)
        lr = float(get_learning_rate_from_optimizer(optimizer))

        data = {
          'epoch': epoch,
          'iteration': step + 1,
          'learning_rate': lr,
          'loss': loss,
          'positive_loss': positive_loss,
          'negative_loss': negative_loss,
          'time': train_timer.tok(clear=True),
        }
        wandb.log({f"train/{k}": v for k, v in data.items()}, commit=not do_validation)

        print(
          'Epoch[{epoch:,}/{max_epoch:,}] iteration={iteration:,} lr={learning_rate:.4f} '
          'loss={loss:.4f} loss_p={positive_loss:.4f} loss_n={negative_loss:.4f} '
          'time={time:.0f}sec'.format(max_epoch=args.max_epoch, **data)
        )

      # endregion

    # region evaluation
    model.eval()
    with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=args.mixed_precision):
      threshold, miou, iou, val_time = saliency_validation_step(model, valid_loader, THRESHOLDS, DEVICE)

    if best_train_mIoU == -1 or best_train_mIoU < miou:
      best_train_mIoU = miou
      wandb.run.summary["train/best_t"] = threshold
      wandb.run.summary["train/best_miou"] = miou
      wandb.run.summary["train/best_iou"] = iou

    data = {
      'iteration': step + 1,
      'threshold': threshold,
      'train_sal_mIoU': miou,
      'train_sal_iou': iou,
      'best_train_sal_mIoU': best_train_mIoU,
      'time': val_time,
    }
    wandb.log({f"val/{k}": v for k, v in data.items()})

    print(
      'iteration={iteration:,}\n'
      'threshold={threshold:.2f}\n'
      'train_sal_mIoU={train_sal_mIoU:.2f}%\n'
      'train_sal_iou ={train_sal_iou}\n'
      'best_train_sal_mIoU={best_train_sal_mIoU:.2f}%\n'
      'time={time:.0f}sec'.format(**data)
    )

    save_model(model, model_path, parallel=GPUS_COUNT > 1)
    # endregion

  print(TAG)
  wb_run.finish()
