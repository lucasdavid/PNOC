# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>
# modified by Sierkinhane <sierkinhane@163.com>

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import wandb
from core.training import saliency_validation_step
from core.ccam import SimMaxLoss, SimMinLoss
from core.networks import *
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
from tools.general.json_utils import *
from tools.general.time_utils import *

parser = argparse.ArgumentParser()

# Dataset
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--dataset', default='voc12', choices=datasets.DATASOURCES)
parser.add_argument('--data_dir', required=True, type=str)
parser.add_argument('--cams_dir', default='/experiments/predictions/resnest101@ra/', type=str)
parser.add_argument('--domain_train', default=None, type=str)
parser.add_argument('--domain_valid', default=None, type=str)

# Network
parser.add_argument('--architecture', default='resnet50', type=str)
parser.add_argument('--weights', default='imagenet', type=str)
parser.add_argument('--mode', default='normal', type=str)  # fix
parser.add_argument('--trainable-stem', default=True, type=str2bool)
parser.add_argument('--dilated', default=False, type=str2bool)
parser.add_argument('--restore', default=None, type=str)

# Hyperparameter
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument("--first_epoch", default=0, type=int)
parser.add_argument('--max_epoch', default=10, type=int)
parser.add_argument('--accumulate_steps', default=1, type=int)
parser.add_argument("--mixed_precision", default=False, type=str2bool)
parser.add_argument('--validate', default=True, type=str2bool)
parser.add_argument('--validate_max_steps', default=None, type=int)

parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--wd', default=1e-4, type=float)
parser.add_argument('--label_smoothing', default=0, type=float)

parser.add_argument('--image_size', default=448, type=int)
parser.add_argument('--print_ratio', default=0.25, type=float)

parser.add_argument('--tag', default='', type=str)
parser.add_argument('--augment', default='', type=str)

parser.add_argument('--alpha', type=float, default=0.25)
parser.add_argument('--hint_w', type=float, default=1.0)

parser.add_argument('--cams_mode', type=str, choices=["npy", "png"], default="npy")
parser.add_argument('--fg_threshold', type=float, default=0.4)
parser.add_argument('--bg_threshold', type=float, default=0.05)

import cv2

cv2.setNumThreads(0)

try:
  GPUS = os.environ["CUDA_VISIBLE_DEVICES"]
except KeyError:
  GPUS = "0"
GPUS = GPUS.split(",")
GPUS_COUNT = len(GPUS)

IS_POSITIVE = True
THRESHOLDS = np.arange(0.05, 0.51, 0.05).astype(float).tolist()

if __name__ == '__main__':
  args = parser.parse_args()

  TAG = args.tag
  SEED = args.seed
  DEVICE = args.device if torch.cuda.is_available() else "cpu"
  BATCH_TRAIN = args.batch_size
  BATCH_VALID = 32

  wb_run = wandb_utils.setup(TAG, args)
  log_config(vars(args), TAG)

  model_dir = create_directory('./experiments/models/')
  model_path = model_dir + f'{TAG}.pth'

  set_seed(args.seed)

  print('[i] {}'.format(TAG))
  print()

  ts = datasets.custom_data_source(args.dataset, args.data_dir, args.domain_train, masks_dir=args.cams_dir, split="train")
  vs = datasets.custom_data_source(args.dataset, args.data_dir, args.domain_valid, split="valid")
  tt, tv = datasets.get_ccam_transforms(int(args.image_size * 1.15), args.image_size)
  if args.cams_mode == "npy":
    train_dataset = datasets.CAMsDataset(ts, transform=tt)
    interp = "bicubic"
  else:
    train_dataset = datasets.SaliencyDataset(ts, transform=tt)
    interp = "nearest"

  valid_dataset = datasets.SaliencyDataset(vs, transform=tv)
  # TODO: test mixup and cutmix in C2AM
  # train_dataset = datasets.apply_augmentation(train_dataset, args.augment, args.image_size, args.cutmix_prob, args.mixup_prob)
  train_loader = DataLoader(train_dataset, batch_size=BATCH_TRAIN, num_workers=args.num_workers, shuffle=True, drop_last=True)
  valid_loader = DataLoader(valid_dataset, batch_size=BATCH_VALID, num_workers=args.num_workers, drop_last=True)
  log_dataset(args.dataset, train_dataset, tt, tv)

  print("Foreground hints threshold:", args.fg_threshold, '(unused, as png cams are being loaded)' if args.cams_mode == "png" else "")

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
  )
  if args.restore:
    print(f"Restoring weights from {args.restore}")
    model.load_state_dict(torch.load(args.restore), strict=True)
  log_model("CCAM", model, args)

  param_groups, param_names = model.get_parameter_groups(with_names=True)
  log_opt_params("CCAM", param_names)

  model = model.to(DEVICE)
  model.train()

  if GPUS_COUNT > 1:
    print('[i] the number of gpu : {}'.format(GPUS_COUNT))
    model = torch.nn.DataParallel(model)

  # Loss, Optimizer
  hint_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none').to(DEVICE)

  criterion = [
    SimMaxLoss(metric='cos', alpha=args.alpha).to(DEVICE),
    SimMinLoss(metric='cos').to(DEVICE),
    SimMaxLoss(metric='cos', alpha=args.alpha).to(DEVICE),
  ]

  optimizer = get_optimizer(args.lr, args.wd, int(step_max // args.accumulate_steps), param_groups)
  scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)

  # Train
  miou_best = -1
  train_timer = Timer()
  train_metrics = MetricsContainer(['loss', 'positive_loss', 'negative_loss', 'hint_loss'])

  for epoch in range(args.max_epoch):
    for step, (_, images, _, cam_hints) in enumerate(tqdm(train_loader, f"Epoch {epoch}", mininterval=2.0)):
      with torch.autocast(device_type=DEVICE, enabled=args.mixed_precision):

        fg_feats, bg_feats, ccams = model(images.to(DEVICE))

        loss1 = criterion[0](fg_feats)
        loss2 = criterion[1](bg_feats, fg_feats)
        loss3 = criterion[2](bg_feats)

        # CAM Hints
        hints_dtype = cam_hints.dtype
        cam_hints = F.interpolate(cam_hints.float(), ccams.shape[2:], mode=interp).to(hints_dtype)

        # Using foreground cues:
        if args.cams_mode == "npy":
          fg_likely = (cam_hints >= args.fg_threshold).to(DEVICE)
        else:
          fg_likely = ((cam_hints != 0) & (cam_hints != 255)).to(DEVICE)

        # loss_h := -log(sigmoid(output[fg_likely]))
        output_fg = ccams[fg_likely]

        target_fg = torch.ones_like(output_fg)
        target_fg = label_smoothing(target_fg, args.label_smoothing).to(DEVICE)

        loss_h = hint_loss_fn(output_fg, target_fg).mean()

        # Using both foreground and background cues:
        # bg_likely = cam_hints < args.bg_threshold
        # fg_likely = cam_hints >= args.fg_threshold
        # mk_likely = (bg_likely | fg_likely).to(DEVICE)
        # target = torch.zeros_like(cam_hints)
        # target[fg_likely] = 1.
        # loss_h = hint_loss_fn(output, target)
        # loss_h = loss_h[mk_likely].sum() / mk_likely.float().sum().detach()

        # Back-propagation
        loss = args.hint_w * loss_h + (loss1 + loss2 + loss3)

      scaler.scale(loss).backward()

      if (step + 1) % args.accumulate_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

      # region logging

      do_logging = (step + 1) % step_log == 0
      do_validation = args.validate and (step + 1) % step_valid == 0

      train_metrics.update(
        {
          'loss': loss.item(),
          'hint_loss': loss_h.item(),
          'positive_loss': loss1.item() + loss3.item(),
          'negative_loss': loss2.item(),
        }
      )

      if do_logging:
        ccams = torch.sigmoid(ccams).cpu().float()
        # visualize_heatmap(TAG, images.clone().detach(), ccams, 0, step)
        loss, positive_loss, negative_loss, loss_h = train_metrics.get(clear=True)
        lr = float(get_learning_rate_from_optimizer(optimizer))

        data = {
          'epoch': epoch,
          'max_epoch': args.max_epoch,
          'iteration': step + 1,
          'learning_rate': lr,
          'loss': loss,
          'positive_loss': positive_loss,
          'negative_loss': negative_loss,
          'hint_loss': loss_h,
          'time': train_timer.tok(clear=True),
        }
        wandb.log({f"train/{k}": v for k, v in data.items()}, commit=not do_validation)

        print(
          'Epoch[{epoch:,}/{max_epoch:,}] iteration={iteration:,} lr={learning_rate:.4f} '
          'loss={loss:.4f} loss_p={positive_loss:.4f} loss_n={negative_loss:.4f} loss_h={hint_loss:.4f} '
          'time={time:.0f}sec'.format(**data)
        )

      # endregion

    # region evaluation
    if do_validation:
      model.eval()
      with torch.autocast(device_type=DEVICE, enabled=args.mixed_precision):
        metric_results = saliency_validation_step(model, valid_loader, THRESHOLDS, DEVICE, args.validate_max_steps)
        metric_results["iteration"] = step + 1
      model.train()

      wandb.log({f"val/{k}": v for k, v in metric_results.items()})
      print(*(f"{metric}={value}" for metric, value in metric_results.items()))

      if metric_results["miou"] > miou_best:
        miou_best = metric_results["miou"]
        for k in ("threshold", "miou", "iou"):
          wandb.run.summary[f"val/best_{k}"] = metric_results[k]

      save_model(model, model_path, parallel=GPUS_COUNT > 1)
      # endregion

  print(TAG)
  wb_run.finish()
