import argparse
from copy import deepcopy
import os
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

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
from tools.general.time_utils import *
from core.training import (
  priors_validation_step,
  classification_validation_step,
)

parser = argparse.ArgumentParser()

# Dataset
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--dataset', default='voc12', choices=datasets.DATASOURCES)
parser.add_argument('--data_dir', required=True, type=str)
parser.add_argument('--domain_train', default=None, type=str)
parser.add_argument('--domain_valid', default=None, type=str)

# Network
parser.add_argument('--architecture', default='resnet50', type=str)
parser.add_argument('--mode', default='normal', type=str)  # fix
parser.add_argument('--trainable-stem', default=True, type=str2bool)
parser.add_argument('--trainable-backbone', default=True, type=str2bool)
parser.add_argument('--dilated', default=False, type=str2bool)
parser.add_argument('--restore', default=None, type=str)
parser.add_argument('--backbone_weights', default="imagenet", type=str)

# Hyperparameter
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument("--first_epoch", default=0, type=int)
parser.add_argument('--max_epoch', default=15, type=int)
parser.add_argument('--accumulate_steps', default=1, type=int)
parser.add_argument('--mixed_precision', default=False, type=str2bool)
parser.add_argument('--amp_min_scale', default=None, type=float)
parser.add_argument('--validate', default=True, type=str2bool)
parser.add_argument('--validate_priors', default=True, type=str2bool)
parser.add_argument('--validate_max_steps', default=None, type=int)
parser.add_argument('--validate_thresholds', default=None, type=str)

parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--wd', default=1e-4, type=float)
parser.add_argument('--label_smoothing', default=0, type=float)
parser.add_argument('--optimizer', default="sgd", choices=OPTIMIZERS_NAMES)
parser.add_argument('--lr_alpha_scratch', default=10., type=float)
parser.add_argument('--lr_alpha_bias', default=2., type=float)
parser.add_argument('--class_weight', default=None, type=str)
parser.add_argument('--ema', default=False, type=str2bool)
parser.add_argument('--ema_steps', default=32, type=int)
parser.add_argument('--ema_warmup', default=1, type=int)
parser.add_argument('--ema_decay', default=0.99, type=float)

parser.add_argument('--image_size', default=512, type=int)
parser.add_argument('--min_image_size', default=320, type=int)
parser.add_argument('--max_image_size', default=640, type=int)

parser.add_argument('--print_ratio', default=0.1, type=float)

parser.add_argument('--tag', default='', type=str)
parser.add_argument('--augment', default='', type=str)
parser.add_argument('--cutmix_prob', default=1.0, type=float)
parser.add_argument('--mixup_prob', default=1.0, type=float)

import cv2

cv2.setNumThreads(0)

try:
  GPUS = os.environ["CUDA_VISIBLE_DEVICES"]
except KeyError:
  GPUS = "0"
GPUS = GPUS.split(",")
GPUS_COUNT = len(GPUS)
THRESHOLDS = list(np.arange(0.10, 0.50, 0.05))

if __name__ == '__main__':
  args = parser.parse_args()

  TAG = args.tag
  SEED = args.seed
  DEVICE = args.device if torch.cuda.is_available() else "cpu"
  if DEVICE == "cpu":
    args.mixed_precision = False
  if args.validate_thresholds:
    THRESHOLDS = list(map(float, args.validate_thresholds.split(",")))

  if args.class_weight and args.class_weight != "none":
    CLASS_WEIGHT = torch.Tensor(list(map(float, args.class_weight.split(",")))).to(DEVICE)
  else:
    CLASS_WEIGHT = None

  wb_run = wandb_utils.setup(TAG, args)
  log_config(vars(args), TAG)

  data_dir = create_directory(f'./experiments/data/')
  model_dir = create_directory('./experiments/models/')
  model_path = model_dir + f'{TAG}.pth'

  set_seed(SEED)

  ts = datasets.custom_data_source(args.dataset, args.data_dir, args.domain_train, split="train")
  vs = datasets.custom_data_source(args.dataset, args.data_dir, args.domain_valid, split="valid")
  tt, tv = datasets.get_classification_transforms(args.min_image_size, args.max_image_size, args.image_size, args.augment, ts.classification_info.normalize_stats)
  train_dataset = datasets.ClassificationDataset(ts, transform=tt)
  valid_dataset = datasets.SegmentationDataset(vs, transform=tv)
  train_dataset = datasets.apply_augmentation(train_dataset, args.augment, args.image_size, args.cutmix_prob, args.mixup_prob)

  train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True, pin_memory=True, shuffle=True)
  valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True, pin_memory=True)
  train_iterator = datasets.Iterator(train_loader)
  log_dataset(args.dataset, train_dataset, tt, tv)

  step_val = len(train_loader)
  step_log = int(step_val * args.print_ratio)
  step_init = args.first_epoch * step_val
  step_max = args.max_epoch * step_val
  ema_warmup_steps = args.ema_warmup * int(step_val // args.accumulate_steps)
  print(f"Iterations: first={step_init} logging={step_log} validation={step_val} max={step_max}")

  # Network
  model = Classifier(
    args.architecture,
    train_dataset.info.num_classes,
    channels=train_dataset.info.channels,
    backbone_weights=args.backbone_weights,
    mode=args.mode,
    dilated=args.dilated,
    trainable_stem=args.trainable_stem,
    trainable_backbone=args.trainable_backbone,
  )
  if args.restore:
    print(f"Restoring weights from {args.restore}")
    model.load_state_dict(torch.load(args.restore), strict=True)
  log_model("Vanilla", model, args)

  param_groups, param_names = model.get_parameter_groups(with_names=True)
  model.train()

  if args.ema:
    # my_ema_avg_fn = partial(ema_avg_fun, optimizer=optimizer, decay=args.ema_decay, warmup=args.ema_warmup)
    # ema_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=my_ema_avg_fn)
    ema_model = deepcopy(model)
    for p in ema_model.parameters():
      p.requires_grad = False

    ema_model = model.to(DEVICE)
  
  model = model.to(DEVICE)

  if GPUS_COUNT > 1:
    print(f"GPUs={GPUS_COUNT}")
    model = torch.nn.DataParallel(model)

    if args.ema:
      ema_model = torch.nn.DataParallel(ema_model)

  # Loss, Optimizer
  class_loss_fn = torch.nn.MultiLabelSoftMarginLoss(weight=CLASS_WEIGHT, reduction='none').to(DEVICE)

  optimizer = get_optimizer(
    args.lr, args.wd, int(step_max // args.accumulate_steps), param_groups,
    algorithm=args.optimizer,
    alpha_scratch=args.lr_alpha_scratch,
    alpha_bias=args.lr_alpha_bias,
    start_step=int(step_init // args.accumulate_steps),
  )
  scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)

  log_opt_params("Vanilla", param_names)

  # Train
  train_meter = MetricsContainer(['loss', 'class_loss'])
  train_timer = Timer()
  miou_best = -1

  tqdm_bar = tqdm(range(step_init, step_max), 'Training', mininterval=2.0, ncols=80)
  for step in tqdm_bar:
    _, images, labels = train_iterator.get()

    with torch.autocast(device_type=DEVICE, enabled=args.mixed_precision):
      logits = model(images.to(DEVICE))

      labels = label_smoothing(labels, args.label_smoothing)
      class_loss = class_loss_fn(logits, labels.to(DEVICE)).mean()
      loss = class_loss

    scaler.scale(loss).backward()

    if (step + 1) % args.accumulate_steps == 0:
      scaler.step(optimizer)
      scaler.update()
      optimizer.zero_grad()

      if args.ema:
        with torch.no_grad():

          if optimizer.global_step+1 <= ema_warmup_steps:
            for t_params, s_params in zip(ema_model.parameters(), model.parameters()):
              t_params.copy_(s_params)
          else:
            ema_decay = min(1 - 1 / (1 + optimizer.global_step - ema_warmup_steps), args.ema_decay)
            for t_params, s_params in zip(ema_model.parameters(), model.parameters()):
              t_params.copy_(ema_decay * t_params + (1 - ema_decay) * s_params)

    train_meter.update({'loss': loss.item(), 'class_loss': class_loss.item()})

    epoch = step // step_val
    do_logging = (step + 1) % step_log == 0
    do_validation = args.validate and (step + 1) % step_val == 0

    loss, class_loss = train_meter.get(clear=True)
    tqdm_bar.set_description(f"[Epoch={epoch} Loss={loss:.5f}] ")

    if do_logging:
      learning_rate = float(get_learning_rate_from_optimizer(optimizer))

      data = {
        'iteration': step + 1,
        'learning_rate': learning_rate,
        'loss': loss,
        'class_loss': class_loss,
        'time': train_timer.tok(clear=True),
      }

      wb_logs = {f"train/{k}": v for k, v in data.items()}
      wb_logs["train/epoch"] = epoch
      wandb.log(wb_logs, commit=not do_validation)

      print(
        'step={iteration:,} '
        'lr={learning_rate:.4f} '
        'loss={loss:.4f} '
        'class_loss={class_loss:.4f} '
        'time={time:.0f}sec'.format(**data)
      )

    if do_validation:
      valid_model = model if not args.ema or optimizer.global_step+1 <= ema_warmup_steps else ema_model
      valid_model.eval()
      with torch.autocast(device_type=DEVICE, enabled=args.mixed_precision):
        if args.validate_priors:
          metric_results = priors_validation_step(
            valid_model, valid_loader, train_dataset.info, THRESHOLDS, DEVICE, args.validate_max_steps
          )
        else:
          metric_results = classification_validation_step(
            valid_model, valid_loader, train_dataset.info, DEVICE, args.validate_max_steps
          )
      metric_results["iteration"] = step + 1
      valid_model.train()

      wandb.log({f"val/{k}": v for k, v in metric_results.items()})
      print(*(f"{metric}={value}" for metric, value in metric_results.items()))

      if args.validate_priors and metric_results["miou"] > miou_best:
        miou_best = metric_results["miou"]
        for k in ("threshold", "miou", "iou"):
          wandb.run.summary[f"val/best_{k}"] = metric_results[k]

      save_model(valid_model, model_path, parallel=GPUS_COUNT > 1)

  print(TAG)
  wb_run.finish()
