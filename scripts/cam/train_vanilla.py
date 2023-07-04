import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import wandb
from core.networks import *
from core.training import priors_validation_step
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
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--dataset', default='voc12', choices=datasets.DATASOURCES)
parser.add_argument('--data_dir', required=True, type=str)
parser.add_argument('--train_domain', default=None, type=str)
parser.add_argument('--valid_domain', default=None, type=str)

# Network
parser.add_argument('--architecture', default='resnet50', type=str)
parser.add_argument('--mode', default='normal', type=str)  # fix
parser.add_argument('--regularization', default=None, type=str)  # kernel_usage
parser.add_argument('--trainable-stem', default=True, type=str2bool)
parser.add_argument('--dilated', default=False, type=str2bool)
parser.add_argument('--restore', default=None, type=str)

# Hyperparameter
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument("--first_epoch", default=0, type=int)
parser.add_argument('--max_epoch', default=15, type=int)
parser.add_argument('--validate', default=True, type=str2bool)

parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--wd', default=1e-4, type=float)
parser.add_argument('--label_smoothing', default=0, type=float)

parser.add_argument('--image_size', default=512, type=int)
parser.add_argument('--min_image_size', default=320, type=int)
parser.add_argument('--max_image_size', default=640, type=int)

parser.add_argument('--print_ratio', default=0.25, type=float)

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
  DEVICE = args.device

  wb_run = wandb_utils.setup(TAG, args, tags=[args.dataset, args.architecture, f"ls:{args.label_smoothing}"])
  log_config(vars(args), TAG)

  log_dir = create_directory(f'./experiments/logs/')
  data_dir = create_directory(f'./experiments/data/')
  model_dir = create_directory('./experiments/models/')

  log_path = log_dir + f'{TAG}.txt'
  model_path = model_dir + f'{TAG}.pth'

  set_seed(SEED)

  ts = datasets.custom_data_source(args.dataset, args.data_dir, args.train_domain, split="train")
  vs = datasets.custom_data_source(args.dataset, args.data_dir, args.valid_domain, split="valid")
  tt, tv = datasets.get_classification_transforms(args.min_image_size, args.max_image_size, args.image_size, args.augment)
  train_dataset = datasets.ClassificationDataset(ts, transform=tt)
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
  print(f"Iterations: first={step_init} logging={step_log} validation={step_val} max={step_max}")

  # Network
  model = Classifier(
    args.architecture,
    train_dataset.info.num_classes,
    mode=args.mode,
    dilated=args.dilated,
    regularization=args.regularization,
    trainable_stem=args.trainable_stem,
  )
  if args.restore:
    print(f"Restoring weights from {args.restore}")
    model.load_state_dict(torch.load(args.restore), strict=True)
  log_model("Vanilla", model, args)

  param_groups = model.get_parameter_groups()
  model = model.to(DEVICE)
  model.train()

  if GPUS_COUNT > 1:
    print(f"GPUs={GPUS_COUNT}")
    model = torch.nn.DataParallel(model)

  # Loss, Optimizer
  class_loss_fn = torch.nn.MultiLabelSoftMarginLoss(reduction='none').to(DEVICE)

  optimizer = get_optimizer(args.lr, args.wd, step_max, param_groups)
  log_opt_params("Vanilla", param_groups)

  # Train
  train_meter = MetricsContainer(['loss', 'class_loss'])
  train_timer = Timer()
  best_train_mIoU = -1

  for step in tqdm(range(step_init, step_max), 'Training', mininterval=2.0):
    image_ids, images, labels = train_iterator.get()

    optimizer.zero_grad()

    logits = model(images.to(DEVICE))

    labels = label_smoothing(labels, args.label_smoothing)
    class_loss = class_loss_fn(logits, labels.to(DEVICE)).mean()
    loss = class_loss

    loss.backward()
    optimizer.step()

    train_meter.update({'loss': loss.item(), 'class_loss': class_loss.item()})

    #################################################################################################
    # For Log
    #################################################################################################
    epoch = step // step_val
    do_logging = (step + 1) % step_log == 0
    do_validation = args.validate and (step + 1) % step_val == 0

    if (step + 1) % step_log == 0:
      loss, class_loss = train_meter.get(clear=True)
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

    #################################################################################################
    # Evaluation
    #################################################################################################
    if do_validation:
      model.eval()
      threshold, miou, iou, val_time = priors_validation_step(model, valid_loader, train_dataset.info.classes, THRESHOLDS, DEVICE)
      model.train()

      if best_train_mIoU == -1 or best_train_mIoU < miou:
        best_train_mIoU = miou
        wandb.run.summary["train/best_t"] = threshold
        wandb.run.summary["train/best_miou"] = miou
        wandb.run.summary["train/best_iou"] = iou

      data = {
        'iteration': step + 1,
        'threshold': threshold,
        'train_mIoU': miou,
        'best_train_mIoU': best_train_mIoU,
        'time': val_time,
      }
      wandb.log({f"val/{k}": v for k, v in data.items()})

      print(
        'iteration={iteration:,} '
        'threshold={threshold:.2f} '
        'train_mIoU={train_mIoU:.2f}% '
        'best_train_mIoU={best_train_mIoU:.2f}% '
        'time={time:.0f}sec'.format(**data)
      )

      print(f'saving weights `{model_path}`')
      save_model(model, model_path, parallel=GPUS_COUNT > 1)

  print(TAG)
  wb_run.finish()
