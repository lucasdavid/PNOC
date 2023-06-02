# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from core.datasets import *
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
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--dataset', default='voc12', choices=['voc12', 'coco14'])
parser.add_argument('--data_dir', default='../VOCtrainval_11-May-2012/', type=str)
parser.add_argument('--masks_dir', default='../VOCtrainval_11-May-2012/SegmentationMasks/', type=str)

# Network
parser.add_argument('--architecture', default='resnest269', type=str)
parser.add_argument('--mode', default='normal', type=str)
parser.add_argument('--dilated', default=False, type=str2bool)
parser.add_argument('--use_gn', default=True, type=str2bool)
parser.add_argument('--restore', default=None, type=str)

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

parser.add_argument('--print_ratio', default=0.5, type=float)

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
  DEVICE = args.device

  wb_run = wandb_utils.setup(TAG, args)
  log_config(vars(args), TAG)

  log_dir = create_directory(f'./experiments/logs/')
  data_dir = create_directory(f'./experiments/data/')
  model_dir = create_directory('./experiments/models/')

  log_path = log_dir + f'{args.tag}.txt'
  data_path = data_dir + f'{args.tag}.json'
  model_path = model_dir + f'{args.tag}.pth'

  set_seed(SEED)

  tt, tv = get_segmentation_transforms(args.min_image_size, args.max_image_size, args.image_size, args.augment)
  train_dataset, valid_dataset = get_segmentation_datasets(
    args.dataset,
    args.data_dir,
    args.augment,
    args.image_size,
    args.masks_dir,
    train_transforms=tt,
    valid_transforms=tv
  )

  train_loader = DataLoader(
    train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True
  )
  valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=1, shuffle=False, drop_last=True)
  log_dataset(args.dataset, train_dataset, tt, tv)

  step_val = len(train_loader)
  step_log = int(step_val * args.print_ratio)
  step_init = args.first_epoch * step_val
  step_max = args.max_epoch * step_val
  print(f'Iterations: first={step_init} logging={step_log} validation={step_val} max={step_max}')

  # Network
  model = DeepLabV3Plus(
    model_name=args.architecture,
    num_classes=train_dataset.info.num_classes + 1,
    mode=args.mode,
    dilated=args.dilated,
    use_group_norm=args.use_gn
  )
  if args.restore:
    print(f'Restoring weights from {args.restore}')
    model.load_state_dict(torch.load(args.restore), strict=True)
  log_model('DeepLabV3+', model, args)

  param_groups = model.get_parameter_groups()

  model = model.to(DEVICE)
  model.train()

  if GPUS_COUNT > 1:
    print(f'GPUs={GPUS_COUNT}')
    model = torch.nn.DataParallel(model)
    # for sync bn
    # patch_replication_callback(model)

  load_model_fn = lambda: load_model(model, model_path, parallel=GPUS_COUNT > 1)
  save_model_fn = lambda: save_model(model, model_path, parallel=GPUS_COUNT > 1)

  # Loss, Optimizer
  class_loss_fn = nn.CrossEntropyLoss(ignore_index=255, label_smoothing=args.label_smoothing).to(DEVICE)

  opt = get_optimizer(args.lr, args.wd, step_max, param_groups)
  log_opt_params('DeepLabV3+', param_groups)

  scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)

  # Train
  data_dic = {
    'train': [],
    'validation': [],
  }

  train_timer = Timer()
  eval_timer = Timer()

  train_meter = MetricsContainer(['loss'])
  miou_best_ = -1

  def evaluate(loader, classes):
    eval_timer.tik()

    iou_meter = Calculator_For_mIoU(train_dataset.info.classes)

    with torch.no_grad():
      for images, labels in loader:
        images = images.to(DEVICE)

        with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=args.mixed_precision):
          logits = model(images)
          predictions = torch.argmax(logits, dim=1)

        for batch_index in range(images.size()[0]):
          pred_mask = to_numpy(predictions[batch_index])
          gt_mask = to_numpy(labels[batch_index])

          h, w = pred_mask.shape
          gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)

          iou_meter.add(pred_mask, gt_mask)

    miou, _, iou, *_ = iou_meter.get(clear=True, detail=True)
    iou = [round(iou[c], 2) for c in classes]

    return miou, iou

  train_iterator = Iterator(train_loader)
  # torch.autograd.set_detect_anomaly(True)

  for step in tqdm(range(step_init, step_max), 'Training'):
    images, targets = train_iterator.get()
    images, targets = images.to(DEVICE), targets.to(DEVICE)

    with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=args.mixed_precision):
      logits = model(images)
      loss = class_loss_fn(logits, targets)

    scaler.scale(loss).backward()

    if (step + 1) % args.accumulate_steps == 0:
      scaler.step(opt)
      scaler.update()
      opt.zero_grad()

    train_meter.update({
      'loss': loss.item(),
    })

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
      data_dic['train'].append(data)
      write_json(data_path, data_dic)

      wandb.log({f'train/{k}': v for k, v in data.items()} | {'train/epoch': epoch}, step=step, commit=not do_validation)

      print('step={iteration:,} lr={learning_rate:.4f} loss={loss:.4f} time={time:.0f} sec'.format(**data))

    if do_validation:
      model.eval()
      miou, iou = evaluate(valid_loader, train_dataset.info.classes)
      model.train()

      if miou_best_ < miou:
        miou_best_ = miou
        wandb.run.summary['val/best_miou'] = miou
        wandb.run.summary['val/best_iou'] = iou

        print(f'saving weights `{model_path}`\n')
        save_model_fn()

      data = {
        'iteration': step + 1,
        'mIoU': miou,
        'iou': iou,
        'best_valid_mIoU': miou_best_,
        'time': eval_timer.tok(clear=True),
      }
      data_dic['validation'].append(data)
      write_json(data_path, data_dic)
      wandb.log({f'val/{k}': v for k, v in data.items()})

      print(
        'step={iteration:,} mIoU={mIoU:.2f}% best_valid_mIoU={best_valid_mIoU:.2f}% time={time:.0f}sec'.format(**data)
      )

  write_json(data_path, data_dic)
  print(TAG)
  wb_run.finish()
