# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import argparse
import copy
import os
import random
import shutil
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from core.datasets import *
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
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--data_dir', default='../VOCtrainval_11-May-2012/', type=str)
parser.add_argument('--masks_dir', default='../VOCtrainval_11-May-2012/SegmentationMasks/', type=str)

###############################################################################
# Network
###############################################################################
parser.add_argument('--backbone', default='resnest269', type=str)
parser.add_argument('--mode', default='fix', type=str)
parser.add_argument('--dilated', default=False, type=str2bool)
parser.add_argument('--use_gn', default=True, type=str2bool)

###############################################################################
# Hyperparameter
###############################################################################
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--max_epoch', default=50, type=int)

parser.add_argument('--lr', default=0.007, type=float)
parser.add_argument('--wd', default=4e-5, type=float)

parser.add_argument('--image_size', default=512, type=int)
parser.add_argument('--min_image_size', default=256, type=int)
parser.add_argument('--max_image_size', default=1024, type=int)

parser.add_argument('--print_ratio', default=0.1, type=float)

parser.add_argument('--tag', default='', type=str)

parser.add_argument(
  '--label_name',
  default='resnet50@seed=0@aug=Affinity_ResNet50@ep=3@nesterov@train_aug@beta=10@exp_times=8@rw',
  type=str
)

if __name__ == '__main__':
  ###################################################################################
  # Arguments
  ###################################################################################
  args = parser.parse_args()

  log_dir = create_directory(f'./experiments/logs/')
  data_dir = create_directory(f'./experiments/data/')
  model_dir = create_directory('./experiments/models/')
  pred_dir = './experiments/predictions/{}/'.format(args.label_name)

  log_path = log_dir + f'{args.tag}.txt'
  data_path = data_dir + f'{args.tag}.json'
  model_path = model_dir + f'{args.tag}.pth'

  set_seed(args.seed)
  log = lambda string='': log_print(string, log_path)

  log('[i] {}'.format(args.tag))
  log()

  ###################################################################################
  # Transform, Dataset, DataLoader
  ###################################################################################

  tt, tv = get_segmentation_transforms(
    args.min_image_size, args.max_image_size, args.image_size, args.augment
  )

  train_dataset, valid_dataset = get_segmentation_datasets(
    args.dataset, args.data_dir, args.augment, args.image_size, args.masks_dir,
    train_transforms=tt,
    valid_transforms=tv
  )

  train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True)
  valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=1, shuffle=False, drop_last=True)

  log('[i] The number of class is {}'.format(train_dataset.info.num_classes))
  log('[i] train_transform is {}'.format(tt))
  log()

  val_iteration = len(train_loader)
  log_iteration = int(val_iteration * args.print_ratio)
  max_iteration = args.max_epoch * val_iteration

  log('[i] log_iteration : {:,}'.format(log_iteration))
  log('[i] val_iteration : {:,}'.format(val_iteration))
  log('[i] max_iteration : {:,}'.format(max_iteration))

  ###################################################################################
  # Network
  ###################################################################################
  model = DeepLabV3Plus(
    model_name=args.backbone,
    num_classes=train_dataset.info.num_classes + 1,
    mode=args.mode,
    dilated=args.dilated,
    use_group_norm=args.use_gn
  )
  param_groups = model.get_parameter_groups()
  model = model.cuda()
  model.train()

  log('[i] Total Params: %.2fM' % (calculate_parameters(model)))
  log()

  try:
    use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
  except KeyError:
    use_gpu = '0'

  the_number_of_gpu = len(use_gpu.split(','))
  if the_number_of_gpu > 1:
    log('[i] the number of gpu : {}'.format(the_number_of_gpu))
    model = nn.DataParallel(model)

    # for sync bn
    # patch_replication_callback(model)

  load_model_fn = lambda: load_model(model, model_path, parallel=the_number_of_gpu > 1)
  save_model_fn = lambda: save_model(model, model_path, parallel=the_number_of_gpu > 1)
  save_model_fn_for_backup = lambda: save_model(
    model, model_path.replace('.pth', f'_backup.pth'), parallel=the_number_of_gpu > 1
  )

  ###################################################################################
  # Loss, Optimizer
  ###################################################################################
  class_loss_fn = nn.CrossEntropyLoss(ignore_index=255).cuda()

  optimizer = get_optimizer(args.lr, args.wd, max_iteration, param_groups)

  #################################################################################################
  # Train
  #################################################################################################
  data_dic = {
    'train': [],
    'validation': [],
  }

  train_timer = Timer()
  eval_timer = Timer()

  train_meter = MetricsContainer(['loss'])

  best_valid_mIoU = -1

  def evaluate(loader):
    model.eval()
    eval_timer.tik()

    meter = Calculator_For_mIoU(train_dataset.info.classes)

    with torch.no_grad():
      length = len(loader)
      for step, (images, labels) in enumerate(loader):
        images = images.cuda()
        labels = labels.cuda()

        logits = model(images)
        predictions = torch.argmax(logits, dim=1)

        for batch_index in range(images.size()[0]):
          pred_mask = to_numpy(predictions[batch_index])
          gt_mask = to_numpy(labels[batch_index])

          h, w = pred_mask.shape
          gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)

          meter.add(pred_mask, gt_mask)

    return meter.get(clear=True)

  train_iterator = Iterator(train_loader)

  torch.autograd.set_detect_anomaly(True)

  for step in range(max_iteration):
    model.train()

    images, labels = train_iterator.get()
    images, labels = images.cuda(), labels.cuda()

    logits = model(images)

    if 'Seg' in args.architecture:
      labels = resize_for_tensors(labels.float().unsqueeze(1),
                                  logits.size()[2:], 'nearest', None)[:, 0, :, :]
      labels = labels.long().cuda()

      # print(labels.size(), labels.min(), labels.max())

    loss = class_loss_fn(logits, labels)
    #################################################################################################

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_meter.update({
      'loss': loss.item(),
    })

    #################################################################################################
    # For Log
    #################################################################################################
    if (step + 1) % log_iteration == 0:
      loss = train_meter.get(clear=True)
      lr = float(get_learning_rate_from_optimizer(optimizer))

      data = {
        'iteration': step + 1,
        'learning_rate': lr,
        'loss': loss,
        'time': train_timer.tok(clear=True),
      }
      data_dic['train'].append(data)
      write_json(data_path, data_dic)

      log('step={iteration:,} lr={learning_rate:.4f} loss={loss:.4f} time={time:.0f} sec'.format(**data))

    #################################################################################################
    # Evaluation
    #################################################################################################
    if (step + 1) % val_iteration == 0:
      mIoU, _ = evaluate(valid_loader)

      if best_valid_mIoU == -1 or best_valid_mIoU < mIoU:
        best_valid_mIoU = mIoU

        save_model_fn()
        log('[i] save model')

      data = {
        'iteration': step + 1,
        'mIoU': mIoU,
        'best_valid_mIoU': best_valid_mIoU,
        'time': eval_timer.tok(clear=True),
      }
      data_dic['validation'].append(data)
      write_json(data_path, data_dic)

      log(
        'step={iteration:,} mIoU={mIoU:.2f}% best_valid_mIoU={best_valid_mIoU:.2f}% time={time:.0f}sec'.format(**data)
      )

  write_json(data_path, data_dic)
  print(args.tag)
