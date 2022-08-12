# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

from torch.backends import cudnn
cudnn.enabled = True

import argparse
import copy
from functools import partial
import os
import random
import shutil
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from torchvision import transforms

from core import occse
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
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--data_dir', default='../VOCtrainval_11-May-2012/', type=str)

###############################################################################
# Network
###############################################################################
parser.add_argument('--architecture', default='resnet50', type=str)
parser.add_argument('--mode', default='normal', type=str)  # fix
parser.add_argument('--regularization', default=None, type=str)  # kernel_usage
parser.add_argument('--trainable-stem', default=True, type=str2bool)
parser.add_argument('--dilated', default=False, type=str2bool)
parser.add_argument('--restore', default=None, type=str)

parser.add_argument('--oc-architecture', default='resnet50', type=str)
parser.add_argument('--oc-regularization', default=None, type=str)
parser.add_argument('--oc-pretrained', required=True, type=str)
parser.add_argument('--oc-strategy', default='random', type=str)
parser.add_argument('--oc-focal-momentum', default=0.9, type=float)
parser.add_argument('--oc-focal-gamma', default=2.0, type=float)

parser.add_argument('--oc-alpha', default=1.0, type=float)
parser.add_argument('--oc-alpha-init', default=0.3, type=float)
parser.add_argument('--oc-alpha-schedule', default=1.0, type=float)
parser.add_argument('--oc-k', default=1.0, type=float)
parser.add_argument('--oc-k-init', default=1.0, type=float)
parser.add_argument('--oc-k-schedule', default=0.0, type=float)

###############################################################################
# Hyperparameter
###############################################################################
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--max_epoch', default=15, type=int)

parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--wd', default=1e-4, type=float)
parser.add_argument('--nesterov', default=True, type=str2bool)

parser.add_argument('--image_size', default=512, type=int)
parser.add_argument('--min_image_size', default=320, type=int)
parser.add_argument('--max_image_size', default=640, type=int)

parser.add_argument('--print_ratio', default=0.5, type=float)

parser.add_argument('--tag', default='', type=str)
parser.add_argument('--augment', default='', type=str)

if __name__ == '__main__':
  ###################################################################################
  # Arguments
  ###################################################################################
  args = parser.parse_args()

  DEVICE = args.device
  SEED = args.seed
  TAG = args.tag
  SIZES = args.image_size
  BATCH = args.batch_size

  set_seed(SEED)

  log_dir = create_directory(f'./experiments/logs/')
  data_dir = create_directory(f'./experiments/data/')
  model_dir = create_directory('./experiments/models/')
  tensorboard_dir = create_directory(f'./experiments/tensorboards/{TAG}/')

  log_path = log_dir + f'{TAG}.txt'
  data_path = data_dir + f'{TAG}.json'
  model_path = model_dir + f'{TAG}.pth'

  log = partial(log_print, path=log_path)

  log_config(vars(args), title=f'Train OC-CSE `{TAG}`', print_fn=log)

  ###################################################################################
  # Transform, Dataset, DataLoader
  ###################################################################################
  in_stats = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

  train_transforms = [
    RandomResize(args.min_image_size, args.max_image_size),
    RandomHorizontalFlip(),
  ]
  if 'colorjitter' in args.augment:
    train_transforms.append(transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1))

  if 'randaugment' in args.augment:
    train_transforms.append(RandAugmentMC(n=2, m=10))

  train_transform = transforms.Compose(train_transforms + [
    Normalize(*in_stats),
    RandomCrop(SIZES),
    Transpose()
  ])
  test_transform = transforms.Compose([
    Normalize_For_Segmentation(*in_stats),
    Top_Left_Crop_For_Segmentation(SIZES),
    Transpose_For_Segmentation()
  ])

  META = read_json('./data/VOC_2012.json')
  CLASSES = META['classes']
  CLASS_NAMES = np.asarray(META['class_names'])

  train_dataset = VOC_Dataset_For_Classification(args.data_dir, 'train_aug', train_transform)

  train_dataset_for_seg = VOC_Dataset_For_Testing_CAM(args.data_dir, 'train', test_transform)
  valid_dataset_for_seg = VOC_Dataset_For_Testing_CAM(args.data_dir, 'val', test_transform)

  train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True)
  train_loader_for_seg = DataLoader(train_dataset_for_seg, batch_size=args.batch_size, num_workers=1, drop_last=True)

  log(f'[i] mean/std values are {in_stats}')
  log(f'[i] The number of class is {CLASSES}')
  log('[i] train_transform is {}'.format(train_transform))
  log('[i] test_transform is {}'.format(test_transform))
  log()

  val_iteration = len(train_loader)
  log_iteration = int(val_iteration * args.print_ratio)
  step_max = args.max_epoch * val_iteration

  # val_iteration = log_iteration
  log('[i] log_iteration : {:,}'.format(log_iteration))
  log('[i] val_iteration : {:,}'.format(val_iteration))
  log('[i] max_iteration : {:,}'.format(step_max))

  # Network
  classifier_args = dict(
    model_name=args.architecture,
    num_classes=CLASSES,
    mode=args.mode,
    dilated=args.dilated,
    regularization=args.regularization,
    trainable_stem=args.trainable_stem,
    # strides=(1, 2, 2, 1),
  )
  log_config(classifier_args, 'Classifier')

  model = Classifier(**classifier_args)
  # model.eval(); summary(model, (3, *SIZES))

  if args.restore:
    print(f'  restoring weights from {args.restore}')
    model.load_state_dict(torch.load(args.restore), strict=True)

  param_groups = model.get_parameter_groups()

  model = model.to(DEVICE)
  model.train()

  log('  total parameters: %.2fM' % (calculate_parameters(model)))
  log()

  # Ordinary Classifier.
  print(f'Build OC {args.oc_architecture} (weights from `{args.oc_pretrained}`)')
  oc_nn = Classifier(args.oc_architecture, CLASSES, mode=args.mode, regularization=args.oc_regularization)
  oc_nn.load_state_dict(torch.load(args.oc_pretrained, map_location=torch.device(DEVICE)), strict=True)
  
  oc_nn = oc_nn.to(DEVICE)
  oc_nn.eval()
  for child in oc_nn.children():
    for param in child.parameters():
      param.requires_grad = False

  try:
    use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
  except KeyError:
    use_gpu = '0'

  the_number_of_gpu = len(use_gpu.split(','))
  if the_number_of_gpu > 1:
    log('[i] the number of gpu : {}'.format(the_number_of_gpu))
    model = nn.DataParallel(model)

  load_model_fn = lambda: load_model(model, model_path, parallel=the_number_of_gpu > 1)
  save_model_fn = lambda: save_model(model, model_path, parallel=the_number_of_gpu > 1)

  ###################################################################################
  # Loss, Optimizer
  ###################################################################################
  class_loss_fn = nn.MultiLabelSoftMarginLoss(reduction='none').to(DEVICE)

  log('[i] The number of pretrained weights : {}'.format(len(param_groups[0])))
  log('[i] The number of pretrained bias : {}'.format(len(param_groups[1])))
  log('[i] The number of scratched weights : {}'.format(len(param_groups[2])))
  log('[i] The number of scratched bias : {}'.format(len(param_groups[3])))

  optimizer = PolyOptimizer(
    [
      {'params': param_groups[0],'lr': args.lr,'weight_decay': args.wd},
      {'params': param_groups[1],'lr': 2 * args.lr,'weight_decay': 0},
      {'params': param_groups[2],'lr': 10 * args.lr,'weight_decay': args.wd},
      {'params': param_groups[3],'lr': 20 * args.lr,'weight_decay': 0},
    ],
    lr=args.lr,
    momentum=0.9,
    weight_decay=args.wd,
    max_step=step_max,
    nesterov=args.nesterov
  )

  #################################################################################################
  # Train
  #################################################################################################
  data_dic = {'train': [], 'validation': []}

  train_timer = Timer()
  eval_timer = Timer()

  train_meter = Average_Meter(['loss', 'c_loss', 'o_loss', 'oc_alpha'])

  best_train_mIoU = -1
  thresholds = list(np.arange(0.1, 0.50, 0.05))

  choices = torch.ones(CLASSES).to(DEVICE)
  focal_factor = torch.ones(CLASSES).to(DEVICE)

  def evaluate(loader):
    model.eval()
    eval_timer.tik()

    meter_dic = {th: Calculator_For_mIoU('./data/VOC_2012.json') for th in thresholds}

    with torch.no_grad():
      length = len(loader)
      for step, (images, labels, gt_masks) in enumerate(loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        _, features = model(images, with_cam=True)

        # features = resize_for_tensors(features, images.size()[-2:])
        # gt_masks = resize_for_tensors(gt_masks, features.size()[-2:], mode='nearest')

        mask = labels.unsqueeze(2).unsqueeze(3)
        cams = (make_cam(features) * mask)

        # for visualization
        if step == 0:
          obj_cams = cams.max(dim=1)[0]

          for b in range(8):
            image = get_numpy_from_tensor(images[b])
            cam = get_numpy_from_tensor(obj_cams[b])

            image = denormalize(image, in_stats, in_s)[..., ::-1]
            h, w, c = image.shape

            cam = (cam * 255).astype(np.uint8)
            cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
            cam = colormap(cam)

            image = cv2.addWeighted(image, 0.5, cam, 0.5, 0)[..., ::-1]
            image = image.astype(np.float32) / 255.

            writer.add_image('CAM/{}'.format(b + 1), image, step, dataformats='HWC')

        for batch_index in range(images.size()[0]):
          # c, h, w -> h, w, c
          cam = get_numpy_from_tensor(cams[batch_index]).transpose((1, 2, 0))
          gt_mask = get_numpy_from_tensor(gt_masks[batch_index])

          h, w, c = cam.shape
          gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)

          for th in thresholds:
            bg = np.ones_like(cam[:, :, 0]) * th
            pred_mask = np.argmax(np.concatenate([bg[..., np.newaxis], cam], axis=-1), axis=-1)

            meter_dic[th].add(pred_mask, gt_mask)

        # break

        sys.stdout.write('\r# Evaluation [{}/{}] = {:.2f}%'.format(step + 1, length, (step + 1) / length * 100))
        sys.stdout.flush()

    print(' ')
    model.train()

    best_th = 0.0
    best_mIoU = 0.0

    for th in thresholds:
      mIoU, mIoU_foreground = meter_dic[th].get(clear=True)
      if best_mIoU < mIoU:
        best_th = th
        best_mIoU = mIoU

    return best_th, best_mIoU

  writer = SummaryWriter(tensorboard_dir)
  train_iterator = Iterator(train_loader)

  for step in range(step_max):
    images, labels = train_iterator.get()
    images, labels = images.to(DEVICE), labels.to(DEVICE)

    ao = linear_schedule(step, step_max, args.oc_alpha_init, args.oc_alpha, args.oc_alpha_schedule)

    #################################################################################################
    logits, features = model(images, with_cam=True)

    c_loss = class_loss_fn(logits, labels).mean()
    #################################################################################################

    # OC-CSE
    labels_mask = occse.split_label(labels, 1, choices, focal_factor, args.oc_strategy)
    labels_oc = labels - labels_mask
    cl_logits = oc_nn(occse.images_with_masked_objects(images, features, labels_mask))
    o_loss = class_loss_fn(cl_logits, labels_oc).mean()

    loss = (
      c_loss
      + ao * o_loss
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    occse.update_focal_factor(
      labels,
      labels_oc,
      cl_logits,
      focal_factor,
      momentum=args.oc_focal_momentum,
      gamma=args.oc_focal_gamma
    )

    # region logging
    train_meter.add({
      'loss': loss.item(),
      'c_loss': c_loss.item(),
      'o_loss': o_loss.item(),
      'oc_alpha': ao,
    })

    if (step + 1) % log_iteration == 0:
      (
        loss,
        c_loss,
        o_loss,
        ao,
      ) = train_meter.get(clear=True)
      
      lr = float(get_learning_rate_from_optimizer(optimizer))

      data = {
        'iteration': step + 1,
        'lr': lr,
        'loss': loss,
        'c_loss': c_loss,
        'o_loss': o_loss,
        'oc_alpha': ao,
        'time': train_timer.tok(clear=True),
        'focal_factor': focal_factor.cpu().detach().numpy().round(2).tolist()
      }
      data_dic['train'].append(data)
      write_json(data_path, data_dic)

      log(
        '\niteration  = {iteration:,}\n'
        'time         = {time:.0f} sec\n'
        'lr           = {lr:.4f}\n'
        'loss         = {loss:.4f}\n'
        'c_loss       = {c_loss:.4f}\n'
        'o_loss       = {o_loss:.4f}\n'
        'oc_alpha     = {oc_alpha:.4f}\n'
        'focal_factor = {focal_factor}'
        .format(**data)
      )

      writer.add_scalar('Train/loss', loss, step)
      writer.add_scalar('Train/c_loss', c_loss, step)
      writer.add_scalar('Train/o_loss', o_loss, step)
      writer.add_scalar('Train/learning_rate', lr, step)
      writer.add_scalar('Train/oc_alpha', ao, step)
    # endregion

    # region evaluation
    if (step + 1) % val_iteration == 0:
      threshold, mIoU = evaluate(train_loader_for_seg)

      if best_train_mIoU == -1 or best_train_mIoU < mIoU:
        best_train_mIoU = mIoU

      data = {
        'iteration': step + 1,
        'threshold': threshold,
        'train_mIoU': mIoU,
        'best_train_mIoU': best_train_mIoU,
        'time': eval_timer.tok(clear=True),
      }
      data_dic['validation'].append(data)
      write_json(data_path, data_dic)

      log(
        '\niteration       = {iteration:,}\n'
        'time            = {time:.0f} sec'
        'threshold       = {threshold:.2f}\n'
        'train_mIoU      = {train_mIoU:.2f}%\n'
        'best_train_mIoU = {best_train_mIoU:.2f}%\n'
        .format(**data)
      )

      writer.add_scalar('Evaluation/threshold', threshold, step)
      writer.add_scalar('Evaluation/train_mIoU', mIoU, step)
      writer.add_scalar('Evaluation/best_train_mIoU', best_train_mIoU, step)

      save_model_fn()
      log('[i] save model')
    # endregion

  write_json(data_path, data_dic)
  writer.close()
