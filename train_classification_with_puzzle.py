# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import argparse
import copy
import math
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

from core.datasets import *
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

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--dataset', default='voc12', choices=['voc12', 'coco14'])
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

###############################################################################
# Hyperparameter
###############################################################################
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--max_epoch', default=15, type=int)

parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--wd', default=1e-4, type=float)

parser.add_argument('--image_size', default=512, type=int)
parser.add_argument('--min_image_size', default=320, type=int)
parser.add_argument('--max_image_size', default=640, type=int)

parser.add_argument('--print_ratio', default=0.25, type=float)

parser.add_argument('--tag', default='', type=str)
parser.add_argument('--augment', default='', type=str)
parser.add_argument('--cutmix_prob', default=1.0, type=float)

# For Puzzle-CAM
parser.add_argument('--num_pieces', default=4, type=int)

# 'cl_pcl'
# 'cl_re'
# 'cl_conf'
# 'cl_pcl_re'
# 'cl_pcl_re_conf'
parser.add_argument('--loss_option', default='cl_pcl_re', type=str)

parser.add_argument('--level', default='feature', type=str)

parser.add_argument('--re_loss', default='L1_Loss', type=str)  # 'L1_Loss', 'L2_Loss'
parser.add_argument('--re_loss_option', default='masking', type=str)  # 'none', 'masking', 'selection'

# parser.add_argument('--branches', default='0,0,0,0,0,1', type=str)

parser.add_argument('--alpha', default=1.0, type=float)
parser.add_argument('--alpha_schedule', default=0.50, type=float)

if __name__ == '__main__':
  ###################################################################################
  # Arguments
  ###################################################################################
  args = parser.parse_args()

  log_dir = create_directory(f'./experiments/logs/')
  data_dir = create_directory(f'./experiments/data/')
  model_dir = create_directory('./experiments/models/')
  tensorboard_dir = create_directory(f'./experiments/tensorboards/{args.tag}/')

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
  META = read_json(f'./data/{args.dataset}/meta.json')
  CLASSES = np.asarray(META['class_names'])
  NUM_CLASSES = META['classes']

  tt, tv = get_transforms(args.min_image_size, args.max_image_size, args.image_size, args.augment)
  train_dataset, valid_dataset = get_dataset_classification(
    args.dataset, args.data_dir, args.augment, args.image_size, args.cutmix_prob, tt, tv
  )

  train_loader = DataLoader(
    train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True
  )
  valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=1, drop_last=True)

  log('[i] The number of class is {}'.format(NUM_CLASSES))
  log('[i] train_transform is {}'.format(tt))
  log('[i] test_transform is {}'.format(tv))
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
  model = Classifier(
    args.architecture,
    NUM_CLASSES,
    mode=args.mode,
    dilated=args.dilated,
    regularization=args.regularization,
    trainable_stem=args.trainable_stem,
  )
  param_groups = model.get_parameter_groups()

  gap_fn = model.global_average_pooling_2d

  model = model.cuda()
  model.train()

  log('[i] Architecture is {}'.format(args.architecture))
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

  ###################################################################################
  # Loss, Optimizer
  ###################################################################################
  class_loss_fn = nn.MultiLabelSoftMarginLoss(reduction='none').cuda()

  if args.re_loss == 'L1_Loss':
    re_loss_fn = L1_Loss
  else:
    re_loss_fn = L2_Loss

  log('[i] The number of pretrained weights : {}'.format(len(param_groups[0])))
  log('[i] The number of pretrained bias : {}'.format(len(param_groups[1])))
  log('[i] The number of scratched weights : {}'.format(len(param_groups[2])))
  log('[i] The number of scratched bias : {}'.format(len(param_groups[3])))

  optimizer = get_optimizer(args.lr, args.wd, max_iteration, param_groups)

  #################################################################################################
  # Train
  #################################################################################################
  data_dic = {'train': [], 'validation': []}

  train_timer = Timer()
  eval_timer = Timer()

  train_meter = Average_Meter(['loss', 'class_loss', 'p_class_loss', 're_loss', 'conf_loss', 'alpha'])

  best_train_mIoU = -1
  thresholds = list(np.arange(0.10, 0.50, 0.05))

  def evaluate(loader):
    imagenet_mean, imagenet_std = imagenet_stats()

    model.eval()
    eval_timer.tik()

    meter_dic = {th: Calculator_For_mIoU(CLASSES) for th in thresholds}

    with torch.no_grad():
      length = len(loader)
      for step, (images, labels, gt_masks) in enumerate(loader):
        images = images.cuda()
        labels = labels.cuda()

        _, features = model(images, with_cam=True)

        # features = resize_for_tensors(features, images.size()[-2:])
        # gt_masks = resize_for_tensors(gt_masks, features.size()[-2:], mode='nearest')

        mask = labels.unsqueeze(2).unsqueeze(3)
        cams = (make_cam(features) * mask)

        # for visualization
        if step == 0:
          obj_cams = cams.max(dim=1)[0]

          for b in range(8):
            image = to_numpy(images[b])
            cam = to_numpy(obj_cams[b])

            image = denormalize(image, imagenet_mean, imagenet_std)[..., ::-1]
            h, w, c = image.shape

            cam = (cam * 255).astype(np.uint8)
            cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
            cam = colormap(cam)

            image = cv2.addWeighted(image, 0.5, cam, 0.5, 0)[..., ::-1]
            image = image.astype(np.float32) / 255.

            writer.add_image('CAM/{}'.format(b + 1), image, iteration, dataformats='HWC')

        for batch_index in range(images.size()[0]):
          # c, h, w -> h, w, c
          cam = to_numpy(cams[batch_index]).transpose((1, 2, 0))
          gt_mask = to_numpy(gt_masks[batch_index])

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
    best_iou = {}

    for th in thresholds:
      mIoU, mIoU_foreground, iou, *_ = meter_dic[th].get(clear=True, detail=True)
      if best_mIoU < mIoU:
        best_th = th
        best_mIoU = mIoU
        best_iou = iou

    return best_th, best_mIoU, best_iou

  writer = SummaryWriter(tensorboard_dir)
  train_iterator = Iterator(train_loader)

  loss_option = args.loss_option.split('_')

  for iteration in range(max_iteration):
    images, labels = train_iterator.get()
    images, labels = images.cuda(), labels.cuda()

    ###############################################################################
    # Normal
    ###############################################################################
    logits, features = model(images, with_cam=True)

    ###############################################################################
    # Puzzle Module
    ###############################################################################
    tiled_images = tile_features(images, args.num_pieces)

    tiled_logits, tiled_features = model(tiled_images, with_cam=True)

    re_features = merge_features(tiled_features, args.num_pieces, args.batch_size)

    ###############################################################################
    # Losses
    ###############################################################################
    if args.level == 'cam':
      features = make_cam(features)
      re_features = make_cam(re_features)

    class_loss = class_loss_fn(logits, labels).mean()

    if 'pcl' in loss_option:
      p_class_loss = class_loss_fn(gap_fn(re_features), labels).mean()
    else:
      p_class_loss = torch.zeros(1).cuda()

    if 're' in loss_option:
      if args.re_loss_option == 'masking':
        class_mask = labels.unsqueeze(2).unsqueeze(3)
        re_loss = re_loss_fn(features, re_features) * class_mask
        re_loss = re_loss.mean()
      elif args.re_loss_option == 'selection':
        re_loss = 0.
        for b_index in range(labels.size()[0]):
          class_indices = labels[b_index].nonzero(as_tuple=True)
          selected_features = features[b_index][class_indices]
          selected_re_features = re_features[b_index][class_indices]

          re_loss_per_feature = re_loss_fn(selected_features, selected_re_features).mean()
          re_loss += re_loss_per_feature
        re_loss /= labels.size()[0]
      else:
        re_loss = re_loss_fn(features, re_features).mean()
    else:
      re_loss = torch.zeros(1).cuda()

    if 'conf' in loss_option:
      conf_loss = shannon_entropy_loss(tiled_logits)
    else:
      conf_loss = torch.zeros(1).cuda()

    if args.alpha_schedule == 0.0:
      alpha = args.alpha
    else:
      alpha = min(args.alpha * iteration / (max_iteration * args.alpha_schedule), args.alpha)

    loss = class_loss + p_class_loss + alpha * re_loss + conf_loss
    #################################################################################################

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_meter.add(
      {
        'loss': loss.item(),
        'class_loss': class_loss.item(),
        'p_class_loss': p_class_loss.item(),
        're_loss': re_loss.item(),
        'conf_loss': conf_loss.item(),
        'alpha': alpha,
      }
    )

    #################################################################################################
    # For Log
    #################################################################################################
    if (iteration + 1) % log_iteration == 0:
      loss, class_loss, p_class_loss, re_loss, conf_loss, alpha = train_meter.get(clear=True)
      learning_rate = float(get_learning_rate_from_optimizer(optimizer))

      data = {
        'iteration': iteration + 1,
        'learning_rate': learning_rate,
        'alpha': alpha,
        'loss': loss,
        'class_loss': class_loss,
        'p_class_loss': p_class_loss,
        're_loss': re_loss,
        'conf_loss': conf_loss,
        'time': train_timer.tok(clear=True),
      }
      data_dic['train'].append(data)
      write_json(data_path, data_dic)

      log(
        '[i] \
                iteration={iteration:,}, \
                learning_rate={learning_rate:.4f}, \
                alpha={alpha:.2f}, \
                loss={loss:.4f}, \
                class_loss={class_loss:.4f}, \
                p_class_loss={p_class_loss:.4f}, \
                re_loss={re_loss:.4f}, \
                conf_loss={conf_loss:.4f}, \
                time={time:.0f}sec'.format(**data)
      )

      writer.add_scalar('Train/loss', loss, iteration)
      writer.add_scalar('Train/class_loss', class_loss, iteration)
      writer.add_scalar('Train/p_class_loss', p_class_loss, iteration)
      writer.add_scalar('Train/re_loss', re_loss, iteration)
      writer.add_scalar('Train/conf_loss', conf_loss, iteration)
      writer.add_scalar('Train/learning_rate', learning_rate, iteration)
      writer.add_scalar('Train/alpha', alpha, iteration)

    #################################################################################################
    # Evaluation
    #################################################################################################
    if (iteration + 1) % val_iteration == 0:
      threshold, mIoU, iou = evaluate(valid_loader)

      if best_train_mIoU == -1 or best_train_mIoU < mIoU:
        best_train_mIoU = mIoU

      data = {
        'iteration': iteration + 1,
        'threshold': threshold,
        'train_mIoU': mIoU,
        'train_iou': iou,
        'best_train_mIoU': best_train_mIoU,
        'time': eval_timer.tok(clear=True),
      }
      data_dic['validation'].append(data)
      write_json(data_path, data_dic)

      log(
        'iteration       = {iteration:,} '
        'threshold       = {threshold:.2f}'
        'best_train_mIoU = {best_train_mIoU:.2f}%'
        'train_mIoU      = {train_mIoU:.2f}%'
        'train_iou       = {train_iou}'
        'time={time:.0f}sec'.format(**data)
      )

      writer.add_scalar('Evaluation/threshold', threshold, iteration)
      writer.add_scalar('Evaluation/train_mIoU', mIoU, iteration)
      writer.add_scalar('Evaluation/best_train_mIoU', best_train_mIoU, iteration)

  write_json(data_path, data_dic)
  writer.close()

  log(f'[i] {args.tag} saved at {model_path}')
  save_model_fn()
