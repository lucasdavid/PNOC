import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

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
parser.add_argument('--label_smoothing', default=0, type=float)

parser.add_argument('--image_size', default=512, type=int)
parser.add_argument('--min_image_size', default=320, type=int)
parser.add_argument('--max_image_size', default=640, type=int)

parser.add_argument('--print_ratio', default=0.5, type=float)

parser.add_argument('--tag', default='', type=str)
parser.add_argument('--augment', default='', type=str)
parser.add_argument('--cutmix_prob', default=1.0, type=float)
parser.add_argument('--mixup_prob', default=1.0, type=float)

if __name__ == '__main__':
  ###################################################################################
  # Arguments
  ###################################################################################
  args = parser.parse_args()

  TAG = args.tag
  SEED = args.seed
  DEVICE = args.device

  print('Train Configuration')
  pad = max(map(len, vars(args))) + 1
  for k, v in vars(args).items():
    print(f'{k.ljust(pad)}: {v}')
  print('===================')

  log_dir = create_directory(f'./experiments/logs/')
  data_dir = create_directory(f'./experiments/data/')
  model_dir = create_directory('./experiments/models/')
  tensorboard_dir = create_directory(f'./experiments/tensorboards/{TAG}/')

  log_path = log_dir + f'{TAG}.txt'
  data_path = data_dir + f'{TAG}.json'
  model_path = model_dir + f'{TAG}.pth'

  set_seed(SEED)
  log = lambda string='': log_print(string, log_path)

  log('[i] {}'.format(TAG))
  log()

  ###################################################################################
  # Transform, Dataset, DataLoader
  ###################################################################################
  tt, tv = get_transforms(args.min_image_size, args.max_image_size, args.image_size, args.augment)
  train_dataset, valid_dataset = get_dataset_classification(
    args.dataset, args.data_dir, args.augment, args.image_size, args.cutmix_prob, args.mixup_prob, tt, tv,
  )

  train_loader = DataLoader(
    train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True
  )
  valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=1, drop_last=True)

  log('[i] The number of class is {}'.format(train_dataset.info.num_classes))
  log('[i] train_transform is {}'.format(tt))
  log('[i] test_transform is {}'.format(tv))
  log()

  val_iteration = len(train_loader)
  log_iteration = int(val_iteration * args.print_ratio)
  max_iteration = args.max_epoch * val_iteration

  # val_iteration = log_iteration

  log('[i] log_iteration : {:,}'.format(log_iteration))
  log('[i] val_iteration : {:,}'.format(val_iteration))
  log('[i] max_iteration : {:,}'.format(max_iteration))

  ###################################################################################
  # Network
  ###################################################################################
  model = Classifier(
    args.architecture,
    train_dataset.info.num_classes,
    mode=args.mode,
    dilated=args.dilated,
    regularization=args.regularization,
    trainable_stem=args.trainable_stem,
  )
  param_groups = model.get_parameter_groups()

  model = model.to(DEVICE)
  model.train()

  log('[i] Architecture is {}'.format(args.architecture))
  log('[i] Regularization is {}'.format(args.regularization))
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

  optimizer = get_optimizer(args.lr, args.wd, max_iteration, param_groups)

  #################################################################################################
  # Train
  #################################################################################################
  data_dic = {'train': [], 'validation': []}

  train_timer = Timer()
  eval_timer = Timer()

  train_meter = Average_Meter(['loss', 'class_loss'])

  best_train_mIoU = -1
  thresholds = list(np.arange(0.1, 0.50, 0.05))

  def evaluate(loader):
    imagenet_mean, imagenet_std = imagenet_stats()
    
    model.eval()
    eval_timer.tik()

    meter_dic = {th: Calculator_For_mIoU(train_dataset.info.classes) for th in thresholds}

    outputs = {'labels': [], 'preds': []}

    with torch.no_grad():
      for step, (images, labels, gt_masks) in enumerate(loader):
        logits, features = model(images.to(DEVICE), with_cam=True)

        # features = resize_for_tensors(features, images.size()[-2:])
        # gt_masks = resize_for_tensors(gt_masks, features.size()[-2:], mode='nearest')

        mask = labels.unsqueeze(2).unsqueeze(3)
        cams = make_cam(features).cpu() * mask

        images = to_numpy(images)
        labels = to_numpy(labels)
        gt_masks = to_numpy(gt_masks)
        preds = to_numpy(torch.sigmoid(logits))
        cams = to_numpy(cams)

        outputs['labels'].append(labels)
        outputs['preds'].append(preds)

        # for visualization
        if step == 0:
          obj_cams = cams.max(axis=1)

          for b in range(8):
            image = images[b]
            cam = obj_cams[b]

            image = denormalize(image, imagenet_mean, imagenet_std)[..., ::-1]
            h, w, c = image.shape

            cam = (cam * 255).astype(np.uint8)
            cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
            cam = colormap(cam)

            image = cv2.addWeighted(image, 0.5, cam, 0.5, 0)[..., ::-1]
            image = image.astype(np.float32) / 255.

            writer.add_image('CAM/{}'.format(b + 1), image, iteration, dataformats='HWC')

        for b in range(len(images)):
          # c, h, w -> h, w, c
          cam = cams[b].transpose((1, 2, 0))
          gt_mask = gt_masks[b]

          h, w, c = cam.shape
          gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)

          for th in thresholds:
            bg = np.ones_like(cam[:, :, 0]) * th
            pred_mask = np.argmax(np.concatenate([bg[..., np.newaxis], cam], axis=-1), axis=-1)

            meter_dic[th].add(pred_mask, gt_mask)

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

  for iteration in range(max_iteration):
    images, labels = train_iterator.get()

    #################################################################################################
    logits = model(images.to(DEVICE))

    labels = label_smoothing(labels, args.label_smoothing)
    class_loss = class_loss_fn(logits, labels.to(DEVICE)).mean()
    loss = class_loss
    #################################################################################################

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_meter.add({'loss': loss.item(), 'class_loss': class_loss.item()})

    #################################################################################################
    # For Log
    #################################################################################################
    if (iteration + 1) % log_iteration == 0:
      loss, class_loss = train_meter.get(clear=True)
      learning_rate = float(get_learning_rate_from_optimizer(optimizer))

      data = {
        'iteration': iteration + 1,
        'learning_rate': learning_rate,
        'loss': loss,
        'class_loss': class_loss,
        'time': train_timer.tok(clear=True),
      }
      data_dic['train'].append(data)
      write_json(data_path, data_dic)

      log(
        'step={iteration:,} '
        'lr={learning_rate:.4f} '
        'loss={loss:.4f} '
        'class_loss={class_loss:.4f} '
        'time={time:.0f}sec'.format(**data)
      )

      writer.add_scalar('Train/loss', loss, iteration)
      writer.add_scalar('Train/class_loss', class_loss, iteration)
      writer.add_scalar('Train/learning_rate', learning_rate, iteration)

    #################################################################################################
    # Evaluation
    #################################################################################################
    if (iteration + 1) % val_iteration == 0:
      threshold, mIoU = evaluate(valid_loader)

      if best_train_mIoU == -1 or best_train_mIoU < mIoU:
        best_train_mIoU = mIoU

      data = {
        'iteration': iteration + 1,
        'threshold': threshold,
        'train_mIoU': mIoU,
        'best_train_mIoU': best_train_mIoU,
        'time': eval_timer.tok(clear=True),
      }
      data_dic['validation'].append(data)
      write_json(data_path, data_dic)

      log(
        'iteration={iteration:,} '
        'threshold={threshold:.2f} '
        'train_mIoU={train_mIoU:.2f}% '
        'best_train_mIoU={best_train_mIoU:.2f}% '
        'time={time:.0f}sec'.format(**data)
      )

      writer.add_scalar('Evaluation/threshold', threshold, iteration)
      writer.add_scalar('Evaluation/train_mIoU', mIoU, iteration)
      writer.add_scalar('Evaluation/best_train_mIoU', best_train_mIoU, iteration)

  write_json(data_path, data_dic)
  writer.close()

  log(f'[i] {TAG} saved at {model_path}')
  save_model_fn()
