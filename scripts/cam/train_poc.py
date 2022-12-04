import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core import occse
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

# Dataset
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--dataset', default='voc12', choices=['voc12', 'coco14'])
parser.add_argument('--data_dir', default='../VOCtrainval_11-May-2012/', type=str)

# Network
parser.add_argument('--architecture', default='resnet50', type=str)
parser.add_argument('--mode', default='normal', type=str)  # fix
parser.add_argument('--regularization', default=None, type=str)  # kernel_usage
parser.add_argument('--trainable-stem', default=True, type=str2bool)
parser.add_argument('--dilated', default=False, type=str2bool)
parser.add_argument('--restore', default=None, type=str)

parser.add_argument('--oc-architecture', default='resnet50', type=str)
parser.add_argument('--oc-regularization', default=None, type=str)
parser.add_argument('--oc-pretrained', required=True, type=str)
parser.add_argument('--oc-strategy', default='random', type=str, choices=list(occse.STRATEGIES))
parser.add_argument('--oc-focal-momentum', default=0.9, type=float)
parser.add_argument('--oc-focal-gamma', default=2.0, type=float)

# Hyperparameter
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--first_epoch', default=0, type=int)
parser.add_argument('--max_epoch', default=15, type=int)

parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--wd', default=1e-4, type=float)
parser.add_argument('--label_smoothing', default=0, type=float)

parser.add_argument('--image_size', default=512, type=int)
parser.add_argument('--min_image_size', default=320, type=int)
parser.add_argument('--max_image_size', default=640, type=int)

parser.add_argument('--print_ratio', default=1.0, type=float)

parser.add_argument('--tag', default='', type=str)
parser.add_argument('--augment', default='', type=str)
parser.add_argument('--cutmix_prob', default=1.0, type=float)
parser.add_argument('--mixup_prob', default=1.0, type=float)

# For Puzzle-CAM
parser.add_argument('--num_pieces', default=4, type=int)
parser.add_argument('--loss_option', default='cl_pcl_re', type=str)
parser.add_argument('--re_loss', default='L1_Loss', type=str)  # 'L1_Loss', 'L2_Loss'

parser.add_argument('--alpha', default=1.0, type=float)
parser.add_argument('--alpha_init', default=0., type=float)
parser.add_argument('--alpha_schedule', default=0.50, type=float)

parser.add_argument('--oc-alpha', default=1.0, type=float)
parser.add_argument('--oc-alpha-init', default=0.3, type=float)
parser.add_argument('--oc-alpha-schedule', default=1.0, type=float)
# parser.add_argument('--oc-k', default=1.0, type=float)
# parser.add_argument('--oc-k-init', default=1.0, type=float)
# parser.add_argument('--oc-k-schedule', default=0.0, type=float)

if __name__ == '__main__':
  # Arguments
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
    args.dataset, args.data_dir, args.augment, args.image_size, args.cutmix_prob, args.mixup_prob, tt, tv
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
  step_init = args.first_epoch * val_iteration
  step_max = args.max_epoch * val_iteration

  log('[i] log_iteration : {:,}'.format(log_iteration))
  log('[i] val_iteration : {:,}'.format(val_iteration))
  log('[i] max_iteration : {:,}'.format(step_max))

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
    print(f'Restoring weights from {args.restore}')
    model.load_state_dict(torch.load(args.restore), strict=True)
  param_groups = model.get_parameter_groups()  # (exclude_partial_names=['bn'])

  gap_fn = model.global_average_pooling_2d

  model = model.to(DEVICE)
  model.train()

  log('[i] Architecture is {}'.format(args.architecture))
  log('[i] Regularization is {}'.format(args.regularization))
  log('[i] Total Params: %.2fM' % (calculate_parameters(model)))
  log()

  # Ordinary Classifier.
  print(f'Build OC {args.oc_architecture} (weights from `{args.oc_pretrained}`)')
  if 'mcar' in args.oc_architecture:
    ps = 'avg'
    topN = 4
    threshold = 0.5
    oc_nn = mcar_resnet101(train_dataset.info.num_classes, ps, topN, threshold, inference_mode=True, with_logits=True)
    ckpt = torch.load(args.oc_pretrained)
    oc_nn.load_state_dict(ckpt['state_dict'], strict=True)
  else:
    oc_nn = Classifier(args.oc_architecture, train_dataset.info.num_classes, mode=args.mode, regularization=args.oc_regularization)
    oc_nn.load_state_dict(torch.load(args.oc_pretrained), strict=True)

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
    oc_nn = nn.DataParallel(oc_nn)

  load_model_fn = lambda: load_model(model, model_path, parallel=the_number_of_gpu > 1)
  save_model_fn = lambda: save_model(model, model_path, parallel=the_number_of_gpu > 1)

  # Loss, Optimizer
  class_loss_fn = nn.MultiLabelSoftMarginLoss(reduction='none').to(DEVICE)

  if args.re_loss == 'L1_Loss':
    r_loss_fn = L1_Loss
  else:
    r_loss_fn = L2_Loss

  log('[i] The number of pretrained weights : {}'.format(len(param_groups[0])))
  log('[i] The number of pretrained bias : {}'.format(len(param_groups[1])))
  log('[i] The number of scratched weights : {}'.format(len(param_groups[2])))
  log('[i] The number of scratched bias : {}'.format(len(param_groups[3])))

  optimizer = get_optimizer(args.lr, args.wd, step_max, param_groups)

  # Train
  data_dic = {'train': [], 'validation': []}

  train_timer = Timer()
  eval_timer = Timer()

  train_meter = Average_Meter(['loss', 'c_loss', 'p_loss', 're_loss', 'o_loss', 'alpha', 'oc_alpha', 'k'])

  best_train_mIoU = -1
  thresholds = list(np.arange(0.10, 0.50, 0.05))

  choices = torch.ones(train_dataset.info.num_classes)
  focal_factor = torch.ones(train_dataset.info.num_classes)

  def evaluate(loader):
    imagenet_mean, imagenet_std = imagenet_stats()
    
    model.eval()
    eval_timer.tik()

    meter_dic = {th: Calculator_For_mIoU(train_dataset.info.classes) for th in thresholds}

    with torch.no_grad():
      for step, (images, labels, gt_masks) in enumerate(loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        _, features = model(images, with_cam=True)

        # features = resize_for_tensors(features, images.size()[-2:])
        # gt_masks = resize_for_tensors(gt_masks, features.size()[-2:], mode='nearest')

        mask = labels.unsqueeze(2).unsqueeze(3)
        cams = (make_cam(features) * mask)

        for b in range(images.size()[0]):
          # c, h, w -> h, w, c
          cam = to_numpy(cams[b]).transpose((1, 2, 0))
          gt_mask = to_numpy(gt_masks[b])

          h, w, c = cam.shape
          gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)

          for th in thresholds:
            bg = np.ones_like(cam[:, :, 0]) * th
            pred_mask = np.argmax(np.concatenate([bg[..., np.newaxis], cam], axis=-1), axis=-1)

            meter_dic[th].add(pred_mask, gt_mask)

    model.train()

    best_th = 0.0
    best_mIoU = 0.0
    best_iou = None

    for th in thresholds:
      mIoU, mIoU_foreground, iou, *_ = meter_dic[th].get(clear=True, detail=True)
      if best_mIoU < mIoU:
        best_th = th
        best_mIoU = mIoU
        best_iou = [round(iou[c], 2) for c in train_dataset.info.classes]

    return best_th, best_mIoU, best_iou

  train_iterator = Iterator(train_loader)

  for step in range(step_init, step_max):
    images, targets = train_iterator.get()

    images = images.to(DEVICE)
    targets = targets.float()

    ap = linear_schedule(step, step_max, args.alpha_init, args.alpha, args.alpha_schedule)
    ao = linear_schedule(step, step_max, args.oc_alpha_init, args.oc_alpha, args.oc_alpha_schedule)
    k = 1  # round(linear_schedule(step, step_max, args.oc_k_init, args.oc_k, args.oc_k_schedule))

    optimizer.zero_grad()

    # Normal
    logits, features = model(images, with_cam=True)

    # Puzzle Module
    tiled_images = tile_features(images, args.num_pieces)
    tiled_logits, tiled_features = model(tiled_images, with_cam=True)
    re_features = merge_features(tiled_features, args.num_pieces, args.batch_size)

    labels_sm = label_smoothing(targets, args.label_smoothing).to(DEVICE)
    c_loss = class_loss_fn(logits, labels_sm).mean()
    p_loss = class_loss_fn(gap_fn(re_features), labels_sm).mean()

    re_mask = targets.unsqueeze(2).unsqueeze(3)
    re_loss = (r_loss_fn(features, re_features) * re_mask.to(features)).mean()

    # OC-CSE
    labels_mask, _ = occse.split_label(targets, k, choices, focal_factor, args.oc_strategy)
    labels_oc = targets - labels_mask

    cl_logits = oc_nn(occse.images_with_masked_objects(images, features, labels_mask))
    o_loss = class_loss_fn(
      cl_logits,
      label_smoothing(labels_oc, args.label_smoothing).to(cl_logits)
    ).mean()

    loss = (c_loss + p_loss + ap * re_loss + ao * o_loss)

    loss.backward()
    optimizer.step()

    occse.update_focal_factor(
      targets, labels_oc, cl_logits.to(targets), focal_factor, momentum=args.oc_focal_momentum, gamma=args.oc_focal_gamma
    )

    # region logging
    train_meter.add(
      {
        'loss': loss.item(),
        'c_loss': c_loss.item(),
        'p_loss': p_loss.item(),
        're_loss': re_loss.item(),
        'o_loss': o_loss.item(),
        'alpha': ap,
        'oc_alpha': ao,
        'k': k
      }
    )

    if (step + 1) % log_iteration == 0:
      (loss, c_loss, p_loss, re_loss, o_loss, ap, ao, k) = train_meter.get(clear=True)

      lr = float(get_learning_rate_from_optimizer(optimizer))
      cs = to_numpy(choices).tolist()
      ffs = to_numpy(focal_factor).astype(float).round(2).tolist()

      data = {
        'iteration': step + 1,
        'lr': lr,
        'alpha': ap,
        'loss': loss,
        'c_loss': c_loss,
        'p_loss': p_loss,
        're_loss': re_loss,
        'o_loss': o_loss,
        'oc_alpha': ao,
        'k': k,
        'time': train_timer.tok(clear=True),
        'choices': cs,
        'focal_factor': ffs,
      }
      data_dic['train'].append(data)
      write_json(data_path, data_dic)

      log(
        'iteration    = {iteration:,}\n'
        'time         = {time:.0f} sec\n'
        'lr           = {lr:.4f}\n'
        'alpha        = {alpha:.2f}\n'
        'loss         = {loss:.4f}\n'
        'c_loss       = {c_loss:.4f}\n'
        'p_loss       = {p_loss:.4f}\n'
        're_loss      = {re_loss:.4f}\n'
        'o_loss       = {o_loss:.4f}\n'
        'oc_alpha     = {oc_alpha:.4f}\n'
        'k            = {k}\n'
        'focal_factor = {focal_factor}\n'
        'choices      = {choices}\n'.format(**data)
      )

      # endregion

    # region evaluation
    if (step + 1) % val_iteration == 0:
      threshold, mIoU, iou = evaluate(valid_loader)

      if best_train_mIoU == -1 or best_train_mIoU < mIoU:
        best_train_mIoU = mIoU

      data = {
        'iteration': step + 1,
        'threshold': threshold,
        'train_mIoU': mIoU,
        'train_iou': iou,
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
        'train_iou       = {train_iou}\n'.format(**data)
      )
    # endregion

  write_json(data_path, data_dic)

  log(f'[i] {TAG} saved at {model_path}')
  save_model_fn()
