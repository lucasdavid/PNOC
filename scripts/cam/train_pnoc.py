import argparse
import os

import numpy as np
import torch
from torch import multiprocessing
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import wandb
from core import occse
from core.networks import *
from core.puzzle_utils import *
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


# region Arguments
parser.add_argument('--tag', default='', type=str)
parser.add_argument('--print_ratio', default=0.25, type=float)

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
parser.add_argument('--mode', default='normal', type=str)  # fix
parser.add_argument('--trainable-stem', default=True, type=str2bool)
parser.add_argument('--trainable-stage4', default=True, type=str2bool)
parser.add_argument('--trainable-backbone', default=True, type=str2bool)
parser.add_argument('--dilated', default=False, type=str2bool)
parser.add_argument('--restore', default=None, type=str)
parser.add_argument('--backbone_weights', default="imagenet", type=str)

parser.add_argument('--oc-architecture', default='resnet50', type=str)
parser.add_argument('--oc-pretrained', required=True, type=str)
parser.add_argument('--oc-strategy', default='random', type=str, choices=list(occse.STRATEGIES))
parser.add_argument('--oc-focal-momentum', default=0.9, type=float)
parser.add_argument('--oc-focal-gamma', default=2.0, type=float)
parser.add_argument('--oc-persist', default=False, type=str2bool)

# Hyperparameter
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--first_epoch', default=0, type=int)
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
parser.add_argument('--max_grad_norm', default=None, type=float)
parser.add_argument('--max_grad_norm_oc', default=None, type=float)
parser.add_argument('--optimizer', default="sgd", choices=OPTIMIZERS_NAMES)
parser.add_argument('--lr_alpha_scratch', default=10., type=float)
parser.add_argument('--lr_alpha_bias', default=2., type=float)
parser.add_argument('--lr_alpha_oc', default=1., type=float)
parser.add_argument('--class_weight', default=None, type=str)

parser.add_argument('--image_size', default=512, type=int)
parser.add_argument('--min_image_size', default=320, type=int)
parser.add_argument('--max_image_size', default=640, type=int)

# Augmentation
parser.add_argument('--augment', default='', type=str)
parser.add_argument('--cutmix_prob', default=1.0, type=float)
parser.add_argument('--mixup_prob', default=1.0, type=float)

# For Puzzle-CAM
parser.add_argument('--num_pieces', default=4, type=int)
parser.add_argument('--loss_option', default='cl_pcl_re', type=str)
parser.add_argument('--re_loss', default='L1_Loss', type=str)  # 'L1_Loss', 'L2_Loss'

parser.add_argument('--alpha', default=4.0, type=float)
parser.add_argument('--alpha_init', default=0.0, type=float)
parser.add_argument('--alpha_schedule', default=0.50, type=float)

# For OC-CSE
parser.add_argument('--oc-alpha', default=1.0, type=float)
parser.add_argument('--oc-alpha-init', default=0.3, type=float)
parser.add_argument('--oc-alpha-schedule', default=1.0, type=float)
parser.add_argument('--oc-k', default=1.0, type=float)
parser.add_argument('--oc-k-init', default=1.0, type=float)
parser.add_argument('--oc-k-schedule', default=0, type=float)

# For NOC
parser.add_argument('--ow', default=0.5, type=float)
parser.add_argument('--ow-init', default=0, type=float)
parser.add_argument('--ow-schedule', default=1.0, type=float)
parser.add_argument('--oc-train-interval-steps', default=1, type=int)
parser.add_argument('--oc-train-masks', default='cams', choices=['features', 'cams'], type=str)
parser.add_argument('--oc_train_mask_t', default=0.2, type=float)

# endregion

import cv2

cv2.setNumThreads(0)

try:
  GPUS = os.environ['CUDA_VISIBLE_DEVICES']
except KeyError:
  GPUS = '0'
GPUS = GPUS.split(',')
GPUS_COUNT = len(GPUS)
THRESHOLDS = list(np.arange(0.10, 0.50, 0.05))


def train_step(train_iterator, step):
  _, inputs, targets = train_iterator.get()

  images = inputs.to(DEVICE)
  targets = targets.float()
  targets_sm = label_smoothing(targets, args.label_smoothing).to(DEVICE)

  ap = linear_schedule(step, step_max, args.alpha_init, args.alpha, args.alpha_schedule)
  ao = linear_schedule(step, step_max, args.oc_alpha_init, args.oc_alpha, args.oc_alpha_schedule)
  ow = linear_schedule(step, step_max, args.ow_init, args.ow, args.ow_schedule)
  k = round(linear_schedule(step, step_max, args.oc_k_init, args.oc_k, args.oc_k_schedule))
  schedules = {'alpha': ap, 'oc_alpha': ao, 'ot_weight': ow, 'k': k}

  # CAM Generator Training
  (cg_features, images_mask, labels_mask, cg_metrics) = train_step_cg(step, images, targets, targets_sm, ap, ao, k)

  # Ordinary Classifier Training
  if cg_features is not None and (step + 1) % args.oc_train_interval_steps == 0:
    oc_step = int((step + 1) // args.oc_train_interval_steps) - 1

    if args.oc_train_masks == "cams":
      del images_mask
      images_mask = occse.hard_mask_images(images, cg_features.detach(), labels_mask, t=args.oc_train_mask_t)

    images_mask = images_mask.detach()

    del images, targets, cg_features

    ocnet.train()
    oc_metrics = train_step_oc(oc_step, images_mask, targets_sm, ow)
    ocnet.eval()
  else:
    oc_metrics = {}

  return dict(**cg_metrics, **oc_metrics, **schedules)


def train_step_cg(step, images, targets, targets_sm, ap, ao, k):
  # with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=args.mixed_precision):
  with torch.autocast(device_type=DEVICE, enabled=args.mixed_precision):
    # Normal
    logits, features = cgnet(images, with_cam=True)

    # Puzzle Module
    tiled_images = tile_features(images, args.num_pieces)
    tiled_logits, tiled_features = cgnet(tiled_images, with_cam=True)
    re_features = merge_features(tiled_features, args.num_pieces, args.batch_size)

    c_loss = class_loss_fn(logits, targets_sm).mean()
    p_loss = class_loss_fn(gap2d(re_features), targets_sm).mean()

    re_mask = targets.unsqueeze(2).unsqueeze(3)
    re_loss = (r_loss_fn(features, re_features) * re_mask.to(features)).mean()

    # OC-CSE
    labels_mask, _ = occse.split_label(targets, k, choices, focal_factor, args.oc_strategy)
    labels_oc = (targets - labels_mask).clip(min=0)

    images_mask = occse.soft_mask_images(images, features, labels_mask)
    cl_logits = ocnet(images_mask)

    o_loss = class_loss_fn(cl_logits, label_smoothing(labels_oc, args.label_smoothing).to(cl_logits)).mean()

    cg_loss = c_loss + p_loss + ap * re_loss + ao * o_loss

  # print(f"step={step} cg_loss={cg_loss} o-loss={o_loss}")
  cg_scaler.scale(cg_loss).backward()

  if (step + 1) % args.accumulate_steps == 0:
    if args.max_grad_norm:
      cg_scaler.unscale_(cgopt)
      torch.nn.utils.clip_grad_norm_(cgnet.parameters(), args.max_grad_norm)

    cg_scaler.step(cgopt)
    cg_scaler.update()
    cgopt.zero_grad()  # set_to_none=False  # TODO: Try it with True and check performance.

    if args.amp_min_scale and cg_scaler._scale < args.amp_min_scale:
        cg_scaler._scale = torch.as_tensor(args.amp_min_scale, dtype=cg_scaler._scale.dtype, device=cg_scaler._scale.device)

  occse.update_focal(
    targets,
    labels_oc,
    cl_logits.to(targets),
    focal_factor,
    args.oc_focal_momentum,
    args.oc_focal_gamma,
  )

  return (
    features, images_mask, labels_mask, {
      'loss': cg_loss.item(),
      'c_loss': c_loss.item(),
      'p_loss': p_loss.item(),
      're_loss': re_loss.item(),
      'o_loss': o_loss.item(),
    }
  )


def train_step_oc(step, images_mask, targets_sm, ow):
  with torch.autocast(device_type=DEVICE, enabled=args.mixed_precision):
    cl_logits = ocnet(images_mask)
    ot_loss = ow * class_loss_fn(cl_logits, targets_sm).mean()

  # print(f"step={step} ow={ow} ot-loss={ot_loss}")
  oc_scaler.scale(ot_loss).backward()

  if (step + 1) % args.accumulate_steps == 0:
    if args.max_grad_norm_oc:
      oc_scaler.unscale_(ocopt)
      torch.nn.utils.clip_grad_norm_(ocnet.parameters(), args.max_grad_norm_oc)

    oc_scaler.step(ocopt)
    oc_scaler.update()
    ocopt.zero_grad()

    if args.amp_min_scale and oc_scaler._scale < args.amp_min_scale:
        oc_scaler._scale = torch.as_tensor(args.amp_min_scale, dtype=oc_scaler._scale.dtype, device=oc_scaler._scale.device)

  return {'ot_loss': ot_loss.item()}


if __name__ == '__main__':
  try:
    multiprocessing.set_start_method('spawn')
  except RuntimeError:
    ...

  args = parser.parse_args()

  TAG = args.tag
  SEED = args.seed
  DEVICE = args.device if torch.cuda.is_available() else "cpu"
  if args.validate_thresholds:
    THRESHOLDS = list(map(float, args.validate_thresholds.split(",")))
  if args.class_weight and args.class_weight != "none":
    CLASS_WEIGHT = torch.Tensor(list(map(float, args.class_weight.split(",")))).to(DEVICE)
  else:
    CLASS_WEIGHT = None

  wb_run = wandb_utils.setup(TAG, args)
  log_config(vars(args), TAG)

  model_path = os.path.join('./experiments/models', f'{TAG}.pth')
  oc_model_path = os.path.join('./experiments/models', f'{TAG}-oc.pth')

  create_directory(os.path.dirname(model_path))
  set_seed(SEED)

  ts = datasets.custom_data_source(args.dataset, args.data_dir, args.domain_train, split="train")
  vs = datasets.custom_data_source(args.dataset, args.data_dir, args.domain_valid, split="valid")
  tt, tv = datasets.get_classification_transforms(args.min_image_size, args.max_image_size, args.image_size, args.augment, ts.classification_info.normalize_stats)
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
  print(f'Iterations: first={step_init} logging={step_log} validation={step_val} max={step_max}')

  # Network
  cgnet = Classifier(
    args.architecture,
    train_dataset.info.num_classes,
    channels=train_dataset.info.channels,
    backbone_weights=args.backbone_weights,
    mode=args.mode,
    dilated=args.dilated,
    trainable_stem=args.trainable_stem,
    trainable_stage4=args.trainable_stage4,
    trainable_backbone=args.trainable_backbone,
  )
  if args.restore:
    print(f'Restoring weights from {args.restore}')
    cgnet.load_state_dict(torch.load(args.restore, map_location=torch.device('cpu')))
  log_model('CGNet', cgnet, args)

  # Ordinary Classifier.
  print(f'Build OC {args.oc_architecture} (weights from `{args.oc_pretrained}`)')
  ocnet = Classifier(
    args.oc_architecture,
    train_dataset.info.num_classes,
    channels=train_dataset.info.channels,
    backbone_weights=args.backbone_weights,
    mode="fix",
    trainable_stem=args.trainable_stem,
    trainable_stage4=args.trainable_stage4,
    trainable_backbone=args.trainable_backbone,
  )
  ocnet.load_state_dict(torch.load(args.oc_pretrained, map_location=torch.device('cpu')))

  cg_param_groups, cg_param_names = cgnet.get_parameter_groups(with_names=True)
  oc_param_groups, oc_param_names = ocnet.get_parameter_groups(with_names=True)
  cgnet = cgnet.to(DEVICE)
  ocnet = ocnet.to(DEVICE)
  cgnet.train()
  ocnet.eval()

  if GPUS_COUNT > 1:
    print(f'GPUs={GPUS_COUNT}')
    cgnet = torch.nn.DataParallel(cgnet)
    ocnet = torch.nn.DataParallel(ocnet)

  # Loss, Optimizer
  class_loss_fn = torch.nn.MultiLabelSoftMarginLoss(weight=CLASS_WEIGHT, reduction='none').to(DEVICE)

  if args.re_loss == 'L1_Loss':
    r_loss_fn = L1_Loss
  else:
    r_loss_fn = L2_Loss

  cgopt = get_optimizer(
    args.lr, args.wd, int(step_max // args.accumulate_steps), cg_param_groups,
    algorithm=args.optimizer,
    alpha_scratch=args.lr_alpha_scratch,
    alpha_bias=args.lr_alpha_bias,
    start_step=int(step_init // args.accumulate_steps),
  )
  ocopt = get_optimizer(
    args.lr * args.lr_alpha_oc, args.wd, int(step_max // args.accumulate_steps), oc_param_groups,
    algorithm=args.optimizer,
    alpha_scratch=args.lr_alpha_scratch,
    alpha_bias=args.lr_alpha_bias,
    start_step=int(step_init // args.accumulate_steps),
  )
  log_opt_params('CGNet', cg_param_names)
  log_opt_params('OCNet', oc_param_names)

  cg_scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)
  oc_scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)

  ## Train
  # torch.autograd.set_detect_anomaly(True)
  train_metrics = MetricsContainer(['loss', 'c_loss', 'p_loss', 're_loss', 'o_loss', 'ot_loss', 'alpha', 'oc_alpha', 'ot_weight', 'k'])
  train_timer = Timer()
  miou_best = -1

  choices = torch.ones(train_dataset.info.num_classes)
  focal_factor = torch.ones(train_dataset.info.num_classes)

  for step in tqdm(range(step_init, step_max), 'Training', mininterval=2.0):
    metrics_values = train_step(train_iterator, step)
    train_metrics.update(metrics_values)

    epoch = step // step_val
    do_logging = (step + 1) % step_log == 0
    do_validation = args.validate and (step + 1) % step_val == 0

    if do_logging:
      (cg_loss, c_loss, p_loss, re_loss, o_loss, ot_loss, ap, ao, ow, k) = train_metrics.get(clear=True)

      lr = float(get_learning_rate_from_optimizer(cgopt))
      cs = to_numpy(choices).tolist()
      ffs = to_numpy(focal_factor).astype(float).round(2).tolist()

      data = {
        'iteration': step + 1,
        'lr': lr,
        'alpha': ap,
        'loss': cg_loss,
        'c_loss': c_loss,
        'p_loss': p_loss,
        're_loss': re_loss,
        'o_loss': o_loss,
        'ot_loss': ot_loss,
        'oc_alpha': ao,
        'ow': ow,
        'k': k,
        'choices': cs,
        'focal_factor': ffs,
        'time': train_timer.tok(clear=True),
      }

      wb_logs = {f"train/{k}": v for k, v in data.items()}
      wb_logs["train/epoch"] = epoch
      wandb.log(wb_logs, commit=not do_validation)

      print(
        '\niteration    = {iteration:,}\n'
        'time         = {time:.0f} sec\n'
        'lr           = {lr:.4f}\n'
        'alpha        = {alpha:.2f}\n'
        'loss         = {loss:.4f}\n'
        'c_loss       = {c_loss:.4f}\n'
        'p_loss       = {p_loss:.4f}\n'
        're_loss      = {re_loss:.4f}\n'
        'o_loss       = {o_loss:.4f}\n'
        'o_train_loss = {ot_loss:.4f}\n'
        'o_train_w    = {ow:.4f}\n'
        'oc_alpha     = {oc_alpha:.4f}\n'
        'k            = {k}\n'
        'focal_factor = {focal_factor}\n'
        'choices      = {choices}\n'.format(**data)
      )

    if do_validation:
      cgnet.eval()
      with torch.autocast(device_type=DEVICE, enabled=args.mixed_precision):
        if args.validate_priors:
          metric_results = priors_validation_step(
            cgnet, valid_loader, train_dataset.info, THRESHOLDS, DEVICE, args.validate_max_steps
          )
        else:
          metric_results = classification_validation_step(
            cgnet, valid_loader, train_dataset.info, DEVICE, args.validate_max_steps
          )
      metric_results["iteration"] = step + 1
      cgnet.train()

      wandb.log({f"val/{k}": v for k, v in metric_results.items()})
      print(*(f"{metric}={value}" for metric, value in metric_results.items()))

      if args.validate_priors and metric_results["miou"] > miou_best:
        miou_best = metric_results["miou"]
        for k in ("threshold", "miou", "iou"):
          wandb.run.summary[f"val/best_{k}"] = metric_results[k]

      save_model(cgnet, model_path, parallel=GPUS_COUNT > 1)
      if args.oc_persist:
        save_model(ocnet, oc_model_path, parallel=GPUS_COUNT > 1)

  print(TAG)
  wb_run.finish()
