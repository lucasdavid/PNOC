import argparse
import os

import numpy as np
import torch
import wandb
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
from tools.general import wandb_utils
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
parser.add_argument('--oc-mask-globalnorm', default=True, type=str2bool)
parser.add_argument('--oc-focal-momentum', default=0.9, type=float)
parser.add_argument('--oc-focal-gamma', default=2.0, type=float)

# Hyperparameter
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--first_epoch', default=0, type=int)
parser.add_argument('--max_epoch', default=15, type=int)
parser.add_argument('--accumulate_steps', default=1, type=int)

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
parser.add_argument('--oc-k', default=1.0, type=float)
parser.add_argument('--oc-k-init', default=1.0, type=float)
parser.add_argument('--oc-k-schedule', default=0.0, type=float)

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

  wb_run = wandb_utils.setup(TAG, args)
  log_config(vars(args), TAG)

  data_dir = create_directory(f'./experiments/data/')
  model_dir = create_directory('./experiments/models/')

  data_path = data_dir + f'{TAG}.json'
  model_path = model_dir + f'{TAG}.pth'

  set_seed(SEED)

  tt, tv = get_classification_transforms(args.min_image_size, args.max_image_size, args.image_size, args.augment)
  train_dataset, valid_dataset = get_classification_datasets(
    args.dataset, args.data_dir, args.augment, args.image_size, args.cutmix_prob, args.mixup_prob, tt, tv
  )
  train_loader = DataLoader(
    train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True
  )
  valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=1, drop_last=True)
  log_dataset(args.dataset, train_dataset, tt, tv)

  step_valid = len(train_loader)
  step_log = int(step_valid * args.print_ratio)
  step_init = args.first_epoch * step_valid
  step_max = args.max_epoch * step_valid
  print(f"Iterations: first={step_init} logging={step_log} validation={step_valid} max={step_max}")

  # Network
  cgnet = Classifier(
    args.architecture,
    train_dataset.info.num_classes,
    mode=args.mode,
    dilated=args.dilated,
    regularization=args.regularization,
    trainable_stem=args.trainable_stem,
  )
  if args.restore:
    print(f'Restoring weights from {args.restore}')
    cgnet.load_state_dict(torch.load(args.restore), strict=True)
  log_model("CGNet", cgnet, args)

  # Ordinary Classifier.
  print(f'Build OC {args.oc_architecture} (weights from `{args.oc_pretrained}`)')
  ocnet = Classifier(
    args.oc_architecture, train_dataset.info.num_classes, mode=args.mode, regularization=args.oc_regularization
  )
  ocnet.load_state_dict(torch.load(args.oc_pretrained), strict=True)

  cg_param_groups = cgnet.get_parameter_groups()

  cgnet = cgnet.to(DEVICE)
  ocnet = ocnet.to(DEVICE)
  cgnet.train()
  ocnet.eval()

  for child in ocnet.children():
    for param in child.parameters():
      param.requires_grad = False

  if GPUS_COUNT > 1:
    print(f"GPUs={GPUS_COUNT}")
    cgnet = torch.nn.DataParallel(cgnet)
    ocnet = torch.nn.DataParallel(ocnet)

  load_model_fn = lambda: load_model(cgnet, model_path, parallel=GPUS_COUNT > 1)
  save_model_fn = lambda: save_model(cgnet, model_path, parallel=GPUS_COUNT > 1)

  # Loss, Optimizer
  class_loss_fn = torch.nn.MultiLabelSoftMarginLoss(reduction='none').to(DEVICE)

  if args.re_loss == 'L1_Loss':
    r_loss_fn = L1_Loss
  else:
    r_loss_fn = L2_Loss

  optimizer = get_optimizer(args.lr, args.wd, step_max, cg_param_groups)

  log_opt_params("CGNet", cg_param_groups)

  # Train
  data_dic = {'train': [], 'validation': []}

  train_timer = Timer()
  eval_timer = Timer()

  train_metrics = MetricsContainer(['loss', 'c_loss', 'p_loss', 're_loss', 'o_loss', 'alpha', 'oc_alpha', 'k'])

  best_train_mIoU = -1

  choices = torch.ones(train_dataset.info.num_classes)
  focal_factor = torch.ones(train_dataset.info.num_classes)

  def evaluate(loader):
    classes = train_dataset.info.classes

    cgnet.eval()
    eval_timer.tik()

    iou_meters = {th: Calculator_For_mIoU(classes) for th in THRESHOLDS}

    with torch.no_grad():
      for step, (images_batch, targets_batch, masks_batch) in enumerate(loader):
        images_batch = images_batch.to(DEVICE)
        targets_batch = to_numpy(targets_batch)
        masks_batch = to_numpy(masks_batch)

        _, features = cgnet(images_batch, with_cam=True)

        labels_mask = targets_batch[..., np.newaxis, np.newaxis]
        cams_batch = to_numpy(make_cam(features.cpu().float())) * labels_mask
        cams_batch = cams_batch.transpose(0, 2, 3, 1)

        if step == 0:
          images_batch = to_numpy(images_batch)
          preds_batch = to_numpy(torch.sigmoid(logits))
          wandb_utils.log_cams(classes, images_batch, targets_batch, cams_batch, preds_batch)

        accumulate_batch_iou_lowres(masks_batch, cams_batch, iou_meters)

    cgnet.train()

    return result_miou_from_thresholds(iou_meters, classes)

  train_iterator = Iterator(train_loader)

  for step in range(step_init, step_max):
    images, targets = train_iterator.get()

    images = images.to(DEVICE)
    targets = targets.float()

    ap = linear_schedule(step, step_max, args.alpha_init, args.alpha, args.alpha_schedule)
    ao = linear_schedule(step, step_max, args.oc_alpha_init, args.oc_alpha, args.oc_alpha_schedule)
    k = 1  # round(linear_schedule(step, step_max, args.oc_k_init, args.oc_k, args.oc_k_schedule))

    # Normal
    logits, features = cgnet(images, with_cam=True)

    # Puzzle Module
    tiled_images = tile_features(images, args.num_pieces)
    tiled_logits, tiled_features = cgnet(tiled_images, with_cam=True)
    re_features = merge_features(tiled_features, args.num_pieces, args.batch_size)

    labels_sm = label_smoothing(targets, args.label_smoothing).to(DEVICE)
    c_loss = class_loss_fn(logits, labels_sm).mean()
    p_loss = class_loss_fn(gap2d(re_features), labels_sm).mean()

    re_mask = targets.unsqueeze(2).unsqueeze(3)
    re_loss = (r_loss_fn(features, re_features) * re_mask.to(features)).mean()

    # OC-CSE
    labels_mask, _ = occse.split_label(targets, k, choices, focal_factor, args.oc_strategy)
    labels_oc = (targets - labels_mask).clip(min=0)

    cl_logits = ocnet(occse.soft_mask_images(images, features, labels_mask, globalnorm=args.oc_mask_globalnorm))
    o_loss = class_loss_fn(cl_logits, label_smoothing(labels_oc, args.label_smoothing).to(cl_logits)).mean()

    loss = (c_loss + p_loss + ap * re_loss + ao * o_loss)

    loss.backward()

    if (step + 1) % args.accumulate_steps == 0:
      optimizer.step()
      optimizer.zero_grad()

    occse.update_focal(
      targets,
      labels_oc,
      cl_logits.to(targets),
      focal_factor,
      momentum=args.oc_focal_momentum,
      gamma=args.oc_focal_gamma
    )

    epoch = step // step_valid
    do_logging = (step + 1) % step_log == 0
    do_validation = (step + 1) % step_valid == 0

    train_metrics.update(
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

    if do_logging:
      (loss, c_loss, p_loss, re_loss, o_loss, ap, ao, k) = train_metrics.get(clear=True)

      lr = float(get_learning_rate_from_optimizer(optimizer))
      cs = to_numpy(choices.int()).tolist()
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
        'choices': cs,
        'focal_factor': ffs,
        'time': train_timer.tok(clear=True),
      }
      data_dic['train'].append(data)
      write_json(data_path, data_dic)

      wandb.log({f"train/{k}": v for k, v in data.items()} | {"train/epoch": epoch}, commit=not do_validation)

      print(
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

    if do_validation:
      threshold, mIoU, iou = evaluate(valid_loader)

      if best_train_mIoU == -1 or best_train_mIoU < mIoU:
        best_train_mIoU = mIoU
        wandb.run.summary["train/best_t"] = threshold
        wandb.run.summary["train/best_miou"] = mIoU
        wandb.run.summary["train/best_iou"] = iou

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
      wandb.log({f"val/{k}": v for k, v in data.items()})

      print(
        '\niteration       = {iteration:,}\n'
        'time            = {time:.0f} sec\n'
        'threshold       = {threshold:.2f}\n'
        'train_mIoU      = {train_mIoU:.2f}%\n'
        'best_train_mIoU = {best_train_mIoU:.2f}%\n'
        'train_iou       = {train_iou}\n'.format(**data)
      )

      print(f'saving weights `{model_path}`')
      save_model_fn()

  print(TAG)
  wb_run.finish()
