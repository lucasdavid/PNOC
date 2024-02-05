import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

import datasets
import wandb
from core import occse
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
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--dataset', default='voc12', choices=datasets.DATASOURCES)
parser.add_argument('--data_dir', required=True, type=str)
parser.add_argument('--domain_train', default=None, type=str)
parser.add_argument('--domain_valid', default=None, type=str)

# Network
parser.add_argument('--architecture', default='resnet50', type=str)
parser.add_argument('--mode', default='normal', type=str)  # fix
parser.add_argument('--trainable-stem', default=True, type=str2bool)
parser.add_argument('--dilated', default=False, type=str2bool)
parser.add_argument('--restore', default=None, type=str)

parser.add_argument('--oc-architecture', default='resnet50', type=str)
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
parser.add_argument('--mixed_precision', default=False, type=str2bool)
parser.add_argument('--amp_min_scale', default=None, type=float)
parser.add_argument('--validate', default=True, type=str2bool)
parser.add_argument('--validate_max_steps', default=None, type=int)
parser.add_argument('--validate_thresholds', default=None, type=str)

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
  DEVICE = args.device if torch.cuda.is_available() else "cpu"
  if args.validate_thresholds:
      THRESHOLDS = list(map(float, args.validate_thresholds.split(",")))

  wb_run = wandb_utils.setup(TAG, args)
  log_config(vars(args), TAG)

  MODEL_PATH = os.path.join('./experiments/models/', f"{TAG}.pth")

  set_seed(SEED)

  ts = datasets.custom_data_source(args.dataset, args.data_dir, args.domain_train, split="train")
  vs = datasets.custom_data_source(args.dataset, args.data_dir, args.domain_valid, split="valid")
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
  cgnet = Classifier(
    args.architecture,
    train_dataset.info.num_classes,
    mode=args.mode,
    dilated=args.dilated,
    trainable_stem=args.trainable_stem,
  )
  if args.restore:
    print(f'Restoring weights from {args.restore}')
    cgnet.load_state_dict(torch.load(args.restore), strict=True)
  log_model("CGNet", cgnet, args)

  # Ordinary Classifier.
  print(f'Build OC {args.oc_architecture} (weights from `{args.oc_pretrained}`)')
  ocnet = Classifier(
    model_name=args.oc_architecture,
    num_classes=train_dataset.info.num_classes,
    mode=args.mode,
  )
  ocnet.load_state_dict(torch.load(args.oc_pretrained), strict=True)

  cg_param_groups, param_names = cgnet.get_parameter_groups(with_names=True)

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

  class_loss_fn = torch.nn.MultiLabelSoftMarginLoss(reduction='none').to(DEVICE)

  optimizer = get_optimizer(args.lr, args.wd, int(step_max // args.accumulate_steps), cg_param_groups)
  scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)

  log_opt_params("CGNet", param_names)

  # Train
  train_metrics = MetricsContainer(['loss', 'c_loss', 'o_loss', 'oc_alpha', 'k'])
  train_timer = Timer()
  miou_best = -1

  choices = torch.ones(train_dataset.info.num_classes)
  focal_factor = torch.ones(train_dataset.info.num_classes)

  for step in range(step_init, step_max):
    _, images, targets = train_iterator.get()

    images = images.to(DEVICE)
    targets = targets.float()

    ao = linear_schedule(step, step_max, args.oc_alpha_init, args.oc_alpha, args.oc_alpha_schedule)
    k = 1  # round(linear_schedule(step, step_max, args.oc_k_init, args.oc_k, args.oc_k_schedule))

    with torch.autocast(device_type=DEVICE, enabled=args.mixed_precision):
      # Normal
      logits, features = cgnet(images, with_cam=True)

      labels_sm = label_smoothing(targets, args.label_smoothing).to(DEVICE)
      c_loss = class_loss_fn(logits, labels_sm).mean()

      labels_mask, _ = occse.split_label(targets, k, choices, focal_factor, args.oc_strategy)
      labels_oc = (targets - labels_mask).clip(min=0)

      cl_logits = ocnet(occse.soft_mask_images(images, features, labels_mask, globalnorm=args.oc_mask_globalnorm))
      o_loss = class_loss_fn(cl_logits, label_smoothing(labels_oc, args.label_smoothing).to(cl_logits)).mean()

      loss = c_loss + ao * o_loss

    scaler.scale(loss).backward()

    if (step + 1) % args.accumulate_steps == 0:
      scaler.step(optimizer)
      scaler.update()
      optimizer.zero_grad()  # set_to_none=False  # TODO: Try it with True and check performance.

    occse.update_focal(
      targets,
      labels_oc,
      cl_logits.to(targets),
      focal_factor,
      momentum=args.oc_focal_momentum,
      gamma=args.oc_focal_gamma
    )

    epoch = step // step_val
    do_logging = (step + 1) % step_log == 0
    do_validation = args.validate and (step + 1) % step_val == 0

    train_metrics.update(
      {
        'loss': loss.item(),
        'c_loss': c_loss.item(),
        'o_loss': o_loss.item(),
        'oc_alpha': ao,
        'k': k
      }
    )

    if do_logging:
      (loss, c_loss, o_loss, ap, ao, k) = train_metrics.get(clear=True)

      lr = float(get_learning_rate_from_optimizer(optimizer))
      cs = to_numpy(choices.int()).tolist()
      ffs = to_numpy(focal_factor).astype(float).round(2).tolist()

      data = {
        'iteration': step + 1,
        'lr': lr,
        'loss': loss,
        'c_loss': c_loss,
        'o_loss': o_loss,
        'oc_alpha': ao,
        'k': k,
        'choices': cs,
        'focal_factor': ffs,
        'time': train_timer.tok(clear=True),
      }

      wb_logs = {f"train/{k}": v for k, v in data.items()}
      wb_logs["train/epoch"] = epoch
      wandb.log(wb_logs, commit=not do_validation)

      print(
        'iteration    = {iteration:,}\n'
        'time         = {time:.0f} sec\n'
        'lr           = {lr:.4f}\n'
        'loss         = {loss:.4f}\n'
        'c_loss       = {c_loss:.4f}\n'
        'o_loss       = {o_loss:.4f}\n'
        'oc_alpha     = {oc_alpha:.4f}\n'
        'k            = {k}\n'
        'focal_factor = {focal_factor}\n'
        'choices      = {choices}\n'.format(**data)
      )

    if do_validation:
      cgnet.eval()
      with torch.autocast(device_type=DEVICE, enabled=args.mixed_precision):
        metric_results = priors_validation_step(
          cgnet, valid_loader, train_dataset.info, THRESHOLDS, DEVICE, args.validate_max_steps
        )
      metric_results["iteration"] = step + 1
      cgnet.train()

      wandb.log({f"val/{k}": v for k, v in metric_results.items()})
      print(*(f"{metric}={value}" for metric, value in metric_results.items()))

      if metric_results["miou"] > miou_best:
        miou_best = metric_results["miou"]
        for k in ("threshold", "miou", "iou"):
          wandb.run.summary[f"val/best_{k}"] = metric_results[k]

      save_model(cgnet, MODEL_PATH, parallel=GPUS_COUNT > 1)

  print(TAG)
  wb_run.finish()
