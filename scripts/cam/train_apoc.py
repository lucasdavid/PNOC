import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn import metrics as skmetrics

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
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--num_workers", default=4, type=int)
parser.add_argument("--dataset", default="voc12", choices=["voc12", "coco14"])
parser.add_argument("--data_dir", default="../VOCtrainval_11-May-2012/", type=str)

# Network
parser.add_argument("--architecture", default="resnet50", type=str)
parser.add_argument("--mode", default="normal", type=str)  # fix
parser.add_argument("--regularization", default=None, type=str)  # kernel_usage
parser.add_argument("--trainable-stem", default=True, type=str2bool)
parser.add_argument("--dilated", default=False, type=str2bool)
parser.add_argument("--restore", default=None, type=str)

parser.add_argument("--oc-architecture", default="resnet50", type=str)
parser.add_argument("--oc-regularization", default=None, type=str)
parser.add_argument("--oc-pretrained", required=True, type=str)
parser.add_argument("--oc-strategy", default="random", type=str, choices=list(occse.STRATEGIES))
parser.add_argument("--oc-focal-momentum", default=0.9, type=float)
parser.add_argument("--oc-focal-gamma", default=2.0, type=float)

# Hyperparameter
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--first_epoch", default=0, type=int)
parser.add_argument("--max_epoch", default=15, type=int)

parser.add_argument("--lr", default=0.1, type=float)
parser.add_argument("--wd", default=1e-4, type=float)
parser.add_argument("--label_smoothing", default=0, type=float)

parser.add_argument("--image_size", default=512, type=int)
parser.add_argument("--min_image_size", default=320, type=int)
parser.add_argument("--max_image_size", default=640, type=int)

parser.add_argument("--print_ratio", default=1.0, type=float)

parser.add_argument("--tag", default="", type=str)
parser.add_argument("--augment", default="", type=str)
parser.add_argument("--cutmix_prob", default=1.0, type=float)
parser.add_argument("--mixup_prob", default=1.0, type=float)

# For Puzzle-CAM
parser.add_argument("--num_pieces", default=4, type=int)
parser.add_argument("--loss_option", default="cl_pcl_re", type=str)
parser.add_argument("--re_loss", default="L1_Loss", type=str)  # 'L1_Loss', 'L2_Loss'

parser.add_argument("--alpha", default=1.0, type=float)
parser.add_argument("--alpha_init", default=0.0, type=float)
parser.add_argument("--alpha_schedule", default=0.50, type=float)

parser.add_argument("--oc-alpha", default=1.0, type=float)
parser.add_argument("--oc-alpha-init", default=0.3, type=float)
parser.add_argument("--oc-alpha-schedule", default=1.0, type=float)
parser.add_argument("--oc-k", default=1.0, type=float)
parser.add_argument("--oc-k-init", default=1.0, type=float)
parser.add_argument("--oc-k-schedule", default=0, type=float)

parser.add_argument("--ow", default=1.0, type=float)
parser.add_argument("--ow-init", default=0, type=float)
parser.add_argument("--ow-schedule", default=0.5, type=float)
parser.add_argument("--oc-train-interval-steps", default=1, type=int)
parser.add_argument("--oc-train-masks", default="features", choices=["features", "cams"], type=str)
parser.add_argument("--oc_train_mask_threshold", default=0.2, type=float)

if __name__ == "__main__":
  # Arguments
  args = parser.parse_args()

  TAG = args.tag
  SEED = args.seed
  DEVICE = args.device

  print("Train Configuration")
  pad = max(map(len, vars(args))) + 1
  for k, v in vars(args).items():
    print(f"{k.ljust(pad)}: {v}")
  print("===================")

  log_dir = create_directory(f"./experiments/logs/")
  data_dir = create_directory(f"./experiments/data/")
  model_dir = create_directory("./experiments/models/")

  log_path = log_dir + f"{TAG}.txt"
  data_path = data_dir + f"{TAG}.json"
  model_path = model_dir + f"{TAG}.pth"

  set_seed(SEED)
  log = lambda string="": log_print(string, log_path)
  log("[i] {}".format(TAG))
  log()

  tt, tv = get_classification_transforms(args.min_image_size, args.max_image_size, args.image_size, args.augment)
  train_dataset, valid_dataset = get_classification_datasets(
    args.dataset,
    args.data_dir,
    args.augment,
    args.image_size,
    args.cutmix_prob,
    args.mixup_prob,
    tt,
    tv,
  )

  train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=True,
    drop_last=True,
  )
  valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=1, drop_last=True)

  log(
    f"Dataset {args.dataset}\n"
    f"  classes={train_dataset.info.num_classes}\n"
    f"  train transforms={tt}\n"
    f"  valid transforms={tv}\n\n"
  )

  step_valid = len(train_loader)
  step_log = int(step_valid * args.print_ratio)
  step_init = args.first_epoch * step_valid
  step_max = args.max_epoch * step_valid
  log(f"Iterations: first={step_init} logging={step_log} validation={step_valid} max={step_max}")

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
    print(f"Restoring weights from {args.restore}")
    cgnet.load_state_dict(torch.load(args.restore), strict=True)

  log("[i] Architecture is {}".format(args.architecture))
  log("[i] Regularization is {}".format(args.regularization))
  log("[i] Total Params: %.2fM" % (calculate_parameters(cgnet)))
  log()

  # Ordinary Classifier.
  print(f"Build OC {args.oc_architecture} (weights from `{args.oc_pretrained}`)")
  if "mcar" in args.oc_architecture:
    ps = "avg"
    topN = 4
    threshold = 0.5
    ocnet = mcar_resnet101(
      train_dataset.info.num_classes,
      ps,
      topN,
      threshold,
      inference_mode=True,
      with_logits=True,
    )
    ckpt = torch.load(args.oc_pretrained)
    ocnet.load_state_dict(ckpt["state_dict"], strict=True)
  else:
    ocnet = Classifier(
      args.oc_architecture,
      train_dataset.info.num_classes,
      mode=args.mode,
      regularization=args.oc_regularization,
    )
    ocnet.load_state_dict(torch.load(args.oc_pretrained), strict=True)

  cg_param_groups = cgnet.get_parameter_groups()
  oc_param_groups = ocnet.get_parameter_groups()
  cgnet = cgnet.to(DEVICE)
  ocnet = ocnet.to(DEVICE)
  cgnet.train()
  ocnet.train()

  try:
    GPUS = os.environ["CUDA_VISIBLE_DEVICES"]
  except KeyError:
    GPUS = "0"
  GPUS = GPUS.split(",")
  GPUS_COUNT = len(GPUS)
  if GPUS_COUNT > 1:
    log("[i] the number of gpu : {}".format(GPUS_COUNT))
    cgnet = nn.DataParallel(cgnet)
    ocnet = nn.DataParallel(ocnet)

  load_model_fn = lambda: load_model(cgnet, model_path, parallel=GPUS_COUNT > 1)
  save_model_fn = lambda: save_model(cgnet, model_path, parallel=GPUS_COUNT > 1)

  # Loss, Optimizer
  class_loss_fn = nn.MultiLabelSoftMarginLoss(reduction="none").to(DEVICE)

  if args.re_loss == "L1_Loss":
    r_loss_fn = L1_Loss
  else:
    r_loss_fn = L2_Loss

  log(
    f"CGNet Parameters\n"
    f"  pretrained:\n    w={len(cg_param_groups[0])} b={len(cg_param_groups[1])}\n"
    f"  scratch:\n    w={len(cg_param_groups[2])} b={len(cg_param_groups[3])}\n"
  )
  log(
    f"OCNet Parameters\n"
    f"  pretrained:\n    w={len(oc_param_groups[0])} b={len(oc_param_groups[1])}\n"
    f"  scratch:\n    w={len(oc_param_groups[2])} b={len(oc_param_groups[3])}\n"
  )

  cgopt = get_optimizer(args.lr, args.wd, step_max, cg_param_groups)
  ocopt = get_optimizer(args.lr, args.wd, step_max, oc_param_groups)

  log(f"CGNet Optimizer\n{cgopt}")
  log(f"OCNet Optimizer\n{ocopt}")

  # Train
  data_dic = {"train": [], "validation": []}

  train_timer = Timer()
  eval_timer = Timer()

  train_metrics = MetricsContainer([
    "loss", "c_loss", "p_loss", "re_loss", "o_loss", "ot_loss", "alpha", "oc_alpha", "ot_weight", "k"
  ])

  best_train_mIoU = -1
  thresholds = list(np.arange(0.10, 0.50, 0.05))

  choices = torch.ones(train_dataset.info.num_classes)
  focal_factor = torch.ones(train_dataset.info.num_classes)

  def evaluate(loader):
    
    cgnet.eval()
    ocnet.eval()
    eval_timer.tik()

    meter_dic = {th: Calculator_For_mIoU(train_dataset.info.classes) for th in thresholds}

    preds = []
    targets = []

    with torch.no_grad():
      for step, (images, labels, gt_masks) in enumerate(loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        _, features = cgnet(images, with_cam=True)

        preds += [to_numpy(torch.sigmoid(ocnet(images)))]
        targets += [to_numpy(labels)]

        # features = resize_for_tensors(features, images.size()[-2:])
        # gt_masks = resize_for_tensors(gt_masks, features.size()[-2:], mode='nearest')

        mask = labels.unsqueeze(2).unsqueeze(3)
        cams = make_cam(features) * mask

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
    
    preds, targets = (np.concatenate(preds, axis=0),
                      np.concatenate(targets, axis=0))
    
    r = skmetrics.precision_recall_fscore_support(targets, preds.round(), average='macro')
    log(f"oc predictions (macro):    precision={r[0]:.5f} recall={r[1]:.5f} f1={r[2]:.5f} support={r[3]}")
    r = skmetrics.precision_recall_fscore_support(targets, preds.round(), average='weighted')
    log(f"oc predictions (weighted): precision={r[0]:.5f} recall={r[1]:.5f} f1={r[2]:.5f} support={r[3]}")

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
    
    cgnet.train()
    ocnet.eval()

    images = images.to(DEVICE)
    targets = targets.float()

    ap =       linear_schedule(step, step_max, args.alpha_init,    args.alpha,    args.alpha_schedule   )
    ao =       linear_schedule(step, step_max, args.oc_alpha_init, args.oc_alpha, args.oc_alpha_schedule)
    ow =       linear_schedule(step, step_max, args.ow_init,       args.ow,       args.ow_schedule      )
    k  = round(linear_schedule(step, step_max, args.oc_k_init,     args.oc_k,     args.oc_k_schedule    ))

    ##############################
    # CAM Generator Training
    ##############################
    cgopt.zero_grad()  # set_to_none=False  # TODO: Try it with True and check performance.

    # Normal
    logits, features = cgnet(images, with_cam=True)
    
    # Puzzle Module
    tiled_images = tile_features(images, args.num_pieces)
    tiled_logits, tiled_features = cgnet(tiled_images, with_cam=True)
    re_features = merge_features(tiled_features, args.num_pieces, args.batch_size)

    targets_sm = label_smoothing(targets, args.label_smoothing).to(logits)
    c_loss = class_loss_fn(logits, targets_sm).mean()
    p_loss = class_loss_fn(gap2d(re_features), targets_sm).mean()

    re_mask = targets.unsqueeze(2).unsqueeze(3)
    re_loss = (r_loss_fn(features, re_features) * re_mask.to(features)).mean()

    # OC-CSE
    labels_mask, _ = occse.split_label(targets, k, choices, focal_factor, args.oc_strategy)
    labels_oc = (targets - labels_mask).clip(min=0)

    images_mask = occse.images_with_masked_objects(images, features, labels_mask)
    cl_logits = ocnet(images_mask)

    o_loss = class_loss_fn(cl_logits, label_smoothing(labels_oc, args.label_smoothing).to(cl_logits)).mean()

    cg_loss = c_loss + p_loss + ap * re_loss + ao * o_loss
    cg_loss.backward()
    cgopt.step()

    occse.update_focal(
      targets,
      labels_oc,
      cl_logits.to(targets),
      focal_factor,
      args.oc_focal_momentum,
      args.oc_focal_gamma,
    )

    ##############################
    # Ordinary Classifier Training
    ##############################
    if (step + 1) % args.oc_train_interval_steps != 0:
      # Skip OC-CSE training this step.
      ot_loss = ow * class_loss_fn(cl_logits, targets_sm).mean()
    else:
      ocnet.train()
      ocopt.zero_grad()

      if args.oc_train_masks == "cams":
        features = features.cpu()
        mask = features[labels_mask == 1, :, :].unsqueeze(1)
        mask = F.relu(mask)
        mask = resize_for_tensors(mask, images.size()[2:])
        mask /= F.adaptive_max_pool2d(mask, (1, 1)) + 1e-5
        mask = mask > args.oc_train_mask_threshold
        images_mask = (images * (1 - mask.to(images)))

      images_mask = images_mask.detach()

      cl_logits = ocnet(images_mask.detach())

      ot_loss = ow * class_loss_fn(cl_logits, targets_sm).mean()
      ot_loss.backward()
      ocopt.step()

    # region logging
    train_metrics.update(
      {
        "loss": cg_loss.item(),
        "c_loss": c_loss.item(),
        "p_loss": p_loss.item(),
        "re_loss": re_loss.item(),
        "o_loss": o_loss.item(),
        "ot_loss": ot_loss.item(),
        "alpha": ap,
        "oc_alpha": ao,
        "ot_weight": ow,
        "k": k,
      }
    )

    if (step + 1) % step_log == 0:
      (cg_loss, c_loss, p_loss, re_loss, o_loss, ot_loss, ap, ao, ow, k) = train_metrics.get(clear=True)

      lr = float(get_learning_rate_from_optimizer(cgopt))
      cs = to_numpy(choices).tolist()
      ffs = to_numpy(focal_factor).astype(float).round(2).tolist()

      data = {
        "iteration": step + 1,
        "lr": lr,
        "alpha": ap,
        "loss": cg_loss,
        "c_loss": c_loss,
        "p_loss": p_loss,
        "re_loss": re_loss,
        "o_loss": o_loss,
        "ot_loss": ot_loss,
        "oc_alpha": ao,
        "ow": ow,
        "k": k,
        "time": train_timer.tok(clear=True),
        "choices": cs,
        "focal_factor": ffs,
      }
      data_dic["train"].append(data)
      write_json(data_path, data_dic)

      log(
        "iteration    = {iteration:,}\n"
        "time         = {time:.0f} sec\n"
        "lr           = {lr:.4f}\n"
        "alpha        = {alpha:.2f}\n"
        "loss         = {loss:.4f}\n"
        "c_loss       = {c_loss:.4f}\n"
        "p_loss       = {p_loss:.4f}\n"
        "re_loss      = {re_loss:.4f}\n"
        "o_loss       = {o_loss:.4f}\n"
        "o_train_loss = {ot_loss:.4f}\n"
        "o_train_w    = {ow:.4f}\n"
        "oc_alpha     = {oc_alpha:.4f}\n"
        "k            = {k}\n"
        "focal_factor = {focal_factor}\n"
        "choices      = {choices}\n".format(**data)
      )

      # endregion

    # region evaluation
    if (step + 1) % step_valid == 0:
      threshold, mIoU, iou = evaluate(valid_loader)

      if best_train_mIoU == -1 or best_train_mIoU < mIoU:
        best_train_mIoU = mIoU

      data = {
        "iteration": step + 1,
        "threshold": threshold,
        "train_mIoU": mIoU,
        "train_iou": iou,
        "best_train_mIoU": best_train_mIoU,
        "time": eval_timer.tok(clear=True),
      }
      data_dic["validation"].append(data)
      write_json(data_path, data_dic)

      log(
        "iteration       = {iteration:,}\n"
        "time            = {time:.0f} sec\n"
        "threshold       = {threshold:.2f}\n"
        "train_mIoU      = {train_mIoU:.2f}%\n"
        "best_train_mIoU = {best_train_mIoU:.2f}%\n"
        "train_iou       = {train_iou}".format(**data)
      )

      log(f"saving weights `{model_path}`\n")
      save_model_fn()
    # endregion

  write_json(data_path, data_dic)
  log(TAG)
