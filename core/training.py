import time
from typing import List, Optional

import numpy as np
import sklearn.metrics as skmetrics
import torch
from torch.utils.data import DataLoader

from datasets import DatasetInfo
from tools.ai.evaluate_utils import (MIoUCalculator,
                                     accumulate_batch_iou_priors,
                                     accumulate_batch_iou_saliency,
                                     maximum_miou_from_thresholds)
from tools.ai.torch_utils import make_cam, resize_tensor, to_numpy
from tools.general import wandb_utils


def classification_validation_step(
    model: torch.nn.Module,
    loader: DataLoader,
    info: DatasetInfo,
    device: str,
    max_steps: Optional[int] = None,
  ):
  """Run Classification Validation Step.

  Evaluate Multi-label Classification produced by a `model`,
  when presented with samples from a `loader`.

  """

  start = time.time()

  preds_ = []
  targets_ = []

  with torch.no_grad():
    for step, (ids, inputs, targets, _) in enumerate(loader):
      targets = to_numpy(targets)
      logits, features = model(inputs.to(device), with_cam=True)

      labels_mask = targets[..., np.newaxis, np.newaxis]
      cams = to_numpy(make_cam(features.cpu().float())) * labels_mask
      cams = cams.transpose(0, 2, 3, 1)

      preds = to_numpy(torch.sigmoid(logits.cpu().float()))
      preds_.append(preds)
      targets_.append(targets)

      if step == 0:
        inputs = to_numpy(inputs)
        wandb_utils.log_cams(ids, inputs, targets, cams, preds, classes=info.classes, normalize_stats=info.normalize_stats)

      if max_steps and step >= max_steps:
        break

  elapsed = time.time() - start

  preds_ = np.concatenate(preds_, axis=0)
  targets_ = np.concatenate(targets_, axis=0)

  try:
    precision, recall, f_score, _ = skmetrics.precision_recall_fscore_support(targets_, preds_.round(), average="macro")
    roc = skmetrics.roc_auc_score(targets_, preds_, average="macro")
  except ValueError:
    precision = recall = f_score = roc = 0.

  results = {
    "precision": round(100 * precision, 3),
    "recall": round(100 * recall, 3),
    "f_score": round(100 * f_score, 3),
    "roc_auc": round(100 * roc, 3),
    "time": int(round(elapsed)),
  }

  return results


def priors_validation_step(
    model: torch.nn.Module,
    loader: DataLoader,
    info: DatasetInfo,
    thresholds: List[float],
    device: str,
    max_steps: Optional[int] = None,
  ):
  """Run Validation Step.

  Evaluate CAMs priors produced by a classification `model`,
  when presented with samples from a `loader`.

  """

  meters = {t: MIoUCalculator.from_dataset_info(info) for t in thresholds}

  start = time.time()

  preds_ = []
  targets_ = []

  with torch.no_grad():
    for step, (ids, inputs, targets, masks) in enumerate(loader):
      targets = to_numpy(targets)
      masks = to_numpy(masks)
      logits, features = model(inputs.to(device), with_cam=True)

      labels_mask = targets[..., np.newaxis, np.newaxis]
      cams = to_numpy(make_cam(features.cpu().float())) * labels_mask
      cams = cams.transpose(0, 2, 3, 1)

      preds = to_numpy(torch.sigmoid(logits.cpu().float()))
      preds_.append(preds)
      targets_.append(targets)

      if step == 0:
        inputs = to_numpy(inputs)
        wandb_utils.log_cams(ids, inputs, targets, cams, preds, classes=info.classes)

      accumulate_batch_iou_priors(masks, cams, meters, include_bg=info.bg_class is None)

      if max_steps and step >= max_steps:
        break

  elapsed = time.time() - start

  preds_ = np.concatenate(preds_, axis=0)
  targets_ = np.concatenate(targets_, axis=0)

  try:
    precision, recall, f_score, _ = skmetrics.precision_recall_fscore_support(targets_, preds_.round(), average="macro")
    roc = skmetrics.roc_auc_score(targets_, preds_, average="macro")
  except ValueError:
    precision = recall = f_score = roc = 0.

  results = maximum_miou_from_thresholds(meters)
  results.update({
    "precision": round(100 * precision, 3),
    "recall": round(100 * recall, 3),
    "f_score": round(100 * f_score, 3),
    "roc_auc": round(100 * roc, 3),
    "time": int(round(elapsed)),
  })

  return results


def segmentation_validation_step(
    model: torch.nn.Module,
    loader: DataLoader,
    info: DatasetInfo,
    device: str,
    log_samples: bool = True,
):
  start = time.time()
  meter = MIoUCalculator(info.classes, bg_class=info.bg_class, include_bg=False)

  with torch.no_grad():
    for step, (ids, inputs, targets, masks) in enumerate(loader):
      _, H, W = masks.shape

      logits = model(inputs.to(device))
      preds = torch.argmax(logits, dim=1).cpu()
      preds = resize_tensor(
        preds.float().unsqueeze(1), (H, W), mode="nearest"
      ).squeeze().to(masks)
      preds = to_numpy(preds)
      masks = to_numpy(masks)

      meter.add_many(preds, masks)

      if step == 0 and log_samples:
        inputs = to_numpy(inputs)
        wandb_utils.log_masks(
          ids, inputs, targets, masks, preds,
          classes=info.classes,
          normalize_stats=info.normalize_stats,
          void_class=info.void_class)

  miou, miou_fg, iou, FP, FN = meter.get(clear=True, detail=True)
  iou = [round(iou[c], 2) for c in info.classes]

  elapsed = time.time() - start

  return {
    "miou": miou,
    "miou_fg": miou_fg,
    "iou": iou,
    "fp": FP,
    "fn": FN,
    "time": elapsed,
  }


def saliency_validation_step(
    model: torch.nn.Module,
    loader: DataLoader,
    thresholds: List[float],
    device: str,
    max_steps: Optional[int] = None,
):
  start = time.time()

  classes = ['background', 'foreground']
  iou_meters = {th: MIoUCalculator(classes, bg_class=0, include_bg=False) for th in thresholds}

  with torch.no_grad():
    for step, (_, images, _, masks) in enumerate(loader):
      B, C, H, W = images.size()

      _, _, preds = model(images.to(device))

      preds = resize_tensor(preds.cpu().float(), (H, W))
      preds = to_numpy(make_cam(preds).squeeze())
      masks = to_numpy(masks)

      accumulate_batch_iou_saliency(masks, preds, iou_meters)

      if max_steps and step >= max_steps:
        break

  elapsed = time.time() - start

  results = maximum_miou_from_thresholds(iou_meters)
  results.update({
    "time" : elapsed,
  })

  return results
