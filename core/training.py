import time
from typing import List, Optional

import numpy as np
import sklearn.metrics as skmetrics
import torch
from torch.utils.data import DataLoader

from datasets import DatasetInfo
from tools.ai.evaluate_utils import (MIoUCalcFromNames, MIoUCalculator,
                                     accumulate_batch_iou_priors,
                                     accumulate_batch_iou_saliency,
                                     maximum_miou_from_thresholds)
from tools.ai.torch_utils import make_cam, resize_for_tensors, to_numpy
from tools.general import wandb_utils


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

  # if dataset does not have a background class, add it at i=0.
  include_bg = info.bg_class is None
  bg_class = info.bg_class or 0
  meters = {t: MIoUCalculator(info.classes, bg_class=bg_class, include_bg=include_bg) for t in thresholds}

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

      accumulate_batch_iou_priors(masks, cams, meters, include_bg=include_bg)

      if max_steps and step >= max_steps:
        break

  elapsed = time.time() - start

  preds_ = np.concatenate(preds_, axis=0)
  targets_ = np.concatenate(targets_, axis=0)
  precision, recall, f_score, _ = skmetrics.precision_recall_fscore_support(targets_, preds_.round(), average="macro")
  roc = skmetrics.roc_auc_score(targets_, preds_, average="macro")

  results = maximum_miou_from_thresholds(meters, info.classes)
  results.update({
    "precision": precision,
    "recall": recall,
    "f_score": f_score,
    "roc_auc": roc,
    "time": elapsed,
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
      preds = resize_for_tensors(
        preds.float().unsqueeze(1), (H, W), mode="nearest"
      ).squeeze().to(masks)
      preds = to_numpy(preds)
      masks = to_numpy(masks)

      meter.add_many(preds, masks)

      if step == 0 and log_samples:
        inputs = to_numpy(inputs)
        wandb_utils.log_masks(ids, inputs, targets, masks, preds, info.classes)

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
    info: DatasetInfo,
    thresholds: List[float],
    device: str,
):
  start = time.time()

  classes = ['background', 'foreground']
  iou_meters = {th: MIoUCalcFromNames(classes, bg_class=0) for th in thresholds}

  with torch.no_grad():
    for _, (_, images, _, masks) in enumerate(loader):
      B, C, H, W = images.size()

      _, _, ccams = model(images.to(device))

      ccams = resize_for_tensors(ccams.cpu().float(), (H, W))
      ccams = to_numpy(make_cam(ccams).squeeze())
      masks = to_numpy(masks)

      accumulate_batch_iou_saliency(masks, ccams, iou_meters, bg_class=info.bg_class)

  elapsed = time.time() - start

  results = maximum_miou_from_thresholds(iou_meters, classes)
  results.update({
    "time" : elapsed,
  })

  return results
