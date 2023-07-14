import time
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from tools.ai.evaluate_utils import (MIoUCalcFromNames, MIoUCalculator,
                                     accumulate_batch_iou,
                                     accumulate_batch_iou_saliency,
                                     maximum_miou_from_thresholds)
from tools.ai.torch_utils import make_cam, resize_for_tensors, to_numpy
from tools.general import wandb_utils


def priors_validation_step(
    model: torch.nn.Module,
    loader: DataLoader,
    classes: List[str],
    thresholds: List[float],
    device: str,
    max_steps: Optional[int] = None,
    bg_class: Optional[int] = None,
  ):
  """Run Validation Step.

  Evaluate CAMs priors produced by a classification `model`,
  when presented with samples from a `loader`.

  """

  # if dataset does not have a background class, add it at i=0.
  include_bg = bg_class is None
  bg_class = bg_class or 0
  meters = {t: MIoUCalculator(classes, bg_class=bg_class, include_bg=include_bg) for t in thresholds}

  start = time.time()

  with torch.no_grad():
    for step, (ids, inputs, targets, masks) in enumerate(loader):
      targets = to_numpy(targets)
      masks = to_numpy(masks)
      logits, features = model(inputs.to(device), with_cam=True)

      labels_mask = targets[..., np.newaxis, np.newaxis]
      cams = to_numpy(make_cam(features.cpu().float())) * labels_mask
      cams = cams.transpose(0, 2, 3, 1)

      if step == 0:
        inputs = to_numpy(inputs)
        preds = to_numpy(torch.sigmoid(logits).float())
        wandb_utils.log_cams(ids, inputs, targets, cams, preds, classes=classes)

      accumulate_batch_iou(masks, cams, meters, include_bg=include_bg)

      if max_steps and step >= max_steps:
        break

  val_time = time.time() - start

  return (*maximum_miou_from_thresholds(meters, classes), val_time)

  # preds_ = np.concatenate(preds_, axis=0)
  # targets_ = np.concatenate(targets_, axis=0)
  # rm = skmetrics.precision_recall_fscore_support(targets_, preds_.round(), average='macro')
  # rw = skmetrics.precision_recall_fscore_support(targets_, preds_.round(), average='weighted')

def segmentation_validation_step(
    model: torch.nn.Module,
    loader: DataLoader,
    classes: List[str],
    device: str,
    bg_class: int = 0,
    log_samples: bool = True,
):
  start = time.time()

  meter = MIoUCalculator(classes, bg_class=bg_class, include_bg=False)

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
        wandb_utils.log_masks(ids, inputs, targets, masks, preds, classes)

  miou, _, iou, *_ = meter.get(clear=True, detail=True)
  iou = [round(iou[c], 2) for c in classes]

  val_time = time.time() - start

  return miou, iou, val_time

def saliency_validation_step(
    model: torch.nn.Module,
    loader: DataLoader,
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

      accumulate_batch_iou_saliency(masks, ccams, iou_meters)

  val_time = time.time() - start

  return (*maximum_miou_from_thresholds(iou_meters, classes), val_time)
