import time
from typing import List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from tools.ai.evaluate_utils import (MIoUCalcFromNames, MIoUCalculator,
                                     accumulate_batch_iou,
                                     accumulate_batch_iou_saliency,
                                     result_miou_from_thresholds)
from tools.ai.torch_utils import make_cam, resize_for_tensors, to_numpy
from tools.general import wandb_utils


def priors_validation_step(
    model: torch.nn.Module,
    loader: DataLoader,
    classes: List[str],
    thresholds: List[float],
    device: str,
    max_steps: Optional[int] = None,
    bg_index: int = 0,
    include_bg: bool = True,
  ):
  """Run Validation Step.

  Evaluate CAMs priors produced by a classification `model`,
  when presented with samples from a `loader`.

  """

  iou_meters = {th: MIoUCalculator(classes, bg_index=bg_index, include_bg=include_bg) for th in thresholds}

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
        preds = to_numpy(torch.sigmoid(logits).float())  # TODO: check if `to_numpy(...).astype(np.float32)` is better.
        wandb_utils.log_cams(classes, inputs, targets, cams, preds)

      accumulate_batch_iou(masks, cams, iou_meters)

      if max_steps and step >= max_steps:
        break

  val_time = time.time() - start

  return (*result_miou_from_thresholds(iou_meters, classes), val_time)

  # preds_ = np.concatenate(preds_, axis=0)
  # targets_ = np.concatenate(targets_, axis=0)
  # rm = skmetrics.precision_recall_fscore_support(targets_, preds_.round(), average='macro')
  # rw = skmetrics.precision_recall_fscore_support(targets_, preds_.round(), average='weighted')

def segmentation_validation_step(
    model: torch.nn.Module,
    loader: DataLoader,
    classes: List[str],
    device: str,
):
  start = time.time()

  # Set bg_index=None as we don't use FG mIoU information.
  iou_meter = MIoUCalculator(classes, bg_index=None, include_bg=False)

  with torch.no_grad():
    for _, images, _, masks in loader:
      logits = model(images.to(device))
      preds = torch.argmax(logits, dim=1)

      masks = to_numpy(masks)
      preds = to_numpy(preds)

      for i in range(images.shape[0]):
        pred_mask = preds[i]
        gt_mask = masks[i]

        h, w = pred_mask.shape
        gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        iou_meter.add(pred_mask, gt_mask)

  miou, _, iou, *_ = iou_meter.get(clear=True, detail=True)
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
  iou_meters = {th: MIoUCalcFromNames(classes, bg_index=0) for th in thresholds}

  with torch.no_grad():
    for _, (_, images, _, masks) in enumerate(loader):
      B, C, H, W = images.size()

      _, _, ccams = model(images.to(device))

      ccams = resize_for_tensors(ccams.cpu().float(), (H, W))
      ccams = to_numpy(make_cam(ccams).squeeze())
      masks = to_numpy(masks)

      accumulate_batch_iou_saliency(masks, ccams, iou_meters)

  val_time = time.time() - start

  return (*result_miou_from_thresholds(iou_meters, classes), val_time)
