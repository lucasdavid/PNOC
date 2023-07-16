from typing import Dict, Optional

import cv2
import numpy as np


def accumulate_batch_iou_priors(masks, cams, meters, include_bg: bool = True):
  for i in range(len(masks)):
    yi = masks[i]
    ci = cams[i]

    h, w, _ = ci.shape
    yi = cv2.resize(yi, (w, h), interpolation=cv2.INTER_NEAREST)

    for t, meter in meters.items():
      pred = ci
      if include_bg:
        bg = np.ones_like(ci[:, :, 0]) * t
        pred = np.concatenate([bg[..., np.newaxis], pred], axis=-1)

      p = np.argmax(pred, axis=-1)
      meter.add(p, yi)


def accumulate_batch_iou_saliency(masks, ccams, meters: Dict[float, "MIoUCalculator"]):
  for i in range(len(masks)):
    yi = masks[i]
    pi = ccams[i]

    for t, meter in meters.items():
      pi_b = (pi > t).astype(yi.dtype)
      meter.add(pi_b, yi)


def maximum_miou_from_thresholds(iou_meters: Dict[float, "MIoUCalculator"]):
  th_ = iou_ = None
  miou_ = miou_fg_ = 0.0

  for th, meter in iou_meters.items():
    miou, miou_fg, iou, *_ = meter.get(clear=True, detail=True)
    if miou_ < miou:
      th_ = th
      miou_ = miou
      miou_fg_ = miou_fg
      iou_ = [round(iou[c], 2) for c in meter.class_names]

  return {
    "threshold": th_,
    "miou": round(miou_, 3),
    "miou_fg": round(miou_fg_, 3),
    "iou": iou_,
  }


class MIoUCalculator:

  def __init__(self, classes, bg_class: Optional[int] = 0, include_bg: bool = True):
    if isinstance(classes, np.ndarray):
      classes = classes.tolist()

    if include_bg:
      classes = classes[:bg_class] + ["background"] + classes[bg_class:]

    self.bg_class = bg_class
    self.include_bg = include_bg
    self.class_names = classes
    self.classes = len(self.class_names)

    self.clear()

  def get_data(self, pred_mask, gt_mask):
    obj_mask = gt_mask < 255
    correct_mask = (pred_mask == gt_mask) * obj_mask

    P_list, T_list, TP_list = [], [], []
    for i in range(self.classes):
      P_list.append(np.sum((pred_mask == i) * obj_mask))
      T_list.append(np.sum((gt_mask == i) * obj_mask))
      TP_list.append(np.sum((gt_mask == i) * correct_mask))

    return (P_list, T_list, TP_list)

  def add_using_data(self, data):
    P_list, T_list, TP_list = data
    for i in range(self.classes):
      self.P[i] += P_list[i]
      self.T[i] += T_list[i]
      self.TP[i] += TP_list[i]

  def add(self, pred_mask, gt_mask):
    obj_mask = gt_mask < 255
    correct_mask = (pred_mask == gt_mask) * obj_mask

    for i in range(self.classes):
      self.P[i] += np.sum((pred_mask == i) * obj_mask)
      self.T[i] += np.sum((gt_mask == i) * obj_mask)
      self.TP[i] += np.sum((gt_mask == i) * correct_mask)

  def add_many(self, predictions, masks):
    for p, m in zip(predictions, masks):
      self.add(p, m)

  def get(self, detail=False, clear=True):
    IoU_dic = {}
    IoU_list = []

    FP_list = []  # over activation
    FN_list = []  # under activation

    for i in range(self.classes):
      IoU = self.TP[i] / (self.T[i] + self.P[i] - self.TP[i] + 1e-10) * 100
      FP = (self.P[i] - self.TP[i]) / (self.T[i] + self.P[i] - self.TP[i] + 1e-10)
      FN = (self.T[i] - self.TP[i]) / (self.T[i] + self.P[i] - self.TP[i] + 1e-10)

      IoU_dic[self.class_names[i]] = IoU

      IoU_list.append(IoU)
      FP_list.append(FP)
      FN_list.append(FN)

    iou = np.asarray(IoU_list)
    miou = np.mean(iou)

    if self.bg_class is None:
      miou_fg = miou
    else:
      miou_fg = (sum(iou[:self.bg_class]) + sum(iou[self.bg_class + 1:])) / (len(iou) - 1)

    FP = np.mean(FP_list)
    FN = np.mean(FN_list)

    if clear:
      self.clear()

    if detail:
      return miou, miou_fg, IoU_dic, FP, FN
    else:
      return miou, miou_fg

  def clear(self):
    self.TP = []
    self.P = []
    self.T = []

    for _ in range(self.classes):
      self.TP.append(0)
      self.P.append(0)
      self.T.append(0)


class MIoUCalcFromNames(MIoUCalculator):

  def __init__(self, class_names, bg_class: int = 0, include_bg: bool = False):
    self.bg_class = bg_class
    self.include_bg = include_bg
    self.class_names = class_names
    self.classes = len(self.class_names)

    self.clear()
