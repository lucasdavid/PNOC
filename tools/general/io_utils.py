# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import argparse
import os
from typing import Tuple

import numpy as np
from PIL import Image


def create_directory(path):
  if not os.path.isdir(path):
    os.makedirs(path, exist_ok=True)
  return path


def str2bool(v):
  if isinstance(v, bool):
    return v
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')


def load_cam_file(cam_file: str, allow_pickle: bool = True) -> Tuple[np.ndarray, np.ndarray]:
  data = np.load(cam_file, allow_pickle=allow_pickle).item()

  if "keys" in data:
    # Affinity/Puzzle/PNOC
    keys = data["keys"]

    if "hr_cam" in data.keys():
      cam = data["hr_cam"]
    elif "rw" in data.keys():
      cam = data["rw"]
  else:
    # OC-CSE
    keys = list(data.keys())
    cam = np.stack([data[k] for k in keys], 0)
    keys = np.asarray([0] + [k+1 for k in keys])

  return cam, keys


def load_saliency_file(sal_file: str, kind: str = 'saliency') -> np.ndarray:

  with Image.open(sal_file) as s:
    s = np.asarray(s)

  if kind == 'saliency':
    s = s.astype(float) / 255.
  elif kind == 'segmentation':
    s = (~np.isin(s, [0, 255])).astype(float)

  if len(s.shape) == 2:
    s = s[np.newaxis, ...]

  return s


def load_background_file(sal_file, kind='saliency'):
  return 1 - load_saliency_file(sal_file, kind=kind)


def load_dlv2_seg_file(npy_file, sizes):
  import torch
  import torch.nn.functional as F

  logit = np.load(npy_file)
  logit = torch.FloatTensor(logit)[None, ...]
  logit = F.interpolate(logit, size=sizes, mode="bilinear", align_corners=False)
  prob = F.softmax(logit, dim=1)[0].numpy()
  y_pred = np.argmax(prob, axis=0)

  keys, cam = np.unique(y_pred, return_inverse=True)
  cam = cam.reshape(y_pred.shape)

  return cam, keys, prob
