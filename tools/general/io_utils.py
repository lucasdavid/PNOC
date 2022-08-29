# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import argparse
import os
import random

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


def load_saliency_file(sal_file, kind='saliency'):
  if kind == 'saliency':
    s = np.array(Image.open(sal_file)).astype(float)
    s = s / 255.
  elif kind == 'segmentation':
    s = np.array(Image.open(sal_file))
    s = (~np.isin(s, [0, 255])).astype(float)

  if len(s.shape) == 2:
    s = s[np.newaxis, ...]

  return s


def load_background_file(sal_file, kind='saliency'):
  return 1 - load_saliency_file(sal_file, kind=kind)
