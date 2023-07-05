import os
from typing import List, Optional, Union

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from core.aff_utils import *

from . import base

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'coco14')


def _decode_id(sample_id):
  s = str(sample_id).split('\n')[0]
  if len(s) != 12:
    s = '%012d' % int(s)
  return s


def _load_labels_from_npy(sample_ids, root_dir):
  filepath = os.path.join(root_dir, 'cls_labels_coco.npy')
  data = np.load(filepath, allow_pickle=True).item()
  return {_id: data[int(_id)] for _id in sample_ids}


class COCO14DataSource(base.CustomDataSource):

  NAME = "coco14"
  DOMAINS = {
    "train": "train2014",
    "valid": "val2014",
  }

  UNKNOWN_CLASS = 81

  def __init__(
    self,
    root_dir: str,
    domain: str,
    split: Optional[str] = None,
    images_dir: Optional[str] = None,
    masks_dir: Optional[str] = None,
    sample_ids: Union[str, List[str]] = None,
  ):
    domain = domain or split and self.DOMAINS.get(split, self.DOMAINS[self.DEFAULT_SPLIT])

    super().__init__(
      images_dir=images_dir or os.path.join(root_dir, domain),
      masks_dir=masks_dir or os.path.join(root_dir, "coco_seg_anno"),
      sample_ids=sample_ids,
      domain=domain,
      split=split,
    )

    self.root_dir = root_dir
    self.sample_labels = _load_labels_from_npy(self.sample_ids, root_dir)

  def get_image_path(self, sample_id) -> str:
    return os.path.join(self.image_dir, f"COCO_{self.domain}_{sample_id}.jpg")

  def get_label(self, sample_id) -> np.ndarray:
    return self.sample_labels[sample_id]

  def get_sample_ids(self, domain) -> List[str]:
    sample_ids = super().get_sample_ids(domain)
    return [_decode_id(_id) for _id in sample_ids if _id]


base.DATASOURCES[COCO14DataSource.NAME] = COCO14DataSource
