import os
from typing import List, Optional

import numpy as np

from core.aff_utils import *
from tools.ai.augment_utils import *

from . import base

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'deepglobe')
CLASSES = ['urban_land', 'agriculture_land', 'rangeland', 'forest_land', 'water', 'barren_land', 'unknown']
COLORS  = [[0, 255, 255], [255, 255, 0], [255, 0, 255], [0, 255, 0], [0, 0, 255], [255, 255, 255], [0, 0, 0]]


class DeepGlobeLandCoverDataSource(base.CustomDataSource):

  NAME = "deepglobe"
  DOMAINS = {
    "train": "train75",
    "valid": "test",
    "test": "test",
  }

  VOID_CLASS = 6

  def __init__(
    self,
    root_dir,
    domain: str,
    split: Optional[str] = None,
    images_dir=None,
    masks_dir: str = None,
    xml_dir: str = None,
    sample_ids: List[str] = None,
    segmentation: bool = False,
  ):
    super().__init__(
      domain=domain,
      split=split,
      images_dir=images_dir or os.path.join(root_dir, "JPEGImages"),
      masks_dir=masks_dir or os.path.join(root_dir, "SegmentationClassAug"),
      sample_ids=sample_ids,
      segmentation=segmentation,
    )
    self.root_dir = root_dir
    self.sample_labels = self._load_labels_from_npy()

    self.colors = np.asarray(COLORS)
    self.color_ids = self.colors[:, 0] * 256**2 + self.colors[:, 1] * 256 + self.colors[:, 2]

  def get_label(self, sample_id) -> np.ndarray:
    return self.sample_labels[sample_id].astype("float32")

  @classmethod
  def _load_labels_from_npy(cls):
    filepath = os.path.join(DATA_DIR, 'cls_labels_unbalanced.npy')
    return np.load(filepath, allow_pickle=True).item()


base.DATASOURCES[DeepGlobeLandCoverDataSource.NAME] = DeepGlobeLandCoverDataSource
