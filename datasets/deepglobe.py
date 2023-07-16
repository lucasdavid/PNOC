import os
from typing import List, Optional

import numpy as np

from . import base

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'deepglobe')
CLASSES = ['urban_land', 'agriculture_land', 'rangeland', 'forest_land', 'water', 'barren_land', 'unknown']
COLORS = [[0, 255, 255], [255, 255, 0], [255, 0, 255], [0, 255, 0], [0, 0, 255], [255, 255, 255], [0, 0, 0]]


class DeepGlobeLandCoverDataSource(base.CustomDataSource):

  NAME = "deepglobe"
  DOMAINS = {
    "train": "train75",
    "valid": "test",
    "test": "test",
  }

  @classmethod
  def _load_labels_from_npy(cls):
    filepath = os.path.join(DATA_DIR, 'cls_labels_unbalanced.npy')
    return np.load(filepath, allow_pickle=True).item()

  def __init__(
    self,
    root_dir,
    domain: str,
    split: Optional[str] = None,
    images_dir=None,
    masks_dir: str = None,
    sample_ids: List[str] = None,
  ):
    super().__init__(
      domain=domain,
      split=split,
      images_dir=images_dir or os.path.join(root_dir, "JPEGImages"),
      masks_dir=masks_dir or os.path.join(root_dir, "SegmentationClassAug"),
      sample_ids=sample_ids,
    )
    self.root_dir = root_dir
    self.sample_labels = self._load_labels_from_npy()

  def get_label(self, sample_id) -> np.ndarray:
    return self.sample_labels[sample_id].astype("float32")

  def get_info(self, task: str) -> base.DatasetInfo:
    if task == "segmentation":
      void_class = 7
      bg_class = 1  # can I call this pinoptic?
    else:
      void_class = None
      bg_class = 1

    return base.DatasetInfo(
      classes=CLASSES,
      colors=COLORS,
      bg_class=bg_class,
      void_class=void_class,
    )


base.DATASOURCES[DeepGlobeLandCoverDataSource.NAME] = DeepGlobeLandCoverDataSource
