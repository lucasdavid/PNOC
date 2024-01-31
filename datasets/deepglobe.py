import os
from typing import List, Optional

import numpy as np

from . import base

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "deepglobe")
CLASSES = ["agriculture_land", "urban_land", "rangeland", "forest_land", "water", "barren_land", "unknown"]
COLORS = [[255, 255, 0], [0, 255, 255], [255, 0, 255], [0, 255, 0], [0, 0, 255], [255, 255, 255], [0, 0, 0], [127, 127, 127]]

BG_CLASS = "agriculture_land"
FG_NPY_INDICES = [0, 2, 3, 4, 5, 6]  # Agriculture (our bg) is the 2nd class in the stored .npy vectors.


class DeepGlobeLandCoverDataSource(base.CustomDataSource):

  NAME = "deepglobe"
  DOMAINS = {
    "train": "train75",
    "valid": "test",
    "test": "test",
  }

  @classmethod
  def _load_labels_from_npy(cls):
    filepath = os.path.join(DATA_DIR, "cls_labels_unbalanced.npy")
    return {k: v.astype("float32") for k, v in np.load(filepath, allow_pickle=True).item().items()}

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
    label = self.sample_labels[sample_id]
    return label[FG_NPY_INDICES]

  def get_info(self, task: str) -> base.DatasetInfo:
    if task == "segmentation":
      num_classes = len(CLASSES)
      classes = CLASSES
      colors = COLORS
      void_class = 7
      bg_class = 0
    else:
      num_classes = len(CLASSES) - 1
      classes = CLASSES[1:]
      colors = COLORS[1:]
      void_class = None
      bg_class = None

    return base.DatasetInfo(
      num_classes=num_classes,
      classes=classes,
      colors=colors,
      bg_class=bg_class,
      void_class=void_class,
    )


base.DATASOURCES[DeepGlobeLandCoverDataSource.NAME] = DeepGlobeLandCoverDataSource
