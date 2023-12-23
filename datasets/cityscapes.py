import os
from typing import List, Optional

import numpy as np

from . import base

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cityscapes")
CLASSES = [
  "road",
  "sidewalk",
  "building",
  "wall",
  "fence",
  "pole",
  "traffic_light",
  "traffic_sign",
  "vegetation",
  "terrain",
  "sky",
  "person",
  "rider",
  "car",
  "truck",
  "bus",
  "train",
  "motorcycle",
  "bicycle",
  # "unlabelled",
]
COLORS = [
  [0, 0, 0],
  [128, 64, 128],
  [244, 35, 232],
  [70, 70, 70],
  [102, 102, 156],
  [190, 153, 153],
  [153, 153, 153],
  [250, 170, 30],
  [220, 220, 0],
  [107, 142, 35],
  [152, 251, 152],
  [0, 130, 180],
  [220, 20, 60],
  [255, 0, 0],
  [0, 0, 142],
  [0, 0, 70],
  [0, 60, 100],
  [0, 80, 100],
  [0, 0, 230],
  [119, 11, 32],
  [127, 127, 127],
]


class CityscapesDataSource(base.CustomDataSource):

  NAME = "cityscapes"
  DOMAINS = {
    "train": "train75",
    "valid": "test",
    "test": "test",
  }

  @classmethod
  def _load_labels_from_npy(cls):
    filepath = os.path.join(DATA_DIR, "labels.npy")
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
      masks_dir=masks_dir or os.path.join(root_dir, "SegmentationClass"),
      sample_ids=sample_ids,
    )
    self.root_dir = root_dir
    self.sample_labels = self._load_labels_from_npy()

  def get_label(self, sample_id) -> np.ndarray:
    label = self.sample_labels[sample_id]
    return label

  def get_info(self, task: str) -> base.DatasetInfo:
    if task == "segmentation":
      num_classes = 19
      classes = CLASSES
      colors = COLORS
      void_class = 250
      bg_class = 0
    else:
      num_classes = 19
      classes = CLASSES
      colors = COLORS
      void_class = None
      bg_class = 0

    return base.DatasetInfo(
      num_classes=num_classes,
      classes=classes,
      colors=colors,
      bg_class=bg_class,
      void_class=void_class,
    )


base.DATASOURCES[CityscapesDataSource.NAME] = CityscapesDataSource
