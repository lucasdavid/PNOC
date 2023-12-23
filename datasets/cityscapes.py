import os
from typing import List, Optional

from PIL import Image
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
]
COLORS = [
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


def onehot_encode(indices, n=19):
  target = np.zeros(n, dtype=np.float32)
  target[indices] = 1

  return target


class CityscapesDataSource(base.CustomDataSource):

  NAME = "cityscapes"
  DOMAINS = {
    "train": "train",
    "valid": "val",
    "test": "test",
  }

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
      images_dir=images_dir or os.path.join(root_dir, "leftImg8bit"),
      masks_dir=masks_dir or os.path.join(root_dir, "gtFine"),
      sample_ids=sample_ids,
    )
    self.root_dir = root_dir
    self.sample_labels = self.get_sample_labels(self.domain)

    kind = "gtCoarse" if "extra" in self.domain else "gtFine"
    self._image_ext = '_leftImg8bit.png'
    self._mask_ext = f'_{kind}_labelTrainIds.png'
  
  def get_sample_ids(self, domain) -> List[str]:
    with open(self.get_sample_ids_path(domain)) as f:
      return [sid.strip().split(",")[0] for sid in f.readlines()]

  def get_sample_labels(self, domain):
    with open(self.get_sample_ids_path(domain)) as f:
      ids_and_labels = [sid.strip().split(",") for sid in f.readlines()]
    
    return {k: onehot_encode(list(map(int, v.split("|")))) for k, v in ids_and_labels}

  def get_label(self, sample_id) -> np.ndarray:
    label = self.sample_labels[sample_id]
    return label

  def get_image_path(self, sample_id) -> str:
    return os.path.join(self.images_dir, self.domain, sample_id + self._image_ext)
  
  def get_mask_path(self, sample_id) -> str:
    return os.path.join(self.masks_dir, self.domain, sample_id + self._mask_ext)

  def get_info(self, task: str) -> base.DatasetInfo:
    if task == "segmentation":
      num_classes = 19
      classes = CLASSES
      colors = COLORS
      void_class = 255
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
