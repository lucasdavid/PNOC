import os
from typing import List, Optional

import numpy as np

from tools.ai.torch_utils import one_hot_embedding
from tools.general.xml_utils import read_xml

from . import base

CLASSES = [
  'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
  'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', 'void'
]
COLORS = [
  [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
  [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128],
  [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128], [224, 224, 192],
]


class VOC12DataSource(base.CustomDataSource):

  NAME = "voc12"
  DOMAINS = {
    "train": "train_aug",
    "valid": "val",
  }

  def __init__(
    self,
    root_dir,
    domain: str,
    split: Optional[str] = None,
    images_dir=None,
    masks_dir: str = None,
    xml_dir: str = None,
    sample_ids: List[str] = None,
  ):
    super().__init__(
      domain=domain,
      split=split,
      images_dir=images_dir or os.path.join(root_dir, "JPEGImages"),
      masks_dir=masks_dir or os.path.join(root_dir, "SegmentationClass"),
      sample_ids=sample_ids,
    )
    self.xml_dir = xml_dir or os.path.join(root_dir, 'Annotations/')
    self.root_dir = root_dir

  def get_label(self, sample_id, task: Optional[str] = None) -> np.ndarray:
    info = self.classification_info

    xml_file = self.xml_dir + sample_id + '.xml'
    if not os.path.exists(xml_file):
      return None

    _, tags = read_xml(xml_file)
    label = [info.class_ids[tag] for tag in tags]
    label = one_hot_embedding(label, len(info.classes))

    return label

  def get_info(self, task: str) -> base.DatasetInfo:
    if task == "segmentation":
      classes = CLASSES
      colors = COLORS
      bg_class = 0
      void_class = 21
    else:
      # without bg and void:
      classes = CLASSES[1:-1]
      colors = COLORS[1:-1]
      bg_class = None
      void_class = None

    return base.DatasetInfo(
      classes=classes,
      colors=colors,
      bg_class=bg_class,
      void_class=void_class,
    )


base.DATASOURCES[VOC12DataSource.NAME] = VOC12DataSource
