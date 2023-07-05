import os
from typing import List, Optional

import numpy as np

from core.aff_utils import *
from tools.ai.augment_utils import *
from tools.ai.torch_utils import one_hot_embedding
from tools.general.json_utils import read_json
from tools.general.xml_utils import read_xml

from . import base

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'voc12')
CLASSES = [
  'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
  'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', 'void'
]
COLORS = [
  [0, 0, 0], [0, 0, 128], [0, 128, 0], [0, 128, 128], [128, 0, 0], [128, 0, 128], [128, 128, 0], [128, 128, 128],
  [0, 0, 64], [0, 0, 192], [0, 128, 64], [0, 128, 192], [128, 0, 64], [128, 0, 192], [128, 128, 64],
  [128, 128, 192], [0, 64, 0], [0, 64, 128], [0, 192, 0], [0, 192, 128], [128, 64, 0], [192, 224, 224]
]

class VOC12DataSource(base.CustomDataSource):

  NAME = "voc12"
  DOMAINS = {
    "train": "train_aug",
    "valid": "train",
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

    data = read_json(os.path.join(DATA_DIR, "meta.json"))
    self.class_dic = data['class_dic']
    self.classes = np.asarray(CLASSES)
    self.colors = np.asarray(COLORS)
    # Reversed access ([2, 1, 0]) because it was stored as BGR:
    self.color_ids = self.colors[:, 2] * 256**2 + self.colors[:, 1] * 256 + self.colors[:, 0]

  def get_label(self, sample_id) -> np.ndarray:
    _, tags = read_xml(self.xml_dir + sample_id + '.xml')
    label = one_hot_embedding([self.class_dic[tag] for tag in tags], 20)

    return label


base.DATASOURCES[VOC12DataSource.NAME] = VOC12DataSource
