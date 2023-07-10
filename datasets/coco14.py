import os
from typing import List, Optional, Union

import numpy as np

from . import base

CLASSES = [
  "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
  "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
  "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
  "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
  "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
  "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


def _decode_id(sample_id):
  s = str(sample_id).split('\n')[0]
  if len(s) != 12:
    s = '%012d' % int(s)
  return s


class COCO14DataSource(base.CustomDataSource):

  NAME = "coco14"
  DOMAINS = {
    "train": "train2014",
    "valid": "val2014",
  }

  VOID_CLASS = 81

  def __init__(
    self,
    root_dir: str,
    domain: str,
    split: Optional[str] = None,
    images_dir: Optional[str] = None,
    masks_dir: Optional[str] = None,
    sample_ids: Union[str, List[str]] = None,
    segmentation: bool = False,
  ):
    domain = domain or split and self.DOMAINS.get(split, self.DOMAINS[self.DEFAULT_SPLIT])

    super().__init__(
      images_dir=images_dir or os.path.join(root_dir, domain),
      masks_dir=masks_dir or os.path.join(root_dir, "coco_seg_anno"),
      sample_ids=sample_ids,
      domain=domain,
      split=split,
      segmentation=segmentation,
    )

    self.root_dir = root_dir
    self.sample_labels = self._load_labels_from_npy()

  def get_image_path(self, sample_id) -> str:
    return os.path.join(self.images_dir, f"COCO_{self.domain}_{sample_id}.jpg")

  def get_label(self, sample_id) -> np.ndarray:
    return self.sample_labels[sample_id]

  def get_sample_ids(self, domain) -> List[str]:
    sample_ids = super().get_sample_ids(domain)
    return [_decode_id(_id) for _id in sample_ids if _id]

  def get_info(self):
    if self.segmentation:
      classes = ["background"] + CLASSES
      bg_class = 0
      void_class = 81
    else:
      # without bg and void:
      classes = CLASSES
      bg_class = None
      void_class = None

    return base.DatasetInfo(
      classes=classes,
      colors=None,
      bg_class=bg_class,
      void_class=void_class,
    )

  def _load_labels_from_npy(self):
    filepath = os.path.join(self.root_dir, 'cls_labels_coco.npy')
    data = np.load(filepath, allow_pickle=True).item()
    return {_id: data[int(_id)] for _id in self.sample_ids}


base.DATASOURCES[COCO14DataSource.NAME] = COCO14DataSource
