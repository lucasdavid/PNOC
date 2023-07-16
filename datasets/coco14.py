import os
from typing import List, Optional, Union

import numpy as np

from . import base

CLASSES = [
  "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
  "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
  "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
  "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
  "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
  "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "void"
]

COLORS = [
  [0, 0, 0], [172, 47, 117], [192, 67, 195], [103, 9, 211], [21, 36, 87], [70, 216, 88], [140, 58, 193], [39, 87, 174],
  [88, 81, 165], [25, 77, 72], [9, 148, 115], [208, 197, 79], [175, 192, 82], [99, 216, 177], [29, 147, 147],
  [142, 167, 32], [193, 9, 185], [127, 32, 31], [202, 151, 163], [203, 114, 183], [28, 34, 128], [128, 164, 53],
  [133, 38, 17], [79, 132, 105], [42, 186, 31], [120, 1, 65], [169, 57, 35], [102, 119, 11], [174, 82, 91],
  [128, 142, 99], [53, 140, 121], [170, 84, 203], [68, 6, 196], [47, 127, 131], [204, 100, 180], [78, 143, 148],
  [186, 23, 207], [141, 117, 85], [48, 49, 69], [169, 163, 192], [95, 197, 94], [0, 113, 178], [36, 162, 48],
  [93, 131, 98], [42, 205, 112], [149, 201, 127], [0, 138, 114], [43, 186, 127], [23, 187, 130], [121, 98, 62],
  [163, 222, 123], [195, 82, 174], [148, 209, 50], [155, 14, 41], [58, 193, 36], [10, 86, 43], [104, 11, 2],
  [51, 80, 32], [182, 128, 38], [19, 174, 42], [115, 184, 188], [77, 30, 24], [125, 2, 3], [94, 107, 13],
  [112, 40, 72], [19, 95, 72], [154, 194, 180], [67, 61, 14], [96, 4, 195], [139, 86, 205], [121, 109, 75],
  [184, 16, 152], [157, 149, 110], [25, 208, 188], [121, 118, 117], [189, 83, 161], [104, 160, 121], [70, 213, 31],
  [13, 71, 184], [152, 79, 41], [18, 40, 182], [224, 224, 192],
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

  def __init__(
    self,
    root_dir: str,
    domain: str,
    split: Optional[str] = None,
    images_dir: Optional[str] = None,
    masks_dir: Optional[str] = None,
    sample_ids: Union[str, List[str]] = None,
    task: Optional[str] = "classification",
  ):
    domain = domain or split and self.DOMAINS.get(split, self.DOMAINS[self.DEFAULT_SPLIT])

    super().__init__(
      images_dir=images_dir or os.path.join(root_dir, domain),
      masks_dir=masks_dir or os.path.join(root_dir, "coco_seg_anno"),
      sample_ids=sample_ids,
      domain=domain,
      split=split,
      task=task,
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
    if self.task == "segmentation":
      classes = CLASSES
      colors = COLORS
      bg_class = 0
      void_class = 81
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

  def _load_labels_from_npy(self):
    filepath = os.path.join(self.root_dir, 'cls_labels_coco.npy')
    data = np.load(filepath, allow_pickle=True).item()
    return {_id: data[int(_id)] for _id in self.sample_ids}


base.DATASOURCES[COCO14DataSource.NAME] = COCO14DataSource
