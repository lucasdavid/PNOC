import os
from typing import List, Optional

import numpy as np
from PIL import Image

from core.aff_utils import *
from tools.ai.augment_utils import *
from tools.ai.torch_utils import one_hot_embedding
from tools.general.json_utils import read_json
from tools.general.xml_utils import read_xml

from . import base

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'voc12')


def color_map(N=256):

  def bitget(byteval, idx):
    return ((byteval & (1 << idx)) != 0)

  cmap = np.zeros((N, 3), dtype=np.uint8)
  for i in range(N):
    r = g = b = 0
    c = i
    for j in range(8):
      r = r | (bitget(c, 0) << 7 - j)
      g = g | (bitget(c, 1) << 7 - j)
      b = b | (bitget(c, 2) << 7 - j)
      c = c >> 3

    cmap[i] = np.array([b, g, r])

  return cmap


def get_color_map_dic():
  labels = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
    'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', 'void'
  ]
  # colors = color_map()
  colors = np.asarray(
    [
      [0, 0, 0], [0, 0, 128], [0, 128, 0], [0, 128, 128], [128, 0, 0], [128, 0, 128], [128, 128, 0], [128, 128, 128],
      [0, 0, 64], [0, 0, 192], [0, 128, 64], [0, 128, 192], [128, 0, 64], [128, 0, 192], [128, 128, 64],
      [128, 128, 192], [0, 64, 0], [0, 64, 128], [0, 192, 0], [0, 192, 128], [128, 64, 0], [192, 224, 224]
    ]
  )

  # n_classes = 21
  n_classes = len(labels)

  h = 20
  w = 500

  color_index_list = [index for index in range(n_classes)]

  cmap_dic = {label: colors[color_index] for label, color_index in zip(labels, range(n_classes))}
  cmap_image = np.empty((h * len(labels), w, 3), dtype=np.uint8)

  for color_index in color_index_list:
    cmap_image[color_index * h:(color_index + 1) * h, :] = colors[color_index]

  return cmap_dic, cmap_image, labels


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
    domain = domain or split and self.DOMAINS.get(split, "train_aug")

    if sample_ids is None:
      with open(os.path.join(DATA_DIR, f"{domain}.txt")) as f:
        sample_ids = [image_id.strip() for image_id in f.readlines()]

    super().__init__(
      domain=domain,
      split=split,
      images_dir=images_dir or os.path.join(root_dir, "JPEGImages"),
      masks_dir=masks_dir or os.path.join(root_dir, "SegmentationClass"),
      sample_ids=sample_ids
    )
    self.xml_dir = xml_dir or os.path.join(root_dir, 'Annotations/')
    self.root_dir = root_dir

    data = read_json(os.path.join(DATA_DIR, "meta.json"))
    self.class_dic = data['class_dic']
    self.classes = data['classes']

    cmap_dic, _, class_names = get_color_map_dic()
    self.colors = np.asarray([cmap_dic[class_name] for class_name in class_names])
    # Reversed, because it was stored as BGR:
    self.color_ids = self.colors[:, 2] * 256**2 + self.colors[:, 1] * 256 + self.colors[:, 0]

  def get_label(self, sample_id) -> np.ndarray:
    _, tags = read_xml(self.xml_dir + sample_id + '.xml')
    label = one_hot_embedding([self.class_dic[tag] for tag in tags], self.classes)

    return label

  def get_mask(self, sample_id):
    mask_path = self.get_mask_path(sample_id)

    if not os.path.isfile(mask_path):
      return None

    mask = Image.open(mask_path)

    if mask.mode == "RGB":  # Correction for SBD dataset.
      with mask:
        mask = np.array(mask)

      mask = mask[..., 0] * 256**2 + mask[..., 1] * 256 + mask[..., 2]
      mask = np.argmax(mask[..., np.newaxis] == self.color_ids, axis=-1)
      mask[mask == 21] = 255
      mask = Image.fromarray(mask.astype('uint8'))

    return mask


base.DATASOURCES[VOC12DataSource.NAME] = VOC12DataSource
