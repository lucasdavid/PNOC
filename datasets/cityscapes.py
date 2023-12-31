import os
from typing import List, Optional

from PIL import Image
import numpy as np

from . import base

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cityscapes")

TRAIN_CLASSES = (
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
)
TRAIN_COLORS = (
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
)

BG_CLASSES = (
  "road",
  "sidewalk",
  "building",
  "pole",
  "vegetation",
  "sky",
)
FG_INDICES = tuple((i for i, c in enumerate(TRAIN_CLASSES) if c not in BG_CLASSES))
BG_INDICES = tuple((i for i, c in enumerate(TRAIN_CLASSES) if c in BG_CLASSES))

CLSF_MAP = dict(map(reversed, enumerate(FG_INDICES)))   # w/o bg indices ({3: 0, 4: 1, 6: 2, 7: 3, ...})
SEGM_MAP = -1 * np.ones(256, dtype="int32")  # Map bg indices to 0 and shift remaining ones.
SEGM_MAP[255] = 255
for i in BG_INDICES: SEGM_MAP[i] = 0
for k, v in CLSF_MAP.items(): SEGM_MAP[k] = v + 1       # +1 accounts for the BG class.

CLASSES = ["background"] + [c for i, c in enumerate(TRAIN_CLASSES) if i in FG_INDICES]
COLORS  = [[0, 0, 0]]    + [c for i, c in enumerate(TRAIN_COLORS)  if i in FG_INDICES]


def _onehot(indices, n=19):
  target = np.zeros(n, dtype=np.float32)
  target[indices] = 1

  return target

def _decode_classification(indices):
  return [CLSF_MAP[i] for i in indices if i in FG_INDICES]

def _decode_segmentation(indices):
  return SEGM_MAP[indices]


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
      ids_and_labels = (line.strip().split(",") for line in f.readlines())
      ids_and_labels = ((_id, list(map(int, indices.split("|")))) for _id, indices in ids_and_labels)

    n = len(CLSF_MAP)

    return {_id: _onehot(_decode_classification(indices), n) for _id, indices in ids_and_labels}

  def get_label(self, sample_id) -> np.ndarray:
    return self.sample_labels[sample_id]

  def get_mask(self, sample_id) -> np.ndarray:
    with Image.open(self.get_mask_path(sample_id)).convert("L") as mask:
      mask = np.array(mask)
      mask = _decode_segmentation(mask)
    return Image.fromarray(mask)

  def get_image_path(self, sample_id) -> str:
    return os.path.join(self.images_dir, self.domain, sample_id + self._image_ext)

  def get_mask_path(self, sample_id) -> str:
    return os.path.join(self.masks_dir, self.domain, sample_id + self._mask_ext)

  def get_info(self, task: str) -> base.DatasetInfo:
    if task == "segmentation":
      num_classes = len(CLASSES)
      classes = CLASSES
      colors = COLORS
      bg_class = 0
      void_class = 15
    else:
      num_classes = len(CLASSES) - 1
      classes = CLASSES[1:]
      colors = COLORS[1:]
      bg_class = None
      void_class = None

    return base.DatasetInfo(
      num_classes=num_classes,
      classes=classes,
      colors=colors,
      bg_class=bg_class,
      void_class=void_class,
    )


base.DATASOURCES[CityscapesDataSource.NAME] = CityscapesDataSource
