import os
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from core.aff_utils import GetAffinityLabelFromIndices
from tools.general.json_utils import read_json

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
DATASOURCES: Dict[str, "CustomDataSource"] = {}


class DatasetInfo:

  def __init__(self, meta, classes, num_classes: int = None, num_classes_segm: int = None):
    self.meta = meta
    self.classes = np.asarray(classes)
    self.num_classes = num_classes or len(classes)

  @classmethod
  def from_metafile(cls, dataset, segmentation=False):
    META = read_json(os.path.join(DATA_DIR, dataset, 'meta_segm.json' if segmentation else 'meta.json'))
    CLASSES = META['class_names']
    NUM_CLASSES = META['classes']

    return cls(META, CLASSES, NUM_CLASSES)


class CustomDataSource(metaclass=ABCMeta):

  NAME: str = "custom"
  DEFAULT_SPLIT = "train"
  DOMAINS = {
    "train": "train",
    "valid": "valid",
    "test": "test",
  }

  VOID_CLASS: int = None

  def __init__(
    self,
    images_dir,
    domain: str,
    split: Optional[str] = None,
    masks_dir: Optional[str] = None,
    sample_ids: Optional[Union[str, List[str]]] = None,
    segmentation: bool = False,
  ):
    domain = domain or split and self.DOMAINS.get(split, self.DOMAINS[self.DEFAULT_SPLIT])

    if sample_ids is None:
      sample_ids = self.get_sample_ids(domain)

    self.images_dir = images_dir
    self.masks_dir = masks_dir
    self.split = split
    self.domain = domain
    self.sample_ids = np.asarray(
      sample_ids.split(",")
      if isinstance(sample_ids, str)
      else sample_ids
    )
    self.segmentation = segmentation

  _info: DatasetInfo = None

  @property
  def info(self):
    if self._info is None:
      self._info = DatasetInfo.from_metafile(self.NAME, segmentation=self.segmentation)
    return self._info

  def __len__(self) -> int:
    return len(self.sample_ids)

  def get_sample_ids_path(self, domain) -> str:
    return os.path.join(DATA_DIR, self.NAME, f"{domain}.txt")

  def get_image_path(self, sample_id) -> str:
    return os.path.join(self.images_dir, sample_id + '.jpg')

  def get_mask_path(self, sample_id) -> str:
    return os.path.join(self.masks_dir, sample_id + '.png')

  def get_sample_ids(self, domain) -> List[str]:
    with open(self.get_sample_ids_path(domain)) as f:
      return [sid.strip() for sid in f.readlines()]

  def get_image(self, sample_id) -> Image.Image:
    return Image.open(self.get_image_path(sample_id)).convert('RGB')

  def get_mask(self, sample_id) -> Image.Image:
    mask = Image.open(self.get_mask_path(sample_id))

    if mask.mode == "RGB":  # Correction for SBD dataset.
      with mask:
        mask = np.array(mask)

      mask = mask[..., 0] * 256**2 + mask[..., 1] * 256 + mask[..., 2]
      mask = np.argmax(mask[..., np.newaxis] == self.color_ids, axis=-1)
      mask[mask == self.VOID_CLASS] = 255
      mask = Image.fromarray(mask.astype('uint8'))

    return mask

  @abstractmethod
  def get_label(self, sample_id) -> np.ndarray:
    raise NotImplementedError


# region Datasets

class ClassificationDataset(torch.utils.data.Dataset):

  IGNORE_BG_IMAGES: bool = True

  def __init__(self, data_source: CustomDataSource, transform: transforms.Compose = None, ignore_bg_images: bool = None):
    self.data_source = data_source
    self.transform = transform
    self.ignore_bg_images = (
      ignore_bg_images
      if ignore_bg_images is not None
      else self.IGNORE_BG_IMAGES
    )

  @property
  def info(self):
    return self.data_source.info

  def __len__(self) -> int:
    return len(self.data_source)

  def get_valid_sample(self, index):
    sample_id = self.data_source.sample_ids[index]
    label = self.data_source.get_label(sample_id)

    if self.ignore_bg_images and label.sum() == 0:
      return self.get_valid_sample(index+1)

    return sample_id, label

  def __getitem__(self, index):
    sample_id, label = self.get_valid_sample(index)
    image = self.data_source.get_image(sample_id)

    if self.transform is not None:
      image = self.transform(image)

    return sample_id, image, label


class SegmentationDataset(ClassificationDataset):

  IGNORE_BG_IMAGES: bool = False

  def __getitem__(self, index):
    sample_id, label = self.get_valid_sample(index)

    image = self.data_source.get_image(sample_id)
    mask = self.data_source.get_mask(sample_id)

    if self.transform is not None:
      data = self.transform({"image": image, "mask": mask})
      image, mask = data["image"], data["mask"]

    return sample_id, image, label, mask


class PathsDataset(ClassificationDataset):

  IGNORE_BG_IMAGES: bool = False

  def __getitem__(self, index):
    sample_id, _ = self.get_valid_sample(index)

    image_path = self.data_source.get_image_path(sample_id)
    mask_path = self.data_source.get_mask_path(sample_id)

    return sample_id, image_path, mask_path


class AffinityDataset(SegmentationDataset):

  IGNORE_BG_IMAGES: bool = True

  def __init__(self, path_index, **kwargs):
    super().__init__(**kwargs)
    self.path_index = path_index
    self.get_affinities = GetAffinityLabelFromIndices(
      self.path_index.src_indices,
      self.path_index.dst_indices,
      classes=self.info.num_classes + 1,
    )

  def __getitem__(self, idx):
    sample_id, image, label, mask = super().__getitem__(idx)

    return sample_id, image, label, self.get_affinities(mask)


class CAMsDataset(ClassificationDataset):

  IGNORE_BG_IMAGES: bool = True

  def __getitem__(self, index):
    sample_id, label = self.get_valid_sample(index)

    image = self.data_source.get_image(sample_id)

    cams_path = os.path.join(self.masks_dir, f'{sample_id}.npy')
    cams = np.load(cams_path, allow_pickle=True).item()
    cams = torch.from_numpy(cams['hr_cam'].max(0, keepdims=True))

    if self.transform is not None:
      data = self.transform({'image': image, 'mask': cams})
      image, cams = data['image'], data['mask']

    return sample_id, image, label, cams

# endregion
