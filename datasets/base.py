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

  def __init__(self, meta, classes, num_classes):
    self.meta = meta
    self.classes = classes
    self.num_classes = num_classes

  @classmethod
  def from_metafile(cls, dataset):
    META = read_json(os.path.join(DATA_DIR, dataset, 'meta.json'))
    CLASSES = np.asarray(META['class_names'])
    NUM_CLASSES = META['classes']

    return cls(META, CLASSES, NUM_CLASSES)


class CustomDataSource(metaclass=ABCMeta):

  NAME = None
  DOMAINS = {
    "train": "train",
    "valid": "valid",
    "test": "test",
  }

  def __init__(
    self,
    images_dir,
    domain: str,
    split: Optional[str] = None,
    masks_dir: Optional[str] = None,
    sample_ids: Optional[Union[str, List[str]]] = None,
  ):
    domain = domain or split and self.DOMAINS.get(split, None)

    self.images_dir = images_dir
    self.masks_dir = masks_dir
    self.domain = domain
    self.split = split
    self.sample_ids = np.asarray(
      sample_ids.split(",")
      if isinstance(sample_ids, str)
      else sample_ids
    )

  def __len__(self) -> int:
    return len(self.sample_ids)

  def get_image_path(self, sample_id) -> str:
    return os.path.join(self.images_dir, sample_id + '.jpg')

  def get_mask_path(self, sample_id) -> str:
    return os.path.join(self.masks_dir, sample_id + '.png')

  def get_image(self, sample_id) -> Image.Image:
    return Image.open(self.get_image_path(sample_id)).convert('RGB')

  def get_mask(self, sample_id) -> Image.Image:
    return Image.open(self.get_mask_path(sample_id))
    # if not os.path.isfile(mask_path):
    #   return None

  @abstractmethod
  def get_label(self, sample_id) -> np.ndarray:
    raise NotImplementedError

  _info: DatasetInfo = None

  @property
  def info(self):
    if self._info is None:
      self._info = DatasetInfo.from_metafile(self.NAME)
    return self._info

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

    mask_path = os.path.join(self.masks_dir, f'{sample_id}.npy')
    mask_pack = np.load(mask_path, allow_pickle=True).item()
    cams = torch.from_numpy(mask_pack['hr_cam'].max(0, keepdims=True))

    if self.transform is not None:
      data = self.transform({'image': image, 'mask': cams})
      image, cams = data['image'], data['mask']

    return sample_id, image, label, cams

# endregion
