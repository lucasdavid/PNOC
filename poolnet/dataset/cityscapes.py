import os
from typing import List, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cityscapes")


class CityscapesDataset(Dataset):

  def __init__(self, root_dir, domain, masks_dir=None, transform=None, sample_ids=None):
    self.root_dir = root_dir
    self.domain = domain
    self.transform = transform

    self.images_dir = os.path.join(root_dir, "leftImg8bit", domain)
    self.masks_dir = masks_dir or os.path.join(root_dir, "gtFine", domain)
    self.sample_ids = sample_ids or self.get_sample_ids(domain)

    self._image_ext = '_leftImg8bit.png'
    self._mask_ext = f'.png'

  def get_sample_ids_path(self, domain) -> str:
    return os.path.join(DATA_DIR, f"{domain}.txt")

  def get_sample_ids(self, domain) -> List[str]:
    with open(self.get_sample_ids_path(domain)) as f:
      return [sid.strip().split(",")[0] for sid in f.readlines()]

  def __len__(self):
    return len(self.sample_ids)

  def get_image_path(self, sample_id) -> str:
    city = sample_id.split("_")[0]  # images are organized in city-labeled sub-folders
    return os.path.join(self.images_dir, city, sample_id + self._image_ext)

  def get_mask_path(self, sample_id) -> str:
    return os.path.join(self.masks_dir, sample_id + self._mask_ext)


class PathsDataset(CityscapesDataset):

  def __init__(self, root_dir, domain, masks_dir=None):
    super().__init__(root_dir, domain)

    self.masks_dir = masks_dir or os.path.join(self.root_dir, MASKS_DIR)

  def __getitem__(self, idx):
    image_id = self.sample_ids[idx]
    image_path = self.get_image_path(image_id)
    mask_path = self.get_mask_path(image_id)

    return image_id, image_path, mask_path
