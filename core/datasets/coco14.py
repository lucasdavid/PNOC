import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from core.aff_utils import *

IMAGES_DIR = "JPEGImages"
MASKS_DIR = "coco_seg_anno"
IGNORE = 255

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'coco14')


def _decode_id(sample_id):
  s = str(sample_id).split('\n')[0]
  if len(s) != 12:
    s = '%012d' % int(s)
  return s


def _load_labels_from_npy(images):
  filepath = os.path.join(DATA_DIR, 'cls_labels_coco.npy')
  cls_labels_dict = np.load(filepath, allow_pickle=True).item()
  return np.array([cls_labels_dict[int(img_name)] for img_name in images])


def _load_images_list(domain):
  filepath = os.path.join(DATA_DIR, f"{domain}.txt")
  return np.loadtxt(filepath, dtype=np.int32)[::-1]


def load_sample_with_labels(root_dir, domain, images, labels, idx):
  label = labels[idx]
  
  if label.sum() == 0:
    return load_sample_with_labels(root_dir, domain, images, labels, idx+1)

  image_id = _decode_id(images[idx])
  filename = f"COCO_{domain}_{image_id}.jpg"
  filepath = os.path.join(root_dir, IMAGES_DIR, filename)

  image = Image.open(filepath).convert('RGB')

  return image_id, image, label


class COCO14Dataset(Dataset):

  def __init__(self, root_dir, domain, transform=None):
    self.root_dir = root_dir
    self.domain = domain
    self.img_name_list = _load_images_list(domain)
    self.transform = transform

    self.label_list = _load_labels_from_npy(self.img_name_list)

  def __len__(self):
    return len(self.img_name_list)

  def __getitem__(self, idx):
    return load_sample_with_labels(
      self.root_dir,
      self.domain,
      self.img_name_list,
      self.label_list,
      idx
    )


class COCO14ClassificationDataset(COCO14Dataset):

  def __getitem__(self, idx):
    _, image, label = super().__getitem__(idx)

    if self.transform:
      image = self.transform(image)

    return image, label


class COCO14InferenceDataset(COCO14Dataset):

  def __getitem__(self, idx):
    image_id, image, label = super().__getitem__(idx)

    if self.transform:
      image = self.transform(image)

    return image, image_id, label


class COCO14CAMEvaluationDataset(COCO14Dataset):

  def __getitem__(self, idx):
    image_id, image, label = super().__getitem__(idx)

    maskpath = os.path.join(self.root_dir, MASKS_DIR, image_id + '.png')
    mask = Image.open(maskpath) if os.path.isfile(maskpath) else None

    if self.transform:
      entry = self.transform({'image': image, 'mask': mask})
      image, mask = entry['image'], entry['mask']

    return image, label, mask


class COCO14SegmentationDataset(COCO14Dataset):

  def __getitem__(self, idx):
    _, image, _, mask = super().__getitem__(idx)

    if self.transform:
      entry = self.transform({'image': image, 'mask': mask})
      image, mask = entry['image'], entry['mask']

    return image, mask


class COCO14AffinityDataset(COCO14SegmentationDataset):

  def __init__(
    self,
    root_dir,
    domain,
    indices_from,
    indices_to,
    transform=None,
  ):
    super().__init__(root_dir, domain, transform=transform)
    self.extract_aff_lab_func = GetAffinityLabelFromIndices(indices_from, indices_to, classes=81)

  def __getitem__(self, idx):
    image, mask = super().__getitem__(idx)

    return image, self.extract_aff_lab_func(mask)
