import os.path
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset

from core.aff_utils import *

IMG_FOLDER_NAME = "JPEGImages"
ANNOT_FOLDER_NAME = "Annotations"
IGNORE = 255


def decode_int_filename(int_filename):
  s = str(int_filename).split('\n')[0]
  if len(s) != 12:
    s = '%012d' % int(s)
  return s


def load_image_label_list_from_npy(img_name_list):
  cls_labels_dict = np.load('data/coco14/cls_labels_coco.npy', allow_pickle=True).item()
  return np.array([cls_labels_dict[int(img_name)] for img_name in img_name_list])


def load_img_name_list(domain):
  return np.loadtxt(f"./data/coco14/{domain}.txt", dtype=np.int32)[::-1]


class COCO14Dataset(Dataset):

  def __init__(self, root_dir, domain, transform=None):
    self.root_dir = root_dir
    self.domain = domain
    self.img_name_list = load_img_name_list(domain)
    self.transform = transform

  def __len__(self):
    return len(self.img_name_list)

  def __getitem__(self, idx):
    filename = decode_int_filename(self.img_name_list[idx])
    filename = f"COCO_{self.domain}_{filename}.jpg"
    filepath = os.path.join(self.root_dir, IMG_FOLDER_NAME, filename)
    
    image = Image.open(filepath).convert('RGB')

    if self.transform:
      image = self.transform(image)

    return filename, image


class COCO14ClassificationDataset(COCO14Dataset):

  def __init__(self, root_dir, domain, transform=None):
    super().__init__(root_dir, domain, transform)
    self.label_list = load_image_label_list_from_npy(self.img_name_list)

  def __getitem__(self, idx):
    _, image = super().__getitem__(idx)
    label = torch.from_numpy(self.label_list[idx])

    return image, label


class COCO14SegmentationDataset(COCO14Dataset):

  def __init__(self, root_dir, domain, label_dir, transform=None):
    super().__init__(root_dir, domain, transform=transform)

    self.label_dir = label_dir

  def __getitem__(self, idx):
    name = decode_int_filename(self.img_name_list[idx])
    filename = f"COCO_{self.domain}_{name}.jpg"
    filepath = os.path.join(self.root_dir, IMG_FOLDER_NAME, filename)
    maskpath = os.path.join(self.label_dir, name + '.png')

    image = Image.open(filepath).convert('RGB')
    if os.path.isfile(maskpath):
      label = Image.open(maskpath)
    else:
      label = None

    if self.transform:
      entry = self.transform({'image': image, 'mask': label})
      image, label = entry['image'], entry['mask']

    return image, label


class COCO14AffinityDataset(COCO14SegmentationDataset):

  def __init__(
    self,
    root_dir,
    domain,
    label_dir,
    indices_from,
    indices_to,
    transform=None,
  ):
    super().__init__(root_dir, domain, label_dir, transform=transform)
    self.extract_aff_lab_func = GetAffinityLabelFromIndices(indices_from, indices_to, classes=81)

  def __getitem__(self, idx):
    image, label = super().__getitem__(idx)

    return image, self.extract_aff_lab_func(label)
