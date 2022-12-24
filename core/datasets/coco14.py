import os

import imageio
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


class COCO14Dataset(Dataset):

  def __init__(self, root_dir, domain, transform=None):
    self.root_dir = root_dir
    self.domain = domain
    self.transform = transform
    self.img_name_list = _load_images_list(domain)
    self.label_list = _load_labels_from_npy(self.img_name_list)

  def __len__(self):
    return len(self.img_name_list)

  def __getitem__(self, idx):
    return self.load_sample_with_labels(idx)

  def get_image_path(self, image_id):
    return os.path.join(self.root_dir, IMAGES_DIR, f"COCO_{self.domain}_{image_id}.jpg")

  def load_sample_with_labels(self, idx, ignore_bg_only=True):
    label = self.label_list[idx]

    if ignore_bg_only and label.sum() == 0:
      return self.load_sample_with_labels(idx + 1, ignore_bg_only)

    image_id = _decode_id(self.img_name_list[idx])
    file_path = self.get_image_path(image_id)
    image = Image.open(file_path).convert('RGB')

    return image_id, image, label


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

  def __init__(self, root_dir, domain, transform=None, masks_dir=None):
    super().__init__(root_dir, domain, transform)

    self.masks_dir = masks_dir or os.path.join(self.root_dir, MASKS_DIR)

  def __getitem__(self, idx):
    image_id, image, label = super().__getitem__(idx)

    maskpath = os.path.join(self.masks_dir, image_id + '.png')
    mask = Image.open(maskpath) if os.path.isfile(maskpath) else None

    if self.transform:
      entry = self.transform({'image': image, 'mask': mask})
      image, mask = entry['image'], entry['mask']

    return image, label, mask


class COCO14SegmentationDataset(COCO14CAMEvaluationDataset):

  def __getitem__(self, idx):
    image, _, mask = super().__getitem__(idx)

    return image, mask


class COCO14PathsDataset(COCO14Dataset):

  def __init__(self, root_dir, domain, masks_dir=None):
    super().__init__(root_dir, domain)

    self.masks_dir = masks_dir or os.path.join(self.root_dir, MASKS_DIR)

  def __getitem__(self, idx):
    image_id = _decode_id(self.img_name_list[idx])
    image_path = self.get_image_path(image_id)
    mask_path = os.path.join(self.masks_dir, image_id + '.png')

    return image_id, image_path, mask_path


class COCO14AffinityDataset(COCO14Dataset):

  def __init__(self, root_dir, domain, path_index, label_dir, transform=None):
    super().__init__(root_dir, domain, transform=transform)

    self.label_dir = label_dir
    self.path_index = path_index

    self.extract_aff_lab_func = GetAffinityLabelFromIndices(path_index.src_indices, path_index.dst_indices, classes=81)

  def __getitem__(self, idx):
    image_id, image, _ = super().__getitem__(idx)

    label = imageio.imread(os.path.join(self.label_dir, image_id + '.png'))
    label = Image.fromarray(label)

    entry = self.transform({'image': image, 'mask': label})
    image, label = entry['image'], entry['mask']

    return image, self.extract_aff_lab_func(label)


class COCO14HRCAMsDataset(COCO14Dataset):

  def __init__(self, root_dir, domain, cams_dir, resize_fn, normalize_fn, transform):
    super().__init__(root_dir, domain, transform)
    self.cams_dir = cams_dir
    self.resize_fn = resize_fn
    self.normalize_fn = normalize_fn

  def __getitem__(self, index):
    image_id, image, label = super().__getitem__(index)

    mask_path = os.path.join(self.cams_dir, f'{image_id}.npy')
    mask_pack = np.load(mask_path, allow_pickle=True).item()
    cams = torch.from_numpy(mask_pack['hr_cam'].max(0, keepdims=True))

    image = self.resize_fn(image)
    image = self.normalize_fn(image)

    cams = self.resize_fn(cams)

    data = self.transform({'image': image, 'masks': cams})
    image, cams = data['image'], data['masks']

    return image, label, cams
