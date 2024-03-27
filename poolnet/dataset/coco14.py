import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

MASKS_DIR = "coco_seg_anno"
IGNORE = 255

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'coco14')


def _decode_id(sample_id):
  s = str(sample_id).split('\n')[0]
  if len(s) != 12:
    s = '%012d' % int(s)
  return s


def _load_labels_from_npy(images, root_dir):
  filepath = os.path.join(root_dir, 'cls_labels_coco.npy')
  cls_labels_dict = np.load(filepath, allow_pickle=True).item()
  return np.array([cls_labels_dict[int(img_name)] for img_name in images])


def _load_images_list(domain):
  filepath = os.path.join(DATA_DIR, f"{domain}.txt")
  return np.loadtxt(filepath, dtype=np.int32)[::-1]


class COCO14Dataset(Dataset):

  IGNORE_BG_IMAGES = False

  def __init__(self, root_dir, domain, transform=None):
    self.root_dir = root_dir
    self.domain = domain
    self.transform = transform
    self.img_name_list = _load_images_list(domain)
    self.label_list = _load_labels_from_npy(self.img_name_list, root_dir)

  def __len__(self):
    return len(self.img_name_list)

  def __getitem__(self, idx):
    return self.load_sample_with_labels(idx)

  def get_image_path(self, image_id):
    return os.path.join(self.root_dir, self.domain, f"COCO_{self.domain}_{image_id}.jpg")

  def load_sample_with_labels(self, idx):
    label = self.label_list[idx]

    if self.IGNORE_BG_IMAGES and label.sum() == 0:
      return self.load_sample_with_labels(idx + 1)

    image_id = _decode_id(self.img_name_list[idx])
    file_path = self.get_image_path(image_id)
    image = Image.open(file_path).convert('RGB')

    return image_id, image, label


class PathsDataset(COCO14Dataset):

  def __init__(self, root_dir, domain, masks_dir=None):
    super().__init__(root_dir, domain)

    self.masks_dir = masks_dir or os.path.join(self.root_dir, MASKS_DIR)

  def __getitem__(self, idx):
    image_id = _decode_id(self.img_name_list[idx])
    image_path = self.get_image_path(image_id)
    mask_path = os.path.join(self.masks_dir, image_id + '.png')

    return image_id, image_path, mask_path
