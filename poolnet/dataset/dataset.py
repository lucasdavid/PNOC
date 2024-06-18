import os
import random
import cv2
from PIL import Image
import numpy as np

import torch
from torch.utils import data
import torchvision.transforms as T

from . import coco14

# ------------------------------------ added by sierkinhane ------------------------------------#


def load_img_id_list(img_id_file):
  with open(img_id_file) as f:
    return f.read().splitlines()


class VOCImageDataTrain(data.Dataset):

  def __init__(self, data_root, pseudo_root):
    self.sal_root = data_root
    self.pseudo_root = pseudo_root
    self.sal_list = load_img_id_list('../data/voc12/train_aug.txt')

  def __getitem__(self, item):
    # sal data loading
    im_name = self.sal_list[item]
    sal_image = load_image(os.path.join(self.sal_root, 'JPEGImages', im_name + '.jpg'))
    sal_label = load_sal_label(os.path.join(self.pseudo_root, im_name + '.png'))
    sal_image, sal_label = cv_random_flip(sal_image, sal_label)

    h, w = sal_image.shape[1:]

    sal_image = torch.Tensor(sal_image)
    sal_label = torch.Tensor(sal_label)

    return {'sal_image': sal_image, 'sal_label': sal_label, 'image_id': im_name, "height": h, "width": w}

  def __len__(self):
    return len(self.sal_list)


class VOCImageDataTest(data.Dataset):

  def __init__(self, data_root, train_list):
    self.sal_root = data_root
    self.train_list = train_list
    self.sal_list = load_img_id_list(f'../data/voc12/{train_list}.txt')

  def __getitem__(self, item):
    # sal data loading
    im_name = self.sal_list[item]
    sal_image = load_image(os.path.join(self.sal_root, 'JPEGImages', im_name + '.jpg'))
    h, w = sal_image.shape[1:]

    sal_image = torch.Tensor(sal_image)

    return {'sal_image': sal_image, 'image_id': im_name, "height": h, "width": w}

  def __len__(self):
    return len(self.sal_list)


class PadIfSmallerMixin:
  min_size = 224

  def pad_if_smaller_than_min_size(self, image, sal=None):
    if self.min_size:
      _, h, w = image.shape
      if h <= self.min_size or w <= self.min_size:
        sizes_pad = (max(h, self.min_size), max(w, self.min_size))
        i_pad = np.zeros((image.shape[0], *sizes_pad), dtype=image.dtype)
        i_pad[:, :h, :w] = image
        image = i_pad

        if sal is not None:
          sal_pad = np.zeros((sal.shape[0], *sizes_pad), dtype=sal.dtype)
          sal_pad[:, :h, :w] = sal
          sal = sal_pad

    return image, sal


class COCOImageDataTrain(PadIfSmallerMixin, coco14.PathsDataset):

  def __init__(self, root_dir, domain, masks_dir=None, min_size=224):
    super().__init__(root_dir, domain, masks_dir=masks_dir)
    self.min_size = min_size

  def __getitem__(self, item):
    image_id, image_path, mask_path = super().__getitem__(item)

    img = load_image(image_path)
    sal = load_sal_label(mask_path)
    img, sal = cv_random_flip(img, sal)
    h, w = img.shape[1:]

    img, sal = self.pad_if_smaller_than_min_size(img, sal)
    img = torch.Tensor(img)
    sal = torch.Tensor(sal)

    return {'sal_image': img, 'sal_label': sal, 'image_id': image_id, "height": h, "width": w}


class COCOImageDataTest(PadIfSmallerMixin, coco14.PathsDataset):
  def __init__(self, root_dir, domain, min_size=224):
    super().__init__(root_dir, domain)
    self.min_size = min_size

  def __getitem__(self, item):
    image_id, image_path, _ = super().__getitem__(item)

    img = load_image(image_path)
    h, w = img.shape[1:]

    img, _ = self.pad_if_smaller_than_min_size(img)
    img = torch.Tensor(img)

    return {'sal_image': img, 'image_id': image_id, "height": h, "width": w}


# ------------------------------------ added by sierkinhane ------------------------------------#


class ImageDataTrain(data.Dataset):

  def __init__(self, data_root, data_list):
    self.sal_root = data_root
    self.sal_source = data_list

    with open(self.sal_source, 'r') as f:
      self.sal_list = [x.strip() for x in f.readlines()]

    self.sal_num = len(self.sal_list)

  def __getitem__(self, item):
    # sal data loading
    im_name = self.sal_list[item % self.sal_num].split()[0]
    gt_name = self.sal_list[item % self.sal_num].split()[1]
    sal_image = load_image(os.path.join(self.sal_root, im_name))
    sal_label = load_sal_label(os.path.join(self.sal_root, gt_name))
    sal_image, sal_label = cv_random_flip(sal_image, sal_label)
    sal_image = torch.Tensor(sal_image)
    sal_label = torch.Tensor(sal_label)

    sample = {'sal_image': sal_image, 'sal_label': sal_label}
    return sample

  def __len__(self):
    return self.sal_num


class ImageDataTest(data.Dataset):

  def __init__(self, data_root, data_list):
    self.data_root = data_root
    self.data_list = data_list
    with open(self.data_list, 'r') as f:
      self.image_list = [x.strip() for x in f.readlines()]

    self.image_num = len(self.image_list)

  def __getitem__(self, item):
    image, im_size = load_image_test(os.path.join(self.data_root, self.image_list[item]))
    image = torch.Tensor(image)

    return {'image': image, 'name': self.image_list[item % self.image_num], 'size': im_size}

  def __len__(self):
    return self.image_num


def get_loader(config, mode='train', pin=False):
  shuffle = False
  if mode == 'train':
    shuffle = True
    dataset = ImageDataTrain(config.train_root, config.train_list)
    data_loader = data.DataLoader(
      dataset=dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=config.num_thread, pin_memory=pin
    )
  else:
    dataset = ImageDataTest(config.test_root, config.test_list)
    data_loader = data.DataLoader(
      dataset=dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=config.num_thread, pin_memory=pin
    )
  return data_loader


# added by sierkinhane
def get_custom_loader(config, mode='train', pin=False):
  if config.dataset == "voc12":
    dataset = (
      VOCImageDataTrain(config.train_root, config.pseudo_root)
      if mode == 'train'
      else VOCImageDataTest(config.test_root, config.train_list)
    )
  elif config.dataset == "coco14":
    dataset = (
      COCOImageDataTrain(config.train_root, config.train_list, config.pseudo_root)
      if mode == 'train'
      else COCOImageDataTest(config.test_root, config.train_list)
    )
  elif config.dataset == "cityscapes":
    dataset = (
      CityscapesImageDataTrain(config.train_root, config.train_list, config.pseudo_root)
      if mode == 'train'
      else CityscapesImageDataTest(config.test_root, config.train_list)
    )
  else:
    raise ValueError(f'Unknown dataset {config.dataset}. Should either be "voc12" or "coco14".')

  loader = data.DataLoader(
    dataset=dataset,
    batch_size=config.batch_size,
    shuffle=mode == "train",
    num_workers=config.num_thread,
    pin_memory=pin,
  )

  return loader


def load_image(path):
  if not os.path.exists(path):
    print('File {} not exists'.format(path))
  im = cv2.imread(path)
  # print(path)
  # print(im.shape)
  in_ = np.array(im, dtype=np.float32)
  in_ -= np.array((104.00699, 116.66877, 122.67892))
  in_ = in_.transpose((2, 0, 1))
  return in_


def load_image_test(path):
  if not os.path.exists(path):
    print('File {} not exists'.format(path))
  im = cv2.imread(path)
  in_ = np.array(im, dtype=np.float32)
  im_size = tuple(in_.shape[:2])
  in_ -= np.array((104.00699, 116.66877, 122.67892))
  in_ = in_.transpose((2, 0, 1))
  return in_, im_size


def load_sal_label(path):
  if not os.path.exists(path):
    print('File {} not exists'.format(path))
  im = Image.open(path)
  label = np.array(im, dtype=np.float32)
  if len(label.shape) == 3:
    label = label[:, :, 0]
  label = label / 255.
  label = label[np.newaxis, ...]
  return label


def cv_random_flip(img, label):
  flip_flag = random.randint(0, 1)
  if flip_flag == 1:
    img = img[:, :, ::-1].copy()
    label = label[:, :, ::-1].copy()
  return img, label
