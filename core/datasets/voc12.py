import os

import imageio
import numpy as np
import torch
from PIL import Image

from core.aff_utils import *
from tools.ai.augment_utils import *
from tools.ai.torch_utils import one_hot_embedding
from tools.dataset.voc_utils import get_color_map_dic
from tools.general.json_utils import read_json
from tools.general.xml_utils import read_xml

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'voc12')


class VOC12Dataset(torch.utils.data.Dataset):

  def __init__(self, root_dir, domain, with_id=False, with_tags=False, with_mask=False):
    self.root_dir = root_dir
    self.image_dir = os.path.join(self.root_dir, 'JPEGImages/')
    self.xml_dir = os.path.join(self.root_dir, 'Annotations/')
    self.mask_dir = os.path.join(self.root_dir, 'SegmentationClass/')

    filepath = os.path.join(DATA_DIR, f"{domain}.txt")
    self.image_id_list = [image_id.strip() for image_id in open(filepath).readlines()]

    data = read_json(os.path.join(DATA_DIR, "meta.json"))
    self.class_dic = data['class_dic']
    self.classes = data['classes']

    self.with_id = with_id
    self.with_tags = with_tags
    self.with_mask = with_mask

  def __len__(self):
    return len(self.image_id_list)

  def get_image(self, image_id):
    image = Image.open(self.image_dir + image_id + '.jpg').convert('RGB')
    return image

  def get_mask(self, image_id):
    mask_path = self.mask_dir + image_id + '.png'
    if os.path.isfile(mask_path):
      mask = Image.open(mask_path)
    else:
      mask = None
    return mask

  def get_tags(self, image_id):
    _, tags = read_xml(self.xml_dir + image_id + '.xml')
    return tags

  def __getitem__(self, index):
    image_id = self.image_id_list[index]

    data_list = [self.get_image(image_id)]

    if self.with_id:
      data_list.append(image_id)

    if self.with_tags:
      data_list.append(self.get_tags(image_id))

    if self.with_mask:
      data_list.append(self.get_mask(image_id))

    return data_list


class VOC12ClassificationDataset(VOC12Dataset):

  def __init__(self, root_dir, domain, transform=None):
    super().__init__(root_dir, domain, with_tags=True)
    self.transform = transform

  def __getitem__(self, index):
    image, tags = super().__getitem__(index)

    if self.transform is not None:
      image = self.transform(image)

    label = one_hot_embedding([self.class_dic[tag] for tag in tags], self.classes)
    return image, label


class VOC12SegmentationDataset(VOC12Dataset):

  def __init__(self, root_dir, domain, transform=None):
    super().__init__(root_dir, domain, with_mask=True)
    self.transform = transform

    cmap_dic, _, class_names = get_color_map_dic()
    self.colors = np.asarray([cmap_dic[class_name] for class_name in class_names])

  def __getitem__(self, index):
    image, mask = super().__getitem__(index)

    if self.transform is not None:
      input_dic = {'image': image, 'mask': mask}
      output_dic = self.transform(input_dic)

      image = output_dic['image']
      mask = output_dic['mask']

    return image, mask


class VOC12EvaluationDataset(VOC12Dataset):

  def __init__(self, root_dir, domain, transform=None):
    super().__init__(root_dir, domain, with_id=True, with_mask=True)
    self.transform = transform

    cmap_dic, _, class_names = get_color_map_dic()
    self.colors = np.asarray([cmap_dic[class_name] for class_name in class_names])

  def __getitem__(self, index):
    image, image_id, mask = super().__getitem__(index)

    if self.transform is not None:
      input_dic = {'image': image, 'mask': mask}
      output_dic = self.transform(input_dic)

      image = output_dic['image']
      mask = output_dic['mask']

    return image, image_id, mask


class VOC_Dataset_For_WSSS(VOC12Dataset):

  def __init__(self, root_dir, domain, pred_dir, transform=None):
    super().__init__(root_dir, domain, with_id=True)
    self.pred_dir = pred_dir
    self.transform = transform

    cmap_dic, _, class_names = get_color_map_dic()
    self.colors = np.asarray([cmap_dic[class_name] for class_name in class_names])

  def __getitem__(self, index):
    image, image_id = super().__getitem__(index)
    mask = Image.open(self.pred_dir + image_id + '.png')

    if self.transform is not None:
      input_dic = {'image': image, 'mask': mask}
      output_dic = self.transform(input_dic)

      image = output_dic['image']
      mask = output_dic['mask']

    return image, mask


class VOC12CAMEvaluationDataset(VOC12Dataset):

  def __init__(self, root_dir, domain, transform=None):
    super().__init__(root_dir, domain, with_tags=True, with_mask=True)
    self.transform = transform

    cmap_dic, _, class_names = get_color_map_dic()
    self.colors = np.asarray([cmap_dic[class_name] for class_name in class_names])

  def __getitem__(self, index):
    image, tags, mask = super().__getitem__(index)

    if mask.mode == "RGB":
      # Correction for SBD dataset.
      mask = np.asarray(mask)
      mask = (mask[..., np.newaxis] == self.colors[:, ::-1].T).all(axis=-2).argmax(-1)
      mask[mask == 21] = 255
      mask = Image.fromarray(mask.astype('uint8'))

    if self.transform is not None:
      input_dic = {'image': image, 'mask': mask}
      output_dic = self.transform(input_dic)

      image = output_dic['image']
      mask = output_dic['mask']

    label = one_hot_embedding([self.class_dic[tag] for tag in tags], self.classes)
    return image, label, mask


class VOC12InferenceDataset(VOC12Dataset):

  def __init__(self, root_dir, domain, transform=None):
    super().__init__(root_dir, domain, with_id=True, with_tags=True)
    self.transform = transform

  def __getitem__(self, index):
    image, image_id, tags = super().__getitem__(index)

    if self.transform is not None:
      image = self.transform(image)

    label = one_hot_embedding([self.class_dic[tag] for tag in tags], self.classes)
    return image, image_id, label

  def __getitem__(self, index):
    image, image_id, tags = super().__getitem__(index)

    if self.transform is not None:
      image = self.transform(image)

    label = one_hot_embedding([self.class_dic[tag] for tag in tags], self.classes)
    return image, image_id, label


class VOC12AffinityDataset(VOC12Dataset):

  def __init__(self, root_dir, domain, path_index, label_dir, transform=None):
    super().__init__(root_dir, domain, with_id=True)

    self.transform = transform

    self.label_dir = label_dir
    self.path_index = path_index

    self.extract_aff_lab_func = GetAffinityLabelFromIndices(
      self.path_index.src_indices, self.path_index.dst_indices, classes=21
    )

  def __getitem__(self, idx):
    image, image_id = super().__getitem__(idx)

    label = imageio.imread(self.label_dir + image_id + '.png')
    label = Image.fromarray(label)

    entry = self.transform({'image': image, 'mask': label})
    image, label = entry['image'], entry['mask']

    return image, self.extract_aff_lab_func(label)