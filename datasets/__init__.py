from typing import Tuple

from torchvision import transforms

from tools.ai.augment_utils import *
from tools.ai.randaugment import RandAugmentMC

from .base import *
from . import base, voc12, coco14, deepglobe, cityscapes, hpa


class Iterator:

  def __init__(self, loader):
    self.loader = loader
    self.init()

  def init(self):
    self.iterator = iter(self.loader)

  def get(self):
    try:
      data = next(self.iterator)
    except StopIteration:
      self.init()
      data = next(self.iterator)

    return data


def custom_data_source(
  dataset: str, data_dir: str, domain: Optional[str] = None, split: Optional[str] = "train", **kwargs
) -> CustomDataSource:
  data_source_cls = base.DATASOURCES[dataset]
  return data_source_cls(
    root_dir=data_dir,
    domain=domain,
    split=split,
    **kwargs,
  )


def apply_augmentation(
  dataset: ClassificationDataset, augment: str, image_size, cutmix_prob, mixup_prob
) -> ClassificationDataset:
  if 'cutormixup' in augment:
    print(f'Applying cutormixup image_size={image_size}, num_mix=1, beta=1., prob={cutmix_prob}')
    dataset = CutOrMixUp(dataset, image_size, num_mix=1, beta=1., prob=cutmix_prob)
  else:
    if 'cutmix' in augment:
      print(f'Applying cutmix image_size={image_size}, num_mix=1, beta=1., prob={cutmix_prob}')
      dataset = CutMix(dataset, image_size, num_mix=1, beta=1., prob=cutmix_prob)
    if 'mixup' in augment:
      print(f'Applying mixup num_mix=1, beta=1., prob={mixup_prob}')
      dataset = MixUp(dataset, num_mix=1, beta=1., prob=mixup_prob)

  return dataset


def imagenet_stats():
  return (
    [0.485, 0.456, 0.406],
    [0.229, 0.224, 0.225],
  )


def get_classification_transforms(
  min_size,
  max_size,
  crop_size,
  augment,
  normalize_stats = None,
):
  if normalize_stats is None:
    normalize_stats = imagenet_stats()

  tt = []
  tv = []
  if min_size == max_size:
    tt += [transforms.Resize((min_size, min_size))]
    tv += [Resize_For_Segmentation((min_size, min_size))]  # image=(1024,2048) min_size=768 --> (768,768)
  else:
    tt += [RandomResize(min_size, max_size)]
    tv += [Resize_For_Segmentation(crop_size)]  # image=(500,480) crop_size=512 --> (534,512)
  tt += [RandomHorizontalFlip()]
  if "clahe" in augment:
    tt += [CLAHE()]
    tv += [CLAHE()]
  if 'colorjitter' in augment:
    tt += [transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)]
  if 'randaugment' in augment:
    tt += [RandAugmentMC(n=2, m=10)]
  tt += [Normalize(*normalize_stats)]
  if 'cutmix' not in augment:
    tt += [RandomCrop(crop_size)]  # This will happen inside CutMix.
  tt += [Transpose()]

  tv += [
    Normalize_For_Segmentation(*normalize_stats),
    Top_Left_Crop_For_Segmentation(crop_size),  # image=(534,512) crop_size=512 --> (512,512)
    Transpose_For_Segmentation()
  ]

  return tuple(map(transforms.Compose, (tt, tv)))


def get_affinity_transforms(
  min_image_size,
  max_image_size,
  crop_size,
  overcrop: bool = True,
  normalize_stats = None,
):
  if normalize_stats is None:
    normalize_stats = imagenet_stats()

  tt = transforms.Compose(
    [
      RandomResize_For_Segmentation(min_image_size, max_image_size, overcrop=overcrop),
      RandomHorizontalFlip_For_Segmentation(),
      Normalize_For_Segmentation(*normalize_stats),
      RandomCrop_For_Segmentation(crop_size),
      Transpose_For_Segmentation(),
      ResizeMask(crop_size // 4),
    ]
  )

  return tt


def get_segmentation_transforms(
  min_size,
  max_size,
  crop_size,
  augment,
  overcrop: bool = True,
  normalize_stats = None,
) -> Tuple[transforms.Compose]:
  if normalize_stats is None:
    normalize_stats = imagenet_stats()

  tt = [
    RandomResize_For_Segmentation(min_size, max_size, overcrop=overcrop),
    RandomHorizontalFlip_For_Segmentation(),
  ]
  tv = [
    Resize_For_Segmentation(crop_size),
  ]

  if 'colorjitter' in augment:
    tt += [ApplyToImage(transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1))]

  if "clahe" in augment:
    tt += [CLAHE()]
    tv += [CLAHE()]

  tt += [
    Normalize_For_Segmentation(*normalize_stats),
    RandomCrop_For_Segmentation(crop_size),
    Transpose_For_Segmentation(),
  ]

  tv += [
    Normalize_For_Segmentation(*normalize_stats),
    Top_Left_Crop_For_Segmentation(crop_size),
    Transpose_For_Segmentation(),
  ]

  return tuple(map(transforms.Compose, (tt, tv)))


def get_ccam_transforms(
  image_size,
  crop_size,
  normalize_stats = None,
):
  if normalize_stats is None:
    normalize_stats = imagenet_stats()

  size = [image_size, image_size]
  resize = Resize_For_Segmentation(
    size,
    resize_y=transforms.Resize(size)  # CAMs are continuous maps. Bilinear interp. Ok.
  )

  tt = transforms.Compose(
    [
      resize,
      Normalize_For_Segmentation(*normalize_stats, mdtype=np.float32),
      RandomCrop_For_Segmentation(crop_size, ignore_value=0., labels_last=False),
      Transpose_For_Segmentation(),
      random_hflip_fn,
    ]
  )

  tv = transforms.Compose([
    resize,
    Normalize_For_Segmentation(*normalize_stats),
    Transpose_For_Segmentation(),
  ])

  return tt, tv

