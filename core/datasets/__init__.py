import os

from torchvision import transforms

from tools.ai.augment_utils import *
from tools.ai.randaugment import RandAugmentMC
from tools.general.json_utils import read_json

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')


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


def get_classification_datasets(
  dataset,
  data_dir,
  augment,
  image_size,
  cutmix_prob=1.,
  mixup_prob=1.,
  train_transforms=None,
  valid_transforms=None,
):
  print(f'Loading {dataset} Classification Dataset')

  if dataset == 'voc12':
    from . import voc12
    train = voc12.ClassificationDataset(data_dir, 'train_aug', train_transforms)
    valid = voc12.CAMEvaluationDataset(data_dir, 'train', valid_transforms)
  else:
    from . import coco14
    train = coco14.ClassificationDataset(data_dir, 'train2014', train_transforms)
    valid = coco14.CAMEvaluationDataset(data_dir, 'train2014', valid_transforms)

  train = _apply_augmentation_strategies(train, augment, image_size, cutmix_prob, mixup_prob)

  info = DatasetInfo.from_metafile(dataset)
  train.info = info
  valid.info = info

  return train, valid


def get_affinity_datasets(
  dataset,
  data_dir,
  label_dir,
  path_index,
  train_transforms=None,
):
  print(f'Loading {dataset} Affinity Dataset')

  if dataset == 'voc12':
    from . import voc12
    train = voc12.AffinityDataset(data_dir, 'train_aug', path_index, label_dir, train_transforms)
  else:
    from . import coco14
    train = coco14.AffinityDataset(data_dir, 'train2014', path_index, label_dir, train_transforms)

  info = DatasetInfo.from_metafile(dataset)
  train.info = info

  return train


def get_segmentation_datasets(
  dataset,
  data_dir,
  augment,
  image_size,
  pseudo_masks_dir=None,
  cutmix_prob=1.,
  mixup_prob=1.,
  train_transforms=None,
  valid_transforms=None,
):
  print(f'Loading {dataset} Segmentation Dataset')

  if dataset == 'voc12':
    from . import voc12
    train = voc12.SegmentationDataset(data_dir, 'train_aug', train_transforms, pseudo_masks_dir)
    valid = voc12.SegmentationDataset(data_dir, 'val', valid_transforms, pseudo_masks_dir)
  else:
    from . import coco14
    train = coco14.SegmentationDataset(data_dir, 'train2014', train_transforms, pseudo_masks_dir)
    valid = coco14.SegmentationDataset(data_dir, 'val2014', valid_transforms, pseudo_masks_dir)

  info = DatasetInfo.from_metafile(dataset)
  train.info = info
  valid.info = info

  return train, valid


def get_inference_dataset(dataset, data_dir, domain=None, transform=None):
  print(f'Loading {dataset} Inference Dataset')

  if dataset == 'voc12':
    from . import voc12
    infer = voc12.InferenceDataset(data_dir, domain or 'train_aug', transform)
  else:
    from . import coco14
    infer = coco14.InferenceDataset(data_dir, domain or 'train2014', transform)

  infer.info = DatasetInfo.from_metafile(dataset)

  return infer


def get_paths_dataset(dataset, data_dir, domain=None, transform=None):
  print(f'Loading {dataset} Segmentation Evaluation Dataset')

  if dataset == 'voc12':
    from . import voc12
    valid = voc12.PathsDataset(data_dir, domain or 'train_aug', transform)
  else:
    from . import coco14
    valid = coco14.PathsDataset(data_dir, domain or 'train2014', transform)

  valid.info = DatasetInfo.from_metafile(dataset)

  return valid


def get_hrcams_datasets(
  dataset,
  data_dir,
  cams_dir,
  domain=None,
  train_transforms=None,
  valid_transforms=None,
  resize_fn=None,
  normalize_fn=None,
):
  print(f'Loading {dataset} Classification Dataset')

  if dataset == 'voc12':
    from . import voc12
    train = voc12.HRCAMsDataset(data_dir, domain or 'train_aug', cams_dir, resize_fn, normalize_fn, train_transforms)
    valid = voc12.CAMEvaluationDataset(data_dir, 'train', valid_transforms)
  else:
    from . import coco14
    train = coco14.HRCAMsDataset(data_dir, domain or 'train2014', cams_dir, resize_fn, normalize_fn, train_transforms)
    valid = coco14.CAMEvaluationDataset(data_dir, 'train2014', valid_transforms)

  # train = _apply_augmentation_strategies(train, augment, image_size, cutmix_prob, mixup_prob)

  info = DatasetInfo.from_metafile(dataset)
  train.info = info
  valid.info = info

  return train, valid


def _apply_augmentation_strategies(dataset, augment, image_size, cutmix_prob, mixup_prob):
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
):
  mean, std = imagenet_stats()

  tt = []
  if min_size == max_size:
    tt += [transforms.Resize((min_size, min_size))]
  else:
    tt += [RandomResize(min_size, max_size)]
  tt += [RandomHorizontalFlip()]
  if 'colorjitter' in augment:
    tt += [transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)]
  if 'randaugment' in augment:
    tt += [RandAugmentMC(n=2, m=10)]
  tt += [Normalize(mean, std)]
  if 'cutmix' not in augment:
    # This will happen inside CutMix.
    tt += [RandomCrop(crop_size)]
  tt += [Transpose()]

  tt = transforms.Compose(tt)
  tv = transforms.Compose(
    [
      # RandomResize_For_Segmentation(image_size, image_size),
      Normalize_For_Segmentation(mean, std),
      Top_Left_Crop_For_Segmentation(crop_size),
      Transpose_For_Segmentation()
    ]
  )

  return tt, tv


def get_affinity_transforms(
  min_image_size,
  max_image_size,
  crop_size,
):
  mean, std = imagenet_stats()

  tt = transforms.Compose(
    [
      RandomResize_For_Segmentation(min_image_size, max_image_size),
      RandomHorizontalFlip_For_Segmentation(),
      Normalize_For_Segmentation(mean, std),
      RandomCrop_For_Segmentation(crop_size),
      Transpose_For_Segmentation(),
      Resize_For_Mask(crop_size // 4),
    ]
  )

  return tt


def get_segmentation_transforms(
  min_size,
  max_size,
  crop_size,
  augment,
  overcrop: bool = True,
):
  mean, std = imagenet_stats()

  tt = transforms.Compose(
    [
      RandomResize_For_Segmentation(min_size, max_size, overcrop=overcrop),
      RandomHorizontalFlip_For_Segmentation(),
      Normalize_For_Segmentation(mean, std),
      RandomCrop_For_Segmentation(crop_size),
      Transpose_For_Segmentation()
    ]
  )

  tv = transforms.Compose(
    [Normalize_For_Segmentation(mean, std),
     Top_Left_Crop_For_Segmentation(crop_size),
     Transpose_For_Segmentation()]
  )

  return tt, tv


def get_ccam_transforms(
  image_size,
  crop_size,
):
  mean, std = imagenet_stats()

  resize_fn = transforms.Resize(size=[image_size] * 2)
  normalize_fn = Normalize_For_Segmentation(mean, std)

  tt = transforms.Compose([
    normalize_fn,
    random_hflip_fn,
    RandomCrop_For_Segmentation(crop_size, ignore_value=0.),
    Transpose_For_Segmentation(),
  ])
  tv = transforms.Compose(
    [
      normalize_fn,
      Top_Left_Crop_For_Segmentation(crop_size),
      Transpose_For_Segmentation(),
    ]
  )

  return tt, tv, resize_fn, normalize_fn
