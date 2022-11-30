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
  

def imagenet_stats():
  return (
    [0.485, 0.456, 0.406],
    [0.229, 0.224, 0.225],
  )


def get_transforms(
  min_image_size,
  max_image_size,
  image_size,
  augment,
):
  mean, std = imagenet_stats()

  tt = []

  if min_image_size == max_image_size:
    tt.append(transforms.Resize((image_size, image_size)))
  else:
    tt.append(RandomResize(min_image_size, max_image_size))

  tt.append(RandomHorizontalFlip())

  if 'colorjitter' in augment:
    tt.append(transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1))

  if 'randaugment' in augment:
    tt.append(RandAugmentMC(n=2, m=10))

  tt.append(Normalize(mean, std))

  if 'cutmix' not in augment:
    # This will happen inside CutMix.
    tt.append(RandomCrop(image_size))

  tt.append(Transpose())

  tt = transforms.Compose(tt)
  tv = transforms.Compose(
    [
      Normalize_For_Segmentation(mean, std),
      Top_Left_Crop_For_Segmentation(image_size),
      Transpose_For_Segmentation()
    ]
  )

  return tt, tv


def get_dataset_classification(
  dataset,
  data_dir,
  augment,
  image_size,
  cutmix_prob=0.,
  train_transforms=None,
  valid_transforms=None,
):
  print(f'Loading {dataset} dataset')

  if dataset == 'voc12':
    from . import voc12
    train_dataset = voc12.VOC12ClassificationDataset(data_dir, 'train_aug', train_transforms)
    valid_dataset = voc12.VOC12CAMEvaluationDataset(data_dir, 'train', valid_transforms)
  else:
    from . import coco14
    train_dataset = coco14.COCO14ClassificationDataset(data_dir, 'train2014', train_transforms)
    valid_dataset = coco14.COCO14CAMEvaluationDataset(data_dir, 'train2014', valid_transforms)
  
  if 'cutmix' in augment:
    print('[i] Using cutmix')
    train_dataset = CutMix(
      train_dataset, image_size, num_mix=1, beta=1., prob=cutmix_prob
    )
  
  info = DatasetInfo.from_metafile(dataset)
  train_dataset.info = info
  valid_dataset.info = info

  return train_dataset, valid_dataset


def get_dataset_inference(dataset, data_dir, domain=None):
  if dataset == 'voc12':
    from . import voc12
    infer_dataset = voc12.VOC12InferenceDataset(data_dir, domain or 'train_aug')
  else:
    from . import coco14
    infer_dataset = coco14.COCO14InferenceDataset(data_dir, domain or 'train2014')

  infer_dataset.info = DatasetInfo.from_metafile(dataset)

  return infer_dataset


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
