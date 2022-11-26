from torchvision import transforms

from tools.ai.augment_utils import *
from tools.ai.randaugment import RandAugmentMC


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
