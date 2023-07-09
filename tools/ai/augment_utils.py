import random

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def convert_OpenCV_to_PIL(image):
  return Image.fromarray(image[..., ::-1])


def convert_PIL_to_OpenCV(image):
  return np.array(image)[..., ::-1]


class RandomResize:

  def __init__(self, min_image_size, max_image_size):
    self.min_image_size = min_image_size
    self.max_image_size = max_image_size

  def __call__(self, image, mode=Image.BICUBIC):
    rand_image_size = random.randint(self.min_image_size, self.max_image_size)

    w, h = image.size
    if w < h:
      scale = rand_image_size / h
    else:
      scale = rand_image_size / w

    size = (int(round(w * scale)), int(round(h * scale)))
    if size[0] == w and size[1] == h:
      return image

    return image.resize(size, mode)


class RandomResize_For_Segmentation:

  def __init__(self, min_image_size, max_image_size, overcrop=True):
    self.min_image_size = min_image_size
    self.max_image_size = max_image_size
    self.overcrop = overcrop

  def __call__(self, data):
    image, mask = data['image'], data['mask']
    W, H = image.size

    alpha = random.randint(self.min_image_size, self.max_image_size)
    alpha /= max(W, H) if self.overcrop else min(W, H)

    size = (int(round(W * alpha)), int(round(H * alpha)))
    if size[0] != W or size[1] == H:
      data['image'] = image.resize(size, Image.BICUBIC)
      data['mask'] = mask.resize(size, Image.NEAREST)

    return data


class Resize_For_Segmentation:

  def __init__(self, image_size, resize_x = None, resize_y = None):

    self.image_size = image_size
    self.resize_x = resize_x or transforms.Resize(self.image_size)
    self.resize_y = resize_y or transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.NEAREST)

  def __call__(self, data):
    image, mask = data['image'], data['mask']
    return {
      "image": self.resize_x(image),
      "mask": self.resize_y(mask),
    }


class RandomHorizontalFlip:

  def __init__(self):
    pass

  def __call__(self, image):
    if bool(random.getrandbits(1)):
      return image.transpose(Image.FLIP_LEFT_RIGHT)
    return image


class RandomHorizontalFlip_For_Segmentation:

  def __init__(self):
    pass

  def __call__(self, data):
    image, mask = data['image'], data['mask']

    if bool(random.getrandbits(1)):
      data['image'] = image.transpose(Image.FLIP_LEFT_RIGHT)
      data['mask'] = mask.transpose(Image.FLIP_LEFT_RIGHT)

    return data


def random_hflip_fn(data):
  if bool(random.getrandbits(1)):
    data['image'] = np.flip(data['image'], axis=-1).copy()
    data['mask'] = np.flip(data['mask'], axis=-1).copy()
  return data


class Normalize:

  def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    self.mean = mean
    self.std = std

  def __call__(self, image):
    x = np.array(image, dtype=np.float32)
    if isinstance(image, Image.Image):
      image.close()

    norm_image = np.empty_like(x, np.float32)

    norm_image[..., 0] = (x[..., 0] / 255. - self.mean[0]) / self.std[0]
    norm_image[..., 1] = (x[..., 1] / 255. - self.mean[1]) / self.std[1]
    norm_image[..., 2] = (x[..., 2] / 255. - self.mean[2]) / self.std[2]

    return norm_image


class Normalize_For_Segmentation:

  def __init__(
    self,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    idtype=np.float32,
    mdtype=np.int64,
  ):
    self.mean = mean
    self.std = std
    self.idtype = idtype
    self.mdtype = mdtype

  def __call__(self, data):
    image, mask = data['image'], data['mask']

    x = np.array(image, dtype=self.idtype)
    y = np.array(mask, dtype=self.mdtype)

    if isinstance(image, Image.Image):
      image.close()
    if isinstance(mask, Image.Image):
      mask.close()

    z = np.empty_like(x)

    z[..., 0] = (x[..., 0] / 255. - self.mean[0]) / self.std[0]
    z[..., 1] = (x[..., 1] / 255. - self.mean[1]) / self.std[1]
    z[..., 2] = (x[..., 2] / 255. - self.mean[2]) / self.std[2]

    data['image'] = z
    data['mask'] = y

    return data


class Top_Left_Crop:

  def __init__(self, crop_size, channels=3):
    self.bg_value = 0
    self.crop_size = crop_size
    self.crop_shape = (self.crop_size, self.crop_size, channels)

  def __call__(self, image):
    h, w, c = image.shape

    ch = min(self.crop_size, h)
    cw = min(self.crop_size, w)

    cropped_image = np.ones(self.crop_shape, image.dtype) * self.bg_value
    cropped_image[:ch, :cw] = image[:ch, :cw]

    return cropped_image


class Top_Left_Crop_For_Segmentation:

  def __init__(self, crop_size, channels=3):
    self.bg_value = 0
    self.crop_size = crop_size
    self.crop_shape = (self.crop_size, self.crop_size, channels)
    self.crop_shape_for_mask = (self.crop_size, self.crop_size)

  def __call__(self, data):
    image, mask = data['image'], data['mask']

    h, w, c = image.shape

    ch = min(self.crop_size, h)
    cw = min(self.crop_size, w)

    cropped_image = np.ones(self.crop_shape, image.dtype) * self.bg_value
    cropped_image[:ch, :cw] = image[:ch, :cw]

    cropped_mask = np.ones(self.crop_shape_for_mask, mask.dtype) * 255
    cropped_mask[:ch, :cw] = mask[:ch, :cw]

    data['image'] = cropped_image
    data['mask'] = cropped_mask

    return data


def random_crop_box(crop_size, h, w):
  ch = min(crop_size, h)  # (448, 512) -> 448
  cw = min(crop_size, w)  # (448, 300) -> 300

  h_space = h - crop_size  # 512-448 =   64
  w_space = w - crop_size  # 300-448 = -148

  if w_space > 0:
    cont_left = 0
    img_left = random.randrange(w_space + 1)
  else:
    cont_left = random.randrange(-w_space + 1)  # rand(149)  = 20
    img_left = 0

  if h_space > 0:
    cont_top = 0
    img_top = random.randrange(h_space + 1)  # rand(65)   = 10
  else:
    cont_top = random.randrange(-h_space + 1)
    img_top = 0

  dst_bbox = {
    'xmin': cont_left,
    'ymin': cont_top,
    'xmax': cont_left + cw,
    'ymax': cont_top + ch
  }  # 20,  0, 20+300, 0+448
  src_bbox = {'xmin': img_left, 'ymin': img_top, 'xmax': img_left + cw, 'ymax': img_top + ch}  #  0, 10,    300, 10+448

  return dst_bbox, src_bbox


class RandomCrop:

  def __init__(self, crop_size, channels=3, channels_last=True, with_bbox=False, bg_value=0):
    self.bg_value = bg_value
    self.with_bbox = with_bbox
    self.crop_size = crop_size
    self.channels_last = channels_last
    self.crop_shape = (
      (self.crop_size, self.crop_size, channels) if channels_last else (channels, self.crop_size, self.crop_size)
    )

  def __call__(self, x):
    sizes = x.shape[:2] if self.channels_last else x.shape[1:]
    b, a = random_crop_box(self.crop_size, *sizes)

    y = np.ones(self.crop_shape, x.dtype) * self.bg_value

    if self.channels_last:
      crop = x[a['ymin']:a['ymax'], a['xmin']:a['xmax']]
      y[b['ymin']:b['ymax'], b['xmin']:b['xmax']] = crop
    else:
      crop = x[:, a['ymin']:a['ymax'], a['xmin']:a['xmax']]
      y[:, b['ymin']:b['ymax'], b['xmin']:b['xmax']] = crop

    if self.with_bbox:
      return y, (b, a)
    else:
      return y


class RandomCrop_For_Segmentation(RandomCrop):

  def __init__(
    self,
    crop_size,
    channels=3,
    channels_last=True,
    labels_last=True,
    bg_value=0,
    ignore_value=255,
  ):
    super().__init__(crop_size, channels, channels_last, with_bbox=True, bg_value=bg_value)

    self.labels_last = labels_last
    self.ignore_value = ignore_value
    self.mask_crop_shape = (self.crop_size, self.crop_size)

  def __call__(self, data):
    image, mask = data['image'], data['mask']

    ci, (b, a) = super().__call__(image)

    rank = len(mask.shape)
    if self.labels_last:
      l = mask.shape[2:]
      cm = np.ones([*self.mask_crop_shape, *l], dtype=mask.dtype) * self.ignore_value
      cm[b['ymin']:b['ymax'], b['xmin']:b['xmax']] = mask[a['ymin']:a['ymax'], a['xmin']:a['xmax']]
    else:
      l = mask.shape[:rank - 2]
      cm = np.ones([*l, *self.mask_crop_shape], dtype=mask.dtype) * self.ignore_value
      cm[..., b['ymin']:b['ymax'], b['xmin']:b['xmax']] = mask[..., a['ymin']:a['ymax'], a['xmin']:a['xmax']]

    data['image'] = ci
    data['mask'] = cm

    return data


class Transpose:

  def __init__(self):
    pass

  def __call__(self, image):
    return image.transpose((2, 0, 1))


class Transpose_For_Segmentation:

  def __call__(self, data):
    # h, w, c -> c, h, w
    data['image'] = data['image'].transpose((2, 0, 1))
    return data


class ResizeMask:

  def __init__(self, size):
    self.size = (size, size)

  def __call__(self, data):
    mask = Image.fromarray(data['mask'].astype(np.uint8))
    mask = mask.resize(self.size, Image.NEAREST)
    data['mask'] = np.array(mask, dtype=np.uint64)
    return data


class CLAHE:

  def __init__(self, clip_limit: float = 2.0, width: int = 8, height: int = 8):

    self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(width, height))

  def __call__(self, data):
    if isinstance(data, dict):
      x = data["image"]
    else:
      x = data

    if isinstance(x, Image.Image):
      i = np.array(x)
      x.close()
      x = i

    x = cv2.cvtColor(x, cv2.COLOR_RGB2Lab)
    # 0 to 'L' channel, 1 to 'a' channel, and 2 to 'b' channel
    x[:, :, 0] = self.clahe.apply(x[:, :, 0])
    x = cv2.cvtColor(x, cv2.COLOR_Lab2RGB)

    if isinstance(data, dict):
      data["image"] = x
    else:
      data = x

    return data


# region MixUp


class AugmentedDataset(torch.utils.data.Dataset):
  dataset: torch.utils.data.Dataset

  def __len__(self):
    return len(self.dataset)

  @property
  def info(self):
    return self.dataset.info

  def _to_segm_batch(self, batch):
    return (
      (batch, True)
      if len(batch) == 4
      else (batch + (None,), False)
    )


class MixUp(AugmentedDataset):

  def __init__(self, dataset, num_mix=1, beta=1., prob=1.0):
    self.dataset = dataset
    self.num_mix = num_mix
    self.beta = beta
    self.prob = prob

  def __getitem__(self, index):
    batch, is_segm = self._to_segm_batch(self.dataset[index])
    i, x, y, m = batch

    for _ in range(self.num_mix):
      r = np.random.rand(1)
      if self.beta <= 0 or r > self.prob:
        continue

      # Draw random sample.
      r = random.choice(range(len(self)))
      (_, xb, yb, mb), _ = self._to_segm_batch(self.dataset[r])

      alpha = np.random.beta(self.beta, self.beta)
      x = x * alpha + xb * (1. - alpha)
      y = y * alpha + yb * (1. - alpha)

      # TODO: mix int segmentation masks.

    return (i, x, y, m) if is_segm else (i, x, y)

# endregion

# region CutMix


def rand_bbox(h, w, lam):
  ratio = np.sqrt(1. - lam)
  cw = np.int(w * ratio)
  ch = np.int(h * ratio)

  cx = np.random.randint(cw // 2, w - cw // 2)
  cy = np.random.randint(ch // 2, h - ch // 2)

  h1 = np.clip(cy - ch // 2, 0, h)
  h2 = np.clip(cy + ch // 2, 0, h)
  w1 = np.clip(cx - cw // 2, 0, w)
  w2 = np.clip(cx + cw // 2, 0, w)

  return h1, w1, h2, w2


class CutMix(AugmentedDataset):

  def __init__(self, dataset, crop, num_mix=1, beta=1., prob=1.0):
    self.dataset = dataset
    self.num_mix = num_mix
    self.beta = beta
    self.prob = prob

    # This is done here so cut-mixed batches aren't cropped as well.
    self.random_crop = RandomCrop(crop, channels_last=False)

  def do_cutmix(self, batch_a, batch_b, alpha):
    ia, xa, ya, ma = batch_a
    _, xb, yb, mb = batch_b

    # Cut random bbox.
    bH, bW = xb.shape[1:]
    bh1, bw1, bh2, bw2 = rand_bbox(bH, bW, alpha)
    xb = xb[:, bh1:bh2, bw1:bw2]

    # Central crop if B larger than A.
    aH, aW = xa.shape[1:]
    bH, bW = xb.shape[1:]
    bhs = (bH - aH) // 2 if bH > aH else 0
    bws = (bW - aW) // 2 if bW > aW else 0
    xb = xb[:, bhs:bhs + aH, bws:bws + aW]

    # Random (x,y) placement if A larger than B.
    bH, bW = xb.shape[1:]
    bhs, bws = random.randint(0, aH - bH), random.randint(0, aW - bW)
    xa[:, bhs:bhs + bH, bws:bws + bW] = xb

    # targets.
    alpha = 1 - ((bH * bW) / (aH * aW))
    ya = ya * alpha + yb * (1. - alpha)

    # masks.
    if ma is not None:
      mb = mb[bh1:bh2, bw1:bw2]
      ma[bhs:bhs + bH, bws:bws + bW] = mb

    return ia, xa, ya, ma

  def __getitem__(self, index):
    (i, x, y, m), is_segm = self._to_segm_batch(self.dataset[index])
    batch_a = (i, self.random_crop(x), y, m)

    for _ in range(self.num_mix):
      r = np.random.rand(1)
      if self.beta <= 0 or r > self.prob:
        continue

      # Draw random sample.
      l = np.random.beta(self.beta, self.beta)
      r = random.choice(range(len(self)))
      batch_b, _ = self._to_segm_batch(self.dataset[r])

      batch_a = self.do_cutmix(batch_a, batch_b, l)

    return batch_a if is_segm else batch_a[:3]


class CutOrMixUp(CutMix):

  def __getitem__(self, index):
    (i, x, y, m), is_segm = self._to_segm_batch(self.dataset[index])
    batch_a = (i, self.random_crop(x), y, m)

    for _ in range(self.num_mix):
      r = np.random.rand(1)
      if self.beta <= 0 or r > self.prob:
        continue

      # Draw random sample.
      alpha = np.random.beta(self.beta, self.beta)
      r = random.choice(range(len(self)))
      batch_b, _ = self._to_segm_batch(self.dataset[r])

      if np.random.rand(1) > 0.5:
        batch_a = self.do_cutmix(batch_a, batch_b, alpha)
        i, x, y, m = batch_a
      else:
        x, y = batch_a
        xb, yb = batch_b[1:3]
        x = x * alpha + xb * (1. - alpha)
        y = y * alpha + yb * (1. - alpha)
        batch_a = (i, x, y, m)

    return (i, x, y, m) if is_segm else (i, x, y)

# endregion
