import random

import numpy as np
import torch
from PIL import Image


def convert_OpenCV_to_PIL(image):
  return Image.fromarray(image[..., ::-1])


def convert_PIL_to_OpenCV(image):
  return np.asarray(image)[..., ::-1]


class RandomResize:

  def __init__(self, min_image_size, max_image_size):
    self.min_image_size = min_image_size
    self.max_image_size = max_image_size

    self.modes = [Image.BICUBIC, Image.NEAREST]

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

  def __init__(self, min_image_size, max_image_size):
    self.min_image_size = min_image_size
    self.max_image_size = max_image_size

    self.modes = [Image.BICUBIC, Image.NEAREST]

  def __call__(self, data):
    image, mask = data['image'], data['mask']

    rand_image_size = random.randint(self.min_image_size, self.max_image_size)

    w, h = image.size
    if w < h:
      scale = rand_image_size / h
    else:
      scale = rand_image_size / w

    size = (int(round(w * scale)), int(round(h * scale)))
    if size[0] == w and size[1] == h:
      pass
    else:
      data['image'] = image.resize(size, Image.BICUBIC)
      data['mask'] = mask.resize(size, Image.NEAREST)

    return data


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


class Normalize:

  def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    self.mean = mean
    self.std = std

  def __call__(self, image):
    image = np.asarray(image)
    norm_image = np.empty_like(image, np.float32)

    norm_image[..., 0] = (image[..., 0] / 255. - self.mean[0]) / self.std[0]
    norm_image[..., 1] = (image[..., 1] / 255. - self.mean[1]) / self.std[1]
    norm_image[..., 2] = (image[..., 2] / 255. - self.mean[2]) / self.std[2]

    return norm_image


class Normalize_For_Segmentation:

  def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    self.mean = mean
    self.std = std

  def __call__(self, data):
    image, mask = data['image'], data['mask']

    image = np.asarray(image, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.int64)

    norm_image = np.empty_like(image, np.float32)

    norm_image[..., 0] = (image[..., 0] / 255. - self.mean[0]) / self.std[0]
    norm_image[..., 1] = (image[..., 1] / 255. - self.mean[1]) / self.std[1]
    norm_image[..., 2] = (image[..., 2] / 255. - self.mean[2]) / self.std[2]

    data['image'] = norm_image
    data['mask'] = mask

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
    img_top = random.randrange(h_space + 1)     # rand(65)   = 10
  else:
    cont_top = random.randrange(-h_space + 1)
    img_top = 0

  dst_bbox = {'xmin': cont_left, 'ymin': cont_top, 'xmax': cont_left + cw, 'ymax': cont_top + ch}  # 20,  0, 20+300, 0+448
  src_bbox = {'xmin': img_left, 'ymin': img_top, 'xmax': img_left + cw, 'ymax': img_top + ch}      #  0, 10,    300, 10+448

  return dst_bbox, src_bbox


class RandomCrop:

  def __init__(self, crop_size, channels=3, channels_last=True, with_bbox=False, bg_value=0):
    self.bg_value = bg_value
    self.with_bbox = with_bbox
    self.crop_size = crop_size
    self.channels_last = channels_last
    self.crop_shape = (
      (self.crop_size, self.crop_size, channels)
      if channels_last
      else (channels, self.crop_size, self.crop_size)
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

  def __init__(self, crop_size, channels=3, channels_last=True, with_bbox=False, bg_value=0, ignore_value=255):
    super().__init__(crop_size, channels, channels_last, with_bbox=True, bg_value=bg_value)

    self.ignore_value
    self.mask_crop_shape = (self.crop_size, self.crop_size)

  def __call__(self, data):
    image, mask = data['image'], data['mask']

    ci, (b, a) = super()(image)

    cm = np.ones(self.mask_crop_shape, mask.dtype) * self.ignore_value
    cm[b['ymin']:b['ymax'], b['xmin']:b['xmax']] = mask[
      a['ymin']:a['ymax'],
      a['xmin']:a['xmax'],
    ]

    data['image'] = ci
    data['mask'] = cm

    return data


class Transpose:

  def __init__(self):
    pass

  def __call__(self, image):
    return image.transpose((2, 0, 1))


class Transpose_For_Segmentation:

  def __init__(self):
    pass

  def __call__(self, data):
    # h, w, c -> c, h, w
    data['image'] = data['image'].transpose((2, 0, 1))
    return data


class Resize_For_Mask:

  def __init__(self, size):
    self.size = (size, size)

  def __call__(self, data):
    mask = Image.fromarray(data['mask'].astype(np.uint8))
    mask = mask.resize(self.size, Image.NEAREST)
    data['mask'] = np.asarray(mask, dtype=np.uint64)
    return data


# region MixUp


class MixUp(torch.utils.data.Dataset):
  def __init__(self, dataset, num_mix=1, beta=1., prob=1.0):
    self.dataset = dataset
    self.num_mix = num_mix
    self.beta = beta
    self.prob = prob
  
  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, index):
    x, y = self.dataset[index]

    for _ in range(self.num_mix):
      r = np.random.rand(1)
      if self.beta <= 0 or r > self.prob:
        continue

      # Draw random sample.
      r = random.choice(range(len(self)))
      xb, yb = self.dataset[r]

      alpha = np.random.beta(self.beta, self.beta)
      x = x * alpha + xb * (1. - alpha)
      y = y * alpha + yb * (1. - alpha)

    return x, y


# endregion


# region CutMix


def rand_bbox(h, w, lam):
  ratio = np.sqrt(1. - lam)
  cw = np.int(w * ratio)
  ch = np.int(h * ratio)

  cx = np.random.randint(cw//2, w - cw//2)
  cy = np.random.randint(ch//2, h - ch//2)

  h1 = np.clip(cy - ch // 2, 0, h)
  h2 = np.clip(cy + ch // 2, 0, h)
  w1 = np.clip(cx - cw // 2, 0, w)
  w2 = np.clip(cx + cw // 2, 0, w)

  return h1, w1, h2, w2


class CutMix(torch.utils.data.Dataset):

  def __init__(self, dataset, crop, num_mix=1, beta=1., prob=1.0):
    self.dataset = dataset
    self.num_mix = num_mix
    self.beta = beta
    self.prob = prob
    self.random_crop = RandomCrop(crop, channels_last=False)

  def __len__(self):
    return len(self.dataset)
  
  def do_cutmix(self, x, y, xb, yb, lam):
    # Cut random bbox.
    bH, bW = xb.shape[1:]
    bh1, bw1, bh2, bw2 = rand_bbox(bH, bW, lam)
    xb = xb[:, bh1:bh2, bw1:bw2]

    # Central crop if B larger than A.
    aH, aW = x.shape[1:]
    bH, bW = xb.shape[1:]
    bhs = (bH-aH) // 2 if bH > aH else 0
    bws = (bW-aW) // 2 if bW > aW else 0
    xb = xb[:, bhs:bhs+aH, bws:bws+aW]

    # Random (x,y) placement if A larger than B.
    bH, bW = xb.shape[1:]
    bhs, bws = random.randint(0, aH-bH), random.randint(0, aW-bW)
    x[:, bhs:bhs+bH, bws:bws+bW] = xb

    lam = 1 - ((bH * bW) / (aH * aW))
    y = y * lam + yb * (1. - lam)

    return x, y

  def __getitem__(self, index):
    x, y = self.dataset[index]

    x = self.random_crop(x)

    for _ in range(self.num_mix):
      r = np.random.rand(1)
      if self.beta <= 0 or r > self.prob:
        continue

      # Draw random sample.
      lam = np.random.beta(self.beta, self.beta)
      r = random.choice(range(len(self)))
      xb, yb = self.dataset[r]

      x, y = self.do_cutmix(x, y, xb, yb, lam)

    return x, y


class CutOrMixUp(CutMix):
  def __getitem__(self, index):
    x, y = self.dataset[index]

    x = self.random_crop(x)

    for _ in range(self.num_mix):
      r = np.random.rand(1)
      if self.beta <= 0 or r > self.prob:
        continue

      # Draw random sample.
      alpha = np.random.beta(self.beta, self.beta)
      r = random.choice(range(len(self)))
      xb, yb = self.dataset[r]

      if np.random.rand(1) > 0.5:
        x, y = self.do_cutmix(x, y, xb, yb, alpha)
      else:
        x = x * alpha + xb * (1. - alpha)
        y = y * alpha + yb * (1. - alpha)

    return x, y

# endregion
