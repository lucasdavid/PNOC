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


class ApplyToImage:
  def __init__(self, transform):
    self.transform = transform

  def __call__(self, data):
    data["image"] = self.transform(data["image"])
    return data

  def __str__(self) -> str:
    return f"ApplyToImage({self.transform})"


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
    self.resize_y = resize_y or transforms.Resize(self.image_size, interpolation=Image.NEAREST)

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


def at_least_3d(data, axis=0):
  for k in ("image", "mask"):
    if k in data:
      t = data[k]
      if len(t.shape) < 3:
        data[k] = np.expand_dims(t, axis)
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
    channels = norm_image.shape[-1]

    for i in range(channels):
      norm_image[..., i] = (x[..., i] / 255. - self.mean[i]) / self.std[i]

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
    channels = x.shape[-1]

    for i in range(channels):
      z[..., i] = (x[..., i] / 255. - self.mean[i]) / self.std[i]

    data['image'] = z
    data['mask'] = y

    return data


class Top_Left_Crop:

  def __init__(self, crop_size):
    self.bg_value = 0
    self.crop_size = crop_size

  def __call__(self, image):
    h, w, c = image.shape
    ch = min(self.crop_size, h)
    cw = min(self.crop_size, w)
    shape = (self.crop_size, self.crop_size, c)

    cropped_image = np.ones(shape, image.dtype) * self.bg_value
    cropped_image[:ch, :cw] = image[:ch, :cw]

    return cropped_image


class Top_Left_Crop_For_Segmentation:

  def __init__(self, crop_size):
    self.bg_value = 0
    self.crop_size = crop_size

  def __call__(self, data):
    image, mask = data['image'], data['mask']

    h, w, c = image.shape
    ch = min(self.crop_size, h)
    cw = min(self.crop_size, w)
    crop_shape = (self.crop_size, self.crop_size, c)
    mask_shape = (self.crop_size, self.crop_size)

    cropped_image = np.ones(crop_shape, image.dtype) * self.bg_value
    cropped_image[:ch, :cw] = image[:ch, :cw]

    cropped_mask = np.ones(mask_shape, mask.dtype) * 255
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

  def __init__(self, crop_size, channels_last=True, with_bbox=False, bg_value=0):
    self.bg_value = bg_value
    self.with_bbox = with_bbox
    self.crop_size = crop_size
    self.channels_last = channels_last

  def __call__(self, x):
    if self.channels_last:
      h, w, c = x.shape
    else:
      c, h, w = x.shape

    crop_shape = (
      (self.crop_size, self.crop_size, c)
      if self.channels_last
      else (c, self.crop_size, self.crop_size)
    )

    b, a = random_crop_box(self.crop_size, h, w)

    y = np.ones(crop_shape, x.dtype) * self.bg_value

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
    channels_last=True,
    labels_last=True,
    bg_value=0,
    ignore_value=255,
    idtype=np.float32,
    mdtype=np.int64,
  ):
    super().__init__(crop_size, channels_last, with_bbox=True, bg_value=bg_value)

    self.labels_last = labels_last
    self.ignore_value = ignore_value
    self.mask_crop_shape = (self.crop_size, self.crop_size)
    self.idtype = idtype
    self.mdtype = mdtype

  def __call__(self, data):
    image, mask = data['image'], data['mask']

    if isinstance(image, Image.Image):
      x = np.array(image, dtype=self.idtype)
      y = np.array(mask, dtype=self.mdtype)
      image.close()
      mask.close()
      image, mask = x, y

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
    self.clip_limit = clip_limit
    self.width = width
    self.height = height

  _clahe: "cv2.CLAHE" = None

  @property
  def clahe(self):
    if self._clahe is None:
      self._clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(self.width, self.height))
    return self._clahe

  def __call__(self, data):
    if isinstance(data, dict):
      x = data["image"]
    else:
      x = data

    if isinstance(x, Image.Image):
      from_pil = True
      i = np.array(x)
      x.close()
      x = i
    else:
      from_pil = False

    channels = x.shape[-1]

    if channels == 3:
      x = cv2.cvtColor(x, cv2.COLOR_RGB2Lab)

    # 0 to 'L' channel, 1 to 'a' channel, and 2 to 'b' channel
    x[:, :, 0] = self.clahe.apply(x[:, :, 0])

    if channels == 3:
      x = cv2.cvtColor(x, cv2.COLOR_Lab2RGB)

    if from_pil:
      x = Image.fromarray(x)

    if isinstance(data, dict):
      data["image"] = x
    else:
      data = x

    return data


class QuantileChannelIndependentNormalization:

  def __init__(self, q_min=0.00, q_max=0.99, valid_variances=25):
    self.q_min = q_min
    self.q_max = q_max
    self.valid_variances = valid_variances

  def __call__(self, data):
    if isinstance(data, dict):
      x = data["image"]
    else:
      x = data

    from_pil = isinstance(x, Image.Image)
    if from_pil:
      with x:
        x = np.array(x)
    odtype = x.dtype
    x = x.astype("float32")
    if self.q_min:
      x_min_ = np.quantile(x, self.q_min, axis=(0, 1), keepdims=True)
      x -= x_min_
    x_max_ = np.quantile(x, self.q_max, axis=(0, 1), keepdims=True)
    valid_mask = (x_max_ != 0) & (x_max_ >= self.valid_variances)
    x = np.divide(x, x_max_, out=x.copy(), where=valid_mask)
    x = np.clip(x, 0, 1)
    x *= 255
    x = x.astype("uint8")
    if from_pil: x = Image.fromarray(x)
    if isinstance(data, dict): data["image"] = x
    else: data = x
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
  cw = int(w * ratio)
  ch = int(h * ratio)

  cx = np.random.randint(cw // 2, w - cw // 2)
  cy = np.random.randint(ch // 2, h - ch // 2)

  h1 = np.clip(cy - ch // 2, 0, h)
  h2 = np.clip(cy + ch // 2, 0, h)
  w1 = np.clip(cx - cw // 2, 0, w)
  w2 = np.clip(cx + cw // 2, 0, w)

  return h1, w1, h2, w2


def _cutmix(sample_a, sample_b, alpha):
  id_a, image_a, label_a, mask_a = sample_a
  id_b, image_b, label_b, mask_b = sample_b

  # Cut random bbox.
  Hb, Wb = image_b.shape[1:]
  bh1, bw1, bh2, bw2 = rand_bbox(Hb, Wb, alpha)
  image_b = image_b[:, bh1:bh2, bw1:bw2]

  # Central crop if B larger than A.
  Ha, Wa = image_a.shape[1:]
  Hb, Wb = image_b.shape[1:]
  bhs = (Hb - Ha) // 2 if Hb > Ha else 0
  bws = (Wb - Wa) // 2 if Wb > Wa else 0
  image_b = image_b[:, bhs:bhs + Ha, bws:bws + Wa]

  # Random (x,y) placement if A larger than B.
  Hb, Wb = image_b.shape[1:]
  bhs, bws = random.randint(0, Ha - Hb), random.randint(0, Wa - Wb)
  image_a[:, bhs:bhs + Hb, bws:bws + Wb] = image_b

  # targets.
  alpha = 1 - ((Hb * Wb) / (Ha * Wa))
  label_a *= alpha
  label_a += label_b * (1. - alpha)

  # masks.
  if mask_a is not None:
    mask_b = mask_b[bh1:bh2, bw1:bw2]
    mask_a[bhs:bhs + Hb, bws:bws + Wb] = mask_b

  return id_a, image_a, label_a, mask_a


class CutMix(AugmentedDataset):

  def __init__(self, dataset, crop, num_mix=1, beta=1., prob=1.0):
    self.dataset = dataset
    self.num_mix = num_mix
    self.beta = beta
    self.prob = prob

    # This is done here so cut-mixed batches aren't cropped as well.
    self.random_crop = RandomCrop_For_Segmentation(crop, channels_last=False)

  def __getitem__(self, index):
    (i, x, y, m), is_segm = self._to_segm_batch(self.dataset[index])
    entry = self.random_crop({"image": x, "mask": m})
    x, m = entry["image"], entry["mask"]
    batch_a = (i, x, y, m)

    for _ in range(self.num_mix):
      r = np.random.rand(1)
      if self.beta <= 0 or r > self.prob:
        continue

      # Draw random sample.
      l = np.random.beta(self.beta, self.beta)
      r = random.choice(range(len(self)))
      batch_b, _ = self._to_segm_batch(self.dataset[r])

      batch_a = _cutmix(batch_a, batch_b, l)

    return batch_a if is_segm else batch_a[:3]


class CutOrMixUp(CutMix):

  def __getitem__(self, index):
    (i, x, y, m), is_segm = self._to_segm_batch(self.dataset[index])
    entry = self.random_crop({"image": x, "mask": m})
    x, m = entry["image"], entry["mask"]
    batch_a = (i, x, y, m)

    for _ in range(self.num_mix):
      r = np.random.rand(1)
      if self.beta <= 0 or r > self.prob:
        continue

      # Draw random sample.
      alpha = np.random.beta(self.beta, self.beta)
      r = random.choice(range(len(self)))
      batch_b, _ = self._to_segm_batch(self.dataset[r])

      if np.random.rand(1) > 0.5:
        batch_a = _cutmix(batch_a, batch_b, alpha)
        i, x, y, m = batch_a
      else:
        x, y = batch_a
        xb, yb = batch_b[1:3]
        x = x * alpha + xb * (1. - alpha)
        y = y * alpha + yb * (1. - alpha)
        batch_a = (i, x, y, m)

    return (i, x, y, m) if is_segm else (i, x, y)

# endregion
