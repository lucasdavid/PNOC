import random

import cv2
import numpy as np
from PIL import Image


def get_strided_size(orig_size, stride):
  return ((orig_size[0] - 1) // stride + 1, (orig_size[1] - 1) // stride + 1)


def get_strided_up_size(orig_size, stride):
  strided_size = get_strided_size(orig_size, stride)
  return strided_size[0] * stride, strided_size[1] * stride


def imshow(image, delay=0, mode='RGB', title='show'):
  if mode == 'RGB':
    demo_image = image[..., ::-1]
  else:
    demo_image = image

  cv2.imshow(title, demo_image)
  if delay >= 0:
    cv2.waitKey(delay)


def transpose(image):
  return image.transpose((1, 2, 0))


def denormalize(image, mean=None, std=None, dtype=np.uint8, tp=True):
  if tp:
    image = transpose(image)

  if mean is not None:
    image = (image * std) + mean

  if dtype == np.uint8:
    return (image.clip(0, 1) * 255).astype(np.uint8)
  else:
    return image


def colormap(cam, shape=None, mode=cv2.COLORMAP_JET):
  cam = (cam.clip(0, 1) * 255).astype(np.uint8)
  if shape is not None:
    h, w = shape[:2]
    cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
  cam = cv2.applyColorMap(cam, mode)
  return cam


def decode_from_colormap(data, colors, ignore_index=21):
  # ignore = (data == ignore_index).astype(data.dtype)
  # mask = 1 - ignore
  # data *= mask

  h, w = data.shape
  image = colors[data.reshape((h * w))].reshape((h, w, 3))

  # ignore = np.concatenate([ignore[..., np.newaxis], ignore[..., np.newaxis], ignore[..., np.newaxis]], axis=-1)
  # image[ignore.astype('bool')] = 255
  return image


def normalize(cam, epsilon=1e-5):
  cam = np.maximum(cam, 0)
  max_value = np.max(cam, axis=(0, 1), keepdims=True)
  return np.maximum(cam - epsilon, 0) / (max_value + epsilon)


def crf_inference(image, probs, t=10, scale_factor=1, gt_prob=1):
  import pydensecrf.densecrf as dcrf
  from pydensecrf.utils import unary_from_softmax

  H, W = image.shape[:2]
  C = probs.shape[0]

  d = dcrf.DenseCRF2D(W, H, C)

  unary = unary_from_softmax(probs, scale=gt_prob)
  unary = np.ascontiguousarray(unary)
  image = np.ascontiguousarray(image)

  d.setUnaryEnergy(unary)
  d.addPairwiseGaussian(sxy=3 / scale_factor, compat=3)
  d.addPairwiseBilateral(sxy=50 / scale_factor, srgb=5, rgbim=image, compat=10)
  Q = d.inference(t)

  return np.array(Q).reshape((C, H, W))


def crf_with_alpha(ori_image, cams, alpha):
  # h, w, c -> c, h, w
  # cams = cams.transpose((2, 0, 1))

  bg_score = np.power(1 - np.max(cams, axis=0, keepdims=True), alpha)
  bgcam_score = np.concatenate((bg_score, cams), axis=0)

  cams_with_crf = crf_inference(ori_image, bgcam_score)
  # return cams_with_crf.transpose((1, 2, 0))
  return cams_with_crf


def crf_inference_label(img, labels, t=10, n_labels=21, gt_prob=0.7):
  import pydensecrf.densecrf as dcrf
  from pydensecrf.utils import unary_from_labels

  h, w = img.shape[:2]

  d = dcrf.DenseCRF2D(w, h, n_labels)

  unary = unary_from_labels(labels, n_labels, gt_prob=gt_prob, zero_unsure=False)

  d.setUnaryEnergy(unary)
  d.addPairwiseGaussian(sxy=3, compat=3)
  d.addPairwiseBilateral(sxy=50, srgb=5, rgbim=np.ascontiguousarray(np.copy(img)), compat=10)

  q = d.inference(t)

  return np.argmax(np.array(q).reshape((n_labels, h, w)), axis=0)


def crf_inference_dlv2_label(img, labels, t=10, n_labels=21, gt_prob=0.7, pos_w=3, pos_xy_std=1, bi_w=4, bi_xy_std=67, bi_rgb_std=3):
  # Parameters from DeepLabV2 <https://github.com/kazuto1011/deeplab-pytorch/>.

  import pydensecrf.densecrf as dcrf
  from pydensecrf.utils import unary_from_labels

  h, w = img.shape[:2]

  d = dcrf.DenseCRF2D(w, h, n_labels)

  unary = unary_from_labels(labels, n_labels, gt_prob=gt_prob, zero_unsure=False)

  d.setUnaryEnergy(unary)
  d.addPairwiseGaussian(sxy=pos_xy_std, compat=pos_w)
  d.addPairwiseBilateral(sxy=bi_xy_std, srgb=bi_rgb_std, rgbim=np.ascontiguousarray(np.copy(img)), compat=bi_w)

  q = d.inference(t)

  return np.argmax(np.array(q).reshape((n_labels, h, w)), axis=0)


def crf_inference_dlv2_softmax(image, probs, t=10, pos_w=3, pos_xy_std=1, bi_w=4, bi_xy_std=67, bi_rgb_std=3):
  # Parameters from DeepLabV2 <https://github.com/kazuto1011/deeplab-pytorch/>.

  import pydensecrf.densecrf as dcrf
  from pydensecrf.utils import unary_from_softmax

  C, H, W = probs.shape

  U = unary_from_softmax(probs)
  U = np.ascontiguousarray(U)

  image = np.ascontiguousarray(image)

  d = dcrf.DenseCRF2D(W, H, C)
  d.setUnaryEnergy(U)
  d.addPairwiseGaussian(sxy=pos_xy_std, compat=pos_w)
  d.addPairwiseBilateral(sxy=bi_xy_std, srgb=bi_rgb_std, rgbim=image, compat=bi_w)

  Q = d.inference(t)
  Q = np.array(Q).reshape((C, H, W))

  return Q
