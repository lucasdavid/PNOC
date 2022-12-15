import cv2
import numpy as np
import wandb

from tools.ai.demo_utils import colormap, denormalize
from tools.general.txt_utils import add_txt
from core.datasets import imagenet_stats


def cams_to_wb_images(images, cams):
  results = []
  mu_std = imagenet_stats()
  cams = cams.max(-1)

  for b in range(8):
    image = images[b]
    cam = cams[b]

    image = denormalize(image, *mu_std)[..., ::-1]
    h, w, c = image.shape

    cam = (cam.clip(0, 1) * 255).astype(np.uint8)
    cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    cam = colormap(cam)

    image = cv2.addWeighted(image, 0.5, cam, 0.5, 0)[..., ::-1]
    image = image.astype(np.float32) / 255.

    results.append(wandb.Image(image))

  return results


def log_evaluation_table(images, targets, cg_predictions, oc_predictions, cams):
  ...
