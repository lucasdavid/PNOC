import cv2
import numpy as np
import wandb

from tools.ai.demo_utils import colormap, denormalize
from tools.general.txt_utils import add_txt
from core.datasets import imagenet_stats


def cams_to_wb_images(images, cams):
  wb_images, wb_cams = [], []

  mu_std = imagenet_stats()
  cams = cams.max(-1)

  for b in range(8):
    image = denormalize(images[b], *mu_std)[..., ::-1]
    cam = colormap(cams[b], image.shape)
    cam = cv2.addWeighted(image, 0.5, cam, 0.5, 0)

    wb_images.append(wandb.Image(image))
    wb_cams.append(wandb.Image(cam))

  return wb_images, wb_cams


def log_cams(
    images,
    targets,
    predictions,
    oc_predictions,
    cams,
    classes,
    commit=False,
):
  wb_images, wb_cams = cams_to_wb_images(images, cams)
  
  wb_targets = [classes[t > 0.5].tolist() for t in targets]
  wb_predics = [classes[p > 0.5].tolist() for p in predictions]
  wb_oc_preds = [classes[p > 0.5].tolist() for p in oc_predictions]

  table = wandb.Table(
    columns=["Image", "CAM", "Labels", "CG Predictions", "OC Predictions"],
    data=list(zip(wb_images, wb_cams, wb_targets, wb_predics, wb_oc_preds))
  )

  wandb.log({"val/predictions": table}, commit=commit)
