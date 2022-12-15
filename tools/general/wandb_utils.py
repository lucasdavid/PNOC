import cv2
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
    classes,
    images,
    targets,
    cams,
    predictions,
    oc_predictions=None,
    commit=False,
):
  wb_images, wb_cams = cams_to_wb_images(images, cams)
  wb_targets = _predictions_to_names(targets, classes)
  wb_predics = _predictions_to_names(predictions, classes)
  wb_oc_pred = _predictions_to_names(oc_predictions, classes)

  columns = ("Image", "CAM", "Labels", "CG Predictions", "OC Predictions")
  entries = (wb_images, wb_cams, wb_targets, wb_predics, wb_oc_pred)

  table = wandb.Table()
  for c, d in zip(columns, entries):
    if d is not None:
      table.add_column(c, d)

  wandb.log({"val/predictions": table}, commit=commit)


def _predictions_to_names(predictions, classes, threshold=0.5):
  if predictions is not None:
    return [classes[p > threshold].tolist() for p in predictions]
