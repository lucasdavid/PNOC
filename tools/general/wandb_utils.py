import cv2
import numpy as np

import wandb
from datasets import imagenet_stats
from tools.ai.demo_utils import colormap, denormalize


def setup(name, config, job_type="train", tags=None):
  wb_run = wandb.init(
    name=name,
    job_type=job_type,
    entity="lerdl",
    project="research-wsss",
    config=config,
    tags=tags,
  )

  return wb_run


def cams_to_wb_images(images, cams):
  samples = min(len(images), 8)
  stats = imagenet_stats()
  cams = cams.max(-1)

  wb_images, wb_cams = [], []

  for b in range(samples):
    image = denormalize(images[b], *stats)
    img = image[..., ::-1]
    cam = colormap(cams[b], img.shape)
    cam = cv2.addWeighted(img, 0.5, cam, 0.5, 0)
    cam = cam[..., ::-1]

    wb_images.append(wandb.Image(image))
    wb_cams.append(wandb.Image(cam))

  return wb_images, wb_cams


def masks_to_wb_images(images, masks, preds, classes):
  indices = dict(enumerate(classes.tolist()))
  samples = min(len(images), 8)
  stats = imagenet_stats()

  if classes[0] != "background":
    # If BG is not first class, then display all classes.
    masks = masks + 1
    preds = preds + 1
    indices = {i+1: c for i, c in indices.items()}

  return [
    wandb.Image(denormalize(images[b], *stats), masks={
      "ground_truth": {
        "mask_data": masks[b],
        "class_labels": {i: indices[i] for i in np.unique(masks.ravel()).tolist()},
      },
      "predictions": {
        "mask_data": preds[b],
        "class_labels": {i: indices[i] for i in np.unique(preds.ravel()).tolist()},
      },
    })
    for b in range(samples)
  ]


def log_cams(
  ids,
  images,
  targets,
  cams,
  predictions,
  classes,
  oc_predictions=None,
  commit=False,
):
  wb_images, wb_cams = cams_to_wb_images(images, cams)
  wb_targets = _predictions_to_names(targets, classes)
  wb_predics = _predictions_to_names(predictions, classes)
  wb_oc_pred = _predictions_to_names(oc_predictions, classes)

  columns = ("Id", "Image", "CAM", "Labels", "CG Predictions", "OC Predictions")
  entries = (ids, wb_images, wb_cams, wb_targets, wb_predics, wb_oc_pred)

  columns_v, entries_v = [], []

  for c, e in zip(columns, entries):
    if e is not None:
      columns_v += [c]
      entries_v += [e]

  data = [list(row) for row in zip(*entries_v)]
  table = wandb.Table(columns=columns_v, data=data)

  wandb.log({"val/predictions": table, "val/cams": wb_cams}, commit=commit)


def log_masks(
  ids,
  images,
  targets,
  masks,
  preds,
  classes,
  commit=False,
):
  wb_images = masks_to_wb_images(images, masks, preds, classes)
  wb_targets = _predictions_to_names(targets, classes)

  columns = ("Id", "Image", "Labels")
  entries = (ids, wb_images, wb_targets)

  columns_v, entries_v = [], []

  for c, e in zip(columns, entries):
    if e is not None:
      columns_v += [c]
      entries_v += [e]

  data = [list(row) for row in zip(*entries_v)]
  table = wandb.Table(columns=columns_v, data=data)

  wandb.log({"val/predictions": table, "val/masks": wb_images}, commit=commit)


def _predictions_to_names(predictions, classes, threshold=0.5):
  if predictions is not None:
    return [classes[p > threshold].tolist() for p in predictions]
