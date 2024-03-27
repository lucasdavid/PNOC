import argparse
import multiprocessing
import os
import sys

import numpy as np
from PIL import Image, UnidentifiedImageError

import datasets
from tqdm import tqdm

from tools.ai.demo_utils import crf_inference_dlv2_softmax, crf_inference_label
from tools.ai.log_utils import log_config
from tools.general.io_utils import load_cam_file, load_saliency_file, load_dlv2_seg_file, str2bool

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", type=str, required=True)
parser.add_argument("--num_workers", default=48, type=int)
parser.add_argument("--verbose", default=1, type=int)

parser.add_argument("--dataset", default="voc12", choices=datasets.DATASOURCES)
parser.add_argument("--domain", default="train", type=str)
parser.add_argument("--data_dir", default="../VOCtrainval_11-May-2012/", type=str)
parser.add_argument("--pred_dir", default="", type=str)
parser.add_argument("--sal_dir", default=None, type=str)
parser.add_argument("--output_dir", default=None, type=str)

parser.add_argument("--sal_mode", default="saliency", type=str, choices=("saliency", "segmentation"))
parser.add_argument("--sal_threshold", default=None, type=float)

parser.add_argument("--crf_t", default=0, type=int)
parser.add_argument("--crf_gt_prob", default=0.7, type=float)

parser.add_argument("--mode", default="npy", type=str, choices=["png", "npy", "rw"])
parser.add_argument("--threshold", default=0.25, type=float)
parser.add_argument("--ignore_bg_cam", default=False, type=str2bool)


def compare(dataset: datasets.PathsDataset, classes, start, step):
  compared = 0
  corrupted = []

  steps = range(start, len(dataset), step)

  if start == 0:
    steps = tqdm(steps, mininterval=2.0)

  for idx in steps:
    image_id, image_path, mask_path = dataset[idx]

    # y_true = np.asarray(dataset.data_source.get_mask(image_id))

    npy_file = os.path.join(PRED_DIR, image_id + ".npy")
    png_file = os.path.join(PRED_DIR, image_id + ".png")
    sal_file = os.path.join(SAL_DIR, image_id + ".png") if SAL_DIR else None
    out_file = os.path.join(OUT_DIR, image_id + ".png")

    if args.mode == "png":
      try:
        with Image.open(png_file) as y_pred:
          y_pred = np.asarray(y_pred)
      except UnidentifiedImageError:
        corrupted.append(image_id)
        continue

      keys, cam = np.unique(y_pred, return_inverse=True)
      cam = cam.reshape(y_pred.shape)
      prob = None

    elif args.mode in ("npy", "rw"):
      cam, keys = load_cam_file(npy_file)

      if len(cam) == len(keys) and args.ignore_bg_cam:
        # Background map is here, but we want to ignore it in favor of thresholding.
        cam = cam[1:]

      if len(cam) < len(keys):  # background map is missing. Perform thresholding.
        if not sal_file:
          cam = np.pad(cam, ((1, 0), (0, 0), (0, 0)), mode="constant", constant_values=args.threshold)
        else:
          sal = load_saliency_file(sal_file, args.sal_mode)
          bg = ((sal < args.sal_threshold).astype(float) if args.sal_threshold else (1 - sal))
          cam = np.concatenate((bg, cam), axis=0)

      prob = cam
      cam = np.argmax(cam, axis=0)

    # elif args.mode == "deeplab-pytorch":
    #   cam, keys, prob = load_dlv2_seg_file(npy_file, sizes=y_true.shape)

    # elif args.mode == "deeplab-pytorch-threshold":
    #   cam, keys, prob = load_dlv2_seg_file(npy_file, sizes=y_true.shape)
    #   prob[0, ...] = args.threshold
    #   cam = prob.argmax(0)

    if args.crf_t:
      with dataset.data_source.get_image(image_id) as img:
        img = np.asarray(img).astype(np.uint8)

        if prob is not None and args.crf_gt_prob == 1.0:
          # DeepLab-pytorch's CRF
          prob = crf_inference_dlv2_softmax(img, prob, t=args.crf_t)
          cam = prob.argmax(0)
        else:
          cam = crf_inference_label(img, cam, n_labels=max(len(keys), 2), t=args.crf_t, gt_prob=args.crf_gt_prob)

    y_pred = keys[cam]

    try:
      with Image.fromarray(y_pred.astype(np.uint8)) as p:
        p.save(out_file)
    except:
      if os.path.exists(out_file):
        os.remove(out_file)
      raise


def run(args, dataset: datasets.PathsDataset):
  classes = dataset.info.classes.tolist()
  bg_class = classes[dataset.info.bg_class]

  columns = ["threshold", *classes, "overall", "foreground"]
  report_iou = []

  index_ = miou_ = None
  threshold_ = fp_ = 0.
  iou_ = {}
  miou_history = []
  fp_history = []

  p_list = []
  for i in range(args.num_workers):
    p = multiprocessing.Process(target=compare, args=(dataset, classes, i, args.num_workers))
    p.start()
    p_list.append(p)
  for p in p_list:
    p.join()


if __name__ == "__main__":
  args = parser.parse_args()
  TAG = args.experiment_name
  PRED_DIR = args.pred_dir or f"./experiments/predictions/{TAG}/masks"
  OUT_DIR = args.output_dir or f"./experiments/predictions/{TAG}/pseudo_labels"
  SAL_DIR = args.sal_dir

  if not os.path.exists(PRED_DIR) or not os.listdir(PRED_DIR):
    raise ValueError(f"Predictions cannot be found at `{PRED_DIR}`. Directory does not exist or is empty.")
  
  if os.path.exists(OUT_DIR) and os.listdir(OUT_DIR):
    raise ValueError(f"Output directory `{PRED_DIR}` already exists, and contains files. Directory must not exist or be empty.")
  os.makedirs(OUT_DIR, exist_ok=True)

  log_config(vars(args), TAG)

  ds = datasets.custom_data_source(args.dataset, args.data_dir, args.domain)
  dataset = datasets.PathsDataset(ds, ignore_bg_images=False)

  try:
    run(args, dataset)
  except KeyboardInterrupt:
    print("\ninterrupted")
