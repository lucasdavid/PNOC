import argparse
import multiprocessing
import os
import sys

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

import datasets
import wandb
from tqdm import tqdm

from tools.ai.demo_utils import crf_inference_dlv2_softmax, crf_inference_label
from tools.ai.log_utils import log_config
from tools.general import wandb_utils
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
parser.add_argument("--sal_mode", default="saliency", type=str, choices=("saliency", "segmentation"))
parser.add_argument("--sal_threshold", default=None, type=float)

parser.add_argument("--crf_t", default=0, type=int)
parser.add_argument("--crf_gt_prob", default=0.7, type=float)

parser.add_argument("--mode", default="npy", type=str, choices=["png", "npy", "rw", "deeplab-pytorch"])
parser.add_argument("--threshold", default=None, type=float)
parser.add_argument("--min_th", default=0.05, type=float)
parser.add_argument("--max_th", default=0.81, type=float)
parser.add_argument("--step_th", default=0.05, type=float)
parser.add_argument("--ignore_bg_cam", default=False, type=str2bool)
parser.add_argument("--force_align", default=True, type=str2bool)


def compare(dataset: datasets.PathsDataset, classes, start, step, TP, P, T):
  compared = 0
  corrupted = []

  steps = range(start, len(dataset), step)

  if args.verbose > 1 and start == 0:
    steps = tqdm(steps, mininterval=2.0)

  try:
    for idx in steps:
      image_id, image_path, mask_path = dataset[idx]

      y_true = np.asarray(dataset.data_source.get_mask(image_id))
      valid_mask = y_true < 255

      npy_file = os.path.join(PRED_DIR, image_id + ".npy")
      png_file = os.path.join(PRED_DIR, image_id + ".png")
      sal_file = os.path.join(SAL_DIR, image_id + ".png") if SAL_DIR else None

      if args.mode == "png":
        try:
          with Image.open(png_file) as y_pred:
            y_pred = np.asarray(y_pred)
        except (UnidentifiedImageError, FileNotFoundError):
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

      elif args.mode == "deeplab-pytorch":
        cam, keys, prob = load_dlv2_seg_file(npy_file, sizes=y_true.shape)
        prob = prob[keys, :, :]

      elif args.mode == "deeplab-pytorch-threshold":
        cam, keys, prob = load_dlv2_seg_file(npy_file, sizes=y_true.shape)
        prob[0, ...] = args.threshold
        cam = prob.argmax(0)
        prob = prob[keys, :, :]

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

      if args.force_align and y_pred.shape != y_true.shape:
        H, W = y_true.shape
        if args.verbose > 2:
          print(f"Prediction shape {y_pred.shape} diverges from ground-truth {y_true.shape} for sample {image_id}.")
        y_pred = cv2.resize(y_pred, (W, H), interpolation=cv2.INTER_NEAREST)

      for i in range(len(classes)):
        P[i].acquire()
        P[i].value += np.sum((y_pred == i) * valid_mask)
        P[i].release()
        T[i].acquire()
        T[i].value += np.sum((y_true == i) * valid_mask)
        T[i].release()
        TP[i].acquire()
        TP[i].value += np.sum((y_true == i) * (y_pred == y_true) * valid_mask)
        TP[i].release()

      compared += 1

  except KeyboardInterrupt:
    ...

  if start == 0 and args.verbose > 0 and corrupted:
    print(f"Corrupted files ({len(corrupted)}): {' '.join(corrupted[:30])}")
  if start == 0 and args.verbose > 0 and corrupted:
    read = compared + len(corrupted)
    print(f"{compared} ({compared/read:.3%}) predictions evaluated.")


def do_python_eval(dataset, classes, num_workers=8):
  TP = []
  P = []
  T = []
  for i in range(len(classes)):
    TP.append(multiprocessing.Value("L", 0, lock=True))
    P.append(multiprocessing.Value("L", 0, lock=True))
    T.append(multiprocessing.Value("L", 0, lock=True))

  p_list = []
  for i in range(num_workers):
    p = multiprocessing.Process(target=compare, args=(dataset, classes, i, num_workers, TP, P, T))
    p.start()
    p_list.append(p)
  for p in p_list:
    p.join()

  IoU = []
  T_TP = []
  P_TP = []
  FP_ALL = []
  FN_ALL = []
  for i in range(len(classes)):
    IoU.append(TP[i].value / (T[i].value + P[i].value - TP[i].value + 1e-10))
    T_TP.append(T[i].value / (TP[i].value + 1e-10))
    P_TP.append(P[i].value / (TP[i].value + 1e-10))
    FP_ALL.append((P[i].value - TP[i].value) / (T[i].value + P[i].value - TP[i].value + 1e-10))
    FN_ALL.append((T[i].value - TP[i].value) / (T[i].value + P[i].value - TP[i].value + 1e-10))

  loglist = {}
  for i in range(len(classes)):
    loglist[classes[i]] = IoU[i] * 100

  miou = np.mean(np.array(IoU))
  t_tp = np.mean(np.array(T_TP)[1:])
  p_tp = np.mean(np.array(P_TP)[1:])
  fp_all = np.mean(np.array(FP_ALL)[1:])
  fn_all = np.mean(np.array(FN_ALL)[1:])
  miou_foreground = np.mean(np.array(IoU)[1:])
  loglist["mIoU"] = miou * 100
  loglist["t_tp"] = t_tp
  loglist["p_tp"] = p_tp
  loglist["fp_all"] = fp_all
  loglist["fn_all"] = fn_all
  loglist["miou_foreground"] = miou_foreground * 100
  return loglist


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

  thresholds = np.arange(args.min_th, args.max_th, args.step_th).tolist()
  if args.threshold or SAL_DIR or args.mode in ("png", "deeplab-pytorch"):
    # No threshold-range used in these cases. Assume the default (None) or the passed one.
    thresholds = [args.threshold]

  try:
    for i, t in enumerate(thresholds):
      args.threshold = t
      r = do_python_eval(dataset, classes, num_workers=args.num_workers)

      if args.verbose: print(f"Th={t or 0.:.3f} mIoU={r['mIoU']:.3f}% FP={r['fp_all']:.3%}")

      fp_history.append(r["fp_all"])
      miou_history.append(r["mIoU"])

      report_iou.append([t] + [r[c] for c in classes] + [r["mIoU"], r["miou_foreground"]])

      if miou_ is None or r["mIoU"] > miou_:
        index_ = i
        threshold_ = t
        miou_ = r["mIoU"]
        fp_ = r["fp_all"]
        iou_ = r

      wandb.log({
        "evaluation/t": t,
        "evaluation/miou": r["mIoU"],
        "evaluation/miou_fg": r["miou_foreground"],
        "evaluation/miou_bg": r[bg_class],
        "evaluation/fp": r["fp_all"],
        "evaluation/fn": r["fn_all"],
        "evaluation/iou": [r[c] for c in classes],
      })

  except KeyboardInterrupt:
    print("\ninterrupted")

  if args.verbose: print(
    f"Best Th={threshold_ or 0.:.3f} mIoU={miou_:.5f}% FP={fp_:.3%}",
    "-" * 80,
    *(f"{k:<12}\t{v:.5f}" for k, v in iou_.items()),
    "-" * 80,
    sep="\n"
  )

  wandb.run.summary[f"evaluation/iou"] = wandb.Table(columns=columns, data=report_iou)

  wandb.run.summary[f"evaluation/best_t"] = threshold_
  wandb.run.summary[f"evaluation/best_miou"] = miou_
  wandb.run.summary[f"evaluation/best_fp"] = fp_
  wandb.run.summary[f"evaluation/best_iou"] = wandb.Table(columns=columns, data=[report_iou[index_]])

  if args.mode == "rw":
    a_over = 1.60
    a_under = 0.60

    fp_over = fp_ * a_over
    fp_under = fp_ * a_under

    if args.verbose: print("Over FP : {:.4f}, Under FP : {:.4f}".format(fp_over, fp_under))

    over_loss_list = [np.abs(FP - fp_over) for FP in fp_history]
    under_loss_list = [np.abs(FP - fp_under) for FP in fp_history]

    over_index = np.argmin(over_loss_list)
    t_over = thresholds[over_index]
    miou_over = miou_history[over_index]
    fp_over = fp_history[over_index]

    under_index = np.argmin(under_loss_list)
    t_under = thresholds[under_index]
    miou_under = miou_history[under_index]
    fp_under = fp_history[under_index]

    if args.verbose: print("Best Th={:.2f}, mIoU={:.3f}%, FP={:.4f}".format(threshold_ or 0., miou_, fp_))
    if args.verbose: print("Over Th={:.2f}, mIoU={:.3f}%, FP={:.4f}".format(t_over or 0., miou_over, fp_over))
    if args.verbose: print("Under Th={:.2f}, mIoU={:.3f}%, FP={:.4f}".format(t_under or 0., miou_under, fp_under))

    wandb.run.summary[f"evaluation/over_t"] = t_over
    wandb.run.summary[f"evaluation/over_miou"] = miou_over
    wandb.run.summary[f"evaluation/over_fp"] = fp_over

    wandb.run.summary[f"evaluation/under_t"] = t_under
    wandb.run.summary[f"evaluation/under_miou"] = miou_under
    wandb.run.summary[f"evaluation/under_fp"] = fp_under


if __name__ == "__main__":
  args = parser.parse_args()
  TAG = args.experiment_name
  PRED_DIR = args.pred_dir or f"./experiments/predictions/{args.experiment_name}/"
  SAL_DIR = args.sal_dir

  if not os.path.exists(PRED_DIR) or not os.listdir(PRED_DIR):
    raise ValueError(f"Predictions cannot be found at `{PRED_DIR}`. Directory does not exist or is empty.")

  wb_run = wandb_utils.setup(TAG, args, job_type="evaluation")
  wandb.define_metric("evaluation/t")
  log_config(vars(args), TAG)

  ds = datasets.custom_data_source(args.dataset, args.data_dir, args.domain)
  dataset = datasets.PathsDataset(ds, ignore_bg_images=False)

  try:
    run(args, dataset)
  except KeyboardInterrupt:
    print("\ninterrupted")

  wb_run.finish()
