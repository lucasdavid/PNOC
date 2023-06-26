import traceback
import argparse
import multiprocessing
import os
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

import wandb
from core.datasets import get_paths_dataset
from tools.general import wandb_utils
from tools.ai.demo_utils import crf_inference_label
from tools.general.io_utils import load_saliency_file, str2bool

parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", default="resnet50@seed=0@nesterov@train@bg=0.20@scale=0.5,1.0,1.5,2.0@png", type=str)
parser.add_argument("--num_workers", default=48, type=int)
parser.add_argument("--progbar", default=False, type=str2bool)

parser.add_argument("--dataset", default="voc12", choices=["voc12", "coco14"])
parser.add_argument("--domain", default="train", type=str)
parser.add_argument("--data_dir", default="../VOCtrainval_11-May-2012/", type=str)
parser.add_argument("--max_iterations", default=None, type=int)

parser.add_argument("--threshold", default=None, type=float)

parser.add_argument("--pred_dir", default=None, type=str)
parser.add_argument("--pred_mode", default="logit", type=str, choices=["logit", "sigmoid"])
parser.add_argument("--pred_flip", default=False, type=str2bool)
parser.add_argument("--crf_t", default=0, type=int)
parser.add_argument("--crf_gt_prob", default=0.7, type=float)

parser.add_argument("--mode", default="npy", type=str, choices=["npy", "png"])
parser.add_argument("--eval_mode", default="saliency", type=str, choices=["saliency", "segmentation"])
parser.add_argument("--min_th", default=0.05, type=float)
parser.add_argument("--max_th", default=0.50, type=float)
parser.add_argument("--step_th", default=0.05, type=float)


def compare(args, dataset, classes, start, step, TP, P, T):
  compared = 0
  corrupted = []
  missing = []

  indices = range(start, len(dataset), step)
  if start == 0:
    indices = tqdm(indices, f"Evaluate t={args.threshold:.0%}", disable=not args.progbar)

  try:
    for idx in indices:
      image_id, image_path, mask_path = dataset[idx]
      pred_npy_path = os.path.join(PRED_DIR, image_id + ".npy")
      pred_png_path = os.path.join(PRED_DIR, image_id + ".png")

      if os.path.exists(pred_png_path):
        s_pred = load_saliency_file(pred_png_path)
      elif os.path.exists(pred_npy_path):
        try:
          data = np.load(pred_npy_path, allow_pickle=True).item()
        except:
          corrupted.append(image_id)
          continue

        if "hr_cam" in data.keys():
          s_pred = data["hr_cam"]
        elif "rw" in data.keys():
          s_pred = data["rw"]
        s_pred = s_pred
      else:
        missing.append(image_id)
        continue

      if PRED_MODE == "logit":
        s_pred = np.pad(s_pred, ((1, 0), (0, 0), (0, 0)), mode="constant", constant_values=args.threshold)
      else:
        s_pred = np.concatenate((1 - s_pred, s_pred), axis=0)

      s_pred = np.argmax(s_pred, axis=0)

      if args.pred_flip:
        s_pred = 1 - s_pred

      if args.crf_t:
        with Image.open(image_path) as img:
          img = np.asarray(img.convert("RGB"))
        s_pred = crf_inference_label(img, s_pred, n_labels=2, t=args.crf_t, gt_prob=args.crf_gt_prob)

      with Image.open(mask_path) as y_true:
        y_true = np.asarray(y_true)
      valid_mask = y_true < 255

      if args.eval_mode == "saliency":
        # Segmentation to saliency map:
        bg_mask = y_true == 0
        y_true = np.zeros_like(y_true)
        y_true[bg_mask] = 0
        y_true[~bg_mask] = 1
        y_true[~valid_mask] = 255
        y_pred = s_pred
      else:
        # Predicted saliency to segmentation map, assuming perfect pixel segm.:
        fg_pred = s_pred == 1
        fg_true = ~np.isin(y_true, [0, 255])  # [[0, 1], [2, 0]] --> [[0, 1], [1, 0]]

        # Do not leak true bg by simulating the prediction of a random
        # class in the label set. If it only contains bg, take any class.
        random_pixels = np.unique(y_true[fg_true].ravel())
        if not random_pixels.shape or random_pixels.shape[0] == 0:
          random_pixels = list(range(1, len(classes)))
        random_pixels = np.random.choice(random_pixels, size=y_true.shape)

        y_pred = fg_true * y_true + ~fg_true * random_pixels
        y_pred[~fg_pred] = 0  # remove all random pixels if the prediction was correct.

      mask = (y_pred == y_true) * valid_mask

      for i in range(len(classes)):
        P[i].acquire()
        P[i].value += np.sum((y_pred == i) * valid_mask)
        P[i].release()
        T[i].acquire()
        T[i].value += np.sum((y_true == i) * valid_mask)
        T[i].release()
        TP[i].acquire()
        TP[i].value += np.sum((y_true == i) * mask)
        TP[i].release()

      compared += 1

      if args.max_iterations and compared >= args.max_iterations:
        break
  except KeyboardInterrupt:
    ...

  # if start == 0:
  #   read = compared + len(missing) + len(corrupted)
  #   print(f"{compared} ({compared/read:.0%}) predictions evaluated (corrupted={len(corrupted)} missing={missing}).")
  if corrupted: print(f"{len(corrupted)} corrupted samples: {', '.join(corrupted)}", file=sys.stderr)
  if missing: print(f"{len(missing)} corrupted samples: {', '.join(missing)}", file=sys.stderr)


def do_python_eval(args, dataset, classes, num_workers=8):
  TP = []
  P = []
  T = []
  for i in range(len(classes)):
    TP.append(multiprocessing.Value("L", 0, lock=True))
    P.append(multiprocessing.Value("L", 0, lock=True))
    T.append(multiprocessing.Value("L", 0, lock=True))

  p_list = []
  for i in range(num_workers):
    p = multiprocessing.Process(target=compare, args=(args, dataset, classes, i, num_workers, TP, P, T))
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
  loglist["fp_bg"] = FP_ALL[0]
  loglist["fn_bg"] = FN_ALL[0]
  loglist["miou_foreground"] = miou_foreground
  return loglist


def run(args, dataset):
  if args.eval_mode == "saliency":
    classes = ["background", "foreground"]
  else:
    classes = ["background"] + dataset.info.classes.tolist()
  columns = ["threshold", *classes, "overall", "foreground"]
  report_iou = []

  if not os.path.exists(PRED_DIR):
    print(f"Predicted saliency maps folder `{PRED_DIR}` does not exist.", file=sys.stderr)
    exit(1)

  miou_ = threshold_ = fp_ = fn_ = 0.
  iou_ = {}
  miou_history = []
  fp_history = []

  thresholds = (
    np.arange(args.min_th, args.max_th, args.step_th).tolist()
    if args.threshold is None and PRED_MODE == "logit" else [args.threshold]
  )

  try:
    for t in thresholds:
      args.threshold = t
      r = do_python_eval(args, dataset, classes, num_workers=args.num_workers)

      print(
        f"Th={t or 0.:.3f} mIoU={r['mIoU']:.3f}% "
        f"iou=[{r['background']:.3f}, {100*r['miou_foreground']:.3f}] "
        f"FP_bg={r['fp_bg']:.3%} "
        f"FN_bg={r['fn_bg']:.3%} "
        f"FP_fg={r['fp_all']:.3%} "
        f"FN_fg={r['fn_all']:.3%}"
      )

      fp_history.append(r["fp_all"])
      miou_history.append(r["mIoU"])

      report_iou.append([t] + [r[c] for c in classes] + [r["mIoU"], r["miou_foreground"]])

      logs = {
        "evaluation/t": t,
        "evaluation/miou": r["mIoU"],
        "evaluation/miou_fg": r["miou_foreground"],
        "evaluation/miou_bg": r["background"],
        "evaluation/fp": r["fp_all"],
        "evaluation/fn": r["fn_all"],
        "evaluation/iou": wandb.Table(columns=columns, data=report_iou)
      }

      if r["mIoU"] > miou_:
        threshold_ = t
        miou_ = r["mIoU"]
        fp_ = r["fp_all"]
        fn_ = r["fn_all"]
        iou_ = r

      wandb.log(logs)

  except KeyboardInterrupt:
    print("\ninterrupted")

  print(
    f"Best Th={threshold_ or 0.:.3f} mIoU={miou_:.5f}% FP={fp_:.3%} FN={fn_:.3%}",
    "-" * 80,
    *(f"{k:<12}\t{v:.5f}" for k, v in iou_.items()),
    "-" * 80,
    sep="\n"
  )

  wandb.run.summary[f"evaluation/best_t"] = threshold_
  wandb.run.summary[f"evaluation/best_miou"] = miou_
  wandb.run.summary[f"evaluation/best_fp"] = fp_


if __name__ == "__main__":

  args = parser.parse_args()
  TAG = args.experiment_name
  PRED_MODE = args.pred_mode
  PRED_DIR = args.pred_dir or f"./experiments/predictions/{args.experiment_name}/"

  wb_run = wandb_utils.setup(
    TAG,
    args,
    job_type="evaluation-saliency",
  )
  wandb.define_metric("evaluation/t")

  dataset = get_paths_dataset(args.dataset, args.data_dir, args.domain)
  run(args, dataset)

  wb_run.finish()
