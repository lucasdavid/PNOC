import argparse
import multiprocessing
import os
import sys
from core.datasets import get_segmentation_evaluation_dataset

import numpy as np
import pandas as pd
from PIL import Image

from tools.ai.demo_utils import crf_inference_label
from tools.general.io_utils import load_saliency_file

SAL_MODES = ('saliency', 'segmentation')

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', type=str, required=True)
parser.add_argument('--num_workers', default=24, type=int)
parser.add_argument("--threshold", default=None, type=float)

parser.add_argument('--dataset', default='voc12', choices=['voc12', 'coco14'])
parser.add_argument("--domain", default='train', type=str)
parser.add_argument('--data_dir', default='../VOCtrainval_11-May-2012/', type=str)
parser.add_argument("--pred_dir", default='', type=str)
parser.add_argument('--sal_dir', default=None, type=str)
parser.add_argument('--sal_mode', default='saliency', type=str, choices=SAL_MODES)
parser.add_argument("--sal_threshold", default=None, type=float)

parser.add_argument('--crf_t', default=0, type=int)
parser.add_argument('--crf_gt_prob', default=0.7, type=float)

parser.add_argument('--mode', default='npy', type=str)  # png, rw
parser.add_argument('--min_th', default=0.05, type=float)
parser.add_argument('--max_th', default=0.50, type=float)
parser.add_argument('--step_th', default=0.05, type=float)


def compare(dataset, classes, start, step, TP, P, T):
  skipped = 0
  for idx in range(start, len(dataset), step):
    image_id, image_path, mask_path = dataset[idx]

    npy_file = os.path.join(PRED_DIR, image_id + '.npy')
    png_file = os.path.join(PRED_DIR, image_id + '.png')
    sal_file = os.path.join(SAL_DIR, image_id + '.png') if SAL_DIR else None

    if os.path.exists(png_file):
      y_pred = np.array(Image.open(PRED_DIR + image_id + '.png'))

      keys, cam = np.unique(y_pred, return_inverse=True)
      cam = cam.reshape(y_pred.shape)

    elif os.path.exists(npy_file):
      try:
        data = np.load(npy_file, allow_pickle=True).item()
      except:
        print(f'  {image_id}.npy is corrupted', file=sys.stderr)
        continue

      keys = data['keys']

      if 'hr_cam' in data.keys():
        cam = data['hr_cam']
      elif 'rw' in data.keys():
        cam = data['rw']

      if sal_file:
        sal = load_saliency_file(sal_file, args.sal_mode)
        bg = (
          (sal < args.sal_threshold).astype(float)
          if args.sal_threshold
          else (1 - sal)
        )

        cam = np.concatenate((bg, cam), axis=0)
      else:
        cam = np.pad(cam, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.threshold)

      cam = np.argmax(cam, axis=0)
    else:
      skipped += 1
      continue

    if args.crf_t:
      with Image.open(image_path) as img:
        cam = crf_inference_label(np.asarray(img), cam, n_labels=max(len(keys), 2), t=args.crf_t, gt_prob=args.crf_gt_prob)

    y_pred = keys[cam]

    with Image.open(mask_path) as y_true:
      y_true = np.array(y_true)

    valid_mask = y_true < 255
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

  if skipped:
    print(f"  {skipped} files were skipped because their predictions were not found")


def do_python_eval(dataset, classes, num_workers=8):
  TP = []
  P = []
  T = []
  for i in range(len(classes)):
    TP.append(multiprocessing.Value('i', 0, lock=True))
    P.append(multiprocessing.Value('i', 0, lock=True))
    T.append(multiprocessing.Value('i', 0, lock=True))

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
  loglist['mIoU'] = miou * 100
  loglist['t_tp'] = t_tp
  loglist['p_tp'] = p_tp
  loglist['fp_all'] = fp_all
  loglist['fn_all'] = fn_all
  loglist['miou_foreground'] = miou_foreground
  return loglist

def run(args, dataset):
  classes = ['background'] + dataset.info.classes.tolist()

  miou_ = threshold_ = fp_ = 0.
  iou_ = {}
  miou_history = []
  fp_history = []

  thresholds = (
    np.arange(args.min_th, args.max_th, args.step_th).tolist()
    if args.threshold is None and SAL_DIR is None and args.mode != 'png' else [args.threshold]
  )

  for t in thresholds:
    args.threshold = t
    r = do_python_eval(dataset, classes, num_workers=args.num_workers)

    print(f"Th={t or 0.:.3f} mIoU={r['mIoU']:.3f}% FP={r['fp_all']:.3%}")

    fp_history.append(r['fp_all'])
    miou_history.append(r['mIoU'])

    if r['mIoU'] > miou_:
      threshold_ = t
      miou_ = r['mIoU']
      fp_ = r['fp_all']
      iou_ = r

  print(
    f'Best Th={threshold_ or 0.:.3f} mIoU={miou_:.5f}% FP={fp_:.3%}',
    '-' * 80,
    *(f'{k:<12}\t{v:.5f}' for k, v in iou_.items()),
    '-' * 80,
    sep='\n'
  )

  if args.mode == 'rw':
    a_over = 1.60
    a_under = 0.60

    fp_over = fp_ * a_over
    fp_under = fp_ * a_under

    print('Over FP : {:.4f}, Under FP : {:.4f}'.format(fp_over, fp_under))

    over_loss_list = [np.abs(FP - fp_over) for FP in fp_history]
    under_loss_list = [np.abs(FP - fp_under) for FP in fp_history]

    over_index = np.argmin(over_loss_list)
    over_th = thresholds[over_index]
    over_mIoU = miou_history[over_index]
    fp_over = fp_history[over_index]

    under_index = np.argmin(under_loss_list)
    under_th = thresholds[under_index]
    under_mIoU = miou_history[under_index]
    fp_under = fp_history[under_index]

    print('Best Th={:.2f}, mIoU={:.3f}%, FP={:.4f}'.format(threshold_ or 0., miou_, fp_))
    print('Over Th={:.2f}, mIoU={:.3f}%, FP={:.4f}'.format(over_th or 0., over_mIoU, fp_over))
    print('Under Th={:.2f}, mIoU={:.3f}%, FP={:.4f}'.format(under_th or 0., under_mIoU, fp_under))


if __name__ == '__main__':
  args = parser.parse_args()

  PRED_DIR = args.pred_dir or f'./experiments/predictions/{args.experiment_name}/'
  SAL_DIR = args.sal_dir

  dataset = get_segmentation_evaluation_dataset(args.dataset, args.data_dir, args.domain)

  try:
    run(args, dataset)
  except KeyboardInterrupt:
    ...
