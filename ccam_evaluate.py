import argparse
import multiprocessing
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

from tools.general.io_utils import load_saliency_file, str2bool

parser = argparse.ArgumentParser()
parser.add_argument(
  '--experiment_name', default='resnet50@seed=0@nesterov@train@bg=0.20@scale=0.5,1.0,1.5,2.0@png', type=str
)
parser.add_argument("--domain", default='train', type=str)
parser.add_argument("--threshold", default=None, type=float)

parser.add_argument('--predict_mode', default='logit', type=str)
parser.add_argument('--predict_flip', default=False, type=str2bool)

parser.add_argument('--gt_dir', default='../VOCtrainval_11-May-2012/SegmentationClass', type=str)

parser.add_argument('--logfile', default='', type=str)
parser.add_argument('--comment', default='', type=str)

parser.add_argument('--mode', default='npy', type=str, choices=['npy', 'png'])
parser.add_argument('--eval_mode', default='saliency', type=str, choices=['saliency', 'segmentation'])
parser.add_argument('--min_th', default=0.05, type=float)
parser.add_argument('--max_th', default=0.50, type=float)
parser.add_argument('--step_th', default=0.05, type=float)

args = parser.parse_args()

predict_folder = './experiments/predictions/{}/'.format(args.experiment_name)
ground_truth_folder = args.gt_dir
p_mode = args.predict_mode

assert p_mode in ('logit', 'sigmoid')

args.list = './data/' + args.domain + '.txt'

if args.eval_mode == 'saliency':
  CLASSES = ['background', 'foreground']
else:
  CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
    'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
  ]

NUM_CLASSES = len(CLASSES)


def compare(start, step, TP, P, T, name_list):
  for idx in range(start, len(name_list), step):
    name = name_list[idx]

    npy_file = os.path.join(predict_folder, name + '.npy')
    png_file = os.path.join(predict_folder, name + '.png')
    label_file = os.path.join(ground_truth_folder, name + '.png')

    if os.path.exists(png_file):
      s_pred = load_saliency_file(png_file)
    elif os.path.exists(npy_file):
      data = np.load(npy_file, allow_pickle=True).item()

      if 'hr_cam' in data.keys():
        s_pred = data['hr_cam']
      elif 'rw' in data.keys():
        s_pred = data['rw']
      s_pred = s_pred
    else:
      raise FileNotFoundError(f'Cannot find .png or .npy predictions for sample {name}.')

    if p_mode == 'logit':
      s_pred = np.pad(s_pred, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.threshold)
    else:
      s_pred = np.concatenate((1 - s_pred, s_pred), axis=0)

    s_pred = np.argmax(s_pred, axis=0)

    if args.predict_flip:
      s_pred = 1 - s_pred

    y_true = np.array(Image.open(label_file))
    valid_mask = y_true < 255

    if args.eval_mode == 'saliency':
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
      fg_true = ~np.isin(y_true, [0, 255])

      # does not leak true bg:
      random_pixels = np.unique(y_true[fg_true].ravel())
      random_pixels = np.random.choice(random_pixels, size=y_true.shape)
      y_pred = fg_true * y_true + ~fg_true * random_pixels
      y_pred[~fg_pred] = 0

    mask = (y_pred == y_true) * valid_mask

    for i in range(NUM_CLASSES):
      P[i].acquire()
      P[i].value += np.sum((y_pred == i) * valid_mask)
      P[i].release()
      T[i].acquire()
      T[i].value += np.sum((y_true == i) * valid_mask)
      T[i].release()
      TP[i].acquire()
      TP[i].value += np.sum((y_true == i) * mask)
      TP[i].release()


def do_python_eval(name_list, num_cores=8):
  TP = []
  P = []
  T = []
  for i in range(NUM_CLASSES):
    TP.append(multiprocessing.Value('i', 0, lock=True))
    P.append(multiprocessing.Value('i', 0, lock=True))
    T.append(multiprocessing.Value('i', 0, lock=True))

  p_list = []
  for i in range(num_cores):
    p = multiprocessing.Process(target=compare, args=(i, num_cores, TP, P, T, name_list))
    p.start()
    p_list.append(p)
  for p in p_list:
    p.join()

  IoU = []
  T_TP = []
  P_TP = []
  FP_ALL = []
  FN_ALL = []
  for i in range(NUM_CLASSES):
    IoU.append(TP[i].value / (T[i].value + P[i].value - TP[i].value + 1e-10))
    T_TP.append(T[i].value / (TP[i].value + 1e-10))
    P_TP.append(P[i].value / (TP[i].value + 1e-10))
    FP_ALL.append((P[i].value - TP[i].value) / (T[i].value + P[i].value - TP[i].value + 1e-10))
    FN_ALL.append((T[i].value - TP[i].value) / (T[i].value + P[i].value - TP[i].value + 1e-10))

  loglist = {}
  for i in range(NUM_CLASSES):
    loglist[CLASSES[i]] = IoU[i] * 100

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


if __name__ == '__main__':
  if not os.path.exists(predict_folder):
    print(f'Predicted saliency maps folder `{predict_folder}` does not exist.', file=sys.stderr)
    exit(1)

  if not os.path.exists(ground_truth_folder):
    print(f'True saliency maps folder `{ground_truth_folder}` does not exist.', file=sys.stderr)
    exit(1)

  df = pd.read_csv(args.list, names=['filename'])
  filenames = df['filename'].values

  miou_ = threshold_ = fp_ = 0.
  iou_ = {}
  miou_history = []
  fp_history = []

  thresholds = (
    np.arange(args.min_th, args.max_th, args.step_th).tolist()
    if args.threshold is None and p_mode == 'logit' else [args.threshold]
  )

  for t in thresholds:
    args.threshold = t
    r = do_python_eval(filenames)

    print(
      f"Th={t or 0.:.3f} mIoU={r['mIoU']:.3f}% "
      f"iou=[{r['background']:.3f}, {r['miou_foreground']:.3f}] "
      f"FP={r['fp_all']:.3%}"
    )

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
