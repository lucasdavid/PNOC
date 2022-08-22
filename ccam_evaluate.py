import argparse
import multiprocessing
import os

import numpy as np
import pandas as pd
from PIL import Image

from tools.general.io_utils import str2bool

parser = argparse.ArgumentParser()
parser.add_argument(
  '--experiment_name', default='resnet50@seed=0@nesterov@train@bg=0.20@scale=0.5,1.0,1.5,2.0@png', type=str
)
parser.add_argument("--domain", default='train', type=str)
parser.add_argument("--threshold", default=None, type=float)

parser.add_argument('--predict_mode', default='logit', type=str)
parser.add_argument('--predict_flip', default=True, type=str2bool)

parser.add_argument('--gt_dir', default='../VOCtrainval_11-May-2012/SegmentationClass', type=str)

parser.add_argument('--logfile', default='', type=str)
parser.add_argument('--comment', default='', type=str)

parser.add_argument('--mode', default='npy', type=str)  # png, rw
parser.add_argument('--min_th', default=0.05, type=float)
parser.add_argument('--max_th', default=0.50, type=float)
parser.add_argument('--step_th', default=0.05, type=float)

args = parser.parse_args()

predict_folder = './experiments/predictions/{}/'.format(args.experiment_name)
gt_dir = args.gt_dir
p_mode = args.predict_mode

assert p_mode in ('logit', 'sigmoid')

args.list = './data/' + args.domain + '.txt'

CLASSES = ['background', 'foreground']
NUM_CLASSES = len(CLASSES)


def compare(start, step, TP, P, T, name_list):
  for idx in range(start, len(name_list), step):
    name = name_list[idx]

    npy_file = os.path.join(predict_folder, name + '.npy')
    png_file = os.path.join(predict_folder, name + '.png')
    label_file = os.path.join(gt_dir, name + '.png')

    if os.path.exists(png_file):
      y_pred = np.array(Image.open(predict_folder + name + '.png')) / 255.
      y_pred = y_pred.astype(np.uint8)
    elif os.path.exists(npy_file):
      data = np.load(npy_file, allow_pickle=True).item()

      if 'hr_cam' in data.keys():
        cams = data['hr_cam']
      elif 'rw' in data.keys():
        cams = data['rw']
    
      if p_mode == 'logit':
        cams = np.pad(cams, ((0, 1), (0, 0), (0, 0)), mode='constant', constant_values=args.threshold)
      else:
        cams = np.concatenate((cams, 1-cams), axis=0)

      y_pred = data['keys'][np.argmax(cams, axis=0)]
    else:
      raise FileNotFoundError(f'Cannot find .png or .npy predictions for sample {name}.')
    
    if args.predict_flip:
      y_pred = 1 - y_pred

    y_true = np.array(Image.open(label_file))
    valid_mask = y_true < 255
    bg_mask = y_true == 0

    y_true = np.zeros_like(y_true)
    y_true[bg_mask] = 0
    y_true[~bg_mask] = 1
    y_true[~valid_mask] = 255

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
  df = pd.read_csv(args.list, names=['filename'])
  filenames = df['filename'].values

  miou_ = threshold_ = fp_ = 0.
  iou_ = {}
  miou_history = []
  fp_history = []

  thresholds = (
    np.arange(args.min_th, args.max_th, args.step_th).tolist()
    if args.threshold is None and p_mode == 'logit' and args.mode != 'png'
    else [args.threshold]
  )

  for t in thresholds:
    args.threshold = t
    r = do_python_eval(filenames)

    print(f"Th={t or 0.:.3f} mIoU={r['mIoU']:.3f}% FP={r['fp_all']:.3%} iou=[{r['background']:.3f}, {r['foreground']:.3f}]")

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
