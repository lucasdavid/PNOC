import argparse
import multiprocessing
import os

import torch
import numpy as np
import pandas as pd
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument(
  '--experiment_name', default='resnet50@seed=0@nesterov@train@bg=0.20@scale=0.5,1.0,1.5,2.0@png', type=str
)
parser.add_argument("--domain", default='train', type=str)
parser.add_argument("--threshold", default=None, type=float)

parser.add_argument("--predict_dir", default='', type=str)
parser.add_argument('--gt_dir', default='../VOCtrainval_11-May-2012/SegmentationClass', type=str)

parser.add_argument('--logfile', default='', type=str)
parser.add_argument('--comment', default='', type=str)

parser.add_argument('--mode', default='npy', type=str)  # png, rw
parser.add_argument('--min_th', default=0.05, type=float)
parser.add_argument('--max_th', default=0.50, type=float)
parser.add_argument('--step_th', default=0.05, type=float)

args = parser.parse_args()

predict_folder = './experiments/predictions/{}/'.format(args.experiment_name)
gt_folder = args.gt_dir

args.list = './data/' + args.domain + '.txt'
args.predict_dir = predict_folder

categories = [
  'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
  'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
num_cls = len(categories)


def compare(start, step, TP, P, T, name_list):
  for idx in range(start, len(name_list), step):
    name = name_list[idx]

    npy_file = os.path.join(predict_folder, name + '.npy')
    label_file = os.path.join(gt_folder, name + '.png')

    if os.path.exists(npy_file):
      data = np.load(npy_file, allow_pickle=True).item()

      if 'hr_cam' in data.keys():
        cams = data['hr_cam']
        cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.threshold)
      elif 'rw' in data.keys():
        cams = data['rw']
        cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.threshold)

      keys = data['keys']
      pred = keys[np.argmax(cams, axis=0)]
    else:
      pred = np.array(Image.open(predict_folder + name + '.png'))

    gt = np.array(Image.open(label_file))

    cal = gt < 255
    mask = (pred == gt) * cal

    for i in range(num_cls):
      P[i].acquire()
      P[i].value += np.sum((pred == i) * cal)
      P[i].release()
      T[i].acquire()
      T[i].value += np.sum((gt == i) * cal)
      T[i].release()
      TP[i].acquire()
      TP[i].value += np.sum((gt == i) * mask)
      TP[i].release()


def do_python_eval(predict_folder, gt_folder, name_list, num_cls=21, num_cores=8):
  TP = []
  P = []
  T = []
  for i in range(num_cls):
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
  for i in range(num_cls):
    IoU.append(TP[i].value / (T[i].value + P[i].value - TP[i].value + 1e-10))
    T_TP.append(T[i].value / (TP[i].value + 1e-10))
    P_TP.append(P[i].value / (TP[i].value + 1e-10))
    FP_ALL.append((P[i].value - TP[i].value) / (T[i].value + P[i].value - TP[i].value + 1e-10))
    FN_ALL.append((T[i].value - TP[i].value) / (T[i].value + P[i].value - TP[i].value + 1e-10))

  loglist = {}
  for i in range(num_cls):
    # if i%2 != 1:
    #     print('%11s:%7.3f%%'%(categories[i],IoU[i]*100),end='\t')
    # else:
    #     print('%11s:%7.3f%%'%(categories[i],IoU[i]*100))
    loglist[categories[i]] = IoU[i] * 100

  miou = np.mean(np.array(IoU))
  t_tp = np.mean(np.array(T_TP)[1:])
  p_tp = np.mean(np.array(P_TP)[1:])
  fp_all = np.mean(np.array(FP_ALL)[1:])
  fn_all = np.mean(np.array(FN_ALL)[1:])
  miou_foreground = np.mean(np.array(IoU)[1:])
  # print('\n======================================================')
  # print('%11s:%7.3f%%'%('mIoU',miou*100))
  # print('%11s:%7.3f'%('T/TP',t_tp))
  # print('%11s:%7.3f'%('P/TP',p_tp))
  # print('%11s:%7.3f'%('FP/ALL',fp_all))
  # print('%11s:%7.3f'%('FN/ALL',fn_all))
  # print('%11s:%7.3f'%('miou_foreground',miou_foreground))
  loglist['mIoU'] = miou * 100
  loglist['t_tp'] = t_tp
  loglist['p_tp'] = p_tp
  loglist['fp_all'] = fp_all
  loglist['fn_all'] = fn_all
  loglist['miou_foreground'] = miou_foreground
  return loglist


if __name__ == '__main__':
  df = pd.read_csv(args.list, names=['filename'])
  name_list = df['filename'].values

  if args.mode == 'png':
    report = do_python_eval(args.predict_dir, args.gt_dir, name_list, 21)
    print('mIoU={:.3f}%, FP={:.4f}, FN={:.4f}'.format(report['mIoU'], report['fp_all'], report['fn_all']))
  elif args.mode == 'rw':
    thresholds = np.arange(args.min_th, args.max_th, args.step_th).tolist()

    over_activation = 1.60
    under_activation = 0.60

    mIoU_list = []
    FP_list = []

    for t in thresholds:
      args.threshold = t
      report = do_python_eval(args.predict_dir, args.gt_dir, name_list, 21)

      mIoU, FP = report['mIoU'], report['fp_all']

      print('Th={:.2f}, mIoU={:.3f}%, FP={:.4f}'.format(t, mIoU, FP))

      FP_list.append(FP)
      mIoU_list.append(mIoU)

    best_index = np.argmax(mIoU_list)
    threshold_ = thresholds[best_index]
    miou_ = mIoU_list[best_index]
    best_FP = FP_list[best_index]

    over_FP = best_FP * over_activation
    under_FP = best_FP * under_activation

    print('Over FP : {:.4f}, Under FP : {:.4f}'.format(over_FP, under_FP))

    over_loss_list = [np.abs(FP - over_FP) for FP in FP_list]
    under_loss_list = [np.abs(FP - under_FP) for FP in FP_list]

    over_index = np.argmin(over_loss_list)
    over_th = thresholds[over_index]
    over_mIoU = mIoU_list[over_index]
    over_FP = FP_list[over_index]

    under_index = np.argmin(under_loss_list)
    under_th = thresholds[under_index]
    under_mIoU = mIoU_list[under_index]
    under_FP = FP_list[under_index]

    print('Best Th={:.2f}, mIoU={:.3f}%, FP={:.4f}'.format(threshold_, miou_, best_FP))
    print('Over Th={:.2f}, mIoU={:.3f}%, FP={:.4f}'.format(over_th, over_mIoU, over_FP))
    print('Under Th={:.2f}, mIoU={:.3f}%, FP={:.4f}'.format(under_th, under_mIoU, under_FP))
  else:
    if args.threshold is None:
      thresholds = np.arange(args.min_th, args.max_th, args.step_th).tolist()

      threshold_ = 0
      miou_ = 0

      for t in thresholds:
        args.threshold = t
        report = do_python_eval(args.predict_dir, args.gt_dir, name_list, 21)
        print(
          'Th={:.2f}, mIoU={:.3f}%, FP={:.4f}, FN={:.4f}'.format(
            args.threshold, report['mIoU'], report['fp_all'], report['fn_all']
          )
        )

        if report['mIoU'] > miou_:
          threshold_ = t
          miou_ = report['mIoU']
          iou_ = report

      print(f'Best Th={threshold_:.2f}, mIoU={miou_:.3f}%')
      print(*(f'{k}\t{v}' for k, v in iou_.items()), sep='\n')
    else:
      report = do_python_eval(args.predict_dir, args.gt_dir, name_list, 21)
      print(
        'Th={:.2f}, mIoU={:.3f}%, FP={:.4f}, FN={:.4f}'.format(
          args.threshold, report['mIoU'], report['fp_all'], report['fn_all']
        )
      )
