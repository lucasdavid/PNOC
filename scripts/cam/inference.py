# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import argparse
import copy
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.datasets import *
from core.networks import *
from tools.ai.augment_utils import *
from tools.ai.demo_utils import *
from tools.ai.evaluate_utils import *
from tools.ai.log_utils import *
from tools.ai.optim_utils import *
from tools.ai.randaugment import *
from tools.ai.torch_utils import *
from tools.general.io_utils import *
from tools.general.json_utils import *
from tools.general.time_utils import *

parser = argparse.ArgumentParser()

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--dataset', default='voc12', choices=['voc12', 'coco14'])
parser.add_argument('--data_dir', default='../VOCtrainval_11-May-2012/', type=str)

###############################################################################
# Network
###############################################################################
parser.add_argument('--architecture', default='resnet50', type=str)
parser.add_argument('--mode', default='normal', type=str)  # fix
parser.add_argument('--regularization', default=None, type=str)  # kernel_usage
parser.add_argument('--trainable-stem', default=True, type=str2bool)
parser.add_argument('--dilated', default=False, type=str2bool)
parser.add_argument('--restore', default=None, type=str)

###############################################################################
# Inference parameters
###############################################################################
parser.add_argument('--tag', default='', type=str)
parser.add_argument('--weights', default='', type=str)
parser.add_argument('--domain', default='train', type=str)

parser.add_argument('--scales', default='0.5,1.0,1.5,2.0', type=str)

if __name__ == '__main__':
  ###################################################################################
  # Arguments
  ###################################################################################
  args = parser.parse_args()

  TAG = args.tag
  TAG += '@train' if 'train' in args.domain else '@val'
  TAG += '@scale=%s' % args.scales

  pred_dir = create_directory(f'./experiments/predictions/{TAG}/')

  model_path = './experiments/models/' + f'{args.weights or args.tag}.pth'

  set_seed(args.seed)
  log = lambda string='': print(string)

  ###################################################################################
  # Transform, Dataset, DataLoader
  ###################################################################################
  dataset = get_dataset_inference(args.dataset, args.data_dir, args.domain)
  log('[i] The number of class is {}'.format(dataset.info.num_classes))
  log()

  normalize_fn = Normalize(*imagenet_stats())

  ###################################################################################
  # Network
  ###################################################################################
  model = Classifier(
    args.architecture,
    dataset.info.num_classes,
    mode=args.mode,
    dilated=args.dilated,
    regularization=args.regularization,
    trainable_stem=args.trainable_stem,
  )

  model = model.cuda()
  model.eval()

  log('[i] Architecture is {}'.format(args.architecture))
  log('[i] Total Params: %.2fM' % (calculate_parameters(model)))
  log()

  try:
    use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
  except KeyError:
    use_gpu = '0'

  the_number_of_gpu = len(use_gpu.split(','))
  if the_number_of_gpu > 1:
    log('[i] the number of gpu : {}'.format(the_number_of_gpu))
    model = nn.DataParallel(model)

  load_model(model, model_path, parallel=the_number_of_gpu > 1)

  #################################################################################################
  # Evaluation
  #################################################################################################
  eval_timer = Timer()
  scales = [float(scale) for scale in args.scales.split(',')]

  model.eval()
  eval_timer.tik()

  def get_cam(ori_image, scale):
    # preprocessing
    image = copy.deepcopy(ori_image)
    image = image.resize((round(ori_w * scale), round(ori_h * scale)), resample=PIL.Image.CUBIC)

    image = normalize_fn(image)
    image = image.transpose((2, 0, 1))

    image = torch.from_numpy(image)
    flipped_image = image.flip(-1)

    images = torch.stack([image, flipped_image])
    images = images.cuda()

    # inferenece
    _, features = model(images, with_cam=True)

    # postprocessing
    cams = F.relu(features)
    cams = cams[0] + cams[1].flip(-1)

    return cams

  with torch.no_grad():
    length = len(dataset)
    for step, (ori_image, image_id, label) in enumerate(dataset):
      ori_w, ori_h = ori_image.size

      npy_path = pred_dir + image_id + '.npy'
      if os.path.isfile(npy_path):
        continue

      strided_size = get_strided_size((ori_h, ori_w), 4)
      strided_up_size = get_strided_up_size((ori_h, ori_w), 16)

      cams_list = [get_cam(ori_image, scale) for scale in scales]

      strided_cams_list = [resize_for_tensors(cams.unsqueeze(0), strided_size)[0] for cams in cams_list]
      strided_cams = torch.sum(torch.stack(strided_cams_list), dim=0)

      hr_cams_list = [resize_for_tensors(cams.unsqueeze(0), strided_up_size)[0] for cams in cams_list]
      hr_cams = torch.sum(torch.stack(hr_cams_list), dim=0)[:, :ori_h, :ori_w]

      keys = torch.nonzero(torch.from_numpy(label))[:, 0]

      strided_cams = strided_cams[keys]
      strided_cams /= F.adaptive_max_pool2d(strided_cams, (1, 1)) + 1e-5

      hr_cams = hr_cams[keys]
      hr_cams /= F.adaptive_max_pool2d(hr_cams, (1, 1)) + 1e-5

      # save cams
      keys = np.pad(keys + 1, (1, 0), mode='constant')
      np.save(npy_path, {"keys": keys, "cam": strided_cams.cpu(), "hr_cam": hr_cams.cpu().numpy()})

      sys.stdout.write(
        '\r# Make CAM [{}/{}] = {:.2f}%, ({}, {})'.format(
          step + 1, length, (step + 1) / length * 100, (ori_h, ori_w), hr_cams.size()
        )
      )
      sys.stdout.flush()
    print()