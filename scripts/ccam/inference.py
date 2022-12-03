# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>
# modified by Sierkinhane <sierkinhane@163.com>

import argparse
import copy
import os
import sys

import PIL
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
from tools.ai.torch_utils import *
from tools.general.io_utils import *
from tools.general.json_utils import *
from tools.general.time_utils import *

parser = argparse.ArgumentParser()

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--data_dir', default='/data1/xjheng/dataset/VOC2012/', type=str)

###############################################################################
# Network
###############################################################################
parser.add_argument('--architecture', default='resnet50', type=str)
parser.add_argument('--mode', default='normal', type=str)  # fix
parser.add_argument('--weights', default='imagenet', type=str)
parser.add_argument('--trainable-stem', default=True, type=str2bool)
parser.add_argument('--dilated', default=False, type=str2bool)
parser.add_argument('--pretrained', type=str, required=True)
parser.add_argument('--stage4_out_features', default=1024, type=int)

###############################################################################
# Inference parameters
###############################################################################
parser.add_argument('--tag', default='', type=str)
parser.add_argument('--domain', default='train', type=str)
parser.add_argument('--vis_dir', default='vis_cam', type=str)
parser.add_argument('--scales', default='0.5,1.0,1.5,2.0', type=str)
parser.add_argument('--activation', default='relu', type=str, choices=['relu', 'sigmoid'])

GPUS_VISIBLE = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
GPUS_COUNT = len(GPUS_VISIBLE.split(','))

if __name__ == '__main__':
  ###################################################################################
  # Arguments
  ###################################################################################
  args = parser.parse_args()

  DEVICE = args.device

  tag = f'{args.tag}@train' if 'train' in args.domain else f'{args.tag}@val'
  tag += '@scale=%s' % args.scales

  pred_dir = create_directory(f'./experiments/predictions/{tag}/')
  cam_path = create_directory(f'{args.vis_dir}/{tag}')

  model_path = args.pretrained

  set_seed(args.seed)
  log_func = lambda string='': print(string)

  ###################################################################################
  # Transform, Dataset, DataLoader
  ###################################################################################
  imagenet_mean = [0.485, 0.456, 0.406]
  imagenet_std = [0.229, 0.224, 0.225]

  normalize_fn = Normalize(imagenet_mean, imagenet_std)

  # for mIoU
  meta_dic = read_json('./data/voc12/meta.json')
  dataset = VOC12InferenceDataset(args.data_dir, args.domain)

  ###################################################################################
  # Network
  ###################################################################################
  model = CCAM(
    args.architecture,
    weights=args.weights,
    mode=args.mode,
    dilated=args.dilated,
    stage4_out_features=args.stage4_out_features
  )

  log_func('[i] Architecture is {}'.format(args.architecture))
  log_func('[i] Total Params: %.2fM' % (calculate_parameters(model)))
  log_func()

  if GPUS_COUNT > 1:
    log_func('[i] the number of gpu : {}'.format(GPUS_COUNT))
    model = nn.DataParallel(model)

  model = model.to(DEVICE)

  print(f'Loading weights from {model_path}.')
  load_model(model, model_path, parallel=GPUS_COUNT > 1)

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
    _, _, cam = model(images)
    # if flag:
    #     cam = 1 - cam
    # postprocessing
    if args.activation == 'relu':
      cams = F.relu(cam)
    else:
      cams = torch.sigmoid(cam)

    cams = cams[0] + cams[1].flip(-1)

    return cams  # (1, H, W)

  vis_cam = True
  with torch.no_grad():
    length = len(dataset)
    for step, (ori_image, image_id, _, _) in enumerate(dataset):
      ori_w, ori_h = ori_image.size
      npy_path = pred_dir + image_id + '.npy'
      if os.path.isfile(npy_path):
        continue
      strided_size = get_strided_size((ori_h, ori_w), 4)
      strided_up_size = get_strided_up_size((ori_h, ori_w), 16)

      cams_list = [get_cam(ori_image, scale) for scale in scales]

      strided_cams_list = [resize_for_tensors(cams.unsqueeze(0), strided_size)[0] for cams in cams_list]
      strided_cams = torch.mean(torch.stack(strided_cams_list), dim=0) # (1, 1, H, W)

      hr_cams_list = [resize_for_tensors(cams.unsqueeze(0), strided_up_size)[0] for cams in cams_list]
      hr_cams = torch.mean(torch.stack(hr_cams_list), dim=0)[:, :ori_h, :ori_w] # (1, 1, H, W)

      strided_cams = make_cam(strided_cams.unsqueeze(1)).squeeze(1)
      hr_cams = make_cam(hr_cams.unsqueeze(1)).squeeze(1)

      np.save(npy_path, {"keys": [0, 1], "cam": strided_cams.cpu(), "hr_cam": hr_cams.cpu().numpy()})

      sys.stdout.write(
        '\r# Make CAM [{}/{}] = {:.2f}%, ({}, {})'.format(
          step + 1, length, (step + 1) / length * 100, (ori_h, ori_w), hr_cams.size()
        )
      )
      sys.stdout.flush()
    print()

  if args.domain == 'train_aug':
    args.domain = 'train'

  print("python3 inference_crf.py --experiment_name {} --domain {}".format(tag, args.domain))
