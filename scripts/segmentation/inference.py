# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os
import sys
import copy
import shutil
import random
import argparse
import numpy as np

import imageio

import torch
import torch.nn.functional as F

from core.puzzle_utils import *
from core.networks import *
from core.datasets import *

from tools.general.io_utils import *
from tools.general.time_utils import *
from tools.general.json_utils import *

from tools.ai.log_utils import *
from tools.ai.demo_utils import *
from tools.ai.optim_utils import *
from tools.ai.torch_utils import *
from tools.ai.evaluate_utils import *

from tools.ai.augment_utils import *
from tools.ai.randaugment import *

parser = argparse.ArgumentParser()

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--data_dir', default='../VOCtrainval_11-May-2012/', type=str)

###############################################################################
# Network
###############################################################################
parser.add_argument('--architecture', default='DeepLabv3+', type=str)
parser.add_argument('--backbone', default='resnest269', type=str)
parser.add_argument('--mode', default='fix', type=str)
parser.add_argument('--dilated', default=False, type=str2bool)
parser.add_argument('--use_gn', default=True, type=str2bool)

###############################################################################
# Inference parameters
###############################################################################
parser.add_argument('--tag', default='', type=str)

parser.add_argument('--domain', default='val', type=str)

parser.add_argument('--scales', default='0.5,1.0,1.5,2.0', type=str)
parser.add_argument('--iteration', default=0, type=int)

if __name__ == '__main__':
    ###################################################################################
    # Arguments
    ###################################################################################
    args = parser.parse_args()

    model_dir = create_directory('./experiments/models/')
    model_path = model_dir + f'{args.tag}.pth'

    if 'train' in args.domain:
        args.tag += '@train'
    else:
        args.tag += '@' + args.domain
    
    args.tag += '@scale=%s'%args.scales
    args.tag += '@iteration=%d'%args.iteration

    pred_dir = create_directory('./experiments/predictions/{}/'.format(args.tag))

    set_seed(args.seed)
    log = lambda string='': print(string)
    
    ###################################################################################
    # Transform, Dataset, DataLoader
    ###################################################################################
    META = read_json(f'./data/{args.dataset}/meta.json')
    CLASSES = np.asarray(META['class_names'])
    NUM_CLASSES = META['classes']

    dataset = get_dataset_inference(args.dataset, args.data_dir, args.domain)
    normalize_fn = Normalize(*imagenet_stats())
    
    ###################################################################################
    # Network
    ###################################################################################
    if args.architecture == 'DeepLabv3+':
        model = DeepLabv3_Plus(args.backbone, num_classes=NUM_CLASSES + 1, mode=args.mode, use_group_norm=args.use_gn)
    elif args.architecture == 'Seg_Model':
        model = Seg_Model(args.backbone, num_classes=NUM_CLASSES + 1)
    elif args.architecture == 'CSeg_Model':
        model = CSeg_Model(args.backbone, num_classes=NUM_CLASSES + 1)
    
    model = model.cuda()
    model.eval()

    log('[i] Architecture is {}'.format(args.architecture))
    log('[i] Total Params: %.2fM'%(calculate_parameters(model)))
    log()

    load_model(model, model_path, parallel=False)
    
    #################################################################################################
    # Evaluation
    #################################################################################################
    eval_timer = Timer()
    scales = [float(scale) for scale in args.scales.split(',')]
    
    model.eval()
    eval_timer.tik()

    def inference(images, image_size):
        images = images.cuda()
        
        logits = model(images)
        logits = resize_for_tensors(logits, image_size)
        
        logits = logits[0] + logits[1].flip(-1)
        logits = to_numpy(logits).transpose((1, 2, 0))
        return logits

    with torch.no_grad():
        length = len(dataset)
        for step, (ori_image, image_id, _) in enumerate(dataset):
            ori_w, ori_h = ori_image.size

            cams_list = []

            for scale in scales:
                image = copy.deepcopy(ori_image)
                image = image.resize((round(ori_w*scale), round(ori_h*scale)), resample=PIL.Image.CUBIC)
                
                image = normalize_fn(image)
                image = image.transpose((2, 0, 1))

                image = torch.from_numpy(image)
                flipped_image = image.flip(-1)
                
                images = torch.stack([image, flipped_image])

                cams = inference(images, (ori_h, ori_w))
                cams_list.append(cams)

            preds = np.sum(cams_list, axis=0)
            preds = F.softmax(torch.from_numpy(preds), dim=-1).numpy()
            
            if args.iteration > 0:
                # h, w, c -> c, h, w
                preds = crf_inference(np.asarray(ori_image), preds.transpose((2, 0, 1)), t=args.iteration)
                pred_mask = np.argmax(preds, axis=0)
            else:
                pred_mask = np.argmax(preds, axis=-1)

            if args.domain == 'test':
                pred_mask = decode_from_colormap(pred_mask, dataset.colors)[..., ::-1]
            
            imageio.imwrite(pred_dir + image_id + '.png', pred_mask.astype(np.uint8))
            
            sys.stdout.write('\r# Make CAM [{}/{}] = {:.2f}%'.format(step + 1, length, (step + 1) / length * 100))
            sys.stdout.flush()
        print()
    
    if args.domain == 'val':
        print("python3 evaluate.py --experiment_name {} --domain {} --mode png".format(args.tag, args.domain))
