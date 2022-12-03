# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>
# modified by Sierkinhane <sierkinhane@163.com>

import sys

from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from core.ccam import SimMaxLoss, SimMinLoss
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
from tools.general.cam_utils import *

# os.environ["NUMEXPR_NUM_THREADS"] = "8"
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
parser.add_argument('--weights', default='imagenet', type=str)
parser.add_argument('--mode', default='normal', type=str)  # fix
parser.add_argument('--trainable-stem', default=True, type=str2bool)
parser.add_argument('--dilated', default=False, type=str2bool)
parser.add_argument('--stage4_out_features', default=1024, type=int)

###############################################################################
# Hyperparameter
###############################################################################
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--max_epoch', default=10, type=int)
parser.add_argument('--depth', default=50, type=int)
parser.add_argument('--accumule_steps', default=1, type=int)

parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--wd', default=1e-4, type=float)

parser.add_argument('--image_size', default=448, type=int)
parser.add_argument('--min_image_size', default=320, type=int)
parser.add_argument('--max_image_size', default=640, type=int)

parser.add_argument('--print_ratio', default=0.2, type=float)

parser.add_argument('--tag', default='', type=str)
parser.add_argument('--augment', default='', type=str)
parser.add_argument('--cutmix_prob', default=1.0, type=float)
parser.add_argument('--mixup_prob', default=1.0, type=float)

parser.add_argument('--alpha', type=float, default=0.25)

IS_POSITIVE = True

GPUS_VISIBLE = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
GPUS_COUNT = len(GPUS_VISIBLE.split(','))

if __name__ == '__main__':
  # global flag
  ###################################################################################
  # Arguments
  ###################################################################################
  args = parser.parse_args()
  
  DEVICE = args.device
  SIZE = args.image_size
  BATCH_TRAIN = args.batch_size
  BATCH_VALID = 32
  CUTMIX = 'cutmix' in args.augment

  log_dir = create_directory('./experiments/logs/')
  data_dir = create_directory('./experiments/data/')
  model_dir = create_directory('./experiments/models/')

  log_path = log_dir + '{}.txt'.format(args.tag)
  data_path = data_dir + '{}.json'.format(args.tag)
  model_path = model_dir + '{}.pth'.format(args.tag)
  cam_path = './experiments/images/{}'.format(args.tag)
  create_directory(cam_path)
  create_directory(cam_path + '/train')
  create_directory(cam_path + '/test')
  create_directory(cam_path + '/train/colormaps')
  create_directory(cam_path + '/test/colormaps')

  set_seed(args.seed)
  log = lambda string='': log_print(string, log_path)

  log('[i] {}'.format(args.tag))
  log()

  ###################################################################################
  # Transform, Dataset, DataLoader
  ###################################################################################
  imagenet_mean = [0.485, 0.456, 0.406]
  imagenet_std = [0.229, 0.224, 0.225]

  train_transform = transforms.Compose(
    [
      transforms.Resize(size=(512, 512)),
      transforms.RandomHorizontalFlip(),
      transforms.RandomCrop(size=(SIZE, SIZE)),
      transforms.ToTensor(),
      transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ]
  )

  test_transform = transforms.Compose(
    [
      Normalize_For_Segmentation(imagenet_mean, imagenet_std),
      Top_Left_Crop_For_Segmentation(SIZE),
      Transpose_For_Segmentation(),
    ]
  )

  meta_dic = read_json('./data/voc12/meta.json')
  class_names = np.asarray(meta_dic['class_names'])
  classes = len(class_names)

  train_dataset = VOC12ClassificationDataset(args.data_dir, 'train_aug', train_transform)
  if CUTMIX:
    log('[i] Using cutmix')
    train_dataset = CutMix(train_dataset, num_mix=1, beta=1., prob=args.cutmix_prob)

  train_dataset_for_seg = VOC12CAMTestingDataset(args.data_dir, 'train', test_transform)
  valid_dataset_for_seg = VOC12CAMTestingDataset(args.data_dir, 'val', test_transform)

  train_loader = DataLoader(train_dataset, batch_size=BATCH_TRAIN, num_workers=args.num_workers, shuffle=True)
  train_loader_for_seg = DataLoader(train_dataset_for_seg, batch_size=BATCH_VALID, num_workers=args.num_workers)
  # valid_loader_for_seg = DataLoader(valid_dataset_for_seg, batch_size=BATCH_VALID, num_workers=args.num_workers)

  log('[i] mean values is {}'.format(imagenet_mean))
  log('[i] std values is {}'.format(imagenet_std))
  log('[i] The number of class is {}'.format(classes))
  log('[i] train_transform is {}'.format(train_transform))
  log('[i] test_transform is {}'.format(test_transform))
  log('[i] #train data'.format(len(train_dataset)))
  log('[i] #valid data'.format(len(valid_dataset_for_seg)))
  log()

  val_iteration = len(train_loader)
  log_iteration = int(val_iteration * args.print_ratio)
  max_iteration = args.max_epoch * val_iteration

  log('[i] log_iteration : {:,}'.format(log_iteration))
  log('[i] val_iteration : {:,}'.format(val_iteration))
  log('[i] max_iteration : {:,}'.format(max_iteration))

  ###################################################################################
  # Network
  ###################################################################################
  model = CCAM(
    args.architecture,
    weights=args.weights,
    mode=args.mode,
    dilated=args.dilated,
    stage4_out_features=args.stage4_out_features,
  )
  param_groups = model.get_parameter_groups()

  log('[i] Architecture is {}'.format(args.architecture))
  log('[i] Total Params: %.2fM' % (calculate_parameters(model)))
  log()

  if GPUS_COUNT > 1:
    log('[i] the number of gpu : {}'.format(GPUS_COUNT))
    model = nn.DataParallel(model)

  model = model.to(DEVICE)

  load_model_fn = lambda: load_model(model, model_path, parallel=GPUS_COUNT > 1)
  save_model_fn = lambda: save_model(model, model_path, parallel=GPUS_COUNT > 1)

  ###################################################################################
  # Loss, Optimizer
  ###################################################################################
  criterion = [
    SimMaxLoss(metric='cos', alpha=args.alpha).to(DEVICE),
    SimMinLoss(metric='cos').to(DEVICE),
    SimMaxLoss(metric='cos', alpha=args.alpha).to(DEVICE)
  ]

  optimizer = PolyOptimizer([
      {'params': param_groups[0],'lr': args.lr,'weight_decay': args.wd},
      {'params': param_groups[1],'lr': 2 * args.lr,'weight_decay': 0},
      {'params': param_groups[2],'lr': 10 * args.lr,'weight_decay': args.wd},
      {'params': param_groups[3],'lr': 20 * args.lr,'weight_decay': 0}
    ],
    lr=args.lr, momentum=0.9, weight_decay=args.wd, max_step=max_iteration)

  #################################################################################################
  # Train
  #################################################################################################
  data_dic = {'train': [], 'validation': []}

  train_timer = Timer()
  eval_timer = Timer()

  train_meter = Average_Meter(['loss', 'class_loss'])

  best_train_mIoU = -1
  thresholds = np.arange(0., 0.50, 0.05).astype(float).tolist()

  def evaluate(loader):
    length = len(loader)

    print(f'Evaluating over {length} batches...')

    model.eval()
    eval_timer.tik()

    meter_dic = {th: MIoUCalcFromNames(['background', 'foreground']) for th in thresholds}

    with torch.no_grad():
      for _, (images, _, masks) in enumerate(loader):
        B, C, H, W = images.size()
        _, _, ccams = model(images.to(DEVICE))

        ccams = resize_for_tensors(ccams.cpu(), (H, W))
        ccams = make_cam(ccams)
        ccams = ccams.squeeze()
        ccams = to_numpy(ccams)

        for i in range(B):
          y_i = to_numpy(masks[i])
          valid_mask = y_i < 255
          bg_mask = y_i == 0

          # to saliency
          y_i = np.zeros_like(y_i)
          y_i[~bg_mask] = 1
          y_i[~valid_mask] = 255

          ccam_i = ccams[i]

          for t in thresholds:
            ccam_b = (ccam_i <= t).astype(y_i.dtype)
            meter_dic[t].add(ccam_b, y_i)

    best_th = 0.0
    best_mIoU = 0.0
    best_iou = {}

    for t in thresholds:
      mIoU, _, iou, *_ = meter_dic[t].get(clear=True, detail=True)
      if best_mIoU < mIoU:
        best_th = t
        best_mIoU = mIoU
        best_iou = iou  # .astype(float).round(2).float()

    return best_th, best_mIoU, best_iou

  train_meter = Average_Meter(['loss', 'positive_loss', 'negative_loss'])

  for epoch in range(args.max_epoch):
    model.train()

    for step, (images, labels) in enumerate(train_loader):
      fg_feats, bg_feats, ccams = model(images.to(DEVICE))

      loss1 = criterion[0](fg_feats)
      loss2 = criterion[1](bg_feats, fg_feats)
      loss3 = criterion[2](bg_feats)

      loss = loss1 + loss2 + loss3
      loss.backward()

      if (step + 1) % args.accumule_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
      
      if epoch == 0 and step == 600:
        IS_POSITIVE = check_positive(ccams)
        print(f"Is Negative: {IS_POSITIVE}")
      if IS_POSITIVE:
        ccams = 1 - ccams

      train_meter.add(
        {
          'loss': loss.item(),
          'positive_loss': loss1.item() + loss3.item(),
          'negative_loss': loss2.item(),
        }
      )

      #################################################################################################
      # For Log
      #################################################################################################

      if (step + 1) % 100 == 0:
        visualize_heatmap(args.tag, images.clone().detach(), ccams, 0, step)
        loss, positive_loss, negative_loss = train_meter.get(clear=True)
        lr = float(get_learning_rate_from_optimizer(optimizer))

        data = {
          'epoch': epoch,
          'max_epoch': args.max_epoch,
          'iteration': step + 1,
          'learning_rate': lr,
          'loss': loss,
          'positive_loss': positive_loss,
          'negative_loss': negative_loss,
          'time': train_timer.tok(clear=True),
        }
        data_dic['train'].append(data)

        log(
          'Epoch[{epoch:,}/{max_epoch:,}] iteration={iteration:,} lr={learning_rate:.4f} '
          'loss={loss:.4f} loss_p={positive_loss:.4f} loss_n={negative_loss:.4f} '
          'time={time:.0f}sec'.format(**data)
        )

    #################################################################################################
    # Evaluation
    #################################################################################################
    save_model_fn()
    log('[i] save model')

    threshold, mIoU, iou = evaluate(train_loader_for_seg)

    if best_train_mIoU == -1 or best_train_mIoU < mIoU:
      best_train_mIoU = mIoU

      save_model_fn()
      log('[i] save model')

    data = {
      'iteration': step + 1,
      'threshold': threshold,
      'train_sal_mIoU': mIoU,
      'train_sal_iou': iou,
      'best_train_sal_mIoU': best_train_mIoU,
      'time': eval_timer.tok(clear=True),
    }
    data_dic['validation'].append(data)
    write_json(data_path, data_dic)

    log(
      'iteration={iteration:,}\n'
      'threshold={threshold:.2f}\n'
      'train_sal_mIoU={train_sal_mIoU:.2f}%\n'
      'train_sal_iou ={train_sal_iou}\n'
      'best_train_sal_mIoU={best_train_sal_mIoU:.2f}%\n'
      'time={time:.0f}sec'.format(**data)
    )

  print(args.tag)
