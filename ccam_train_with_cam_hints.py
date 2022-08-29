# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>
# modified by Sierkinhane <sierkinhane@163.com>

import sys

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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
parser.add_argument('--data_dir', default='/datasets/VOCdevkit/VOC2012/', type=str)
parser.add_argument('--cams_dir', default='/experiments/predictions/resnest101@ra/', type=str)

###############################################################################
# Network
###############################################################################
parser.add_argument('--architecture', default='resnet50', type=str)
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

parser.add_argument('--alpha', type=float, default=0.25)
parser.add_argument('--hint_w', type=float, default=1.0)

# parser.add_argument('--bg_threshold', type=float, default=0.1)
parser.add_argument('--fg_threshold', type=float, default=0.4)

GPUS_VISIBLE = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
GPUS_COUNT = len(GPUS_VISIBLE.split(','))


class VOCDatasetWithCAMs(VOC_Dataset):

  def __init__(self, root_dir, domain, cams_dir, resize, normalize, aug_transform):
    super().__init__(root_dir, domain, with_id=True, with_tags=True)
    self.cams_dir = cams_dir
    self.resize = resize
    self.normalize = normalize
    self.aug_transform = aug_transform

    cmap_dic, _, class_names = get_color_map_dic()
    self.colors = np.asarray([cmap_dic[class_name] for class_name in class_names])

    data = read_json('./data/VOC_2012.json')

    self.class_dic = data['class_dic']
    self.classes = data['classes']

  def __getitem__(self, index):
    image, image_id, tags = super().__getitem__(index)

    mask_path = os.path.join(self.cams_dir, f'{image_id}.npy')
    mask_pack = np.load(mask_path, allow_pickle=True).item()
    cams = torch.from_numpy(mask_pack['hr_cam'].max(0, keepdims=True))

    # Transforms
    image = self.resize(image)
    cams = self.resize(cams)

    image = self.normalize(image)

    data = self.aug_transform({'image': image, 'cams': cams})
    image, cams = data['image'], data['cams']

    label = one_hot_embedding([self.class_dic[tag] for tag in tags], self.classes)
    return image, label, cams


# Augmentations


def random_horizontal_flip(data):
  if bool(random.getrandbits(1)):
    data['image'] = data['image'].flip(-1)
    data['cams'] = data['cams'].flip(-1)
  return data


class RandomCropForCams(RandomCrop):

  def __init__(self, crop_size):
    super().__init__(crop_size)
    self.crop_shape_for_mask = (self.crop_size, self.crop_size)

  def __call__(self, data):
    _, src = random_crop_box(self.crop_size, *data['image'].shape[1:])
    ymin, ymax, xmin, xmax = src['ymin'], src['ymax'], src['xmin'], src['xmax']

    data['image'] = data['image'][:, ymin:ymax, xmin:xmax]
    data['cams'] = data['cams'][:, ymin:ymax, xmin:xmax]

    return data


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

  META = read_json('./data/VOC_2012.json')
  CLASSES = np.asarray(META['class_names'])
  NUM_CLASSES = len(CLASSES)

  log_dir = create_directory('./experiments/logs/')
  data_dir = create_directory('./experiments/data/')
  model_dir = create_directory('./experiments/models/')
  tensorboard_dir = create_directory('./experiments/tensorboards/{}/'.format(args.tag))

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

  resize_t = transforms.Resize(size=(512, 512))
  normalize_t = transforms.Compose([
    resize_t,
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
  ])
  aug_transform = transforms.Compose([random_horizontal_flip, RandomCropForCams(SIZE)])
  test_transform = transforms.Compose(
    [
      Normalize_For_Segmentation(imagenet_mean, imagenet_std),
      Top_Left_Crop_For_Segmentation(SIZE),
      Transpose_For_Segmentation(),
    ]
  )

  train_dataset = VOCDatasetWithCAMs(
    args.data_dir,
    'train_aug',
    args.cams_dir,
    resize=resize_t,
    normalize=normalize_t,
    aug_transform=aug_transform
  )
  valid_dataset = VOC_Dataset_For_Testing_CAM(args.data_dir, 'train', test_transform)
  train_loader = DataLoader(train_dataset, batch_size=BATCH_TRAIN, num_workers=args.num_workers, shuffle=True)
  valid_loader = DataLoader(valid_dataset, batch_size=BATCH_VALID, num_workers=args.num_workers)

  log('[i] mean values is {}'.format(imagenet_mean))
  log('[i] std values is {}'.format(imagenet_std))
  log('[i] The number of class is {}'.format(NUM_CLASSES))
  log(f'[i] train_transform is {[resize_t, normalize_t, aug_transform]}')
  log('[i] test_transform is {}'.format(test_transform))
  log('[i] #train data'.format(len(train_dataset)))
  log('[i] #valid data'.format(len(valid_dataset)))
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
  model = CCAM(args.architecture, mode=args.mode, dilated=args.dilated, stage4_out_features=args.stage4_out_features)
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
  hint_loss_fn = nn.BCEWithLogitsLoss(reduction='none').to(DEVICE)

  criterion = [
    SimMaxLoss(metric='cos', alpha=args.alpha).to(DEVICE),
    SimMinLoss(metric='cos').to(DEVICE),
    SimMaxLoss(metric='cos', alpha=args.alpha).to(DEVICE),
  ]

  optimizer = PolyOptimizer(
    [
      {
        'params': param_groups[0],
        'lr': args.lr,
        'weight_decay': args.wd
      }, {
        'params': param_groups[1],
        'lr': 2 * args.lr,
        'weight_decay': 0
      }, {
        'params': param_groups[2],
        'lr': 10 * args.lr,
        'weight_decay': args.wd
      }, {
        'params': param_groups[3],
        'lr': 20 * args.lr,
        'weight_decay': 0
      }
    ],
    lr=args.lr,
    momentum=0.9,
    weight_decay=args.wd,
    max_step=max_iteration
  )

  #################################################################################################
  # Train
  #################################################################################################
  data_dic = {'train': [], 'validation': []}

  train_timer = Timer()
  eval_timer = Timer()

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
        _, _, ccam = model(images.to(DEVICE))

        ccam = resize_for_tensors(ccam.cpu(), (H, W))
        ccam = make_cam(ccam)
        ccam = ccam.squeeze()
        ccam = to_numpy(ccam)

        for i in range(B):
          y_i = to_numpy(masks[i])
          valid_mask = y_i < 255
          bg_mask = y_i == 0

          # to saliency
          y_i = np.zeros_like(y_i)
          y_i[~bg_mask] = 1
          y_i[~valid_mask] = 255

          ccam_i = ccam[i]

          for t in thresholds:
            ccam_it = (ccam_i >= t).astype(y_i.dtype)
            meter_dic[t].add(ccam_it, y_i)

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

  train_meter = Average_Meter(['loss', 'positive_loss', 'negative_loss', 'hint_loss'])

  writer = SummaryWriter(tensorboard_dir)

  for epoch in range(args.max_epoch):
    model.train()

    for step, (images, labels, cam_hints) in enumerate(train_loader):

      fg_feats, bg_feats, output = model(images.to(DEVICE))

      loss1 = criterion[0](fg_feats)
      loss2 = criterion[1](bg_feats, fg_feats)
      loss3 = criterion[2](bg_feats)

      # CAM Hints
      cam_hints = F.interpolate(cam_hints, output.shape[2:], mode='bicubic')  # B1HW -> B1hw

      # Using foreground cues:
      fg_likely = (cam_hints >= args.fg_threshold).to(DEVICE)

      # loss_h := -log(sigmoid(output[fg_likely]))
      output_fg = output[fg_likely]
      target_fg = torch.ones_like(output_fg)
      loss_h = hint_loss_fn(target_fg, output_fg).mean()

      # Using both foreground and background cues:
      # bg_likely = cam_hints < args.bg_threshold
      # fg_likely = cam_hints >= args.fg_threshold
      # mk_likely = (bg_likely | fg_likely).to(DEVICE)
      # target = torch.zeros_like(cam_hints)
      # target[fg_likely] = 1.
      # loss_h = hint_loss_fn(target, output)
      # loss_h = loss_h[mk_likely].sum() / mk_likely.float().sum()

      # Back-propagation
      loss = args.hint_w * loss_h + (loss1 + loss2 + loss3)
      loss.backward()

      if (step + 1) % args.accumule_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

      ccam = torch.sigmoid(output)

      train_meter.add(
        {
          'loss': loss.item(),
          'hint_loss': loss_h.item(),
          'positive_loss': loss1.item() + loss3.item(),
          'negative_loss': loss2.item(),
        }
      )

      #################################################################################################
      # For Log
      #################################################################################################

      if (step + 1) % 100 == 0:
        visualize_heatmap(args.tag, images.clone().detach(), ccam, 0, step)
        loss, positive_loss, negative_loss, loss_h = train_meter.get(clear=True)
        lr = float(get_learning_rate_from_optimizer(optimizer))

        data = {
          'epoch': epoch,
          'max_epoch': args.max_epoch,
          'iteration': step + 1,
          'learning_rate': lr,
          'loss': loss,
          'positive_loss': positive_loss,
          'negative_loss': negative_loss,
          'hint_loss': loss_h,
          'time': train_timer.tok(clear=True),
        }
        data_dic['train'].append(data)

        log(
          'Epoch[{epoch:,}/{max_epoch:,}] iteration={iteration:,} lr={learning_rate:.4f} '
          'loss={loss:.4f} loss_p={positive_loss:.4f} loss_n={negative_loss:.4f} loss_h={hint_loss:.4f} '
          'time={time:.0f}sec'.format(**data)
        )

        writer.add_scalar('Train/loss', loss, step)
        writer.add_scalar('Train/learning_rate', lr, step)

    #################################################################################################
    # Evaluation
    #################################################################################################
    save_model_fn()
    log('[i] save model')

    threshold, mIoU, iou = evaluate(valid_loader)

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

    writer.add_scalar('Evaluation/threshold', threshold, step)
    writer.add_scalar('Evaluation/train_sal_mIoU', mIoU, step)
    writer.add_scalar('Evaluation/best_train_sal_mIoU', best_train_mIoU, step)

  print(args.tag)
