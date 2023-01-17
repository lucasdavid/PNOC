# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>
# modified by Sierkinhane <sierkinhane@163.com>

import sys

import matplotlib

matplotlib.use('Agg')
from torchvision import transforms
# from tensorboardX import SummaryWriter

from torch.utils.data import DataLoader

from utils import *
from core.datasets import *
from core.model import *
from tools.general.io_utils import *
from tools.general.time_utils import *
from tools.general.json_utils import *
from core.loss import *

from tools.ai.log_utils import *
from tools.ai.demo_utils import *
from tools.ai.torch_utils import *
from tools.ai.evaluate_utils import *

from tools.ai.augment_utils import *
from tools.ai.randaugment import *
from shutil import copyfile
import matplotlib.pyplot as plt
from optimizer import PolyOptimizer

os.environ["NUMEXPR_NUM_THREADS"] = "8"
parser = argparse.ArgumentParser()

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--data_dir', default='/data1/xjheng/dataset/VOC2012/', type=str)
parser.add_argument('--cams_dir', default='/experiments/predictions/resnest101@ra/', type=str)

###############################################################################
# Network
###############################################################################
parser.add_argument('--architecture', default='resnet50', type=str)
parser.add_argument('--mode', default='normal', type=str)  # fix

###############################################################################
# Hyperparameter
###############################################################################
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--max_epoch', default=10, type=int)
parser.add_argument('--depth', default=50, type=int)

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

parser.add_argument('--pretrained', type=str, required=True, help='adopt different pretrained parameters, [supervised, mocov2, detco]')

flag = True

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


def random_crop_box(crop_size, h, w):
  ch = min(crop_size, h)  # (448, 512) -> 448
  cw = min(crop_size, w)  # (448, 300) -> 300

  h_space = h - crop_size  # 512-448 =   64
  w_space = w - crop_size  # 300-448 = -148

  if w_space > 0:
    cont_left = 0
    img_left = random.randrange(w_space + 1)
  else:
    cont_left = random.randrange(-w_space + 1)  # rand(149)  = 20
    img_left = 0

  if h_space > 0:
    cont_top = 0
    img_top = random.randrange(h_space + 1)     # rand(65)   = 10
  else:
    cont_top = random.randrange(-h_space + 1)
    img_top = 0

  dst_bbox = {'xmin': cont_left, 'ymin': cont_top, 'xmax': cont_left + cw, 'ymax': cont_top + ch}  # 20,  0, 20+300, 0+448
  src_bbox = {'xmin': img_left, 'ymin': img_top, 'xmax': img_left + cw, 'ymax': img_top + ch}      #  0, 10,    300, 10+448

  return dst_bbox, src_bbox


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

    SIZE = args.image_size

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
    log_func = lambda string='': log_print(string, log_path)

    log_func('[i] {}'.format(args.tag))
    log_func()

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

    meta_dic = read_json('./data/VOC_2012.json')
    class_names = np.asarray(meta_dic['class_names'])

    train_dataset = VOCDatasetWithCAMs(
        args.data_dir,
        'train_aug',
        args.cams_dir,
        resize=resize_t,
        normalize=normalize_t,
        aug_transform=aug_transform
    )

    train_dataset_for_seg = VOC_Dataset_For_Testing_CAM(args.data_dir, 'train', test_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    log_func('[i] mean values is {}'.format(imagenet_mean))
    log_func('[i] std values is {}'.format(imagenet_std))
    log_func('[i] The number of class is {}'.format(meta_dic['classes']))
    log_func(f'[i] train_transform is {[resize_t, normalize_t, aug_transform]}')
    log_func('[i] test_transform is {}'.format(test_transform))
    log_func('[i] #train data'.format(len(train_dataset)))
    log_func()

    val_iteration = len(train_loader)
    log_iteration = int(val_iteration * args.print_ratio)
    max_iteration = args.max_epoch * val_iteration

    log_func('[i] log_iteration : {:,}'.format(log_iteration))
    log_func('[i] val_iteration : {:,}'.format(val_iteration))
    log_func('[i] max_iteration : {:,}'.format(max_iteration))

    ###################################################################################
    # Network
    ###################################################################################
    model = get_model(args.architecture, args.pretrained)
    param_groups = model.get_parameter_groups()

    model = model.cuda()
    model.train()
    # model_info(model)

    log_func('[i] Architecture is {}'.format(args.architecture))
    log_func('[i] Total Params: %.2fM' % (calculate_parameters(model)))
    log_func()

    try:
        use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
    except KeyError:
        use_gpu = '0'

    the_number_of_gpu = len(use_gpu.split(','))
    if the_number_of_gpu > 1:
        log_func('[i] the number of gpu : {}'.format(the_number_of_gpu))
        model = nn.DataParallel(model)

    load_model_fn = lambda: load_model(model, model_path, parallel=the_number_of_gpu > 1)
    # save_model_fn = lambda: save_model(model, model_path, parallel=the_number_of_gpu > 1)

    ###################################################################################
    # Loss, Optimizer
    ###################################################################################
    hint_loss_fn = nn.BCEWithLogitsLoss(reduction='none').cuda()
    criterion = [SimMaxLoss(metric='cos', alpha=args.alpha).cuda(), SimMinLoss(metric='cos').cuda(),
                 SimMaxLoss(metric='cos', alpha=args.alpha).cuda()]

    optimizer = PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wd},
        {'params': param_groups[1], 'lr': 2 * args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10 * args.lr, 'weight_decay': args.wd},
        {'params': param_groups[3], 'lr': 20 * args.lr, 'weight_decay': 0},
    ], lr=args.lr, momentum=0.9, weight_decay=args.wd, max_step=max_iteration)

    #################################################################################################
    # Train
    #################################################################################################
    data_dic = {
        'train': [],
        'validation': []
    }

    train_timer = Timer()
    eval_timer = Timer()

    train_meter = Average_Meter(['loss', 'positive_loss', 'negative_loss', 'hint_loss'])

    # writer = SummaryWriter(tensorboard_dir)

    for epoch in range(args.max_epoch):
        for iteration, (images, labels, cam_hints) in enumerate(train_loader):

            optimizer.zero_grad()
            fg_feats, bg_feats, output = model(images.cuda())

            loss1 = criterion[0](fg_feats)
            loss2 = criterion[1](bg_feats, fg_feats)
            loss3 = criterion[2](bg_feats)

            # CAM Hints
            cam_hints = F.interpolate(cam_hints, output.shape[2:], mode='bicubic')  # B1HW -> B1hw

            # Using foreground cues:
            fg_likely = (cam_hints >= args.fg_threshold).cuda()

            # loss_h := -log(sigmoid(output[fg_likely]))
            output_fg = output[fg_likely]
            target_fg = torch.ones_like(output_fg)
            loss_h = hint_loss_fn(output_fg, target_fg).mean()

            loss = args.hint_w * loss_h + (loss1 + loss2 + loss3)
            loss.backward()
            optimizer.step()

            ccam = torch.sigmoid(output.cpu())

            if epoch == 0 and iteration == (len(train_loader)-1):
                flag = check_positive(ccam)
                print(f"Is Negative: {flag}")
            if flag:
                ccam = 1 - ccam

            train_meter.add({
                'loss': loss.item(),
                'positive_loss': loss1.item() + loss3.item(),
                'negative_loss': loss2.item(),
                'hint_loss': loss_h.item(),
            })

            #################################################################################################
            # For Log
            #################################################################################################

            if (iteration + 1) % 100 == 0:
                visualize_heatmap(args.tag, images.clone().detach(), ccam, 0, iteration)
                loss, positive_loss, negative_loss, hint_loss = train_meter.get(clear=True)
                learning_rate = float(get_learning_rate_from_optimizer(optimizer))

                data = {
                    'epoch': epoch,
                    'max_epoch': args.max_epoch,
                    'iteration': iteration + 1,
                    'learning_rate': learning_rate,
                    'loss': loss,
                    'positive_loss': positive_loss,
                    'negative_loss': negative_loss,
                    'hint_loss': hint_loss,
                    'time': train_timer.tok(clear=True),
                }
                data_dic['train'].append(data)

                log_func('[i]\t'
                         'Epoch[{epoch:,}/{max_epoch:,}],\t'
                         'iteration={iteration:,}, \t'
                         'learning_rate={learning_rate:.4f}, \t'
                         'loss={loss:.4f}, \t'
                         'positive_loss={positive_loss:.4f}, \t'
                         'negative_loss={negative_loss:.4f}, \t'
                         'hint_loss={hint_loss:.4f}, \t'
                         'time={time:.0f}sec'.format(**data)
                         )

                # writer.add_scalar('Train/loss', loss, iteration)
                # writer.add_scalar('Train/learning_rate', learning_rate, iteration)
                # break
        #################################################################################################
        # Evaluation
        #################################################################################################
        # save_model_fn()
        torch.save({'state_dict': model.module.state_dict() if (the_number_of_gpu > 1) else model.state_dict(),
                    'flag': flag}, model_path)

        log_func('[i] save model')

    print(args.tag)
