import argparse

import torch
import torchmetrics
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from core.datasets import *
from core.networks import *
from core.mcar import mcar_resnet101
from tools.ai.augment_utils import *
from tools.general.json_utils import *
from tools.general.io_utils import *

parser = argparse.ArgumentParser()

# Dataset
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--dataset', default='voc12', choices=['voc12', 'coco14'])
parser.add_argument('--data_dir', default='../VOCtrainval_11-May-2012/', type=str)

parser.add_argument('--batch_size', default=32, type=int)

parser.add_argument('--architecture', default='resnet50', type=str)
parser.add_argument('--mode', default='normal', type=str)  # fix
parser.add_argument('--regularization', default=None, type=str)  # kernel_usage
parser.add_argument('--trainable-stem', default=True, type=str2bool)
parser.add_argument('--dilated', default=False, type=str2bool)
parser.add_argument('--restore', default=None, type=str)


def main(args):

  # Model
  # ps = 'avg'
  # topN = 4
  # threshold = 0.5
  # model = mcar_resnet101(20, ps, topN, threshold, inference_mode=True)
  # ckpt_file = '/home/ldavid/workspace/logs/sdumont/mcar/model_best.pth.tar'
  # ckpt = torch.load(ckpt_file, map_location=torch.device('cpu'))
  # model.load_state_dict(ckpt['state_dict'], strict=True)

  model = Classifier(
    args.architecture,
    NUM_CLASSES,
    mode=args.mode,
    dilated=args.dilated,
    regularization=args.regularization,
    trainable_stem=args.trainable_stem,
  )

  model.eval()

  # Dataset
  META = read_json(f'./data/{args.dataset}/meta.json')
  CLASSES = np.asarray(META['class_names'])
  NUM_CLASSES = META['classes']

  tt, tv = get_transforms(args.min_image_size, args.max_image_size, args.image_size, args.augment)
  _, valid_dataset = get_dataset_classification(
    args.dataset, args.data_dir, args.augment, args.image_size, args.cutmix_prob, args.mixup_prob, tt, tv
  )

  valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
  steps = len(valid_loader)

  # Metrics
  metrics = {'f1': torchmetrics.F1Score(num_classes=NUM_CLASSES, average='none')}

  for step, (x, y) in enumerate(valid_loader):
    p, = model(x)

    metrics['f1'](p, y.int())

    if (step + 1) % (steps // 10) == 0:
      f1 = metrics['f1'].compute()
      print(f'F1 on step {step / steps:.0%}: {f1}')

  # metric on all batches using custom accumulation
  f1 = metrics['f1']
  f1_values = f1.compute().detach().numpy()
  print(*(f'{c:<12}: {v:.2%}' for c, v in zip(CLASSES, f1_values)), sep='\n')

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
