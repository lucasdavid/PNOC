import argparse

import torch
from sklearn import metrics as skmetrics
from torch.utils.data import DataLoader

import datasets
from core.networks import *
from tools.ai.augment_utils import *
from tools.ai.log_utils import *
from tools.ai.torch_utils import *
from tools.general.io_utils import *
from tools.general.json_utils import *

parser = argparse.ArgumentParser()

# Dataset
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--dataset', default='voc12', choices=datasets.DATASOURCES)
parser.add_argument('--data_dir', required=True, type=str)
parser.add_argument('--max_steps', default=None, type=int)

parser.add_argument('--image_size', default=512, type=int)
parser.add_argument('--batch_size', default=32, type=int)

parser.add_argument('--architecture', default='resnet50', type=str)
parser.add_argument('--mode', default='normal', type=str)  # fix
parser.add_argument('--regularization', default=None, type=str)  # kernel_usage
parser.add_argument('--trainable-stem', default=True, type=str2bool)
parser.add_argument('--dilated', default=False, type=str2bool)

parser.add_argument('--tag', default='', type=str)
parser.add_argument('--weights', default='', type=str)
parser.add_argument('--domain', default='train', type=str)


def main(args):
  TAG = args.tag
  SEED = args.seed
  DEVICE = args.device if torch.cuda.is_available() else "cpu"
  SIZE = args.image_size

  print('Evaluation Configuration')
  pad = max(map(len, vars(args))) + 1
  for k, v in vars(args).items():
    print(f'{k.ljust(pad)}: {v}')
  print('========================')

  log_dir = create_directory(f'./experiments/logs/')
  data_dir = create_directory(f'./experiments/data/')
  model_dir = create_directory('./experiments/models/')

  log_path = log_dir + f'{TAG}.txt'
  model_path = './experiments/models/' + f'{args.weights or args.tag}.pth'

  set_seed(SEED)
  log = lambda string='': log_print(string, log_path)

  transform = transforms.Compose([
    transforms.Resize(size=(SIZE, SIZE)),
    Normalize(*datasets.imagenet_stats()),
    Transpose(),
  ])

  data_source = datasets.custom_data_source(args.dataset, args.data_dir, args.domain)
  dataset = datasets.ClassificationDataset(data_source, transform, ignore_bg_images=False)
  loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
  total_steps = args.max_steps or len(loader)
  logg_steps = max(1, int(total_steps * 0.1))

  print(f"dataset steps = {len(loader)}")
  print(f"total steps   = {total_steps}")
  print(f"logging steps = {logg_steps}")

  model = Classifier(
    args.architecture,
    num_classes=dataset.info.num_classes,
    mode=args.mode,
    dilated=args.dilated,
    regularization=args.regularization,
    trainable_stem=args.trainable_stem,
  )
  print(f'Restoring weights from {model_path}')
  model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")), strict=True)
  model.to(DEVICE)

  try:
    GPUS = os.environ['CUDA_VISIBLE_DEVICES']
  except KeyError:
    GPUS = '0'

  if len(GPUS.split(',')) > 1:
    log(f'GPUs = {GPUS}')
    model = nn.DataParallel(model)

  model.eval()

  # Metrics
  probs = []
  targs = []

  for step, (_, images, y) in enumerate(loader):
    images = images.to(DEVICE)
    p = model(images)
    p = torch.sigmoid(p)

    targs += [to_numpy(y)]
    probs += [to_numpy(p)]

    if (step + 1) % logg_steps == 0:
      print(f'{(step + 1) / total_steps:.0%}', flush=True)

    if step >= total_steps:
      break

  targs = np.concatenate(targs, 0)
  probs = np.concatenate(probs, 0)
  preds = probs.round()

  print("Classification Report")
  print(skmetrics.classification_report(
    targs, preds,
    target_names=dataset.info.classes
  ))

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
