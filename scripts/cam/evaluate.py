import argparse

import torch
from sklearn import metrics as skmetrics
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
import datasets
from core.networks import *
from tools.ai.augment_utils import *
from tools.ai.log_utils import *
from tools.ai.torch_utils import *
from tools.general import wandb_utils
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
parser.add_argument('--mixed_precision', default=False, type=str2bool)

parser.add_argument('--batch_size', default=32, type=int)

parser.add_argument('--image_size', default=512, type=int)
parser.add_argument('--min_image_size', default=320, type=int)
parser.add_argument('--max_image_size', default=640, type=int)
parser.add_argument('--augment', default='', type=str)

parser.add_argument('--architecture', default='resnet50', type=str)
parser.add_argument('--mode', default='normal', type=str)  # fix
parser.add_argument('--trainable-stem', default=True, type=str2bool)
parser.add_argument('--dilated', default=False, type=str2bool)

parser.add_argument('--tag', default='', type=str)
parser.add_argument('--weights', default='', type=str)
parser.add_argument('--save_preds', default=None, type=str)
parser.add_argument('--domain', default='train', type=str)


def main(args):
  TAG = args.tag
  SEED = args.seed
  DEVICE = args.device if torch.cuda.is_available() else "cpu"
  SIZE = args.image_size
  AUG = args.augment

  if DEVICE == "cpu":
    args.mixed_precision = False

  wb_run = wandb_utils.setup(TAG, args, "evaluate")
  log_config(vars(args), TAG)

  log_dir = create_directory(f'./experiments/logs/')
  data_dir = create_directory(f'./experiments/data/')
  model_dir = create_directory('./experiments/models/')

  log_path = log_dir + f'{TAG}.txt'
  model_path = './experiments/models/' + f'{args.weights or args.tag}.pth'

  set_seed(SEED)
  log = lambda string='': log_print(string, log_path)

  data_source = datasets.custom_data_source(args.dataset, args.data_dir, args.domain)
  info = data_source.classification_info
  normalize_stats = info.normalize_stats
  tt, tv = datasets.get_classification_transforms(args.image_size, args.image_size, args.image_size, AUG, normalize_stats)
  dataset = datasets.SegmentationDataset(data_source, transform=tv, ignore_bg_images=False)
  loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
  total_steps = args.max_steps or len(loader)

  print(f"dataset steps = {len(loader)}")
  print(f"total steps   = {total_steps}")

  model = Classifier(
    args.architecture,
    channels=info.channels,
    num_classes=info.num_classes,
    mode=args.mode,
    dilated=args.dilated,
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

  scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)

  # Metrics
  sample_ids_ = []
  logits_ = []
  probs_ = []
  targets_ = []

  with torch.no_grad():
    for step, (sample_ids, images, targets, masks) in tqdm(enumerate(loader), mininterval=2.0, total=total_steps):
      try:
        with torch.autocast(device_type=DEVICE, enabled=args.mixed_precision):
          logit = model(images.to(DEVICE))
          probs = torch.sigmoid(logit)

        sample_ids_ += [sample_ids]
        targets_ += [to_numpy(targets)]
        logits_ += [to_numpy(logit)]
        probs_ += [to_numpy(probs)]

        if step >= total_steps:
          break

      except KeyboardInterrupt:
        print("interrupted")
        break

  targets_ = np.concatenate(targets_, 0)
  sample_ids_ = np.concatenate(sample_ids_, 0)
  logits_ = np.concatenate(logits_, 0)
  probs_ = np.concatenate(probs_, 0)
  preds_ = probs_.round()

  if args.save_preds:
    np.savez_compressed(args.save_preds, sample_id=sample_ids_, logit=logits_)

  print("Classification Report")
  print(skmetrics.classification_report(targets_, preds_, target_names=info.classes))

  try:
    precision, recall, f_score, _ = skmetrics.precision_recall_fscore_support(targets_, preds_, average="macro")
    roc = skmetrics.roc_auc_score(targets_, probs_, average="macro")
  except ValueError:
    precision = recall = f_score = roc = 0.

  metric_results = {
    "precision": round(100 * precision, 3),
    "recall": round(100 * recall, 3),
    "f_score": round(100 * f_score, 3),
    "roc_auc": round(100 * roc, 3),
  }

  wandb.log({f"evaluation/{k}": v for k, v in metric_results.items()})
  print(*(f"{metric}={value}" for metric, value in metric_results.items()))

  wb_run.finish()


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
