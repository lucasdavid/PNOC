import argparse
import copy
import os
import sys
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import multiprocessing
from torch.utils.data import Subset
from tqdm import tqdm

from kmeans_pytorch import kmeans
from torch.utils.data import DataLoader

import datasets
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
parser.add_argument('--device', default='cuda', type=str)
# parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--dataset', default='voc12', choices=datasets.DATASOURCES)
parser.add_argument('--data_dir', required=True, type=str)
parser.add_argument('--pred_dir', default=None, type=str)
parser.add_argument('--sample_ids', default=None, type=str)
parser.add_argument('--architecture', default='resnet50', type=str)
parser.add_argument('--mode', default='normal', type=str)  # fix
parser.add_argument('--dilated', default=False, type=str2bool)
parser.add_argument('--tag', default='', type=str)
parser.add_argument('--weights', default='', type=str)
parser.add_argument('--domain', default='train', type=str)
parser.add_argument('--scales', default='0.5,1.0,1.5,2.0', type=str)
parser.add_argument('--exclude_bg_images', default=True, type=str2bool)

try:
  GPUS = os.environ["CUDA_VISIBLE_DEVICES"]
except KeyError:
  GPUS = "0"
GPUS = GPUS.split(",")
GPUS_COUNT = len(GPUS)

normalize_fn = Normalize(*datasets.imagenet_stats())


def preprocess_input(image, scale=None):
  x = copy.deepcopy(image)
  if scale:
    W, H = image.size
    x = x.resize((round(W * scale), round(H * scale)), resample=PIL.Image.BICUBIC)
  x = normalize_fn(x)
  x = x.transpose((2, 0, 1))
  return torch.from_numpy(x)


def print_tensor(x, n=3):
  for i in range(x.shape[0]):
    for j in range(x.shape[1]):
      print(round(x[i, j], n), end=' ')
    print()


def save_feature(model, dataset, save_dir):
  print(f"Dataset {dataset} length:", len(dataset))
  
  tensor_logits = torch.zeros(len(dataset), 20)
  tensor_label = torch.zeros(len(dataset), 20)
  # tensor_cam = torch.zeros(len(dataset),20,32,32)
  tensor_feature = {}
  name2id = dict()
  with torch.no_grad():
    for i, (sample_id, image, label) in enumerate(dataset):
      image = preprocess_input(image)
      logits, features = model(image[None, ...].to(DEVICE), with_features=True)
      name2id[sample_id] = i
      tensor_logits[i] = logits[0].cpu()
      # tensor_cam[i] = cams[0].cpu()
      tensor_feature[i] = features[0].cpu()
      tensor_label[i] = label[0]
      # print(i)

  os.makedirs(save_dir, exist_ok=True)
  torch.save(tensor_logits, os.path.join(save_dir, 'tensor_logits.pt'))
  torch.save(tensor_feature, os.path.join(save_dir, 'tensor_feature.pt'))
  torch.save(tensor_label, os.path.join(save_dir, 'tensor_label.pt'))
  np.save(os.path.join(save_dir, 'name2id.npy'), name2id)


def load_feature_select_and_cluster(
  model,
  dataset,
  workspace,
  feature_dir,
  mask_dir,
  ckpt_path,
  load_cluster=False,
  num_cluster=12,
  select_thres=0.1,
  class_thres=0.9,
  context_thres=0.9,
  context_thres_low=0.05,
  tol=5
):
  tensor_feature = torch.load(os.path.join(feature_dir, 'tensor_feature.pt'))
  tensor_label = torch.load(os.path.join(feature_dir, 'tensor_label.pt'))
  name2id = np.load(os.path.join(feature_dir, 'name2id.npy'), allow_pickle=True).item()
  id2name = {}
  for key in name2id.keys():
    id2name[name2id[key]] = key

  ##### load model for calc similarity
  w = model.classifier.weight.squeeze().detach()

  class_id_to_name = dataset.data_source.classification_info.classes

  ####### feature cluster #####
  centers = {}
  context = {}
  for class_id in range(20):
    print()
    print('class id: ', class_id, ', class name:', class_id_to_name[class_id])
    cluster_result_dir = os.path.join(workspace, 'cluster_result')
    os.makedirs(cluster_result_dir, exist_ok=True)

    if load_cluster:
      cluster_centers = torch.load(os.path.join(cluster_result_dir, 'cluster_centers_' + str(class_id) + '.pt'))
      cluster_centers2 = torch.load(os.path.join(cluster_result_dir, 'cluster_centers2_' + str(class_id) + '.pt'))
      cluster_ids_x = torch.load(os.path.join(cluster_result_dir, 'cluster_ids_x_' + str(class_id) + '.pt'))
      cluster_ids_x2 = torch.load(os.path.join(cluster_result_dir, 'cluster_ids_x2_' + str(class_id) + '.pt'))
    else:
      img_selected = torch.nonzero(tensor_label[:, class_id])[:, 0].numpy()
      feature_selected = []
      feature_not_selected = []
      for idx in img_selected:
        name = id2name[idx]
        cam = np.load(os.path.join(mask_dir, name + '.npy'), allow_pickle=True).item()
        mask = cam['high_res']
        valid_cat = cam['keys']
        feature_map = tensor_feature[idx].permute(1, 2, 0)
        size = feature_map.shape[:2]
        mask = F.interpolate(torch.tensor(mask).unsqueeze(0), size)[0]
        for i in range(len(valid_cat)):
          if valid_cat[i] == class_id:
            mask = mask[i]
            position_selected = mask > select_thres
            position_not_selected = mask < select_thres
            feature_selected.append(feature_map[position_selected])
            feature_not_selected.append(feature_map[position_not_selected])
      feature_selected = torch.cat(feature_selected, 0)
      feature_not_selected = torch.cat(feature_not_selected, 0)

      cluster_ids_x, cluster_centers = kmeans(
        X=feature_selected, num_clusters=num_cluster, distance='cosine', device=torch.device('cuda:0'), tol=tol
      )
      cluster_ids_x2, cluster_centers2 = kmeans(
        X=feature_not_selected, num_clusters=num_cluster, distance='cosine', device=torch.device('cuda:0'), tol=tol
      )

      torch.save(cluster_centers.cpu(), os.path.join(cluster_result_dir, 'cluster_centers_' + str(class_id) + '.pt'))
      torch.save(cluster_centers2.cpu(), os.path.join(cluster_result_dir, 'cluster_centers2_' + str(class_id) + '.pt'))
      torch.save(cluster_ids_x.cpu(), os.path.join(cluster_result_dir, 'cluster_ids_x_' + str(class_id) + '.pt'))
      torch.save(cluster_ids_x2.cpu(), os.path.join(cluster_result_dir, 'cluster_ids_x2_' + str(class_id) + '.pt'))

    ###### calc similarity
    sim = torch.mm(cluster_centers, w.T)
    prob = F.softmax(sim, dim=1)

    ###### select center
    selected_cluster = prob[:, class_id] > class_thres
    cluster_center = cluster_centers[selected_cluster]
    centers[class_id] = cluster_center.cpu()

    ##### print similarity matrix
    # print_tensor(prob.numpy())
    # for i in range(num_cluster):
    #     print(selected_cluster[i].item(), round(prob[i,class_id].item(),3), torch.sum(cluster_ids_x==i).item())

    ###### calc similarity
    sim = torch.mm(cluster_centers2, w.T)
    prob = F.softmax(sim, dim=1)

    ###### select context
    selected_cluster = (prob[:, class_id] > context_thres_low) * (prob[:, class_id] < context_thres)
    cluster_center2 = cluster_centers2[selected_cluster]
    context[class_id] = cluster_center2.cpu()

    ##### print similarity matrix
    # print_tensor(prob.numpy())
    # for i in range(num_cluster):
    #     print(selected_cluster[i].item(), round(prob[i,class_id].item(),3), torch.sum(cluster_ids_x2==i).item())

  # torch.save(centers.cpu(), os.path.join(workspace+'class_ceneters'+'.pt'))
  torch.save(centers, os.path.join(workspace, 'class_ceneters' + '.pt'))
  torch.save(context, os.path.join(workspace, 'class_context' + '.pt'))


def make_lpcam(model, workspace, lpcam_out_dir, ckpt_path, voc12_root, list_name='voc12/train.txt'):
  cluster_centers = torch.load(os.path.join(workspace, 'class_ceneters' + '.pt'))
  cluster_context = torch.load(os.path.join(workspace, 'class_context' + '.pt'))

  data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
  start_time = time.time()
  with torch.no_grad():
    for i, pack in enumerate(tqdm(data_loader)):
      imgs = pack['img']
      label = pack['label'][0]
      img_name = pack['name'][0]
      size = pack['size']
      valid_cat = torch.nonzero(label)[:, 0].numpy()

      # if os.path.exists(os.path.join(cam_out_dir, img_name+'.npy')):
      #     continue

      strided_size = imutils.get_strided_size(size, 4)
      strided_up_size = imutils.get_strided_up_size(size, 16)

      features = []
      for img in imgs:
        feature = model(img[0].cuda())
        features.append((feature[0] + feature[1].flip(-1)))

      strided_cams = []
      highres_cams = []
      for class_id in valid_cat:
        strided_cam = []
        highres_cam = []
        for feature in features:
          h, w = feature.shape[1], feature.shape[2]
          cluster_feature = cluster_centers[class_id]
          att_maps = []
          for j in range(cluster_feature.shape[0]):
            cluster_feature_here = cluster_feature[j].repeat(h, w, 1).cuda()
            feature_here = feature.permute((1, 2, 0)).reshape(h, w, 2048)
            attention_map = F.cosine_similarity(feature_here, cluster_feature_here, 2).unsqueeze(0).unsqueeze(0)
            att_maps.append(attention_map.cpu())
          att_map = torch.mean(torch.cat(att_maps, 0), 0, keepdim=True).cuda()

          context_feature = cluster_context[class_id]
          if context_feature.shape[0] > 0:
            context_attmaps = []
            for j in range(context_feature.shape[0]):
              context_feature_here = context_feature[j]
              context_feature_here = context_feature_here.repeat(h, w, 1).cuda()
              context_attmap = F.cosine_similarity(feature_here, context_feature_here, 2).unsqueeze(0).unsqueeze(0)
              context_attmaps.append(context_attmap.unsqueeze(0))
            context_attmap = torch.mean(torch.cat(context_attmaps, 0), 0)
            att_map = F.relu(att_map - context_attmap)

          attention_map1 = F.interpolate(att_map, strided_size, mode='bilinear', align_corners=False)[:, 0, :, :]
          attention_map2 = F.interpolate(att_map, strided_up_size, mode='bilinear',
                                         align_corners=False)[:, 0, :size[0], :size[1]]
          strided_cam.append(attention_map1.cpu())
          highres_cam.append(attention_map2.cpu())
        strided_cam = torch.mean(torch.cat(strided_cam, 0), 0)
        highres_cam = torch.mean(torch.cat(highres_cam, 0), 0)
        strided_cam = strided_cam / torch.max(strided_cam)
        highres_cam = highres_cam / torch.max(highres_cam)
        strided_cams.append(strided_cam.unsqueeze(0))
        highres_cams.append(highres_cam.unsqueeze(0))
      strided_cams = torch.cat(strided_cams, 0)
      highres_cams = torch.cat(highres_cams, 0)
      np.save(
        os.path.join(lpcam_out_dir, img_name.replace('jpg', 'npy')), {
          "keys": valid_cat,
          "cam": strided_cams,
          "high_res": highres_cams
        }
      )



def run(args):
  ds = datasets.custom_data_source(args.dataset, args.data_dir, args.domain)
  dataset = datasets.ClassificationDataset(ds, ignore_bg_images=args.exclude_bg_images)
  info = ds.classification_info
  print(f'{TAG} dataset={args.dataset} num_classes={info.num_classes}')

  model = Classifier(args.architecture, info.num_classes, mode=args.mode, dilated=args.dilated)
  load_model(model, WEIGHTS_PATH, map_location=torch.device(DEVICE), strict=False)
  model.eval()

  GPUS_COUNT = 1  # force single worker for now.
  scales = [float(scale) for scale in args.scales.split(',')]

  if GPUS_COUNT > 1:
    dataset = [Subset(dataset, np.arange(i, len(dataset), GPUS_COUNT)) for i in range(GPUS_COUNT)]
  else:
    dataset = [dataset]

  if GPUS_COUNT > 1:
    multiprocessing.spawn(_work, nprocs=GPUS_COUNT, args=(model, dataset, scales, PREDS_DIR, DEVICE), join=True)
  else:
    _work(0, model, dataset, scales, PREDS_DIR, DEVICE)


def _work(
  process_id: int,
  model: Classifier,
  dataset: List[datasets.PathsDataset],
  scales: List[float],
  preds_dir: str,
  device: str,
):
  dataset = dataset[process_id]

  save_feature(model, dataset, preds_dir)
  # load_feature_select_and_cluster(
  #   model,
  #   workspace=args.work_space,
  #   feature_dir=os.path.join(args.work_space, 'cam_feature'),
  #   mask_dir=args.cam_out_dir,
  #   ckpt_path=args.cam_weights_name
  # )
  # make_lpcam(
  #   workspace=args.work_space,
  #   lpcam_out_dir=args.lpcam_out_dir,
  #   ckpt_path=args.cam_weights_name,
  #   voc12_root=args.voc12_root,
  #   list_name='voc12/train_aug.txt'
  # )

  # if process_id == 0:
  #   dataset = tqdm(dataset, mininterval=2.0)

  # with torch.no_grad(), torch.cuda.device(process_id):
  #   model.cuda()

  #   for image_id, _, _ in dataset:
  #     npy_path = os.path.join(preds_dir, image_id + '.npy')
  #     if os.path.isfile(npy_path):
  #       continue

  #     image = data_source.get_image(image_id)
  #     label = data_source.get_label(image_id)

  #     W, H = image.size

  #     strided_size = get_strided_size((H, W), 4)
  #     strided_up_size = get_strided_up_size((H, W), 16)

  #     cams = [forward_tta(model, image, scale, device) for scale in scales]

  #     cams_st = [resize_tensor(c.unsqueeze(0), strided_size)[0] for c in cams]
  #     cams_st = torch.sum(torch.stack(cams_st), dim=0)

  #     cams_hr = [resize_tensor(cams.unsqueeze(0), strided_up_size)[0] for cams in cams]
  #     cams_hr = torch.sum(torch.stack(cams_hr), dim=0)[:, :H, :W]

  #     keys = torch.nonzero(torch.from_numpy(label))[:, 0]
  #     cams_st = cams_st[keys]
  #     cams_st /= F.adaptive_max_pool2d(cams_st, (1, 1)) + 1e-5
  #     cams_hr = cams_hr[keys]
  #     cams_hr /= F.adaptive_max_pool2d(cams_hr, (1, 1)) + 1e-5
  #     keys = np.pad(keys + 1, (1, 0), mode='constant')

  #     try:
  #       np.save(npy_path, {"keys": keys, "cam": cams_st.cpu(), "hr_cam": cams_hr.cpu().numpy()})
  #     except:
  #       if os.path.exists(npy_path):
  #         os.remove(npy_path)
  #       raise


def forward_tta(model, ori_image, scale, DEVICE):
  x = preprocess_input(ori_image, scale)
  xf = x.flip(-1)
  images = torch.stack([x, xf])
  images = images.to(DEVICE)

  _, features = model(images, with_cam=True)
  cams = F.relu(features)
  cams = cams[0] + cams[1].flip(-1)

  return cams


if __name__ == '__main__':
  args = parser.parse_args()

  DEVICE = args.device if torch.cuda.is_available() else "cpu"
  SEED = args.seed
  TAG = args.tag
  TAG += '@train' if 'train' in args.domain else '@val'
  TAG += '@lpcam'

  PREDS_DIR = create_directory(args.pred_dir or f'./experiments/predictions/{TAG}/')
  WEIGHTS_PATH = './experiments/models/' + f'{args.weights or args.tag}.pth'

  set_seed(SEED)
  run(args)
