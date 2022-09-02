import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

import numpy as np
import importlib
import os
import imageio

import voc12.dataloader
from misc import torchutils, indexing

cudnn.enabled = True


def _work(process_id, model, dataset, args):

  n_gpus = torch.cuda.device_count()
  databin = dataset[process_id]
  data_loader = DataLoader(databin, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)

  with torch.no_grad(), cuda.device(process_id):

    model.cuda()

    for iter, pack in enumerate(data_loader):
      img_name = voc12.dataloader.decode_int_filename(pack['name'][0])
      orig_img_size = np.asarray(pack['size'])

      npy_path = os.path.join(args.sem_seg_out_dir, img_name + '.npy')
      if os.path.isfile(npy_path):
        continue

      edge, dp = model(pack['img'][0].cuda(non_blocking=True))

      cam_dict = np.load(os.path.join(args.cam_out_dir, img_name + '.npy'), allow_pickle=True).item()

      cams = cam_dict['cam']
      keys = cam_dict['keys']

      if args.cam_saved_format == 'irn':
        keys = np.pad(keys + 1, (1, 0), mode='constant')

      cam_downsized_values = cams.cuda()

      rw = indexing.propagate_to_edge(cam_downsized_values, edge, beta=args.beta, exp_times=args.exp_times, radius=5)

      rw_up = F.interpolate(rw, scale_factor=4, mode='bilinear', align_corners=False)
      rw_up = rw_up[..., 0, :orig_img_size[0], :orig_img_size[1]]
      rw_up = rw_up / torch.max(rw_up)

      np.save(npy_path, {"keys": cam_dict['keys'], "rw": rw_up.cpu().numpy()})

      if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
        print("%d " % ((5 * iter + 1) // (len(databin) // 20)), end='')


def run(args):
  print('Config')
  print('sem_seg_bg_thres:', args.sem_seg_bg_thres)
  print('sem_seg_out_dir :', args.sem_seg_out_dir)
  print('cam_saved_format:', args.cam_saved_format)

  if args.cam_saved_format not in ('irn', 'puzzle'):
    raise ValueError(f'Unknown cam format {args.cam_saved_format}. Valid options are "irn" and "puzzle".')

  model = getattr(importlib.import_module(args.irn_network), 'EdgeDisplacement')(
    model_name=args.model_name
  )
  model.load_state_dict(torch.load(args.irn_weights_name), strict=False)
  model.eval()

  n_gpus = torch.cuda.device_count()

  dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(args.infer_list, voc12_root=args.voc12_root, scales=(1.0,))
  dataset = torchutils.split_dataset(dataset, n_gpus)

  multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)

  torch.cuda.empty_cache()
