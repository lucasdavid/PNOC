# Copyright (C) 2021 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from tools.ai.torch_utils import freeze_and_eval, resize_for_tensors
from torchvision import models

from . import ccam, regularizers
from .abc_modules import ABC_Model
from .aff_utils import PathIndex
from .deeplab_utils import ASPP, Decoder
from .mcar import mcar_resnet50, mcar_resnet101
from .puzzle_utils import merge_features, tile_features
#######################################################################
# Normalization
#######################################################################
from .sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class FixedBatchNorm(nn.BatchNorm2d):

  def forward(self, x):
    return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, training=False, eps=self.eps)


def group_norm(features):
  return nn.GroupNorm(4, features)


#######################################################################


def build_backbone(name, dilated, strides, norm_fn):
  if dilated:
    dilation, dilated = 4, True
  else:
    dilation, dilated = 2, False

  if 'resnet38d' == name:
    from .arch_resnet import resnet38d
    out_features = 4096

    model = resnet38d.ResNet38d()
    state_dict = resnet38d.convert_mxnet_to_torch('./experiments/models/resnet_38d.params')
    model.load_state_dict(state_dict, strict=True)

    stage1 = nn.Sequential(model.conv1a)
    stage2 = nn.Sequential(model.b2, model.b2_1, model.b2_2)
    stage3 = nn.Sequential(model.b3, model.b3_1, model.b3_2)
    stage4 = nn.Sequential(model.b4, model.b4_1, model.b4_2, model.b4_3, model.b4_4, model.b4_5)
    stage5 = nn.Sequential(model.b5, model.b5_1, model.b5_2, model.b6, model.b7, model.bn7, nn.ReLU())
  else:
    out_features = 2048

    if 'resnet' in name:
      from .arch_resnet import resnet
      model = resnet.ResNet(
        resnet.Bottleneck, resnet.layers_dic[name], strides=strides or (2, 2, 2, 1), batch_norm_fn=norm_fn
      )

      state_dict = model_zoo.load_url(resnet.urls_dic[name])
      state_dict.pop('fc.weight')
      state_dict.pop('fc.bias')

      model.load_state_dict(state_dict)
    elif 'resnest' in name:
      from .arch_resnest import resnest

      model_fn = getattr(resnest, name)
      model = model_fn(pretrained=True, dilated=dilated, dilation=dilation, norm_layer=norm_fn)

      del model.avgpool
      del model.fc
    elif 'res2net' in name:
      from .res2net import res2net_v1b

      model_fn = getattr(res2net_v1b, name)
      model = model_fn(pretrained=True, strides=strides or (1, 2, 2, 1), norm_layer=norm_fn)

      del model.avgpool
      del model.fc

    stage1 = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
    stage2 = nn.Sequential(model.layer1)
    stage3 = nn.Sequential(model.layer2)
    stage4 = nn.Sequential(model.layer3)
    stage5 = nn.Sequential(model.layer4)

  return out_features, model, (stage1, stage2, stage3, stage4, stage5)


class Backbone(nn.Module, ABC_Model):

  def __init__(
      self,
      model_name,
      mode='fix',
      dilated=False,
      strides=None,
      trainable_stem=True,
  ):
    super().__init__()

    self.mode = mode
    self.trainable_stem = trainable_stem
    
    if mode == 'normal':
      self.norm_fn = nn.BatchNorm2d
    elif mode == 'fix':
      self.norm_fn = FixedBatchNorm
    else:
      raise ValueError(f'Unknown mode {mode}. Must be `normal` or `fix`.')

    (out_features, model, stages) = build_backbone(model_name, dilated, strides, self.norm_fn)
    self.model = model
    self.out_features = out_features
    self.stage1, self.stage2, self.stage3, self.stage4, self.stage5 = stages

    self.not_training = []
    self.from_scratch_layers = []

    if not self.trainable_stem:
      self.not_training.extend([self.stage1])


class Classifier(Backbone):

  def __init__(
    self,
    model_name,
    num_classes=20,
    mode='fix',
    dilated=False,
    strides=None,
    regularization=None,
    trainable_stem=True
  ):
    super().__init__(
      model_name,
      mode=mode,
      dilated=dilated,
      strides=strides,
      trainable_stem=trainable_stem
    )

    self.num_classes = num_classes
    self.regularization = regularization

    cin = self.out_features

    if not regularization or regularization.lower() == 'none':
      self.classifier = nn.Conv2d(cin, num_classes, 1, bias=False)
    elif regularization.lower() in ('kernel_usage', 'ku'):
      self.classifier = regularizers.Conv2dKU(cin, num_classes, 1, bias=False)
    elif regularization.lower() in ('minmax', 'minmaxcam'):
      self.classifier = regularizers.MinMaxConv2d(cin, num_classes, 1, bias=False)
    else:
      raise ValueError(f'Unknown regularization strategy {regularization}.')

    self.from_scratch_layers.extend([self.classifier])
    self.initialize([self.classifier])

  def train(self, mode=True):
    super().train(mode)
    freeze_and_eval(self.not_training)

    return self

  def forward(self, x, with_cam=False):
    x = self.stage1(x)
    x = self.stage2(x)
    x = self.stage3(x)
    x = self.stage4(x)
    x = self.stage5(x)

    if with_cam:
      features = self.classifier(x)
      logits = self.global_average_pooling_2d(features)
      return logits, features
    else:
      x = self.global_average_pooling_2d(x, keepdims=True)
      logits = self.classifier(x).view(-1, self.num_classes)
      return logits


class CCAM(Backbone):

  def __init__(
    self,
    model_name,
    mode='fix',
    dilated=False,
    strides=(1, 2, 2, 1),
    trainable_stem=True,
    cin=1024 + 2048
  ):
    super().__init__(
      model_name,
      mode=mode,
      dilated=dilated,
      strides=strides,
      trainable_stem=trainable_stem
    )

    self.ac_head = ccam.Disentangler(cin)
    self.from_scratch_layers += [self.ac_head]

  def forward(self, x, inference=False):
    x = self.stage1(x)
    x = self.stage2(x)
    x = self.stage3(x)
    x1 = self.stage4(x)
    x2 = self.stage5(x1)

    feats = torch.cat([x2, x1], dim=1)

    return self.ac_head(feats, inference=inference)

    # if with_cam:
    #   features = self.classifier(x2)
    #   logits = self.global_average_pooling_2d(features)
    #   return logits, fg_feats, bg_feats, ccam, features
    # else:
    #   x = self.global_average_pooling_2d(x2, keepdims=True)
    #   logits = self.classifier(x).view(-1, self.num_classes)
    #   return logits, fg_feats, bg_feats, ccam


class AffinityNet(Backbone):

  def __init__(self, model_name, path_index=None, mode='fix', dilated=False, strides=None):
    super().__init__(model_name, mode=mode, dilated=dilated, strides=strides)

    in_features = self.out_features

    if '50' in model_name:
      fc_edge1_features = 64
    else:
      fc_edge1_features = 128

    self.fc_edge1 = nn.Sequential(
      nn.Conv2d(fc_edge1_features, 32, 1, bias=False),
      nn.GroupNorm(4, 32),
      nn.ReLU(inplace=True),
    )
    self.fc_edge2 = nn.Sequential(
      nn.Conv2d(256, 32, 1, bias=False),
      nn.GroupNorm(4, 32),
      nn.ReLU(inplace=True),
    )
    self.fc_edge3 = nn.Sequential(
      nn.Conv2d(512, 32, 1, bias=False),
      nn.GroupNorm(4, 32),
      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
      nn.ReLU(inplace=True),
    )
    self.fc_edge4 = nn.Sequential(
      nn.Conv2d(1024, 32, 1, bias=False),
      nn.GroupNorm(4, 32),
      nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
      nn.ReLU(inplace=True),
    )
    self.fc_edge5 = nn.Sequential(
      nn.Conv2d(in_features, 32, 1, bias=False),
      nn.GroupNorm(4, 32),
      nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
      nn.ReLU(inplace=True),
    )
    self.fc_edge6 = nn.Conv2d(160, 1, 1, bias=True)

    self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4, self.stage5])
    self.edge_layers = nn.ModuleList(
      [self.fc_edge1, self.fc_edge2, self.fc_edge3, self.fc_edge4, self.fc_edge5, self.fc_edge6]
    )

    if path_index is not None:
      self.path_index = path_index
      self.n_path_lengths = len(self.path_index.path_indices)
      for i, pi in enumerate(self.path_index.path_indices):
        self.register_buffer("path_indices_" + str(i), torch.from_numpy(pi))

  def train(self, mode=True):
    super().train(mode)
    self.backbone.eval()

  def forward(self, x, with_affinity=False):
    x1 = self.stage1(x).detach()
    x2 = self.stage2(x1).detach()
    x3 = self.stage3(x2).detach()
    x4 = self.stage4(x3).detach()
    x5 = self.stage5(x4).detach()

    edge1 = self.fc_edge1(x1)
    edge2 = self.fc_edge2(x2)
    edge3 = self.fc_edge3(x3)[..., :edge2.size(2), :edge2.size(3)]
    edge4 = self.fc_edge4(x4)[..., :edge2.size(2), :edge2.size(3)]
    edge5 = self.fc_edge5(x5)[..., :edge2.size(2), :edge2.size(3)]

    edge = self.fc_edge6(torch.cat([edge1, edge2, edge3, edge4, edge5], dim=1))

    if with_affinity:
      return edge, self.to_affinity(torch.sigmoid(edge))
    else:
      return edge

  def get_edge(self, x, image_size=512, stride=4):
    feat_size = (x.size(2) - 1) // stride + 1, (x.size(3) - 1) // stride + 1

    x = F.pad(x, [0, image_size - x.size(3), 0, image_size - x.size(2)])
    edge_out = self.forward(x)
    edge_out = edge_out[..., :feat_size[0], :feat_size[1]]
    edge_out = torch.sigmoid(edge_out[0] / 2 + edge_out[1].flip(-1) / 2)

    return edge_out

  """
    aff = self.to_affinity(torch.sigmoid(edge_out))
    pos_aff_loss = (-1) * torch.log(aff + 1e-5)
    neg_aff_loss = (-1) * torch.log(1. + 1e-5 - aff)
    """

  def to_affinity(self, edge):
    aff_list = []
    edge = edge.view(edge.size(0), -1)

    for i in range(self.n_path_lengths):
      ind = self._buffers["path_indices_" + str(i)]
      ind_flat = ind.view(-1)
      dist = torch.index_select(edge, dim=-1, index=ind_flat)
      dist = dist.view(dist.size(0), ind.size(0), ind.size(1), ind.size(2))
      aff = torch.squeeze(1 - F.max_pool2d(dist, (dist.size(2), 1)), dim=2)
      aff_list.append(aff)
    aff_cat = torch.cat(aff_list, dim=1)
    return aff_cat


class DeepLabv3_Plus(Backbone):

  def __init__(self, model_name, num_classes=21, mode='fix', dilated=False, strides=None, use_group_norm=False):
    # model_name, num_classes, mode=mode, dilated=dilated, strides=strides, regularization=regularization
    super().__init__(model_name, mode=mode, dilated=dilated, strides=strides)

    if use_group_norm:
      norm_fn_for_extra_modules = group_norm
    else:
      norm_fn_for_extra_modules = self.norm_fn

    in_features = self.out_features

    self.aspp = ASPP(in_features, output_stride=16, norm_fn=norm_fn_for_extra_modules)
    self.decoder = Decoder(num_classes, 256, norm_fn_for_extra_modules)

  def forward(self, x, with_cam=False):
    inputs = x

    x = self.stage1(x)
    x = self.stage2(x)
    x_low_level = x

    x = self.stage3(x)
    x = self.stage4(x)
    x = self.stage5(x)

    x = self.aspp(x)
    x = self.decoder(x, x_low_level)
    x = resize_for_tensors(x, inputs.size()[2:], align_corners=True)

    return x


class Seg_Model(Backbone):

  def __init__(self, model_name, num_classes=21):
    super().__init__(model_name, num_classes, mode='fix', dilated=False)

    self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False)

  def forward(self, inputs):
    x = self.stage1(inputs)
    x = self.stage2(x)
    x = self.stage3(x)
    x = self.stage4(x)
    x = self.stage5(x)

    logits = self.classifier(x)
    # logits = resize_for_tensors(logits, inputs.size()[2:], align_corners=False)

    return logits


class CSeg_Model(Backbone):

  def __init__(self, model_name, num_classes=21):
    super().__init__(model_name, num_classes, 'fix')

    if '50' in model_name:
      fc_edge1_features = 64
    else:
      fc_edge1_features = 128

    self.fc_edge1 = nn.Sequential(
      nn.Conv2d(fc_edge1_features, 32, 1, bias=False),
      nn.GroupNorm(4, 32),
      nn.ReLU(inplace=True),
    )
    self.fc_edge2 = nn.Sequential(
      nn.Conv2d(256, 32, 1, bias=False),
      nn.GroupNorm(4, 32),
      nn.ReLU(inplace=True),
    )
    self.fc_edge3 = nn.Sequential(
      nn.Conv2d(512, 32, 1, bias=False),
      nn.GroupNorm(4, 32),
      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
      nn.ReLU(inplace=True),
    )
    self.fc_edge4 = nn.Sequential(
      nn.Conv2d(1024, 32, 1, bias=False),
      nn.GroupNorm(4, 32),
      nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
      nn.ReLU(inplace=True),
    )
    self.fc_edge5 = nn.Sequential(
      nn.Conv2d(2048, 32, 1, bias=False),
      nn.GroupNorm(4, 32),
      nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
      nn.ReLU(inplace=True),
    )
    self.fc_edge6 = nn.Conv2d(160, num_classes, 1, bias=True)

  def forward(self, x):
    x1 = self.stage1(x)
    x2 = self.stage2(x1)
    x3 = self.stage3(x2)
    x4 = self.stage4(x3)
    x5 = self.stage5(x4)

    edge1 = self.fc_edge1(x1)
    edge2 = self.fc_edge2(x2)
    edge3 = self.fc_edge3(x3)[..., :edge2.size(2), :edge2.size(3)]
    edge4 = self.fc_edge4(x4)[..., :edge2.size(2), :edge2.size(3)]
    edge5 = self.fc_edge5(x5)[..., :edge2.size(2), :edge2.size(3)]

    logits = self.fc_edge6(torch.cat([edge1, edge2, edge3, edge4, edge5], dim=1))
    # logits = resize_for_tensors(logits, x.size()[2:], align_corners=True)

    return logits
