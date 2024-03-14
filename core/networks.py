# Copyright (C) 2021 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from tools.ai.torch_utils import (gap2d, resize_tensor,
                                  set_trainable_layers)

from . import ccam, regularizers
from .deeplab_utils import ASPP, Decoder
from .mcar import mcar_resnet50, mcar_resnet101


class FixedBatchNorm(nn.BatchNorm2d):

  def forward(self, x):
    return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, training=False, eps=self.eps)


def group_norm(features):
  return nn.GroupNorm(4, features)


#######################################################################


def build_backbone(name, dilated, strides, norm_fn, weights='imagenet', **kwargs):
  if 'resnet38d' == name:
    from .backbones.arch_resnet import resnet38d

    model = resnet38d.ResNet38d()
    state_dict = resnet38d.convert_mxnet_to_torch('./experiments/models/resnet_38d.params')
    model.load_state_dict(state_dict, strict=True)

    stages = (
      nn.Sequential(model.conv1a, model.b2, model.b2_1, model.b2_2),
      nn.Sequential(model.b3, model.b3_1, model.b3_2),
      nn.Sequential(model.b4, model.b4_1, model.b4_2, model.b4_3, model.b4_4, model.b4_5),
      nn.Sequential(model.b5, model.b5_1, model.b5_2),
      nn.Sequential(model.b6, model.b7, model.bn7, nn.ReLU()),
    )

  elif "swin" in name:
    if "swinv2" in name:
      from .backbones import swin_transformer_v2 as swin_mod

      model_fn = getattr(swin_mod, name)
      model = model_fn(**kwargs)

      stages = (
        nn.Sequential(model.patch_embed, model.pos_drop),
        *model.layers[:3],
        nn.Sequential(model.layers[3], model.norm, swin_mod.TransposeLayer((1, 2))),
      )
    else:
      from .backbones import swin_transformer as swin_mod

      model_fn = getattr(swin_mod, name)
      model = model_fn(out_indices=(3,), **kwargs)

      stages = (nn.Sequential(model.patch_embed, model.pos_drop), *model.layers)

    if weights and weights != 'imagenet':
      print(f'loading weights from {weights}')
      checkpoint = torch.load(weights, map_location="cpu")
      checkpoint = checkpoint["model"]
      del checkpoint["head.weight"]
      del checkpoint["head.bias"]

      model.load_state_dict(checkpoint, strict=False)

  elif "mit" in name:
    from .backbones import mix_transformer

    model_fn = getattr(mix_transformer, name)
    model = model_fn(**kwargs)

    stages = (
      nn.Sequential(model.patch_embed1, model.block1, model.norm1),
      nn.Sequential(model.patch_embed2, model.block2, model.norm2),
      nn.Sequential(model.patch_embed3, model.block3, model.norm3),
      nn.Sequential(model.patch_embed4, model.block4, model.norm4),
    )

    if weights and weights != "imagenet":
      print(f'loading weights from {weights}')
      checkpoint = torch.load(weights, map_location="cpu")
      del checkpoint["head.weight"]
      del checkpoint["head.bias"]

      model.load_state_dict(checkpoint)

  else:
    if 'resnet' in name:
      from .backbones.arch_resnet import resnet
      if dilated:
        strides = strides or (1, 2, 1, 1)
        dilations = (1, 1, 2, 4)
      else:
        strides = strides or (1, 2, 2, 1)
        dilations = (1, 1, 1, 2)
      model = resnet.ResNet(resnet.Bottleneck, resnet.layers_dic[name], strides=strides, dilations=dilations, batch_norm_fn=norm_fn)

      if weights == 'imagenet':
        print(f'loading weights from {resnet.urls_dic[name]}')
        state_dict = model_zoo.load_url(resnet.urls_dic[name])
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')

        model.load_state_dict(state_dict)
    elif 'resnest' in name:
      from .backbones.arch_resnest import resnest
      dilation = 4 if dilated else 2

      pretrained = weights == "imagenet"
      model_fn = getattr(resnest, name)
      model = model_fn(pretrained=pretrained, dilated=dilated, dilation=dilation, norm_layer=norm_fn)
      if pretrained:
        print(f'loading weights from {resnest.resnest_model_urls[name]}')

      del model.avgpool
      del model.fc
    elif 'res2net' in name:
      from .backbones.res2net import res2net_v1b

      pretrained = weights == "imagenet"
      model_fn = getattr(res2net_v1b, name)
      model = model_fn(pretrained=pretrained, strides=strides or (1, 2, 2, 2), norm_layer=norm_fn)
      if pretrained:
        print(f'loading pretrained weights')

      del model.avgpool
      del model.fc

    if weights and weights != 'imagenet':
      print(f'loading weights from {weights}')
      checkpoint = torch.load(weights, map_location="cpu")
      model.load_state_dict(checkpoint['state_dict'], strict=False)

    stages = (
      nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool),
      model.layer1,
      model.layer2,
      model.layer3,
      model.layer4,
    )

  return model, stages


class Backbone(nn.Module):

  def __init__(
    self,
    model_name,
    weights='imagenet',
    mode='fix',
    dilated=False,
    strides=None,
    trainable_stem=True,
    trainable_stage4=True,
    trainable_backbone=True,
    backbone_kwargs={},
  ):
    super().__init__()

    self.mode = mode
    self.trainable_stem = trainable_stem
    self.trainable_stage4 = trainable_stage4
    self.trainable_backbone = trainable_backbone
    self.not_training = []
    self.from_scratch_layers = []

    if mode == 'normal':
      self.norm_fn = nn.BatchNorm2d
    elif mode == 'fix':
      self.norm_fn = FixedBatchNorm
    else:
      raise ValueError(f'Unknown mode {mode}. Must be `normal` or `fix`.')

    backbone, stages = build_backbone(
      name=model_name, dilated=dilated, strides=strides, norm_fn=self.norm_fn, weights=weights, **backbone_kwargs,
    )

    self.backbone = backbone
    self.stages = stages

    if not self.trainable_backbone:
      for s in stages:
        set_trainable_layers(s, trainable=False)
      self.not_training.extend(stages)
    else:
      if not self.trainable_stage4:
        self.not_training.extend(stages[:-1])
        for s in stages[:-1]:
          set_trainable_layers(s, trainable=False)

      elif not self.trainable_stem:
        set_trainable_layers(stages[0], trainable=False)
        self.not_training.append(stages[0])

      if self.mode == "fix":
        for s in stages:
          set_trainable_layers(s, torch.nn.BatchNorm2d, trainable=False)
          self.not_training.extend([m for m in s.modules() if isinstance(m, torch.nn.BatchNorm2d)])

  def initialize(self, modules):
    for m in modules:
      if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
      elif isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

  def get_parameter_groups(self, exclude_partial_names=(), with_names=False):
    names = ([], [], [], [])
    groups = ([], [], [], [])

    scratch_parameters = set()
    all_parameters = set()

    for layer in self.from_scratch_layers:
      for name, param in layer.named_parameters():
        if param in all_parameters:
          continue
        scratch_parameters.add(param)
        all_parameters.add(param)

        if not param.requires_grad:
          continue
        for p in exclude_partial_names:
          if p in name:
            continue

        idx = 2 if "weight" in name else 3
        names[idx].append(name)
        groups[idx].append(param)

    for name, param in self.named_parameters():
      if param in all_parameters:
        continue
      all_parameters.add(param)

      if not param.requires_grad or param in scratch_parameters:
        continue
      for p in exclude_partial_names:
        if p in name:
          continue

      idx = 0 if "weight" in name else 1
      names[idx].append(name)
      groups[idx].append(param)

    if with_names:
      return groups, names

    return groups

  def train(self, mode=True):
    super().train(mode)
    for m in self.not_training:
      m.eval()
    return self


class Classifier(Backbone):

  def __init__(
    self,
    model_name,
    num_classes=20,
    backbone_weights="imagenet",
    mode='fix',
    dilated=False,
    strides=None,
    trainable_stem=True,
    trainable_stage4=True,
    trainable_backbone=True,
    **backbone_kwargs,
  ):
    super().__init__(
      model_name,
      weights=backbone_weights,
      mode=mode,
      dilated=dilated,
      strides=strides,
      trainable_stem=trainable_stem,
      trainable_stage4=trainable_stage4,
      trainable_backbone=trainable_backbone,
      backbone_kwargs=backbone_kwargs,
    )

    self.num_classes = num_classes

    cin = self.backbone.outplanes
    self.classifier = nn.Conv2d(cin, num_classes, 1, bias=False)

    self.from_scratch_layers.extend([self.classifier])
    self.initialize([self.classifier])

  def forward(self, x, with_cam=False):
    outs = self.backbone(x)
    x = outs[-1] if isinstance(outs, tuple) else outs

    if with_cam:
      features = self.classifier(x)
      logits = gap2d(features)
      return logits, features
    else:
      x = gap2d(x, keepdims=True)
      logits = self.classifier(x).view(-1, self.num_classes)
      return logits


class CCAM(Backbone):

  def __init__(
    self,
    model_name,
    weights='imagenet',
    mode='fix',
    dilated=False,
    strides=None,
    trainable_stem=True,
  ):
    super().__init__(
      model_name, weights=weights, mode=mode, dilated=dilated, strides=strides, trainable_stem=trainable_stem
    )

    self.ac_head = ccam.Disentangler(self.backbone.stage_features[-2] + self.backbone.outplanes)
    self.from_scratch_layers += [*self.ac_head.modules()]
    self.initialize(self.ac_head.modules())

  def forward(self, x):
    outs = self.backbone(x)
    x1 = outs[-2]
    x2 = outs[-1]

    feats = torch.cat([x2, x1], dim=1)

    return self.ac_head(feats)


class AffinityNet(Backbone):

  def __init__(self, model_name, path_index=None, mode='fix', dilated=False, strides=None, trainable_backbone=False):
    super().__init__(
      model_name,
      mode=mode,
      dilated=dilated,
      strides=strides,
      trainable_backbone=trainable_backbone,
    )

    in_features = self.backbone.outplanes

    self.not_training

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
    outs = self.backbone(x)
    x1, x2, x3, x4, x5 = (o.detach() for o in outs)

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

    H, W = x.shape[2:]
    x = F.pad(x, [0, max(image_size - H, 0), 0, max(image_size - W, 0)])

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


class DeepLabV3Plus(Backbone):

  def __init__(
      self,
      model_name,
      num_classes=21,
      mode='fix',
      backbone_weights="imagenet",
      dilated=False,
      strides=None,
      use_group_norm=False,
      trainable_stem=True,
      trainable_backbone=True,
  ):
    super().__init__(
      model_name,
      mode=mode,
      dilated=dilated,
      strides=strides,
      trainable_stem=trainable_stem,
      trainable_backbone=trainable_backbone,
      weights=backbone_weights,
    )

    in_features = self.backbone.outplanes
    low_level_in_features = self.backbone.stage_features[0]

    norm_fn = group_norm if use_group_norm else nn.BatchNorm2d

    self.aspp = ASPP(in_features, output_stride=16, norm_fn=norm_fn)
    self.decoder = Decoder(num_classes, low_level_in_features, norm_fn)

    self.from_scratch_layers += [*self.aspp.modules(), *self.decoder.modules()]

  def forward(self, x, with_cam=False):
    inputs = x

    outs = self.backbone(x)
    x_low_level, x = outs[0], outs[-1]

    x = self.aspp(x)
    x = self.decoder(x, x_low_level)
    x = resize_tensor(x, inputs.size()[2:], align_corners=True)

    return x



class Segformer(nn.Module):
  """
  Deeplabv3plus implememts
  This module has five components:

  self.backbone
  self.aspp
  self.projector: an 1x1 conv for lowlevel feature projection
  self.preclassifier: an 3x3 conv for feature mixing, before final classification
  self.classifier: last 1x1 conv for output classification results

  Args:
      backbone: Dict, configs for backbone
      decoder: Dict, configs for decoder

  NOTE: The bottleneck has only one 3x3 conv by default, some implements stack
      two 3x3 convs
  """
  def __init__(
    self,
    model_name,
    num_classes=21,
    mode='fix',
    backbone_weights="imagenet",
    dilated=False,
    strides=None,
    use_group_norm=False,
    trainable_stem=True,
    trainable_backbone=True,
    decoder_in_channels=[64, 128, 320, 512],
    decoder_channels=256,
    decoder_feature_strides=[4, 8, 16, 32],
    decoder_in_index=[0, 1, 2, 3],
    decoder_embed_dim=768,
    decoder_dropout_ratio=0.1,
    decoder_norm_layer="BatchNorm2d",
    decoder_align_corners=False,
  ):
    super(Segformer, self).__init__()
    
    self.trainable_backbone = trainable_backbone
    self.trainable_stem = trainable_stem
    self.align_corners = decoder_align_corners
    self.mode = mode
    
    norm_layer = group_norm if use_group_norm else getattr(nn, decoder_norm_layer)

    self.backbone = get_mit(variety=model_name, pretrain=backbone_weights)
    self.decoder = SegFormerHead(
      decoder_in_channels,
      channels=decoder_channels,
      feature_strides=decoder_feature_strides,
      embed_dim=decoder_embed_dim,
      dropout_ratio=decoder_dropout_ratio,
      norm_layer=norm_layer,
      in_index=decoder_in_index,
    )
    #self.projector = nn.Sequential( 
    #    nn.Conv2d(
    #        decoder.settings.lowlevel_in_channels,
    #        decoder.settings.lowlevel_channels,
    #        kernel_size=1, bias=False),
    #    norm_layer(decoder.settings.lowlevel_channels),
    #    nn.ReLU(inplace=True),
    #)
    #self.pre_classifier = DepthwiseSeparableConv(
    #    decoder.settings.norm_layer,
    #    channels + decoder.settings.lowlevel_channels,
    #    channels, 3, padding=1
    #)

    self.classifier = nn.Conv2d(decoder_channels, num_classes, 1, 1)

    stages = (
      nn.Sequential(self.backbone.patch_embed1, self.backbone.block1, self.backbone.norm1),
      nn.Sequential(self.backbone.patch_embed2, self.backbone.block2, self.backbone.norm2),
      nn.Sequential(self.backbone.patch_embed3, self.backbone.block3, self.backbone.norm3),
      nn.Sequential(self.backbone.patch_embed4, self.backbone.block4, self.backbone.norm4),
    )

    if not self.trainable_backbone:
      for s in stages:
        set_trainable_layers(s, trainable=False)
      self.not_training.extend(stages)
    else:
      if not self.trainable_stem:
        set_trainable_layers(stages[0], trainable=False)
        self.not_training.append(stages[0])

      if self.mode == "fix":
        for s in stages:
          set_trainable_layers(s, torch.nn.BatchNorm2d, trainable=False)
          self.not_training.extend([m for m in s.modules() if isinstance(m, torch.nn.BatchNorm2d)])

    #init_weight(self.projector)
    #init_weight(self.pre_classifier)
    init_weight(self.classifier)

  def forward(self, x: Tensor, with_embeddings: bool = False) -> Tensor:
    size = (x.shape[2], x.shape[3])
    output = self.backbone(x)
    output = self.decoder(output)
    #output = self.pre_classifier(output)
    out = {}
    out['embeddings'] = output
    output = self.classifier(output)
    out['pre_logits'] = output
    output = F.interpolate(output, size=size, mode='bilinear', align_corners=self.align_corners)
    out['logits'] = output

    if with_embeddings:
      return out
    else:
      return output
