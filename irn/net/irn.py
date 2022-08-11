import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


class FixedBatchNorm(nn.BatchNorm2d):

  def forward(self, input):
    return F.batch_norm(
      input, self.running_mean, self.running_var, self.weight, self.bias, training=False, eps=self.eps
    )


def build_backbone(name, dilated, strides, norm_fn):
  if dilated:
    dilation, dilated = 4, True
  else:
    dilation, dilated = 2, False

  if 'resnet38d' == name:
    from net.arch_resnet import resnet38d
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
      from net.arch_resnet import resnet
      model = resnet.ResNet(resnet.Bottleneck, resnet.layers_dic[name], strides=strides, batch_norm_fn=norm_fn)

      state_dict = model_zoo.load_url(resnet.urls_dic[name])
      state_dict.pop('fc.weight')
      state_dict.pop('fc.bias')

      model.load_state_dict(state_dict)
    else:
      from net.arch_resnest import resnest

      model_fn = getattr(resnest, name)
      model = model_fn(pretrained=True, dilated=dilated, dilation=dilation, norm_layer=norm_fn)

      del model.avgpool
      del model.fc

    stage1 = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
    stage2 = nn.Sequential(model.layer1)
    stage3 = nn.Sequential(model.layer2)
    stage4 = nn.Sequential(model.layer3)
    stage5 = nn.Sequential(model.layer4)

  return out_features, model, (stage1, stage2, stage3, stage4, stage5)


class Net(nn.Module):

  def __init__(self, model_name, mode='fix', dilated=False, strides=(2, 2, 2, 1)):
    super(Net, self).__init__()

    self.mode = mode
    if mode == 'normal':
      self.norm_fn = nn.BatchNorm2d
    elif mode == 'fix':
      self.norm_fn = FixedBatchNorm
    else:
      raise ValueError(f'Unknown mode {mode}. Must be `normal` or `fix`.')

    if '50' in model_name:
      fc_edge1_features = 64
    else:
      fc_edge1_features = 128

    (out_features, model, stages) = build_backbone(model_name, dilated, strides, self.norm_fn)
    self.model = model
    self.out_features = out_features
    self.stage1, self.stage2, self.stage3, self.stage4, self.stage5 = stages

    self.mean_shift = Net.MeanShift(2)

    # branch: class boundary detection
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
      nn.Conv2d(out_features, 32, 1, bias=False),
      nn.GroupNorm(4, 32),
      nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
      nn.ReLU(inplace=True),
    )
    self.fc_edge6 = nn.Conv2d(160, 1, 1, bias=True)

    # branch: displacement field
    self.fc_dp1 = nn.Sequential(
      nn.Conv2d(fc_edge1_features, 64, 1, bias=False),
      nn.GroupNorm(8, 64),
      nn.ReLU(inplace=True),
    )
    self.fc_dp2 = nn.Sequential(
      nn.Conv2d(256, 128, 1, bias=False),
      nn.GroupNorm(16, 128),
      nn.ReLU(inplace=True),
    )
    self.fc_dp3 = nn.Sequential(
      nn.Conv2d(512, 256, 1, bias=False),
      nn.GroupNorm(16, 256),
      nn.ReLU(inplace=True),
    )
    self.fc_dp4 = nn.Sequential(
      nn.Conv2d(1024, 256, 1, bias=False),
      nn.GroupNorm(16, 256),
      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
      nn.ReLU(inplace=True),
    )
    self.fc_dp5 = nn.Sequential(
      nn.Conv2d(out_features, 256, 1, bias=False),
      nn.GroupNorm(16, 256),
      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
      nn.ReLU(inplace=True),
    )
    self.fc_dp6 = nn.Sequential(
      nn.Conv2d(768, 256, 1, bias=False),
      nn.GroupNorm(16, 256),
      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
      nn.ReLU(inplace=True),
    )
    self.fc_dp7 = nn.Sequential(
      nn.Conv2d(448, 256, 1, bias=False), nn.GroupNorm(16, 256), nn.ReLU(inplace=True),
      nn.Conv2d(256, 2, 1, bias=False), self.mean_shift
    )

    self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4, self.stage5])
    self.edge_layers = nn.ModuleList(
      [self.fc_edge1, self.fc_edge2, self.fc_edge3, self.fc_edge4, self.fc_edge5, self.fc_edge6]
    )
    self.dp_layers = nn.ModuleList(
      [self.fc_dp1, self.fc_dp2, self.fc_dp3, self.fc_dp4, self.fc_dp5, self.fc_dp6, self.fc_dp7]
    )

  class MeanShift(nn.Module):

    def __init__(self, num_features):
      super(Net.MeanShift, self).__init__()
      self.register_buffer('running_mean', torch.zeros(num_features))

    def forward(self, input):
      if self.training:
        return input
      return input - self.running_mean.view(1, 2, 1, 1)

  def forward(self, x):
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
    edge_out = self.fc_edge6(torch.cat([edge1, edge2, edge3, edge4, edge5], dim=1))

    dp1 = self.fc_dp1(x1)
    dp2 = self.fc_dp2(x2)
    dp3 = self.fc_dp3(x3)
    dp4 = self.fc_dp4(x4)[..., :dp3.size(2), :dp3.size(3)]
    dp5 = self.fc_dp5(x5)[..., :dp3.size(2), :dp3.size(3)]

    dp_up3 = self.fc_dp6(torch.cat([dp3, dp4, dp5], dim=1))[..., :dp2.size(2), :dp2.size(3)]
    dp_out = self.fc_dp7(torch.cat([dp1, dp2, dp_up3], dim=1))

    return edge_out, dp_out

  def trainable_parameters(self):
    return (tuple(self.edge_layers.parameters()), tuple(self.dp_layers.parameters()))

  def train(self, mode=True):
    super().train(mode)
    self.backbone.eval()


class AffinityDisplacementLoss(Net):

  path_indices_prefix = "path_indices"

  def __init__(self, path_index, model_name, mode='fix', dilated=False, strides=(2, 2, 2, 1)):

    super(AffinityDisplacementLoss, self).__init__(model_name, mode, dilated, strides)

    self.path_index = path_index

    self.n_path_lengths = len(path_index.path_indices)
    for i, pi in enumerate(path_index.path_indices):
      self.register_buffer(AffinityDisplacementLoss.path_indices_prefix + str(i), torch.from_numpy(pi))

    self.register_buffer(
      'disp_target',
      torch.unsqueeze(torch.unsqueeze(torch.from_numpy(path_index.search_dst).transpose(1, 0), 0), -1).float()
    )

  def to_affinity(self, edge):
    aff_list = []
    edge = edge.view(edge.size(0), -1)

    for i in range(self.n_path_lengths):
      ind = self._buffers[AffinityDisplacementLoss.path_indices_prefix + str(i)]
      ind_flat = ind.view(-1)
      dist = torch.index_select(edge, dim=-1, index=ind_flat)
      dist = dist.view(dist.size(0), ind.size(0), ind.size(1), ind.size(2))
      aff = torch.squeeze(1 - F.max_pool2d(dist, (dist.size(2), 1)), dim=2)
      aff_list.append(aff)
    aff_cat = torch.cat(aff_list, dim=1)

    return aff_cat

  def to_pair_displacement(self, disp):
    height, width = disp.size(2), disp.size(3)
    radius_floor = self.path_index.radius_floor

    cropped_height = height - radius_floor
    cropped_width = width - 2 * radius_floor

    disp_src = disp[:, :, :cropped_height, radius_floor:radius_floor + cropped_width]

    disp_dst = [
      disp[:, :, dy:dy + cropped_height, radius_floor + dx:radius_floor + dx + cropped_width]
      for dy, dx in self.path_index.search_dst
    ]
    disp_dst = torch.stack(disp_dst, 2)

    pair_disp = torch.unsqueeze(disp_src, 2) - disp_dst
    pair_disp = pair_disp.view(pair_disp.size(0), pair_disp.size(1), pair_disp.size(2), -1)

    return pair_disp

  def to_displacement_loss(self, pair_disp):
    return torch.abs(pair_disp - self.disp_target)

  def forward(self, *inputs):
    x, return_loss = inputs
    edge_out, dp_out = super().forward(x)

    if return_loss is False:
      return edge_out, dp_out

    aff = self.to_affinity(torch.sigmoid(edge_out))
    pos_aff_loss = (-1) * torch.log(aff + 1e-5)
    neg_aff_loss = (-1) * torch.log(1. + 1e-5 - aff)

    pair_disp = self.to_pair_displacement(dp_out)
    dp_fg_loss = self.to_displacement_loss(pair_disp)
    dp_bg_loss = torch.abs(pair_disp)

    return pos_aff_loss, neg_aff_loss, dp_fg_loss, dp_bg_loss


class EdgeDisplacement(Net):

  def __init__(self, model_name, crop_size=512, stride=4, mode='fix', dilated=False, strides=(2, 2, 2, 1)):
    super(EdgeDisplacement, self).__init__(model_name, mode, dilated, strides)
    self.crop_size = crop_size
    self.stride = stride

  def forward(self, x):
    feat_size = (x.size(2) - 1) // self.stride + 1, (x.size(3) - 1) // self.stride + 1

    x = F.pad(x, [0, self.crop_size - x.size(3), 0, self.crop_size - x.size(2)])
    edge_out, dp_out = super().forward(x)
    edge_out = edge_out[..., :feat_size[0], :feat_size[1]]
    dp_out = dp_out[..., :feat_size[0], :feat_size[1]]

    edge_out = torch.sigmoid(edge_out[0] / 2 + edge_out[1].flip(-1) / 2)
    dp_out = dp_out[0]

    return edge_out, dp_out
