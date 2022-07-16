import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from .arch_resnet import resnet


def cos_simi(embedded_fg, embedded_bg):
  embedded_fg = F.normalize(embedded_fg, dim=1)
  embedded_bg = F.normalize(embedded_bg, dim=1)
  sim = torch.matmul(embedded_fg, embedded_bg.T)

  return torch.clamp(sim, min=0.0005, max=0.9995)


def cos_distance(embedded_fg, embedded_bg):
  embedded_fg = F.normalize(embedded_fg, dim=1)
  embedded_bg = F.normalize(embedded_bg, dim=1)
  sim = torch.matmul(embedded_fg, embedded_bg.T)

  return 1 - sim


def l2_distance(embedded_fg, embedded_bg):
  N, C = embedded_fg.size()

  # embedded_fg = F.normalize(embedded_fg, dim=1)
  # embedded_bg = F.normalize(embedded_bg, dim=1)

  embedded_fg = embedded_fg.unsqueeze(1).expand(N, N, C)
  embedded_bg = embedded_bg.unsqueeze(0).expand(N, N, C)

  return torch.pow(embedded_fg - embedded_bg, 2).sum(2) / C


# Minimize Similarity, e.g., push representation of foreground and background apart.
class SimMinLoss(nn.Module):

  def __init__(self, metric='cos', reduction='mean'):
    super(SimMinLoss, self).__init__()
    self.metric = metric
    self.reduction = reduction

  def forward(self, embedded_bg, embedded_fg):
    """
        :param embedded_fg: [N, C]
        :param embedded_bg: [N, C]
        :return:
        """
    if self.metric == 'l2':
      raise NotImplementedError
    elif self.metric == 'cos':
      sim = cos_simi(embedded_bg, embedded_fg)
      loss = -torch.log(1 - sim)
    else:
      raise NotImplementedError

    if self.reduction == 'mean':
      return torch.mean(loss)
    elif self.reduction == 'sum':
      return torch.sum(loss)


# Maximize Similarity, e.g., pull representation of background and background together.
class SimMaxLoss(nn.Module):

  def __init__(self, metric='cos', alpha=0.25, reduction='mean'):
    super(SimMaxLoss, self).__init__()
    self.metric = metric
    self.alpha = alpha
    self.reduction = reduction

  def forward(self, embedded_bg):
    """
        :param embedded_fg: [N, C]
        :param embedded_bg: [N, C]
        :return:
        """
    if self.metric == 'l2':
      raise NotImplementedError

    elif self.metric == 'cos':
      sim = cos_simi(embedded_bg, embedded_bg)
      loss = -torch.log(sim)
      loss[loss < 0] = 0
      _, indices = sim.sort(descending=True, dim=1)
      _, rank = indices.sort(dim=1)
      rank = rank - 1
      rank_weights = torch.exp(-rank.float() * self.alpha)
      loss = loss * rank_weights
    else:
      raise NotImplementedError

    if self.reduction == 'mean':
      return torch.mean(loss)
    elif self.reduction == 'sum':
      return torch.sum(loss)


class Disentangler(nn.Module):

  def __init__(self, cin):
    super(Disentangler, self).__init__()

    self.activation_head = nn.Conv2d(cin, 1, kernel_size=3, padding=1, bias=False)
    self.bn_head = nn.BatchNorm2d(1)

  def forward(self, x, inference=False):
    N, C, H, W = x.size()
    if inference:
      ccam = self.bn_head(self.activation_head(x))
    else:
      ccam = torch.sigmoid(self.bn_head(self.activation_head(x)))

    ccam_ = ccam.reshape(N, 1, H * W)  # [N, 1, H*W]
    x = x.reshape(N, C, H * W).permute(0, 2, 1).contiguous()  # [N, H*W, C]
    fg_feats = torch.matmul(ccam_, x) / (H * W)  # [N, 1, C]
    bg_feats = torch.matmul(1 - ccam_, x) / (H * W)  # [N, 1, C]

    return fg_feats.reshape(x.size(0), -1), bg_feats.reshape(x.size(0), -1), ccam


def resnet50(pretrained=False, stride=None, num_classes=1000, **kwargs):
  """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        :param pretrained:
        :param stride:
    """
  if stride is None:
    stride = [1, 2, 2, 1]
  model = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], stride=stride, **kwargs)
  if pretrained:
    model.load_state_dict(model_zoo.load_url(resnet.urls_dic['resnet50']), strict=True)
  model.fc = nn.Linear(512 * resnet.Bottleneck.expansion, num_classes)
  return model


class ResNetSeries(nn.Module):

  def __init__(self, pretrained):
    super(ResNetSeries, self).__init__()

    if pretrained == 'supervised':
      print(f'Loading supervised pretrained parameters!')
      model = resnet50(pretrained=True)
    elif pretrained == 'mocov2':
      print(f'Loading unsupervised {pretrained} pretrained parameters!')
      model = resnet50(pretrained=False)
      checkpoint = torch.load('moco_r50_v2-e3b0c442.pth', map_location="cpu")
      model.load_state_dict(checkpoint['state_dict'], strict=False)
    elif pretrained == 'detco':
      print(f'Loading unsupervised {pretrained} pretrained parameters!')
      model = resnet50(pretrained=False)
      checkpoint = torch.load('detco_200ep.pth', map_location="cpu")
      model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
      raise NotImplementedError

    self.conv1 = model.conv1
    self.bn1 = model.bn1
    self.relu = model.relu
    self.maxpool = model.maxpool
    self.layer1 = model.layer1
    self.layer2 = model.layer2
    self.layer3 = model.layer3
    self.layer4 = model.layer4

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x1 = self.layer3(x)
    x2 = self.layer4(x1)

    return torch.cat([x2, x1], dim=1)


class Network(nn.Module):

  def __init__(self, pretrained='mocov2', cin=2048 + 1024):
    super(Network, self).__init__()

    self.backbone = ResNetSeries(pretrained=pretrained)
    self.ac_head = Disentangler(cin)
    self.from_scratch_layers = [self.ac_head]

  def forward(self, x, inference=False):

    feats = self.backbone(x)
    fg_feats, bg_feats, ccam = self.ac_head(feats, inference=inference)

    return fg_feats, bg_feats, ccam

  def get_parameter_groups(self):
    groups = ([], [], [], [])
    print('======================================================')
    for m in self.modules():
      if isinstance(m, (nn.Conv2d, nn.modules.normalization.GroupNorm)):
        if m.weight.requires_grad:
          if m in self.from_scratch_layers:
            groups[2].append(m.weight)
          else:
            groups[0].append(m.weight)

        if m.bias is not None and m.bias.requires_grad:
          if m in self.from_scratch_layers:
            groups[3].append(m.bias)
          else:
            groups[1].append(m.bias)

    return groups
