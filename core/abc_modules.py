import torch
import torch.nn as nn

from abc import ABC


class ABC_Model(ABC):

  def global_average_pooling_2d(self, x, keepdims=False):
    x = torch.mean(x.view(x.size(0), x.size(1), -1), -1)
    if keepdims:
      x = x.view(x.size(0), x.size(1), 1, 1)
    return x

  def initialize(self, modules):
    for m in modules:
      if isinstance(m, nn.Conv2d):
        # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        # m.weight.data.normal_(0, math.sqrt(2. / n))
        torch.nn.init.kaiming_normal_(m.weight)

      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def get_parameter_groups(self):
    groups = ([], [], [], [])

    for name, param in self.named_parameters():
      if param.requires_grad:
        if 'model' in name:
          if 'weight' in name:
            groups[0].append(param)
          else:
            groups[1].append(param)

        # scracthed weights
        else:
          if 'weight' in name:
            groups[2].append(param)
          else:
            groups[3].append(param)

    return groups
