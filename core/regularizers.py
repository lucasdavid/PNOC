import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def kernel_usage(w, alpha):
  return w * F.softmax(w, dim=1) * alpha


class Conv2dKU(nn.Conv2d):
  def forward(self, input: Tensor) -> Tensor:
    w = kernel_usage(self.weight, self.out_channels)
    
    return self._conv_forward(input, w, self.bias)

class MinMaxConv2d(nn.Conv2d):
  def forward(self, input: Tensor) -> Tensor:
    sW = self.weight
    c = self.out_channels
    
    w = sW                                     # (k,c,1,1)
    wt = torch.sum(w, axis=1, keepdims=True)   # (k,1,1,1) (total)
    wn = wt - w                                # (k,c,1,1) (total - pos = negative)
    w = sW - wn / torch.maximum(c-1, 1.)       # (k,c,1,1)

    return self._conv_forward(input, w, self.bias)


# class MinMaxConv2d(nn.Conv2d):
#   def forward(self, input: Tensor, target: Tensor) -> Tensor:
#     sW = self.weight
#     c = self.out_channels
    
#     c = torch.sum(target, axis=-1)
#     c = c.view((-1, 1, 1))
    
#     w = target.unsqueeze(1) * sW.unsqueeze(0)  # (b,1,c) * (1,k,c,1,1) = (b,k,c,1,1) (detected)
#     wt = torch.sum(w, axis=2, keepdims=True)   # (b,k,1,1,1) (total)
#     wn = wt - w                                # (b,k,c,1,1) (total - pos = negative)
#     w = sW - wn / torch.maximum(c-1, 1.)       # (b,k,c,1,1)
#     w = torch.sum(w, axis=0)

#     return self._conv_forward(input, w, self.bias)
