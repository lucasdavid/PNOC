import torch

from .torch_utils import *


class PolyOptimizerMixin:

  def __init__(self, params, lr, max_step, momentum=0.9, **kwargs):
    super().__init__(params, lr, **kwargs)

    self.global_step = 0
    self.max_step = max_step
    self.momentum = momentum

    self.__initial_lr = [group['lr'] for group in self.param_groups]

  def step(self, closure=None):
    if self.global_step < self.max_step:
      lr_mult = (1 - self.global_step / self.max_step)**self.momentum

      for i in range(len(self.param_groups)):
        self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

    super().step(closure)

    self.global_step += 1


class PolyOptimizer(PolyOptimizerMixin, torch.optim.SGD):
  ...


def get_optimizer(lr, wd, max_step, param_groups, algorithm="sgd", alpha_scratch=10., alpha_bias=2.):
  params = [
    {
      'params': param_groups[0],
      'lr': lr,
      'weight_decay': wd
    },
    {
      'params': param_groups[1],
      'lr': alpha_bias * lr,
      'weight_decay': 0
    },
    {
      'params': param_groups[2],
      'lr': alpha_scratch * lr,
      'weight_decay': wd
    },
    {
      'params': param_groups[3],
      'lr': alpha_scratch * alpha_bias * lr,
      'weight_decay': 0
    },
  ]

  if algorithm == "sgd":
    return PolyOptimizer(params, lr=lr, momentum=0.9, max_step=max_step)
  elif algorithm == "lion":
    from lion_pytorch import Lion
    class LionPolyOptimizer(PolyOptimizerMixin, Lion):
      ...

    return LionPolyOptimizer(params, lr=lr, max_step=max_step, betas=(0.9, 0.99))
  else:
    raise NotImplementedError(f"Optimizer {algorithm} not implemented.")


def linear_schedule(step, max_step, a_0=0., a_n=1.0, schedule=1.0, contraint=min):
  if schedule == 0:
    return a_n

  rate = a_0 + (a_n - a_0) * step / (max_step * schedule)
  rate = contraint(rate, a_n)
  return rate
