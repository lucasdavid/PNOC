import torch

from .torch_utils import *


class PolyOptimizerMixin:

  def __init__(self, params, lr, max_step, poly_power=0.9, start_step=0, **kwargs):
    super().__init__(params, lr, **kwargs)

    self.global_step = start_step
    self.max_step = max_step
    self.poly_power = poly_power

    self.__initial_lr = [group['lr'] for group in self.param_groups]

  def step(self, closure=None):
    if self.global_step < self.max_step:
      lr_mult = (1 - self.global_step / self.max_step)**self.poly_power

      for i in range(len(self.param_groups)):
        self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

    super().step(closure)

    self.global_step += 1


class PolyOptimizer(PolyOptimizerMixin, torch.optim.SGD):
  ...


class PolyAdamW(PolyOptimizerMixin, torch.optim.AdamW):
  ...


def get_optimizer(lr, wd, max_step, param_groups, algorithm="sgd", alpha_scratch=10., alpha_bias=2., **kwargs):
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
    return PolyOptimizer(params, lr=lr, max_step=max_step, **kwargs)

  if algorithm == "momentum":
    return PolyOptimizer(params, lr=lr, max_step=max_step, momentum=0.9, nesterov=True, **kwargs)

  elif algorithm == "adamw":
    return PolyAdamW(params, lr=lr, max_step=max_step, **kwargs)

  elif algorithm == "lion":
    from lion_pytorch import Lion
    class LionPolyOptimizer(PolyOptimizerMixin, Lion):
      ...

    return LionPolyOptimizer(params, lr=lr, max_step=max_step, betas=(0.9, 0.99), **kwargs)

  else:
    raise NotImplementedError(f"Optimizer {algorithm} not implemented.")


def linear_schedule(step, max_step, a_0=0., a_n=1.0, schedule=1.0, constraint=min):
  if schedule == 0:
    return a_n

  rate = a_0 + (a_n - a_0) * step / (max_step * schedule)
  rate = constraint(rate, a_n)
  return rate


def ema_avg_fun(avg_p, p, count, optimizer, decay=0.999, warmup=0):
  if optimizer.global_step < warmup:
    return p  # Copy params from model to EMA.

  ema_decay = min(1 - 1 / (1 + optimizer.global_step), decay)
  return ema_decay * avg_p + (1-ema_decay) * p
