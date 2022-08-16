import torch
from .torch_utils import *

class PolyOptimizer(torch.optim.SGD):
    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9, dampening=0, nesterov=False):
        super().__init__(params=params, lr=lr, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)

        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum
        
        self.__initial_lr = [group['lr'] for group in self.param_groups]
    
    def step(self, closure=None):
        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.global_step += 1


def linear_schedule(step, max_step, a_0=0., a_n=1.0, rate=1.0, contraint=min):
    if rate == 0:
        return a_n

    rate = a_0 + (a_n - a_0) * step / (max_step * rate)
    rate = contraint(rate, a_n)
    return rate
