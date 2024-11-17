from copy import deepcopy

import torch


def inference_model(model, ema_model, optimizer_step, enabled=True, warmup_steps=128):
  step = optimizer_step + 1  # Make sure step=0 isn't incorrectly included.

  if enabled and step > warmup_steps:
    print(f"[Valid] Model=EMA step={step} > warmup={warmup_steps}")
    return ema_model
  else:
    print(f"[Valid] Model=Student step={step} <= warmup={warmup_steps}")
    return model

def init(model, device="cuda", use_ema=True):
  if not use_ema:
    return None

  # my_ema_avg_fn = partial(ema_avg_fun, optimizer=optimizer, decay=args.ema_decay, warmup=args.ema_warmup)
  # ema_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=my_ema_avg_fn)
  ema_model = deepcopy(model)
  for p in ema_model.parameters():
    p.requires_grad = False
  ema_model = model.to(device)

  return ema_model


def copy(model, ema_model, optimizer_step, enabled=True, decay=0.99, ema_steps=1, warmup_steps=128):
  if not enabled:
    return

  opt_step = optimizer_step + 1

  with torch.no_grad():
    if opt_step <= warmup_steps:
      for t_params, s_params in zip(ema_model.parameters(), model.parameters()):
        t_params.copy_(s_params)

    elif opt_step % ema_steps == 0:
      ema_step = opt_step // ema_steps

      ema_decay = min(1 - 1 / (1 + ema_step), decay)
      for t_params, s_params in zip(ema_model.parameters(), model.parameters()):
        t_params.copy_(ema_decay * t_params + (1 - ema_decay) * s_params)
