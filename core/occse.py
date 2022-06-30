import torch
import torch.nn as nn
import torch.nn.functional as F


def random_label_split(label, choices):
  bs = label.shape[0]
  label_mask = torch.zeros_like(label).cuda()
  label_remain = label.clone()
  for i in range(bs):
    label_idx = torch.nonzero(label[i], as_tuple=False)  # -> [0, 1, 14]
    rand_idx = torch.randint(0, len(label_idx), (1,))  # -> 2
    target = label_idx[rand_idx][0]  # -> 14
    label_remain[i, target] = 0
    label_mask[i, target] = 1

    choices[target] += 1

  return label_mask, label_remain


def balanced_label_split(label, choices):
  bs = label.shape[0]
  label_mask = torch.zeros_like(label).cuda()
  label_remain = label.clone()
  for i in range(bs):
    p = (1 / choices)  # inv. prop. to the number of times chosen
    p[choices == 0] = 1  # not chosen so far are a priority
    p = p * label[i]  # suppress if label not present

    target = torch.multinomial(p, 1)[0]
    label_remain[i, target] = 0
    label_mask[i, target] = 1

    choices[target] += 1

  return label_mask, label_remain


def focal_label_split(label, choices, focal_factor):
  bs = label.shape[0]
  label_mask = torch.zeros_like(label).cuda()
  label_remain = label.clone()

  for i in range(bs):
    p = focal_factor  # prop. to the focal factor
    p = p * label[i]  # suppress if label not present

    target = torch.multinomial(p, 1)[0]
    label_remain[i, target] = 0
    label_mask[i, target] = 1

    choices[target] += 1

  return label_mask, label_remain


def split_label(
  label,
  choices,
  focal_factor=None,
  strategy='random',  # args.label_split
):
  if strategy == 'random':
    label_mask, label_remain = random_label_split(label, choices)
  elif strategy == 'balanced':
    label_mask, label_remain = balanced_label_split(label, choices)
  elif strategy == 'focal':
    label_mask, label_remain = focal_label_split(label, choices, focal_factor)
  else:
    raise ValueError('Only `random` and `focus` are available.')

  return label, label_mask, label_remain


def calculate_focal_factor(target, output, gamma=2.0, alpha=0.25, apply_class_balancing=False):
  output = torch.sigmoid(output)
  factor = target * output + (1 - target) * (1 - output)  # high if certain
  factor = (1.0 - factor)**gamma  # low if certain

  if apply_class_balancing:
    weight = target * alpha + (1 - target) * (1 - alpha)
    factor = weight * factor

  return factor.detach()


def update_focal_factor(
    labels,
    label_remain,
    cl_logits,
    focal_factor,
    momentum=0.9,
    gamma=2.0,
):
  samples  = labels.sum(axis=0)
  valid    = labels.any(axis=0)

  cf = calculate_focal_factor(label_remain, cl_logits, gamma).sum(axis=0) / samples
  focal_factor[valid] = (momentum * focal_factor + (1-momentum)*cf)[valid]

  return focal_factor
