# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import numpy as np

from tools.ai.torch_utils import calculate_parameters
from tools.general.txt_utils import add_txt


def log_config(args, title=None, print_fn=print):
  margin = '=' * 80
  print_fn(margin)
  if title:
    print_fn(title)
    print_fn()

  pad = max(map(len, args)) + 1
  for k, v in args.items():
    print_fn(f'{k.ljust(pad)}: {v}')

  print_fn(margin)


def log_dataset(name, dataset, tt, tv, info=None):
  if info is None:
    class _unknown_info():
      def __init__(self):
        self.num_classes = "?"
    info = getattr(dataset, "info", _unknown_info())

  print(
    f"Dataset {name}\n"
    f"  samples={len(dataset)}"
    f"  classes={info.num_classes}\n"
    f"  train transforms={tt}\n"
    f"  valid transforms={tv}\n\n"
  )


def log_loader(tl, ts, check_sampler=False):
  print(
    f"Train Loader\n"
    f"  steps={len(tl)}\n"
    f"  batch={tl.batch_size}\n"
    f"  workers={tl.num_workers}\n"
    f"  sampler={tl.sampler}\n"
  )

  if check_sampler:
    labels = np.asarray([ts.get_label(_id) for _id in ts.sample_ids])
    o = labels.sum(0).astype(int)
    s = labels[[i for i in tl.sampler], :].sum(0).astype(int)
    print(f"Original class frequency for {len(labels)} samples: ({o.mean():.2f} {o.std():.2f})")
    print(f" Sampled Class frequency for {len(labels)} samples: ({s.mean():.2f} {s.std():.2f})")

    for n, _o, _s in zip(ts.classification_info.classes, o, s):
      print(f"  {n:>18} (↑{_s/_o-1:4.0%}) {_s:>5}/{_o}")


def log_model(name, model, args):
  properties = "architecture mode dilated trainable_backbone trainable_stem use_gn".split()
  config = [(p, getattr(args, p)) for p in properties if hasattr(args, p)]

  print(
    f"{name}\n", *(f"{p:>18} = {v}\n" for p, v in config),
    f"  total params = {calculate_parameters(model):.2f}M\n"
  )


def log_opt_params(name, params, verbose=1):
  _names = lambda p: ", ".join(p)

  if verbose == 0:
    _report_fields = (len(params[0]), len(params[1]), len(params[2]), len(params[3]))
  elif verbose == 1:
    _report_fields = (
      f"({len(params[0])}) {_names(params[0][:32])}{'...' if len(params[0]) > 32 else ''}",
      f"({len(params[1])}) {_names(params[1][:32])}{'...' if len(params[1]) > 32 else ''}",
      f"({len(params[2])}) {_names(params[2][:32])}{'...' if len(params[2]) > 32 else ''}",
      f"({len(params[3])}) {_names(params[3][:32])}{'...' if len(params[3]) > 32 else ''}",
    )
  elif verbose > 1:
    _report_fields = (_names(params[0]), _names(params[1]), _names(params[2]), _names(params[3]))

  _report_lines = (
    f"  pretrained:",
    f"     weights: {_report_fields[0]}",
    f"     biases:  {_report_fields[1]}",
    f"  scratch:",
    f"     weights: {_report_fields[2]}",
    f"     biases:  {_report_fields[3]}",
  )
  print("Optimizer Parameters", *_report_lines, sep="\n", end="\n\n",)


def log_print(message='', path=None):
  """This function shows message and saves message.

    Args:
        pred_tags:
            The type of variable is list.
            The type of each element is string.

        gt_tags:
            The type of variable is list.
            the type of each element is string.
    """
  print(message, flush=True)
  if path:
    add_txt(path, message)


class Logger:

  def __init__(self):
    pass


class MetricsContainer:

  def __init__(self, metric_names):
    self.metric_names = metric_names
    self.clear()

  def update(self, data):
    for n, v in data.items():
      self._metrics_history[n].append(v)

  def get(self, metrics=None, clear=False, as_map=False):
    if metrics is None:
      metrics = self.metric_names

    metric_values = [float(np.mean(self._metrics_history[k])) for k in metrics]

    if clear:
      self.clear()

    if as_map:
      return dict(zip(self.metric_names, metric_values))

    if len(metric_values) == 1:
      metric_values = metric_values[0]

    return metric_values

  def clear(self):
    self._metrics_history = {n: [] for n in self.metric_names}
