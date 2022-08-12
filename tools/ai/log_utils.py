# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import numpy as np
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

class Average_Meter:
    def __init__(self, keys):
        self.keys = keys
        self.clear()
    
    def add(self, dic):
        for key, value in dic.items():
            self.data_dic[key].append(value)

    def get(self, keys=None, clear=False):
        if keys is None:
            keys = self.keys
        
        dataset = [float(np.mean(self.data_dic[key])) for key in keys]
        if clear:
            self.clear()

        if len(dataset) == 1:
            dataset = dataset[0]
            
        return dataset
    
    def clear(self):
        self.data_dic = {key : [] for key in self.keys}

