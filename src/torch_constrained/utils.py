import functools

from .extra_optimizers import *
import torch

OPTIM_DICT = {'None': None,
              'SGD': torch.optim.SGD,
              'Adam': torch.optim.Adam,
              'Adagrad': torch.optim.Adagrad,
              'RMSprop': torch.optim.RMSprop,
              'SGDM': functools.partial(torch.optim.SGD, momentum=0.7),
              'ExtraSGD': ExtraSGD,
              'ExtraAdam': ExtraAdam,
              'ExtraAdagrad': ExtraAdagrad,
              'ExtraRMSprop': ExtraRMSprop
              }