import os
import numpy as np
import torch
import torch.multiprocessing as mp
import time
import ipdb
import pickle
import itertools
from collections import namedtuple
from namedlist import namedlist

from settings import *
from utils import *
from rl_utils import *

class AbstractVBN(nn.Module):
  def __init__(self, dim, **kwargs):
    super(AbstractVBN, self).__init__()
    self.settings = kwargs['settings'] if 'settings' in kwargs.keys() else CnfSettings()
    self.dim = dim
    self.scale = nn.Parameter(self.settings.FloatTensor(dim), requires_grad=True)
    self.shift = nn.Parameter(self.settings.FloatTensor(dim), requires_grad=True)
    self.effective_mean = nn.Parameter(self.settings.FloatTensor(dim), requires_grad=False)
    self.effective_std = nn.Parameter(self.settings.FloatTensor(dim), requires_grad=False)
    nn_init.normal_(self.scale)
    nn_init.normal_(self.shift)
    # nn.init.constant_(self.shift,0.)
    # nn_init.constant_(self.scale,1.)
    nn.init.constant_(self.effective_mean,0.)
    nn_init.constant_(self.effective_std,1.)
    
  def forward(self, input, **kwargs):
    n1 = input - self.effective_mean
    normed_input = n1 / (self.effective_std + float(np.finfo(np.float32).eps))
    outputs = normed_input*self.scale + self.shift
    return outputs

  def recompute_moments(self, data):    
    pass

class MovingAverageVBN(AbstractVBN):
  def __init__(self, size, **kwargs):
    super(MovingAverageVBN,self).__init__(size[1])
    self.size = size
    self.total_length = 0
    self.moving_mean = nn.Parameter(self.settings.FloatTensor(*self.size), requires_grad=False)
    nn.init.constant_(self.moving_mean,0.)

  def recompute_moments(self, data):
    dsize = len(data)
    window_size = self.size[0]
    if dsize > window_size:
      self.moving_mean.data = data[-window_size:]
    else:
      size_left = window_size - dsize
      self.moving_mean.data = torch.cat([data,self.moving_mean[:size_left]],dim=0)

    self.total_length = min(self.total_length+dsize, window_size)
    self.effective_mean.data = self.moving_mean[:self.total_length].mean(dim=0)
    self.effective_std.data = self.moving_mean[:self.total_length].std(dim=0)

