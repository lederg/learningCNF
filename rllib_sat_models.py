import ipdb
import ray
import torch.nn as nn

from ray.rllib.agents import a3c
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

import logging
import torch.nn as nn
import cadet_utils
import utils
from policy_base import *
from settings import *
from sat_env import *

class RLLibModel(nn.Module, TorchModelV2):
  def __init__(self, *args, **kwargs):  
    TorchModelV2.__init__(self, *args, **kwargs)
    nn.Module.__init__(self)
    self.settings = kwargs['settings'] if 'settings' in kwargs.keys() else CnfSettings()        
    self.state_dim = self.settings['state_dim']
    self.embedding_dim = self.settings['embedding_dim']
    self.vemb_dim = self.settings['vemb_dim']
    self.cemb_dim = self.settings['cemb_dim']
    self.vlabel_dim = self.settings['vlabel_dim']
    self.clabel_dim = self.settings['clabel_dim']
    self.policy_dim1 = self.settings['policy_dim1']
    self.policy_dim2 = self.settings['policy_dim2']   
    self.max_iters = self.settings['max_iters']   
    self.state_bn = self.settings['state_bn']
    self.use_bn = self.settings['use_bn']
    self.entropy_alpha = self.settings['entropy_alpha']    
    self.lambda_value = self.settings['lambda_value']
    self.lambda_disallowed = self.settings['lambda_disallowed']
    self.lambda_aux = self.settings['lambda_aux']
    self.non_linearity = self.settings['policy_non_linearity']
    self.print_every = self.settings['print_every']
    self.logger = utils.get_logger(self.settings, 'SatModel')
    if self.non_linearity is not None:
      self.activation = eval(self.non_linearity)
    else:
      self.activation = lambda x: x

class SatThresholdModel(RLLibModel):
  def __init__(self, *args, **kwargs):  
    super(SatThresholdModel, self).__init__(*args, **kwargs)
    sublayers = []
    prev = self.input_dim()
    self.policy_layers = nn.Sequential()
    n = 0
    num_layers = len([x for x in self.settings['policy_layers'] if type(x) is int])
    for (i,x) in enumerate(self.settings['policy_layers']):
      if x == 'r':
        self.policy_layers.add_module('activation_{}'.format(i), nn.ReLU())
      elif x == 'lr':
        self.policy_layers.add_module('activation_{}'.format(i), nn.LeakyReLU())
      elif x == 'h':
        self.policy_layers.add_module('activation_{}'.format(i), nn.Tanh())        
      elif x == 's':
        self.policy_layers.add_module('activation_{}'.format(i), nn.Sigmoid())        
      else:
        n += 1
        layer = nn.Linear(prev,x)
        prev = x
        if self.settings['init_threshold'] is not None:          
          if n == num_layers:
            nn.init.constant_(layer.weight,0.)
            nn.init.constant_(layer.bias,self.settings['init_threshold'])
        self.policy_layers.add_module('linear_{}'.format(i), layer)
    self.features_size = prev # Last size
    self.value_layer = nn.Linear(self.features_size,1)
    self.logits_layer = nn.Linear(self.features_size, NUM_ACTIONS)

  def input_dim(self):
    return self.settings['state_dim']

  def forward(self, input_dict, state, seq_lens):
    features = self.policy_layers(input_dict["obs"])
    # self._value_out = self.value_layer(features).view(1)
    self._value_out = self.value_layer(features)
    return self.logits_layer(features), state

  def value_function(self):
    return self._value_out.view(-1)

