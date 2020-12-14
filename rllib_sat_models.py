import ipdb
import ray
import pickle
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
from dgl_encoders import *
from ray.util.sgd.utils import TimerStat


def lit_to_var(lit):
  return lit.reshape(-1,2*lit.shape[1])

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


class AnnotationModelBase(nn.Module):
  def __init__(self, size=10):
    super(AnnotationModelBase, self).__init__()
    self.settings = CnfSettings()
    self.curr_fname = None
    self.time_mapping = None
    self.max_time = 0
    self._size = 0

  @property
  def size(self):
    return self._size
  
  def update_from_file(self):
    fname = os.path.splitext(self.curr_fname)[0]+'.annt'
    with open(fname,'rb') as f:
      a = pickle.load(f)
      self.time_mapping = [y[1] for y in sorted(list(a.items()), key=lambda x:x[0])]
      self.max_time = max(self.time_mapping)


class TimestepModel(AnnotationModelBase):
  def __init__(self):
    super(TimestepModel, self).__init__()
    self.MAX_TIMESTEP = 20
    self._size = 1
    self.invert = self.settings['sat_decode_invert']
  # data is numpy array
  # literal_mapping is Tensor.
  # I know.

  def decode(self, vembs, fname):
    if fname != self.curr_fname:
      self.curr_fname = fname
      self.update_from_file()  

    decoded_vembs = torch.Tensor(self.time_mapping).reshape(-1,1)
    if self.invert:
      decoded_vembs = self.max_time - decoded_vembs
    decoded_vembs = decoded_vembs / self.MAX_TIMESTEP     
    return decoded_vembs

  @property
  def size(self):
    return self._size
  

  # def forward(self, input_tensor):
  #   inp = input_tensor.unsqueeze(0).detach()     # Add batch and channel dimensions    
  #   out = self.layer1(inp)
  #   out = out.squeeze(0).transpose(1,2).transpose(0,2)
  #   rc = self.layer2(out.reshape(-1))
  #   return rc



class SatActivityModel(PolicyBase):
  def __init__(self, *args, **kwargs):
    super(SatActivityModel, self).__init__(*args)
    encoder_class = eval(self.settings['sat_encoder_type'])
    self.encoder = encoder_class(self.settings)
    decode_class = eval(self.settings['sat_decode_class'])
    self.decode = self.settings['sat_decode']    
    if self.decode:
      self.decode_module = decode_class()
    self.curr_fname = None
    inp_size = 0
    if self.settings['sat_add_embedding']:
      inp_size += self.encoder.output_size()
    if self.settings['sat_add_labels']:
      inp_size += self.encoder.vlabel_dim
    inp_size *= 2       # from literal to variables (embedding size is doubled)
    if self.decode:
      inp_size += self.decode_module.size
    self.score_layer = MLPModel([inp_size,256,64,1])
    # self.pad = torch.Tensor([torch.finfo().min])
    self.timers = {k: TimerStat() for k in ["make_graph", "encoder", "decode", "score"]}

  # cmat_net and cmat_pos are already "batched" into a single matrix
  def forward(self, input_dict, state, seq_lens, es=True, **kwargs):
    T = 0.25      # temprature from NeuroCore
    K = 10000     # again from NeuroCore
    with self.timers['make_graph']:
      self.encoder.eval()
      G = input_dict['graph']      
      self._value_out = torch.zeros(1).expand(G.number_of_nodes('literal'))
    with self.timers['encoder']:
      vembs, cembs = self.encoder(G)    
    out = []
    if self.settings['sat_add_embedding']:
      out.append(vembs)
    if self.settings['sat_add_labels']:
      out.append(lit_features)
    prescore = torch.cat(out,dim=1)
    var_prescore = lit_to_var(prescore)
    with self.timers['decode']:
      if self.decode: 
        var_prescore = torch.cat([var_prescore,self.decode_module.decode(var_prescore, self.curr_fname)],dim=1)
    with self.timers['score']:
      scores = self.score_layer(var_prescore).t()
      scores = F.softmax(scores/T,dim=1)*K*scores.shape[1]
    return scores, []

  def value_function(self):
    return self._value_out.view(-1)
