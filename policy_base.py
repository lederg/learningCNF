import torch.nn as nn
from settings import *
import cadet_utils

class PolicyBase(nn.Module):
  def __init__(self, **kwargs):
    super(PolicyBase, self).__init__()
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
    self.lambda_value = self.settings['lambda_value']
    self.lambda_disallowed = self.settings['lambda_disallowed']
    self.lambda_aux = self.settings['lambda_aux']
    self.non_linearity = self.settings['policy_non_linearity']

    if self.non_linearity is not None:
      self.activation = eval(self.non_linearity)
    else:
      self.activation = lambda x: x

  
  def forward(self, obs, **kwargs):
    raise NotImplementedError

  def select_action(self, obs_batch, **kwargs):
    raise NotImplementedError

  def translate_action(self, action, obs, **kwargs):
    raise NotImplementedError

  def combine_actions(self, actions, **kwargs):
    raise NotImplementedError
    
  def get_allowed_actions(self, obs, **kwargs):
    return cadet_utils.get_allowed_actions(obs,**kwargs)



