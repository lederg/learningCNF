import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import init as nn_init
import torch.nn.functional as F
import torch.optim as optim
import utils
import numpy as np
from collections import namedtuple
import ipdb
from settings import *
from sat_encoders import *
from policy_base import *

class SatPolicy(PolicyBase):
  def __init__(self, encoder=None, **kwargs):
    super(SatPolicy, self).__init__(**kwargs)
    self.final_embedding_dim = self.cemb_dim+self.clabel_dim    
    if encoder:
      print('Bootstraping Policy from existing encoder')
      self.encoder = encoder
    else:
      self.encoder = SatEncoder(**kwargs)
    if self.settings['use_global_state']:
      self.linear1 = nn.Linear(self.state_dim+self.final_embedding_dim, self.policy_dim1)
    else:
      self.linear1 = nn.Linear(self.final_embedding_dim, self.policy_dim1)

    if self.policy_dim2:
      self.linear2 = nn.Linear(self.policy_dim1,self.policy_dim2)
      self.action_score = nn.Linear(self.policy_dim2,2)
    else:
      self.action_score = nn.Linear(self.policy_dim1,2)
    if self.state_bn:
      self.state_bn = nn.BatchNorm1d(self.state_dim)
    self.activation = eval(self.settings['policy_non_linearity'])
  
  # state is just a (batched) vector of fixed size state_dim which should be expanded. 
  # vlabels are batch * max_vars * vlabel_dim

  # cmat is already "batched" into a single matrix

  def forward(self, obs, **kwargs):
    state = obs.state
    vlabels = obs.ground
    clabels = obs.clabels
    cmat_pos, cmat_neg = split_sparse_adjacency(obs.cmat)

    aux_losses = []

    if self.settings['cuda']:
      cmat_pos, cmat_neg = cmat_pos.cuda(), cmat_neg.cuda()
      state, vlabels, clabels = state.cuda(), vlabels.cuda(), clabels.cuda()

    ipdb.set_trace()
    size = clabels.size()
    num_learned = obs.ext_data
    self.batch_size=size[0]  
    if self.batch_size > 1:
      ipdb.set_trace()  
    cembs = self.encoder(vlabels.view(-1,self.vlabel_dim), clabels.view(-1,self.clabel_dim), cmat_pos=cmat_pos, cmat_neg=cmat_neg, **kwargs)
    cembs = cembs.view(self.batch_size,-1,self.final_embedding_dim)      
    cembs_processed = []

    # WARNING - This is being done in a loop. gotta' change that.

    for i, (nl1, nl2) in enumerate(num_learned):
      cembs_processed.append(cembs[i,nl1:nl2,:])
    if 'do_debug' in kwargs:
      ipdb.set_trace()
    
    if self.state_bn:
      state = self.state_bn(state)

    
    input = []
    ipdb.set_trace()    
    if self.settings['use_global_state']:
      # if self.batch_size > 1:
      #   ipdb.set_trace()

      for i, s,emb in enumerate(zip(state,cembs_processed)):
        a = s.view(1,self.state_dim)
        reshaped_state = a.expand(len(emb),self.state_dim)
        inputs.append(torch.cat([reshaped_state,emb],dim=1))

      # a = state.view(self.batch_size,1,self.state_dim)
      # reshaped_state = a.expand(self.batch_size,num_learned,self.state_dim) # add the maxvars dimension
      # inputs = torch.cat([reshaped_state, cembs],dim=2).view(-1,self.state_dim+self.final_embedding_dim)
    else:
      inputs = cembs.view(-1,self.final_embedding_dim)

    if self.policy_dim2:      
      outputs = self.action_score(self.activation(self.linear2(self.activation(self.linear1(inputs)))))
    else:
      outputs = self.action_score(self.activation(self.linear1(inputs)))
    outputs = outputs.view(self.batch_size,-1,2)    # batch x num_learned       
    if (outputs!=outputs).any():    # Check nans
      ipdb.set_trace()
    value = None
    return outputs, value, cembs, aux_losses

  def get_allowed_actions(self, obs, **kwargs):
    pass

  def translate_action(self, action, **kwargs):
    print('Action is: {}'.format(action[:10]))
    return action

  def combine_actions(self, actions, **kwargs):    
    return torch.cat(actions)

  def select_action(self, obs_batch, **kwargs):
    logits, *_ = self.forward(obs_batch)
    assert(len(logits)==1)
    logits = logits[0]
    probs = F.softmax(logits,dim=1)
    ps = probs[:,0].detach().numpy()
    action = torch.from_numpy(np.random.binomial(1,p=ps)).unsqueeze(0)

    return action
