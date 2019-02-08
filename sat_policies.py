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
from sat_env import *
from sat_encoders import *
from policy_base import *
from rl_utils import *
from tick_utils import *

class SatPolicy(PolicyBase):
  def __init__(self, encoder=None, **kwargs):
    super(SatPolicy, self).__init__(**kwargs)
    self.final_embedding_dim = self.cemb_dim+self.clabel_dim    
    non_linearity = self.settings['policy_non_linearity']
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
    if non_linearity is not None:
      self.activation = eval(non_linearity)
    else:
      self.activation = lambda x: x
      
  # state is just a (batched) vector of fixed size state_dim which should be expanded. 
  # vlabels are batch * max_vars * vlabel_dim

  # cmat is already "batched" into a single matrix

  def forward(self, obs, **kwargs):
    mt0 = 0
    mt1 = 0
    mt2 = 0
    mt3 = 0
    mt4 = 0
    state = obs.state
    vlabels = obs.ground
    clabels = obs.clabels
    size = clabels.size()
    if size[0] > 1:
      mt0 = time.time()
    cmat_pos, cmat_neg = split_sparse_adjacency(obs.cmat)
    aux_losses = []

    # In MP the processes take care of cudaizing, because each Worker thread can have its own local model on the CPU, and
    # Only the main process does the training and has a global model on GPU.
    # In SP reinforce, we only have one model, so it has to be in GPU (if we're using CUDA)

    if self.settings['cuda'] and not self.settings['mp']:
      cmat_pos, cmat_neg = cmat_pos.cuda(), cmat_neg.cuda()
      state, vlabels, clabels = state.cuda(), vlabels.cuda(), clabels.cuda()

    num_learned = obs.ext_data
    self.batch_size=size[0]
    if size[0] > 1:
      mt1 = time.time()
    cembs = self.encoder(vlabels.view(-1,self.vlabel_dim), clabels.view(-1,self.clabel_dim), cmat_pos=cmat_pos, cmat_neg=cmat_neg, **kwargs)
    if size[0] > 1:
      mt2 = time.time()
    cembs = cembs.view(self.batch_size,-1,self.final_embedding_dim)
    cembs_processed = []

    # WARNING - This is being done in a loop. gotta' change that.

    for i, (nl1, nl2) in enumerate(num_learned):
      cembs_processed.append(cembs[i,nl1:nl2,:])
    if 'do_debug' in kwargs:
      ipdb.set_trace()
    
    if self.state_bn:
      state = self.state_bn(state)
    
    inputs = []
    if self.settings['use_global_state']:
      for i, (s,emb) in enumerate(zip(state,cembs_processed)):
        a = s.view(1,self.state_dim)
        reshaped_state = a.expand(len(emb),self.state_dim)
        inputs.append(torch.cat([reshaped_state,emb],dim=1))
      inputs = torch.cat(inputs,dim=0)
    else:
      inputs = torch.cat(cembs_processed,dim=0)

    # if self.batch_size > 1:
    #   ipdb.set_trace()  
    if self.policy_dim2:      
      outputs = self.action_score(self.activation(self.linear2(self.activation(self.linear1(inputs)))))
    else:
      outputs = self.action_score(self.activation(self.linear1(inputs)))
    
    if size[0] > 1:
      mt3 = time.time()
    outputs_processed = []
    for i, (nl1, nl2) in enumerate(num_learned):
      s = nl2-nl1
      outputs_processed.append(outputs[:s])
      outputs = outputs[s:]
    assert(outputs.shape[0]==0)
    if any((x!=x).any() for x in outputs_processed):    # Check nans
      ipdb.set_trace()
    if size[0] > 1:
      mt4 = time.time()
    value = None

    if size[0] > 1:
      print('Times are: split: {}, encoder: {}, policy: {}, post_process: {}'.format(mt1-mt0,mt2-mt1,mt3-mt2,mt4-mt3))
    return outputs_processed, value, cembs, aux_losses

  def get_allowed_actions(self, obs, **kwargs):
    pass

  def translate_action(self, action, **kwargs):
    # print('Action is: {}'.format(action[:10]))
    return action

  def combine_actions(self, actions, **kwargs):    
    return actions
    # return torch.cat(actions)

  def select_action(self, obs_batch, **kwargs):
    ipdb.set_trace()
    logits, *_ = self.forward(obs_batch)
    assert(len(logits)==1)
    logits = logits[0]
    probs = F.softmax(logits,dim=1)
    ps = probs[:,0].cpu().detach().numpy()
    action = torch.from_numpy(np.random.binomial(1,p=ps)).unsqueeze(0)
    num_learned = obs_batch.ext_data[0]
    locked = obs_batch.clabels[0,num_learned[0]:num_learned[1],CLABEL_LOCKED].long().view(1,-1)
    final_action = torch.max(action,locked)
    return final_action

  def compute_loss(self, transition_data):
    mt1 = time.time()
    _, _, _, rewards, *_ = zip(*transition_data)
    collated_batch = collate_transitions(transition_data,settings=self.settings)
    mt2 = time.time()
    batched_logits, values, _, aux_losses = self.forward(collated_batch.state, prev_obs=collated_batch.prev_obs,do_timing=True)
    mt3 = time.time()
    actions = collated_batch.action
    logprobs = []
    batched_clabels = collated_batch.state.clabels
    num_learned = collated_batch.state.ext_data
    for (action, logits, clabels, learned_idx) in zip(actions,batched_logits, batched_clabels, num_learned):
      probs = F.softmax(logits,dim=1)
      locked = self.settings.cudaize_var(clabels[learned_idx[0]:learned_idx[1],CLABEL_LOCKED])
      pre_logprobs = probs.gather(1,self.settings.cudaize_var(action).view(-1,1)).log().view(-1)
      logprobs.append(((1-locked)*pre_logprobs).sum())
    returns = self.settings.FloatTensor(rewards)
    adv_t = returns
    value_loss = 0.
    logprobs = torch.stack(logprobs)
    # entropies = (-probs*all_logprobs).sum(1)    
    adv_t = (adv_t - adv_t.mean())
    if self.settings['use_sum']:
      pg_loss = (-Variable(adv_t)*logprobs).sum()
    else:
      pg_loss = (-Variable(adv_t)*logprobs).mean()

    total_aux_loss = sum(aux_losses) if aux_losses else 0.    
    loss = pg_loss + self.lambda_value*value_loss + self.lambda_aux*total_aux_loss
    mt4 = time.time()
    print('compute_loss: Collate: {}, forward: {}, compute: {}'.format(mt2-mt1,mt3-mt2,mt4-mt3))
    return loss, logits

class SatLinearPolicy(PolicyBase):
  def __init__(self, encoder=None, **kwargs):
    super(SatLinearPolicy, self).__init__(**kwargs)
    non_linearity = self.settings['policy_non_linearity']
    if self.policy_dim1:
      self.linear1 = nn.Linear(self.clabel_dim, self.policy_dim1)

    if self.policy_dim2:
      self.linear2 = nn.Linear(self.policy_dim1,self.policy_dim2)
      self.action_score = nn.Linear(self.policy_dim2,2)
    elif self.policy_dim1:
      self.action_score = nn.Linear(self.policy_dim1,2) 
    else:
      self.action_score = nn.Linear(self.clabel_dim,2) 

    if non_linearity is not None:
      self.activation = eval(non_linearity)
    else:
      self.activation = lambda x: x

  
  # state is just a (batched) vector of fixed size state_dim which should be expanded. 
  # vlabels are batch * max_vars * vlabel_dim

  # cmat is already "batched" into a single matrix

  def forward(self, obs, **kwargs):
    clabels = obs.clabels
    size = obs.clabels.shape

    aux_losses = []

    if self.settings['cuda'] and not self.settings['mp']:
      clabels = clabels.cuda()

    num_learned = obs.ext_data
    # ipdb.set_trace()
    inputs = clabels.view(-1,self.clabel_dim)

    ipdb.set_trace()
    # if size[0] > 1:
    #   break_every_tick(20)
    if self.policy_dim2:      
      outputs = self.action_score(self.activation(self.linear2(self.activation(self.linear1(inputs)))))
    elif self.policy_dim1:
      outputs = self.action_score(self.activation(self.linear1(inputs)))
    else:
      outputs = self.action_score(inputs)
    outputs_processed = []
    # print(num_learned)
    for i, (nl1, nl2) in enumerate(num_learned):
      outputs_processed.append(outputs[nl1:nl2])
      outputs = outputs[size[1]:]

    assert(outputs.shape[0]==0)    
    if any((x!=x).any() for x in outputs_processed):    # Check nans
      ipdb.set_trace()
    value = None
    return outputs_processed, value, clabels, aux_losses

  def get_allowed_actions(self, obs, **kwargs):
    pass

  def translate_action(self, action, **kwargs):
    # print('Action is: {}'.format(action[:10]))
    return action

  def combine_actions(self, actions, **kwargs):    
    return actions
    # return torch.cat(actions)

  def select_action(self, obs_batch, **kwargs):
    logits, *_ = self.forward(obs_batch)
    assert(len(logits)==1)
    logits = logits[0]
    probs = F.softmax(logits,dim=1)
    ps = probs[:,0].detach().numpy()
    action = torch.from_numpy(np.random.binomial(1,p=ps)).unsqueeze(0)
    num_learned = obs_batch.ext_data[0]
    locked = obs_batch.clabels[0,num_learned[0]:num_learned[1],CLABEL_LOCKED].long().view(1,-1)
    final_action = torch.max(action,locked)

    return final_action

  def compute_loss(self, transition_data):
    _, _, _, rewards, *_ = zip(*transition_data)
    collated_batch = collate_transitions(transition_data,settings=self.settings)
    batched_logits, values, _, aux_losses = self.forward(collated_batch.state, prev_obs=collated_batch.prev_obs)
    actions = collated_batch.action
    logprobs = []
    batched_clabels = collated_batch.state.clabels
    num_learned = collated_batch.state.ext_data
    for (action, logits, clabels, learned_idx) in zip(actions,batched_logits, batched_clabels, num_learned):      
      probs = F.softmax(logits,dim=1).clamp(min=0.001,max=0.999)
      locked = clabels[learned_idx[0]:learned_idx[1],CLABEL_LOCKED]
      pre_logprobs = probs.gather(1,action.view(-1,1)).log().view(-1)
      action_probs = ((1-locked)*pre_logprobs).sum()
      if (action_probs!=action_probs).any():
        ipdb.set_trace()
      logprobs.append(action_probs)
    returns = self.settings.FloatTensor(rewards)
    adv_t = returns
    value_loss = 0.
    logprobs = torch.stack(logprobs)
    # entropies = (-probs*all_logprobs).sum(1)    
    adv_t = (adv_t - adv_t.mean())
    if self.settings['use_sum']:
      pg_loss = (-Variable(adv_t)*logprobs).sum()
    else:
      pg_loss = (-Variable(adv_t)*logprobs).mean()

    total_aux_loss = sum(aux_losses) if aux_losses else 0.    
    loss = pg_loss + self.lambda_value*value_loss + self.lambda_aux*total_aux_loss
    return loss, logits

class SatMiniLinearPolicy(PolicyBase):
  def __init__(self, encoder=None, **kwargs):
    super(SatMiniLinearPolicy, self).__init__(**kwargs)
    non_linearity = self.settings['policy_non_linearity']
    if self.policy_dim1:
      self.linear1 = nn.Linear(1, self.policy_dim1)

    if self.policy_dim2:
      self.linear2 = nn.Linear(self.policy_dim1,self.policy_dim2)
      self.action_score = nn.Linear(self.policy_dim2,2)
    elif self.policy_dim1:
      self.action_score = nn.Linear(self.policy_dim1,2) 
    else:
      self.action_score = nn.Linear(1,2) 

    if non_linearity is not None:
      self.activation = eval(non_linearity)
    else:
      self.activation = lambda x: x

  
  # state is just a (batched) vector of fixed size state_dim which should be expanded. 
  # vlabels are batch * max_vars * vlabel_dim

  # cmat is already "batched" into a single matrix

  def forward(self, obs, **kwargs):
    clabels = obs.clabels
    size = obs.clabels.shape

    aux_losses = []

    if self.settings['cuda'] and not self.settings['mp']:
      clabels = clabels.cuda()

    num_learned = obs.ext_data
    inputs = clabels.view(-1,self.clabel_dim)[:,CLABEL_LBD].unsqueeze(1)

    # if size[0] > 1:
    #   break_every_tick(20)
    if self.policy_dim2:      
      outputs = self.action_score(self.activation(self.linear2(self.activation(self.linear1(inputs)))))
    elif self.policy_dim1:
      outputs = self.action_score(self.activation(self.linear1(inputs)))
    else:
      outputs = self.action_score(inputs)
    outputs_processed = []
    # print(num_learned)
    for i, (nl1, nl2) in enumerate(num_learned):
      outputs_processed.append(outputs[nl1:nl2])
      outputs = outputs[size[1]:]

    # ipdb.set_trace()
    assert(outputs.shape[0]==0)    
    if any((x!=x).any() for x in outputs_processed):    # Check nans
      ipdb.set_trace()
    value = None
    return outputs_processed, value, clabels, aux_losses

  def get_allowed_actions(self, obs, **kwargs):
    pass

  def translate_action(self, action, **kwargs):
    # print('Action is: {}'.format(action[:10]))
    return action

  def combine_actions(self, actions, **kwargs):    
    return actions
    # return torch.cat(actions)

  def select_action(self, obs_batch, **kwargs):
    logits, *_ = self.forward(obs_batch)
    assert(len(logits)==1)
    logits = logits[0]
    probs = F.softmax(logits,dim=1)
    ps = probs[:,0].detach().cpu().numpy()    # cpu() just in case, if we're in SP+cuda
    action = torch.from_numpy(np.random.binomial(1,p=ps)).unsqueeze(0)
    num_learned = obs_batch.ext_data[0]
    locked = obs_batch.clabels[0,num_learned[0]:num_learned[1],CLABEL_LOCKED].long().view(1,-1)
    final_action = torch.max(action,locked)

    return final_action

  def compute_loss(self, transition_data):
    _, _, _, rewards, *_ = zip(*transition_data)
    collated_batch = collate_transitions(transition_data,settings=self.settings)    
    collated_batch.state = cudaize_obs(collated_batch.state, self.settings)
    batched_logits, values, _, aux_losses = self.forward(collated_batch.state, prev_obs=collated_batch.prev_obs)
    actions = collated_batch.action
    logprobs = []
    batched_clabels = collated_batch.state.clabels
    num_learned = collated_batch.state.ext_data
    for (action, logits, clabels, learned_idx) in zip(actions,batched_logits, batched_clabels, num_learned):      
      probs = F.softmax(logits,dim=1).clamp(min=0.001,max=0.999)
      locked = clabels[learned_idx[0]:learned_idx[1],CLABEL_LOCKED]
      action = self.settings.cudaize_var(action)
      pre_logprobs = probs.gather(1,action.view(-1,1)).log().view(-1)
      action_probs = ((1-locked)*pre_logprobs).sum()
      if (action_probs!=action_probs).any():
        ipdb.set_trace()
      logprobs.append(action_probs)
    returns = self.settings.FloatTensor(rewards)
    adv_t = returns
    value_loss = 0.
    logprobs = torch.stack(logprobs)
    # entropies = (-probs*all_logprobs).sum(1)    
    adv_t = (adv_t - adv_t.mean())
    if self.settings['use_sum']:
      pg_loss = (-Variable(adv_t)*logprobs).sum()
    else:
      pg_loss = (-Variable(adv_t)*logprobs).mean()

    total_aux_loss = sum(aux_losses) if aux_losses else 0.    
    loss = pg_loss + self.lambda_value*value_loss + self.lambda_aux*total_aux_loss
    return loss, logits

class SatRandomPolicy(PolicyBase):
  def __init__(self, encoder=None, **kwargs):
    super(SatRandomPolicy, self).__init__(**kwargs)
    self.action_score = nn.Linear(2,1)
  
  def forward(self, obs, **kwargs):
    pass

  def get_allowed_actions(self, obs, **kwargs):
    pass

  def translate_action(self, action, **kwargs):
    # print('Action is: {}'.format(action[:10]))
    return action

  def combine_actions(self, actions, **kwargs):    
    return actions
    # return torch.cat(actions)

  def select_action(self, obs_batch, **kwargs):
    assert(obs_batch.clabels.shape[0]==1)
    num_learned = obs_batch.ext_data[0]
    action = torch.from_numpy(np.random.binomial(1,p=0.5,size=num_learned[1]-num_learned[0])).unsqueeze(0)
    locked = obs_batch.clabels[0,num_learned[0]:num_learned[1],CLABEL_LOCKED].long().view(1,-1)
    final_action = torch.max(action,locked)

    return final_action

  def compute_loss(self, transition_data):
    return None, None

class SatLBDPolicy(PolicyBase):
  def __init__(self, encoder=None, **kwargs):
    super(SatLBDPolicy, self).__init__(**kwargs)
    self.action_score = nn.Linear(2,1)
  
  def forward(self, obs, **kwargs):
    return None

  def get_allowed_actions(self, obs, **kwargs):
    pass

  def translate_action(self, action, **kwargs):
    # print('Action is: {}'.format(action[:10]))
    return action

  def combine_actions(self, actions, **kwargs):    
    return actions
    # return torch.cat(actions)

  def select_action(self, obs_batch, **kwargs):
    return [np.empty(shape=(0, 0), dtype=bool)]

  def compute_loss(self, transition_data):
    return None, None