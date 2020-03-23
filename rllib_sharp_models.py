import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import utils
import numpy as np
from torch.nn import init as nn_init
from collections import namedtuple
from IPython.core.debugger import Tracer
from settings import *
from policy_base import *
from rl_utils import *
from ray.rllib.policy.sample_batch import SampleBatch

from dgl_layers import *
from dgl_encoders import *
from common_components import *


def graph_from_adj(lit_features, adj_matrix):
    ind = adj_matrix.coalesce().indices().t().tolist()
    ind_t = adj_matrix.t().coalesce().indices().t().tolist()

    G = dgl.heterograph(
                {('literal', 'l2c', 'clause') : ind_t,
                 ('clause', 'c2l', 'literal') : ind},
                {'literal': adj_matrix.shape[1],
                 'clause': adj_matrix.shape[0]})
    
    G.nodes['literal'].data['literal_feats'] = lit_features
    return G


  
class SharpModel(RLLibModel):
  def __init__(self, *args, **kwargs):
    super(SharpModel, self).__init__(*args)
    encoder_class = eval(self.settings['sharp_encoder_type'])
    self.encoder = encoder_class(self.settings)
    inp_size = 0
    if self.settings['sharp_add_embedding']:
      inp_size += self.encoder.output_size()
    if self.settings['sharp_add_labels']:
      inp_size += self.encoder.vlabel_dim
    self.decision_layer = MLPModel([inp_size,256,64,1])
    self.pad = torch.Tensor([torch.finfo().min])

  def from_batch(self, train_batch, is_training=True):
    """Convenience function that calls this model with a tensor batch.

    All this does is unpack the tensor batch to call this model with the
    right input dict, state, and seq len arguments.
    """
    def obs_from_input_dict(input_dict):
      z = list(input_dict.items())
      dense_obs = [undensify_obs(DenseState(*x)) for x in list(z[3][1])]
      return dense_obs

    collated_batch = collate_observations(obs_from_input_dict(train_batch),settings=self.settings)
    input_dict = {
      "collated_obs": collated_batch,
      "is_training": is_training,
    }
    if SampleBatch.PREV_ACTIONS in train_batch:
      input_dict["prev_actions"] = train_batch[SampleBatch.PREV_ACTIONS]
    if SampleBatch.PREV_REWARDS in train_batch:
      input_dict["prev_rewards"] = train_batch[SampleBatch.PREV_REWARDS]
    states = []
    i = 0
    while "state_in_{}".format(i) in train_batch:
      states.append(train_batch["state_in_{}".format(i)])
      i += 1
    return self.__call__(input_dict, states, train_batch.get("seq_lens"))


  
  # state is just a (batched) vector of fixed size state_dim which should be expanded. 
  # ground_embeddings are batch * max_vars * ground_embedding

  # cmat_net and cmat_pos are already "batched" into a single matrix
  def forward(self, input_dict, state, seq_lens, es=True, **kwargs):
    def obs_from_input_dict(input_dict):
      z = list(input_dict.items())
      z1 = list(z[0][1][0])
      return undensify_obs(DenseState(*z1))
    if es:
      obs = undensify_obs(input_dict)
    else:
      obs = obs_from_input_dict(input_dict)       # This is an experience rollout
    lit_features = obs.ground
    G = graph_from_adj(lit_features, obs.cmat)
    self._value_out = torch.zeros(1).expand(len(lit_features))
    out = []
    if self.settings['sharp_add_embedding']:
      vembs, cembs = self.encoder(G)    
      out.append(vembs)
    if self.settings['sharp_add_labels']:
      out.append(lit_features)
    logits = self.decision_layer(torch.cat(out,dim=1)).t()
    allowed_actions = self.get_allowed_actions(obs).int().float()
    inf_mask = torch.max(allowed_actions.log(),torch.Tensor([torch.finfo().min]))
    logits = logits + inf_mask
    self.outputs = torch.cat([logits,self.pad.expand((1,self.max_vars-logits.shape[1]))], dim=1)
    return self.outputs, []

  def get_allowed_actions(self, obs, **kwargs):
    def add_other_polarity(indices):
      pos = torch.where(1-indices%2)[0]
      neg = torch.where(indices%2)[0]
      add_pos = indices[pos] + 1
      add_neg = indices[neg] - 1
      return torch.cat([indices,add_pos,add_neg],axis=0).unique()

    literal_indices = torch.unique(obs.cmat.coalesce().indices()[1])
    allowed_indices = add_other_polarity(literal_indices)
    print('***')
    print(literal_indices)
    print('----------------------------------')
    print(allowed_indices)
    print('***')
    allowed_actions = torch.zeros(obs.ground.shape[0])
    allowed_actions[allowed_indices] = 1
    return allowed_actions



  def value_function(self):
    return self._value_out.view(-1)
