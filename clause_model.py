import os
import ipdb
import dgl
import torch
import torch.nn as nn

from dgl_layers import *
from dgl_encoders import *

# class NodeApplyModule(nn.Module):
#     """Update the node feature hv with ReLU(Whv+b)."""
#     def __init__(self, in_feats, out_feats, activation):
#         super(NodeApplyModule, self).__init__()
#         self.linear = nn.Linear(in_feats, out_feats)
#         self.activation = activation

#     def forward(self, node):
#         h = self.linear(node.data['h'])
#         h = self.activation(h)
#         return {'h' : h}

class ClausePredictionModel(nn.Module):
  def __init__(self, settings=None, **kwargs):
    super(ClausePredictionModel, self).__init__(**kwargs)
    self.settings = settings if settings else CnfSettings()
    self.gss_dim = self.settings['state_dim']
    self.encoder = CNFEncoder(settings)
    self.decision_layer = nn.Linear(self.gss_dim+self.encoder.cemb_dim+self.encoder.clabel_dim,2)
    
  def forward(self, input_dict, **kwargs):
    gss = input_dict['gss']
    G = input_dict['graph'].local_var()
    
    feat_dict = {
      'literal': G.nodes['literal'].data['lit_labels'],
      'clause': G.nodes['clause'].data['clause_labels'],        
    }
    # ipdb.set_trace()
    vembs, cembs = self.encoder(G,feat_dict)
    out = torch.cat([gss,cembs],dim=1)
    logits = self.decision_layer(out)

    learnt = (G.nodes['clause'].data['clause_labels'][:,-1]).int()
    learnt_idx = torch.where(learnt)[0]
    return logits[learnt_idx]


