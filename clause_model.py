import os
import ipdb
import dgl
import torch
import torch.nn as nn

from dgl_layers import *
from dgl_encoders import *
from common_components import *

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
    encoder_class = eval(self.settings['cp_encoder_type'])
    self.encoder = encoder_class(settings)
    if settings['cp_add_labels']:
      self.decision_layer = nn.Linear(self.gss_dim+self.encoder.cemb_dim+self.encoder.clabel_dim,2)
    else:
      self.decision_layer = MLPModel([self.gss_dim+self.encoder.d,256,64,2])

  def forward(self, input_dict, **kwargs):
    gss = input_dict['gss']
    G = input_dict['graph'].local_var()
    pred_idx = torch.where(G.nodes['clause'].data['predicted_clauses'])[0]
    feat_dict = {
      'literal': G.nodes['literal'].data['lit_labels'],
      'clause': G.nodes['clause'].data['clause_labels'],        
    }
    # ipdb.set_trace()
    vembs, cembs = self.encoder(G,feat_dict)    
    out = torch.cat([gss,cembs],dim=1)    
    logits = self.decision_layer(out)
    # print("model says, pred_idx (out of {}) is:".format(logits.size(0)))
    # print(pred_idx)
    return logits[pred_idx]


