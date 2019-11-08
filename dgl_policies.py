import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import init as nn_init
import torch.nn.functional as F
import torch.optim as optim
import utils
import numpy as np
import dgl
import dgl.function as fn
from collections import namedtuple
from IPython.core.debugger import Tracer
import cadet_utils
from qbf_data import *
from qbf_model import *
from settings import *
from policy_base import *
from rl_utils import *


class CNFLayer(nn.Module):
  def __init__(self, in_size, clause_size, out_size, activation=None, settings=None):
    super(CNFLayer, self).__init__()
    self.ntypes = ['literal', 'clause']
    self.etypes = ['l2c', 'c2l']
    # W_r for each relation
    self.weight = nn.ModuleDict({
      self.etypes[0] : nn.Linear(in_size, clause_size),
      self.etypes[1] : nn.Linear(clause_size+1, out_size)
    })
    self.settings = settings if settings else CnfSettings()
    self.activation = activation if activation else eval(self.settings['non_linearity'])
    
  def forward(self, G, feat_dict):
    # the input is a dictionary of node features for each type
    Wh_l2c = self.weight['l2c'](feat_dict['literal'])
    G.nodes['literal'].data['Wh_l2c'] = Wh_l2c
    G['l2c'].update_all(fn.copy_src('Wh_l2c', 'm'), fn.mean('m', 'h'))
    cembs = self.activation(G.nodes['clause'].data['h'])            # cembs now holds the half-round embedding

    Wh_c2l = self.weight['c2l'](torch.cat([cembs,feat_dict['clause']], dim=1))
    G.nodes['clause'].data['Wh_c2l'] = Wh_c2l
    G['c2l'].update_all(fn.copy_src('Wh_c2l', 'm'), fn.mean('m', 'h'))
    lembs = self.activation(G.nodes['literal'].data['h'])
                    
    return lembs
    
class AAGLayer(nn.Module):
  def __init__(self, in_size, out_size, activation=None, settings=None):
    super(AAGLayer, self).__init__()
    self.ntype = 'literal'
    self.etypes = ['aag_forward', 'aag_backward']
    # W_r for each relation
    self.weight = nn.ModuleDict({
      self.etypes[0] : nn.Linear(in_size, out_size),
      self.etypes[1] : nn.Linear(in_size, out_size)
    })
    self.settings = settings if settings else CnfSettings()
    self.activation = activation if activation else eval(self.settings['non_linearity'])
  
  def forward(self, G, feat_dict):
    # the input is a dictionary of node features for each type
    Wh_af = self.weight['aag_forward'](feat_dict['literal'])
    G.nodes['literal'].data['Wh_af'] = Wh_af
    Wh_ab = self.weight['aag_backward'](feat_dict['literal'])
    G.nodes['literal'].data['Wh_ab'] = Wh_ab
    
    G['aag_forward'].update_all(fn.copy_src('Wh_af', 'm_af'), fn.sum('m_af', 'h_af'))
    G['aag_backward'].update_all(fn.copy_src('Wh_ab', 'm_ab'), fn.sum('m_ab', 'h_ab'))
    lembs = G.nodes['literal'].data['h_af'] + G.nodes['literal'].data['h_ab']

    # normalize by in_degree(v) + out_degree(v) for each literal node v
    combined_degrees = G.out_degrees(etype="aag_forward") + G.out_degrees(etype="aag_backward")
    combined_degrees[combined_degrees == 0] = 1
    combined_degrees = combined_degrees.float()
    lembs = ((lembs.T)/combined_degrees).T
    G.nodes['literal'].data['h_a'] = lembs
    
    ##########################################################################
    """
    EXAMPLE:   files found in words_test_ryan_mini_1
    10 literals            0,1,2,...,9
    6 aag_forward_edges   (4,0) (4,2) (6,1) (6,4) (8,2) (8,7)
    6 aag_backward_edges  (0,4) (2,4) (1,6) (4,6) (2,8) (7,8)
    The following is what the literal embeddings should be (before the nonlinearity),
        given that we have already passed the cnf_output through a linear layer
        to produce Wh_af, Wh_ab:
    """
    should_be = torch.zeros(10, 5) 
    should_be[0] = Wh_af[4]
    should_be[1] = Wh_af[6]
    should_be[2] = (Wh_af[4] + Wh_af[8])/2
    should_be[4] = (Wh_af[6] + Wh_ab[0] + Wh_ab[2])/3
    should_be[6] = (Wh_ab[1] + Wh_ab[4])/2
    should_be[7] = Wh_af[8]
    should_be[8] = (Wh_ab[2] + Wh_ab[7])/2
    
#    import ipdb
#    ipdb.set_trace()
#    print(lembs)
#    print(should_be)
    ##########################################################################
    
    return self.activation(G.nodes['literal'].data['h_a'])
   

class CnfAagEncoder(nn.Module):
  def __init__(self, in_size, clause_size, out_size, settings=None, **kwargs):
    super(CnfAagLayer, self).__init__()
    if settings is None:
      self.settings = CnfSettings()
    else:
      self.settings = settings

    self.vlabel_dim = self.settings['vlabel_dim']
    self.clabel_dim = self.settings['clabel_dim']
    self.vemb_dim = self.settings['vemb_dim']
    self.cemb_dim = self.settings['cemb_dim']
    self.max_iters = self.settings['max_iters']        
    self.non_linearity = eval(self.settings['non_linearity'])

    self.cnf_layer = CNFLayer(self.vlabel_dim,self.cemb_dim,self.vemb_dim, **kwargs)
    self.aag_layer = AAGLayer(2*self.vemb_dim,vemb_dim, **kwargs)

  def forward(self, G, feat_dict):
    pass #FIXME


    
#class QbfNewEncoder(nn.Module):
#    def __init__(self, **kwargs):
#        super(QbfNewEncoder, self).__init__() 
#        self.settings = kwargs['settings'] if 'settings' in kwargs.keys() else CnfSettings()
#        self.debug = False
#        self.ground_dim = self.settings['ground_dim']
#        self.vlabel_dim = self.settings['vlabel_dim']
#        self.clabel_dim = self.settings['clabel_dim']
#        self.vemb_dim = self.settings['vemb_dim']
#        self.cemb_dim = self.settings['cemb_dim']
#        self.batch_size = self.settings['batch_size']
#        # self.embedding_dim = self.settings['embedding_dim']                
#        self.max_iters = self.settings['max_iters']        
#        self.non_linearity = eval(self.settings['non_linearity'])
#        W_L_params = []
#        B_L_params = []
#        W_C_params = []
#        B_C_params = []
#        # if self.settings['use_bn']:
#        self.vnorm_layers = nn.ModuleList([])
#        for i in range(self.max_iters):
#            W_L_params.append(nn.Parameter(self.settings.FloatTensor(self.cemb_dim,self.vlabel_dim+2*i*self.vemb_dim)))
#            B_L_params.append(nn.Parameter(self.settings.FloatTensor(self.cemb_dim)))
#            W_C_params.append(nn.Parameter(self.settings.FloatTensor(self.vemb_dim,self.clabel_dim+self.cemb_dim)))
#            B_C_params.append(nn.Parameter(self.settings.FloatTensor(self.vemb_dim)))
#            nn_init.normal_(W_L_params[i])
#            nn_init.normal_(B_L_params[i])        
#            nn_init.normal_(W_C_params[i])                
#            nn_init.normal_(B_C_params[i])
#            if self.settings['use_bn']:
#                self.vnorm_layers.append(nn.BatchNorm1d(self.vemb_dim))
#            elif self.settings['use_ln']:
#                self.vnorm_layers.append(nn.LayerNorm(self.vemb_dim))
#
#
#        if self.settings['use_gru']:
#            self.gru = GruOperator(embedding_dim=self.vemb_dim, settings=self.settings)
#
#        self.W_L_params = nn.ParameterList(W_L_params)
#        self.B_L_params = nn.ParameterList(B_L_params)
#        self.W_C_params = nn.ParameterList(W_C_params)
#        self.B_C_params = nn.ParameterList(B_C_params)
#        
#                    
#    def copy_from_encoder(self, other, freeze=False):
#        for i in range(len(other.W_L_params)):
#            self.W_L_params[i] = other.W_L_params[i]
#            self.B_L_params[i] = other.B_L_params[i]
#            self.W_C_params[i] = other.W_C_params[i]
#            self.B_C_params[i] = other.B_C_params[i]
#            if freeze:
#                self.W_L_params[i].requires_grad=False
#                self.B_L_params[i].requires_grad=False
#                self.W_C_params[i].requires_grad=False
#                self.B_C_params[i].requires_grad=False
#            if self.settings['use_bn']:
#                for i, layer in enumerate(other.vnorm_layers):
#                    self.vnorm_layers[i].load_state_dict(layer.state_dict())
#
#
## vlabels are (vars,vlabel_dim)
## clabels are sparse (clauses,clabel_dim)
## cmat_pos and cmat_neg is the bs*v -> bs*c block-diagonal adjacency matrix 
#
#    def forward(self, vlabels, clabels, cmat_pos, cmat_neg, **kwargs):
#        # size = vlabels.size()
#        # bs = size[0]
#        # maxvars = size[1]
#        pos_vars = vlabels
#        neg_vars = vlabels
#        vmat_pos = cmat_pos.t()
#        vmat_neg = cmat_neg.t()
#        
##        import ipdb
##        ipdb.set_trace()
#
#
#        for t, p in enumerate(self.W_L_params):
#            # results is everything we computed so far, its precisely the correct input to W_L_t
#            av = (torch.mm(cmat_pos,pos_vars)+torch.mm(cmat_neg,neg_vars)).t()
#            c_t_pre = self.non_linearity(torch.mm(self.W_L_params[t],av).t() + self.B_L_params[t])
#            c_t = torch.cat([clabels,c_t_pre],dim=1)
#            pv = torch.mm(vmat_pos,c_t).t()
#            nv = torch.mm(vmat_neg,c_t).t()
#            pv_t_pre = self.non_linearity(torch.mm(self.W_C_params[t],pv).t() + self.B_C_params[t])
#            nv_t_pre = self.non_linearity(torch.mm(self.W_C_params[t],nv).t() + self.B_C_params[t])
#            if self.settings['use_bn'] or self.settings['use_ln']:
#                pv_t_pre = self.vnorm_layers[t](pv_t_pre.contiguous())
#                nv_t_pre = self.vnorm_layers[t](nv_t_pre.contiguous())            
#            # if bs>1:
#            #     Tracer()()            
#            pos_vars = torch.cat([pos_vars,pv_t_pre,nv_t_pre],dim=1)
#            neg_vars = torch.cat([neg_vars,nv_t_pre,pv_t_pre],dim=1)
#
#
#        return pos_vars, neg_vars   
    
###############################################################################
"""Test AAGLayer"""
###############################################################################
a = CombinedGraph1Base()
a.load_paired_files(aag_fname = './data/words_test_ryan_mini_1/w.qaiger', qcnf_fname = './data/words_test_ryan_mini_1/w.qaiger.qdimacs')
G = a.G
feat_dict = {
        'literal': G.nodes['literal'].data['lit_embs'],   # num_literals x 9 = 104 x 9
        'clause' : G.nodes['clause'].data['clause_embs']  # num_clauses  x 1 = 109 x 9
}

in_size = feat_dict['literal'].shape[1]
clause_size = 11 
out_size = 7
C = CNFLayer(in_size, clause_size, out_size, activation=F.relu)
C_f = C(G, feat_dict)  

in_size = C_f.shape[1]
out_size = 5 
feat_dict = {'literal': C_f}
A = AAGLayer(in_size, out_size, activation=F.relu)
A_f = A(G, feat_dict)


