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
  
  def forward(self, G, cnf_output):
    # the input is a dictionary of node features for each type
    Wh_af = self.weight['aag_forward'](cnf_output)
    G.nodes['literal'].data['Wh_af'] = Wh_af
    Wh_ab = self.weight['aag_backward'](cnf_output)
    G.nodes['literal'].data['Wh_ab'] = Wh_ab

    G.send(G['aag_forward'].edges(), fn.copy_src('Wh_af', 'm_a'), etype='aag_forward')
    G.send(G['aag_backward'].edges(), fn.copy_src('Wh_ab', 'm_a'), etype='aag_backward')
    G.recv(G.nodes('literal'), fn.mean('m_a', 'h_a'), etype='aag_forward')
    G.recv(G.nodes('literal'), fn.mean('m_a', 'h_a'), etype='aag_backward')
    
    lembs = self.activation(G.nodes['literal'].data['h_a'])
    return lembs
    

class CNFEncoder(nn.Module):
  def __init__(self, settings=None, **kwargs):
    super(CNFEncoder, self).__init__()
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

    self.layers = nn.ModuleList(CNFLayer(self.vlabel_dim,self.cemb_dim,self.vemb_dim, **kwargs))
    for i in range(1,self.max_iters):
      self.layers.append(CNFLayer(2*self.vemb_dim,self.cemb_dim,self.vemb_dim, **kwargs))

  def tie_literals(embs):
    bs = len(embs)
    pos_part = embs[:int(bs/2)]
    neg_part = embs[int(bs/2):]
    return torch.cat([torch.cat([pos_part,neg_part],dim=1), torch.cat([neg_part,pos_part],dim=1)], dim=0)


  def forward(self, G, feat_dict, **kwargs):
    embs = CNFEncoder.tie_literals(self.layers[0](G,feat_dict))
    for i in range(1,self.max_iters):      
      feat_dict['literal'] = embs
      pre_embs = self.layers[i](G, feat_dict)
      embs = CNFEncoder.tie_literals(pre_embs)

    return embs
class CnfAagEncoder(nn.Module):
  def __init__(self, in_size, clause_size, out_size, settings=None, **kwargs):
    super(CnfAagEncoder, self).__init__()
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

  def forward(self, G, feat_dict, **kwargs):



    
class QbfNewEncoder(nn.Module):
    def __init__(self, **kwargs):
        super(QbfNewEncoder, self).__init__() 
        self.settings = kwargs['settings'] if 'settings' in kwargs.keys() else CnfSettings()
        self.debug = False
        self.ground_dim = self.settings['ground_dim']
        self.vlabel_dim = self.settings['vlabel_dim']
        self.clabel_dim = self.settings['clabel_dim']
        self.vemb_dim = self.settings['vemb_dim']
        self.cemb_dim = self.settings['cemb_dim']
        self.batch_size = self.settings['batch_size']
        # self.embedding_dim = self.settings['embedding_dim']                
        self.max_iters = self.settings['max_iters']        
        self.non_linearity = eval(self.settings['non_linearity'])
        W_L_params = []
        B_L_params = []
        W_C_params = []
        B_C_params = []
        # if self.settings['use_bn']:
        self.vnorm_layers = nn.ModuleList([])
        for i in range(self.max_iters):
            W_L_params.append(nn.Parameter(self.settings.FloatTensor(self.cemb_dim,self.vlabel_dim+2*i*self.vemb_dim)))
            B_L_params.append(nn.Parameter(self.settings.FloatTensor(self.cemb_dim)))
            W_C_params.append(nn.Parameter(self.settings.FloatTensor(self.vemb_dim,self.clabel_dim+self.cemb_dim)))
            B_C_params.append(nn.Parameter(self.settings.FloatTensor(self.vemb_dim)))
            nn_init.normal_(W_L_params[i])
            nn_init.normal_(B_L_params[i])        
            nn_init.normal_(W_C_params[i])                
            nn_init.normal_(B_C_params[i])
            if self.settings['use_bn']:
                self.vnorm_layers.append(nn.BatchNorm1d(self.vemb_dim))
            elif self.settings['use_ln']:
                self.vnorm_layers.append(nn.LayerNorm(self.vemb_dim))


        if self.settings['use_gru']:
            self.gru = GruOperator(embedding_dim=self.vemb_dim, settings=self.settings)

        self.W_L_params = nn.ParameterList(W_L_params)
        self.B_L_params = nn.ParameterList(B_L_params)
        self.W_C_params = nn.ParameterList(W_C_params)
        self.B_C_params = nn.ParameterList(B_C_params)
        
                    
    def copy_from_encoder(self, other, freeze=False):
        for i in range(len(other.W_L_params)):
            self.W_L_params[i] = other.W_L_params[i]
            self.B_L_params[i] = other.B_L_params[i]
            self.W_C_params[i] = other.W_C_params[i]
            self.B_C_params[i] = other.B_C_params[i]
            if freeze:
                self.W_L_params[i].requires_grad=False
                self.B_L_params[i].requires_grad=False
                self.W_C_params[i].requires_grad=False
                self.B_C_params[i].requires_grad=False
            if self.settings['use_bn']:
                for i, layer in enumerate(other.vnorm_layers):
                    self.vnorm_layers[i].load_state_dict(layer.state_dict())


# vlabels are (vars,vlabel_dim)
# clabels are sparse (clauses,clabel_dim)
# cmat_pos and cmat_neg is the bs*v -> bs*c block-diagonal adjacency matrix 

    def forward(self, vlabels, clabels, cmat_pos, cmat_neg, **kwargs):
        # size = vlabels.size()
        # bs = size[0]
        # maxvars = size[1]
        pos_vars = vlabels
        neg_vars = vlabels
        vmat_pos = cmat_pos.t()
        vmat_neg = cmat_neg.t()
        
#        import ipdb
#        ipdb.set_trace()


        for t, p in enumerate(self.W_L_params):
            # results is everything we computed so far, its precisely the correct input to W_L_t
            av = (torch.mm(cmat_pos,pos_vars)+torch.mm(cmat_neg,neg_vars)).t()
            c_t_pre = self.non_linearity(torch.mm(self.W_L_params[t],av).t() + self.B_L_params[t])
            c_t = torch.cat([clabels,c_t_pre],dim=1)
            pv = torch.mm(vmat_pos,c_t).t()
            nv = torch.mm(vmat_neg,c_t).t()
            pv_t_pre = self.non_linearity(torch.mm(self.W_C_params[t],pv).t() + self.B_C_params[t])
            nv_t_pre = self.non_linearity(torch.mm(self.W_C_params[t],nv).t() + self.B_C_params[t])
            if self.settings['use_bn'] or self.settings['use_ln']:
                pv_t_pre = self.vnorm_layers[t](pv_t_pre.contiguous())
                nv_t_pre = self.vnorm_layers[t](nv_t_pre.contiguous())            
            # if bs>1:
            #     Tracer()()            
            pos_vars = torch.cat([pos_vars,pv_t_pre,nv_t_pre],dim=1)
            neg_vars = torch.cat([neg_vars,nv_t_pre,pv_t_pre],dim=1)


        return pos_vars, neg_vars   
    
###############################################################################
"""Test CNFLayer and AAGLayer"""
###############################################################################
#a = CombinedGraph1Base()
#a.load_paired_files(aag_fname = './data/words_easy_train/words_0_SAT.qaiger', qcnf_fname = './data/words_easy_train/words_0_SAT.qaiger.qdimacs')
#feat_dict = {
#        'literal': a.G.nodes['literal'].data['lit_embs'],   # num_literals x 9 = 104 x 9
#        'clause' : a.G.nodes['clause'].data['clause_embs']  # num_clauses  x 1 = 109 x 9
#}
#
#in_size = feat_dict['literal'].shape[1]
#clause_size = 11 #a.G.number_of_nodes('clause')
#out_size = 107 #9 #FIXME??
#C = CNFLayer(in_size, clause_size, out_size, activation=F.relu)
#C_f = C(a.G, feat_dict) # shape        in_size x out_size = 
#print(C_f)
#print(C_f.shape)
#
#in_size = C_f.shape[1]
#out_size = 82 #FIXME??
#A = AAGLayer(in_size, out_size, activation=F.relu)
#A_f = A(a.G, C_f)
#print(A_f)
#print(A_f.shape)



