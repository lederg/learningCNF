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
    def __init__(self, in_size, clause_size, out_size, activation):
        super(CNFLayer, self).__init__()
        self.ntypes = ['literal', 'clause']
        self.etypes = ['l2c', 'c2l']
        # W_r for each relation
        self.weight = nn.ModuleDict({
                self.etypes[0] : nn.Linear(in_size, clause_size),
                self.etypes[1] : nn.Linear(clause_size, out_size)
        })
        self.activation = activation
    
    def forward(self, G, feat_dict):
        # the input is a dictionary of node features for each type
#        funcs = {}
        
        Wh_l2c = self.weight['l2c'](feat_dict['literal'])
        Wh_l2c = self.activation(Wh_l2c)
        G.nodes['literal'].data['Wh_l2c'] = Wh_l2c
        G.update_all(message_func=fn.copy_src(src='literal', out='m'),
                     reduce_func=fn.mean(msg='m',out='h'))
        
        
        
            
        #return {'Wh' : Wh}
        
        # save it in graph for message passing 
         G.nodes[srctype].data['Wh_%s' % etype] = Wh

        
        # Specify per-relation message passing functions: (message_func, reduce_func).
        # Note that the results are saved to the same destination feature 'h', which
        # hints the type wise reducer for aggregation.
        funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))
        
        # Trigger message passing of multiple types.
    	# The first argument is the message passing functions for each relation.
    	# The second one is the type wise reducer, could be "sum", "max",
    	# "min", "mean", "stack"
        G.update_all(funcs, 'sum')
    	# return the updated node feature dictionary
        return {ntype : G.nodes[ntype].data['h'] for ntype in G.ntypes}




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


