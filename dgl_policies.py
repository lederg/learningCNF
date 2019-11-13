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
    lembs = self.activation(lembs)
    G.nodes['literal'].data['h_a'] = lembs
    
    return lembs
   

class DGLEncoder(nn.Module):
  def __init__(self, settings=None, **kwargs):
    super(DGLEncoder, self).__init__()
    self.settings = settings if settings else CnfSettings()
    self.vlabel_dim = self.settings['vlabel_dim']
    self.clabel_dim = self.settings['clabel_dim']
    self.vemb_dim = self.settings['vemb_dim']
    self.cemb_dim = self.settings['cemb_dim']
    self.max_iters = self.settings['max_iters']        
    self.non_linearity = eval(self.settings['non_linearity'])

  def tie_literals(embs):    
    n, vembs = int(embs.shape[0]/2), embs.shape[1]
    y = embs.view(n, 2, vembs)
    pos_part = y.transpose(1,2)[:,:,0]
    neg_part = y.transpose(1,2)[:,:,1]
    cp = torch.cat([pos_part,neg_part],dim=1)
    cn = torch.cat([neg_part,pos_part],dim=1)
    return torch.stack((cp,cn), dim=1).view(2*n, 2*vembs)

  def forward(self, G, feat_dict, **kwargs):
    raise NotImplementedError

class CNFEncoder(DGLEncoder):
  def __init__(self, settings=None, **kwargs):
    super(CNFEncoder, self).__init__(settings=settings, **kwargs)

    self.layers = nn.ModuleList([CNFLayer(self.vlabel_dim, self.cemb_dim, self.vemb_dim, activation=self.non_linearity, **kwargs)])
    for i in range(1,self.max_iters):
      self.layers.append(CNFLayer(2*self.vemb_dim, self.cemb_dim, self.vemb_dim, activation=self.non_linearity, **kwargs))

  def forward(self, G, feat_dict, **kwargs):
    embs = DGLEncoder.tie_literals(self.layers[0](G,feat_dict))
    for i in range(1,self.max_iters):      
      feat_dict['literal'] = embs
      pre_embs = self.layers[i](G, feat_dict)
      embs = DGLEncoder.tie_literals(pre_embs)
    return embs

class CnfAagEncoder(DGLEncoder):
  def __init__(self, settings=None, **kwargs):
    super(CnfAagEncoder, self).__init__(settings=settings, **kwargs)
    
    self.cnf_layers = nn.ModuleList([CNFLayer(self.vlabel_dim, self.cemb_dim, self.vemb_dim, activation=self.non_linearity, **kwargs)])
    self.aag_layers = nn.ModuleList([AAGLayer(2*self.vemb_dim, self.vemb_dim, activation=self.non_linearity, **kwargs)])
    for i in range(1,self.max_iters):
      self.cnf_layers.append(CNFLayer(2*self.vemb_dim, self.cemb_dim, self.vemb_dim, activation=self.non_linearity, **kwargs))
      self.aag_layers.append(AAGLayer(2*self.vemb_dim, self.vemb_dim, activation=self.non_linearity, **kwargs))

  def forward(self, G, feat_dict, **kwargs):
    pre_embs_cnf = self.cnf_layers[0](G,feat_dict)
    embs_cnf = DGLEncoder.tie_literals(pre_embs_cnf)
    feat_dict['literal'] = embs_cnf
    pre_embs = self.aag_layers[0](G, feat_dict)
    embs = DGLEncoder.tie_literals(pre_embs)
    for i in range(1,self.max_iters):      
      feat_dict['literal'] = embs
      pre_embs_cnf = self.cnf_layers[i](G,feat_dict)
      embs_cnf = DGLEncoder.tie_literals(pre_embs_cnf)
      feat_dict['literal'] = embs_cnf
      pre_embs = self.aag_layers[i](G, feat_dict)
      embs = DGLEncoder.tie_literals(pre_embs)
    return embs

###############################################################################
    
class DGLPolicy(PolicyBase):
  def __init__(self, encoder=None, **kwargs):
    super(DGLPolicy, self).__init__(**kwargs)
    self.final_embedding_dim = 2*self.max_iters*self.vemb_dim+self.vlabel_dim
    self.hidden_dim = 50
    if encoder:
      print('Bootstraping Policy from existing encoder')
      self.encoder = encoder
    else:
      self.encoder = CNFEncoder(**kwargs)
    if self.settings['use_global_state']:
      self.linear1 = nn.Linear(self.state_dim+self.final_embedding_dim, self.policy_dim1)
    else:
      self.linear1 = nn.Linear(self.final_embedding_dim, self.policy_dim1)

    if self.policy_dim2:
      self.linear2 = nn.Linear(self.policy_dim1,self.policy_dim2)
      self.action_score = nn.Linear(self.policy_dim2,1)
    else:
      self.action_score = nn.Linear(self.policy_dim1,1)
    if self.state_bn:
      self.state_vbn = MovingAverageVBN((self.snorm_window,self.state_dim))
    self.use_global_state = self.settings['use_global_state']
    self.activation = eval(self.settings['policy_non_linearity'])

  
  # state is just a (batched) vector of fixed size state_dim which should be expanded. 
  # ground_embeddings are batch * max_vars * ground_embedding

  # cmat_net and cmat_pos are already "batched" into a single matrix

  def forward(self, obs, **kwargs):
    state = obs.state
#    import ipdb
#    ipdb.set_trace()    
    G = obs.ext_data.local_var_()
    ground_embeddings = G.nodes['literal'].data['lit_labels']

    aux_losses = []
    size = ground_embeddings.size()
    self.batch_size=size[0]
    if 'vs' in kwargs.keys():
      vs = kwargs['vs']   
    else:
      pos_vars, neg_vars = self.encoder(ground_embeddings.view(-1,self.vlabel_dim), clabels.view(-1,self.clabel_dim), cmat_pos=cmat_pos, cmat_neg=cmat_neg, **kwargs)
      vs_pos = pos_vars.view(self.batch_size,-1,self.final_embedding_dim)
      vs_neg = neg_vars.view(self.batch_size,-1,self.final_embedding_dim)
      vs = torch.cat([vs_pos,vs_neg])
      if 'do_debug' in kwargs:
        Tracer()()
    

    if self.use_global_state:
      if self.state_bn:
        state = self.state_vbn(state)
      a = state.unsqueeze(0).expand(2,*state.size()).contiguous().view(2*self.batch_size,1,self.state_dim)
      reshaped_state = a.expand(2*self.batch_size,size[1],self.state_dim) # add the maxvars dimention
      inputs = torch.cat([reshaped_state, vs],dim=2).view(-1,self.state_dim+self.final_embedding_dim)
    else:
      inputs = vs.view(-1,self.final_embedding_dim)

    if self.policy_dim2:      
      outputs = self.action_score(self.activation(self.linear2(self.activation(self.linear1(inputs)))))
    else:
      outputs = self.action_score(self.activation(self.linear1(inputs)))
    # Tracer()()
    outputs = outputs.view(2,self.batch_size,-1)
    outputs = outputs.transpose(2,0).transpose(1,0)     # batch x numvars x pos-neg
    outputs = outputs.contiguous().view(self.batch_size,-1)
    if self.settings['ac_baseline'] and self.batch_size > 1:
      embs = vs.view(2,self.batch_size,-1,self.final_embedding_dim).transpose(0,1).contiguous().view(self.batch_size,-1,self.final_embedding_dim)
      mask = torch.cat([obs.vmask]*2,dim=1)
      graph_embedding, value_aux_loss = self.value_attn(state,embs,attn_mask=mask)
      aux_losses.append(value_aux_loss)
      if self.settings['use_state_in_vn']:
        val_inp = torch.cat([state,graph_embedding.view(self.batch_size,-1)],dim=1)
      else:
        val_inp = graph_embedding.view(self.batch_size,-1)
      value = self.value_score2(self.activation(self.value_score1(val_inp)))
    else:
      value = None
    return outputs, value, vs, aux_losses

  def get_allowed_actions(self, obs, **kwargs):
    rc = cadet_utils.get_allowed_actions(obs,**kwargs)
    s = rc.shape
    rc = rc.unsqueeze(2).expand(*s,2).contiguous()
    rc = rc.view(s[0],-1)
    return rc

  def translate_action(self, action, obs, **kwargs):
    try:
      if action in ['?']:
        return action
    except:
      pass
    return (int(action/2),int(action%2))

  def combine_actions(self, actions, **kwargs):
    return self.settings.LongTensor(actions)

  def select_action(self, obs_batch, **kwargs):
    [logits], *_ = self.forward(collate_observations([obs_batch]))
    allowed_actions = self.get_allowed_actions(obs_batch)[0]
    allowed_idx = self.settings.cudaize_var(torch.from_numpy(np.where(allowed_actions.numpy())[0]))
    l = logits[allowed_idx]
    probs = F.softmax(l.contiguous().view(1,-1),dim=1)
    dist = probs.data.cpu().numpy()[0]
    choices = range(len(dist))
    aux_action = np.random.choice(choices, p=dist)
    action = allowed_idx[aux_action]
    return action, 0

  def compute_loss(self, transition_data, **kwargs):
    _, _, _, rewards, *_ = zip(*transition_data)
    collated_batch = collate_transitions(transition_data,settings=self.settings)
    collated_batch.state = cudaize_obs(collated_batch.state)
    logits, values, _, aux_losses = self.forward(collated_batch.state, prev_obs=collated_batch.prev_obs)
    allowed_actions = Variable(self.get_allowed_actions(collated_batch.state))
    if self.settings['cuda']:
      allowed_actions = allowed_actions.cuda()
    # unpacked_logits = unpack_logits(logits, collated_batch.state.pack_indices[1])
    effective_bs = len(logits)    

    if self.settings['masked_softmax']:
      allowed_mask = allowed_actions.float()      
      probs, debug_probs = masked_softmax2d(logits,allowed_mask)
    else:
      probs = F.softmax(logits, dim=1)
    all_logprobs = safe_logprobs(probs)
    if self.settings['disallowed_aux']:        # Disallowed actions are possible, so we add auxilliary loss
      aux_probs = F.softmax(logits,dim=1)
      disallowed_actions = Variable(allowed_actions.data^1).float()      
      disallowed_mass = (aux_probs*disallowed_actions).sum(1)
      disallowed_loss = disallowed_mass.mean()
      # print('Disallowed loss is {}'.format(disallowed_loss))

    returns = self.settings.FloatTensor(rewards)
    if self.settings['ac_baseline']:
      adv_t = returns - values.squeeze().data      
      value_loss = mse_loss(values.squeeze(), Variable(returns))    
      print('Value loss is {}'.format(value_loss.data.numpy()))
      print('Value Auxilliary loss is {}'.format(sum(aux_losses).data.numpy()))
      if i>0 and i % 60 == 0:
        vecs = {'returns': returns.numpy(), 'values': values.squeeze().data.numpy()}        
        pprint_vectors(vecs)
    else:
      adv_t = returns
      value_loss = 0.
    actions = collated_batch.action    
    try:
      logprobs = all_logprobs.gather(1,Variable(actions).view(-1,1)).squeeze()
    except:
      Tracer()()
    entropies = (-probs*all_logprobs).sum(1)    
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + float(np.finfo(np.float32).eps))
    # Tracer()()
    if self.settings['normalize_episodes']:
      episodes_weights = normalize_weights(collated_batch.formula.cpu().numpy())
      adv_t = adv_t*self.settings.FloatTensor(episodes_weights)    
    if self.settings['use_sum']:
      pg_loss = (-Variable(adv_t)*logprobs).sum()
    else:
      pg_loss = (-Variable(adv_t)*logprobs).mean()
    total_aux_loss = sum(aux_losses) if aux_losses else 0.    
    loss = pg_loss + self.lambda_value*value_loss + self.lambda_aux*total_aux_loss

    if self.use_global_state and self.state_bn:
      self.state_vbn.recompute_moments(collated_batch.state.state.detach())

    return loss, logits
    
###############################################################################
"""Tests"""
###############################################################################
#a = CombinedGraph1Base()
#a.load_paired_files(aag_fname = './data/words_test_ryan_mini_1/w.qaiger', qcnf_fname = './data/words_test_ryan_mini_1/w.qaiger.qdimacs')
#feat_dict = {
#        'literal': a.G.nodes['literal'].data['lit_labels'],   
#        'clause' : a.G.nodes['clause'].data['clause_labels'] 
#}
#in_size = feat_dict['literal'].shape[1]
#clause_size = 11
#out_size = 7
#C = CNFLayer(in_size, clause_size, out_size, activation=F.relu)
#C_f = C(a.G, feat_dict)  
#in_size = C_f.shape[1]
#out_size = 5
#A = AAGLayer(in_size, out_size, activation=F.relu)
#feat_dict = {'literal' : C_f}
#A_f = A(a.G, feat_dict)
#    ##########################################################################
#    ## TO GO INSIDE AAGLayer() Forward()
#    """
#    EXAMPLE:   files found in words_test_ryan_mini_1
#    10 literals            0,1,2,...,9
#    6 aag_forward_edges   (4,0) (4,2) (6,1) (6,4) (8,2) (8,7)
#    6 aag_backward_edges  (0,4) (2,4) (1,6) (4,6) (2,8) (7,8)
#    The following is what the literal embeddings should be (before the nonlinearity),
#        given that we have already passed the cnf_output through a linear layer
#        to produce Wh_af, Wh_ab:
#    """
#    embs = torch.zeros(10, 5) 
#    embs[0] = Wh_af[4]
#    embs[1] = Wh_af[6]
#    embs[2] = (Wh_af[4] + Wh_af[8])/2
#    embs[4] = (Wh_af[6] + Wh_ab[0] + Wh_ab[2])/3
#    embs[6] = (Wh_ab[1] + Wh_ab[4])/2
#    embs[7] = Wh_af[8]
#    embs[8] = (Wh_ab[2] + Wh_ab[7])/2
#    
##    import ipdb
##    ipdb.set_trace()
##    print(lembs)
##    print(should_be)
#    ##########################################################################
###############################################################################
### TEST tie_literals()
#e = torch.zeros(10,4)
#for i in range(10):
#  e[i] = i
#e
#DGLEncoder.tie_literals(e)
###############################################################################
### Test the Encoders:
#a = CombinedGraph1Base()
#a.load_paired_files(aag_fname = './data/words_test_ryan_mini_1/w.qaiger', qcnf_fname = './data/words_test_ryan_mini_1/w.qaiger.qdimacs')
#### put the following inside DGLEncoder __init__()
#self.vlabel_dim = 4
#self.clabel_dim = 1
#self.vemb_dim = 3
#self.cemb_dim = 11 #hidden
#self.max_iters = 2   
#self.non_linearity = F.relu
####
#G = a.G
#feat_dict = {'literal' : torch.ones(10,4), 'clause': torch.ones(10,1)}
#e = CnfAagEncoder()
#v = e(G, feat_dict)
#import ipdb
#ipdb.set_trace()
###############################################################################
#### MAILBOX testing
#def reduce_af(nodes):
#    import ipdb
#    ipdb.set_trace()
#    return { 'haf' : torch.sum(nodes.mailbox['m_af'], dim=1) }
#def reduce_ab(nodes):
#    import ipdb
#    ipdb.set_trace()
#    return { 'hab' : torch.sum(nodes.mailbox['m_ab'], dim=1) }
#
#    G['aag_forward'].update_all(fn.copy_src('Wh_af', 'm_af'), reduce_af)
#    G['aag_backward'].update_all(fn.copy_src('Wh_ab', 'm_ab'), reduce_ab)
#    import ipdb
#    ipdb.set_trace()
###############################################################################