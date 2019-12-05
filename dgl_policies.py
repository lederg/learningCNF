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
from batch_dgl import batched_combined_graph
from dgl_graph_base import DGL_Graph_Base


class CNFLayer(nn.Module):
  def __init__(self, in_size, clause_size, out_size, activation=None, settings=None, **kwargs):
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


class WordLevelLayer(nn.Module):
  def __init__(self, in_size, out_size, activation=None, settings=None):
    super(WordLevelLayer, self).__init__()
    self.ntypes = ['literal', 'wl_node']
    self.etypes = [
#                   'l2w', 'w2l', 
                   '+_f', 'and_f', 'or_f', 'xor_f', 'invert_f', 
                   'abs_f', 'neg_f', '=_f', '!=_f', '-L_f', '-R_f', 
                   '<L_f', '<R_f', '<=L_f', '<=R_f', '>L_f', '>R_f', '>=L_f', '>=R_f', 
                   '+_b', 'and_b', 'or_b', 'xor_b', 'invert_b', 
                   'abs_b', 'neg_b', '=_b',   '!=_b', '-L_b', '-R_b', 
                   '<L_b', '<R_b', '<=L_b', '<=R_b', '>L_b', '>R_b', '>=L_b', '>=R_b']
    self.etypes_f = ['+_f', 'and_f', 'or_f', 'xor_f', 'invert_f', 
                   'abs_f', 'neg_f', '=_f', '!=_f', '-L_f', '-R_f', 
                   '<L_f', '<R_f', '<=L_f', '<=R_f', '>L_f', '>R_f', '>=L_f', '>=R_f']
    self.etypes_b = ['+_b', 'and_b', 'or_b', 'xor_b', 'invert_b', 
                   'abs_b', 'neg_b', '=_b',   '!=_b', '-L_b', '-R_b', 
                   '<L_b', '<R_b', '<=L_b', '<=R_b', '>L_b', '>R_b', '>=L_b', '>=R_b']
    # W_r for each relation
    self.weight = nn.ModuleDict({
      self.etypes[i] : nn.Linear(in_size, out_size) for i in range(len(self.etypes))
    })
    self.settings = settings if settings else CnfSettings()
    self.activation = activation if activation else eval(self.settings['non_linearity'])
    
  def get_edges(self, G, node, forward=True):
      """
      forward == True -> 'f' edges;    forward == False -> 'b' edges
      in the forward case, return the edges ending with `node`
      in the backward case, return the edges beggining with `node`
      """
      if forward:
          edge_list = self.etypes_f
          k = 1
      else: # backward
          edge_list = self.etypes_b
          k = 0
      result = []
      for etype in edge_list:
          if G.number_of_edges(etype=etype) > 0:
               e = G.edges(etype=etype)
               if node in e[k]:
                   edges = [(e[0][i].item(), e[1][i].item()) for i in range(len(e[0])) if e[k][i].item() == node]
                   result.append( (edges, etype) )
      return result
  
  def send_messages(self, G, prop_order, forward=True):    
    for n in prop_order:
        pre_aggregation = []
        for x in self.get_edges(G, n, forward=forward):
            edges, etype = x[0], x[1]
            pre_aggregation += [ self.weight[etype](G.nodes['wl_node'].data['Wh_wl'][edges[i][0]]) for i in range(len(edges)) ]
        
        if forward:
            msg = self.activation( torch.mean(torch.stack(pre_aggregation), dim=0) )
            G.nodes['wl_node'].data['Wh_wl'][n] = msg
        else: # backward propagation
            msgs = [ self.activation(p) for p in pre_aggregation ]
            send_to = []
            for a in self.get_edges(G, n, forward=forward):
                edges = a[0]
                for e in edges:
                    send_to.append(e[1])
            for i, dst in enumerate(send_to):
                G.nodes['wl_node'].data['Wh_wl'][dst] = msgs[i]
            

  def forward(self, G, feat_dict): 
    # the input is a dictionary of node features for each type
    G.nodes['wl_node'].data['Wh_wl'] = feat_dict['wl_node']

    first_prop_node = int(G.nodes['wl_node'].data['begin_propagation'][0].item())
    n = G.number_of_nodes(ntype='wl_node')
    forwards_order = list(range(first_prop_node, n)) # non-leaves in order
    self.send_messages(G, forwards_order, forward=True)
    
    backwards_order = [n - i -1 + first_prop_node for i in forwards_order]
    self.send_messages(G, backwards_order, forward=False)
    
    return G.nodes['wl_node'].data['Wh_wl']
    
    

###############################################################################
##### Test WordLevelLayer
b = DGL_Graph_Base()
b.load_paired_files(
        wordlevel_fname = './data/words_3_levels_1/words_2.wordlevel', 
        aag_fname = './data/words_3_levels_1/words_2.qaiger', 
        qcnf_fname = './data/words_3_levels_1/words_2.qaiger.qdimacs')
x = torch.zeros([b.G.number_of_nodes("wl_node"), 5]) 
for i in range(b.G.number_of_nodes("wl_node")): x[i] = i
feat_dict = {
        #'literal': b.G.nodes['literal'].data['lit_labels'],   
        'wl_node' : x
}
in_size = feat_dict['wl_node'].shape[1]
out_size = in_size # FIXME # as of now, only works for same size

#
#H = dgl.DGLGraph()
#H.add_nodes(b.G.number_of_nodes("wl_node"))
#H.add_edges(b.G.edges(etype="wl_op_f")[0], b.G.edges(etype="wl_op_f")[1])
#
W = WordLevelLayer(in_size, out_size, activation=F.relu)
#W_f = W(H, feat_dict)
W_f = W(b.G, feat_dict)


###############################################################################
   
###############################################################################

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
    vlabels = feat_dict['literal']
    embs = DGLEncoder.tie_literals(self.layers[0](G,feat_dict))
    for i in range(1,self.max_iters):      
      feat_dict['literal'] = embs
      pre_embs = self.layers[i](G, feat_dict)
      embs = DGLEncoder.tie_literals(pre_embs)
    return torch.cat([embs,vlabels],dim=1)

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
    self.final_embedding_dim = 2*self.vemb_dim+self.vlabel_dim
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
  # ground_embeddings are batch * num_lits * ground_embedding

  # cmat_net and cmat_pos are already "batched" into a single matrix

  def forward(self, obs, G, **kwargs):
    state = obs.state 
    self.batch_size=obs.state.shape[0]

    feat_dict = {'literal': G.nodes['literal'].data['lit_labels'], 'clause': torch.zeros(G.number_of_nodes('clause'),1)}
    # feat_dict = {'literal': G.nodes['literal'].data['lit_labels'], 'clause': G.nodes['clause'].data['clause_labels']}
    aux_losses = []    
    
    num_lits = int(feat_dict['literal'].shape[0] / self.batch_size)
    
    if 'vs' in kwargs.keys():
      vs = kwargs['vs']   
    else:
      vs = self.encoder(G, feat_dict, **kwargs)
      if 'do_debug' in kwargs:
        Tracer()()

    if self.use_global_state:
      if self.state_bn:
        state = self.state_vbn(state)
      reshaped_state = state.unsqueeze(1).expand(state.shape[0],num_lits,self.state_dim).contiguous().view(-1,self.state_dim)
      inputs = torch.cat([reshaped_state, vs],dim=1).view(-1,self.state_dim+self.final_embedding_dim)
    else:
      inputs = vs.view(-1,self.final_embedding_dim)

    if self.policy_dim2:      
      outputs = self.action_score(self.activation(self.linear2(self.activation(self.linear1(inputs)))))
    else:
      outputs = self.action_score(self.activation(self.linear1(inputs)))
    outputs = outputs.view(self.batch_size,-1)

    return outputs, vs

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

  def select_action(self, obs, **kwargs):
    [logits], *_ = self.forward(obs, obs.ext_data.G.local_var())
    allowed_actions = self.get_allowed_actions(obs)[0]
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
    G = batched_combined_graph([x.G for x in collated_batch.state.ext_data])
    logits, *_ = self.forward(collated_batch.state, G, prev_obs=collated_batch.prev_obs)
    allowed_actions = self.get_allowed_actions(collated_batch.state)
    if self.settings['cuda']:
      allowed_actions = allowed_actions.cuda()    
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
    actions = collated_batch.action    
    try:
      logprobs = all_logprobs.gather(1,actions.view(-1,1)).squeeze()
    except:
      Tracer()()
    entropies = (-probs*all_logprobs).sum(1)    
    adv_t = (returns - returns.mean()) / (returns.std() + float(np.finfo(np.float32).eps))
    if self.settings['use_sum']:
      pg_loss = (-adv_t*logprobs).sum()
    else:
      pg_loss = (-adv_t*logprobs).mean()
    
    loss = pg_loss

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
