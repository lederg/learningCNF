import torch
from torch.distributions import Categorical
import ipdb

from settings import *
from cadet_env import *
from rl_model import *
from qbf_data import *


CADET_BINARY = './cadet'

all_episode_files = ['data/mvs.qdimacs']

settings = CnfSettings()
policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
env = CadetEnv(CADET_BINARY, **settings.hyperparameters)

def select_action(state, ground_embs, **kwargs):
  state = torch.from_numpy(state).float().unsqueeze(0)
  ground_embs = torch.from_numpy(ground_embs).unsqueeze(0).float()
  probs = policy(Variable(state), Variable(ground_embs), batch_size=1, **kwargs)
  m = Categorical(probs)
  action = m.sample()
  policy.saved_log_probs.append(m.log_prob(action))
  return action.data[0]

  # We return a ground embedding of (self.num_vars,4), where embedding is 0 - universal, 1 - existential, 2 - pad, 3 - determinized
  # 3 is obviously empty here

def get_base_ground(qbf):
  rc = np.zeros([qbf.num_vars,4]).astype(float)
  for j, val in enumerate(qbf.var_types):
    rc[j][val] = True
  return rc

def handle_episode(fname):

  # Set up ground_embeddings and adjacency matrices
  state, vars_add, vars_remove, _ = env.reset(fname)
  assert(len(state)==settings['state_dim'])
  qbf = env.qbf
  ground_embs = get_base_ground(qbf)
  ground_embs[:,3][vars_add] = True

  a = qbf.as_np_dict()
  rc_i = a['sp_indices']
  rc_v = a['sp_vals']
  sp_ind_pos = torch.from_numpy(rc_i[np.where(rc_v>0)])
  sp_ind_neg = torch.from_numpy(rc_i[np.where(rc_v<0)])
  sp_val_pos = torch.ones(len(sp_ind_pos))
  sp_val_neg = torch.ones(len(sp_ind_neg))
  cmat_pos = Variable(torch.sparse.FloatTensor(sp_ind_pos.t(),sp_val_pos,torch.Size([qbf.num_clauses,qbf.num_vars])),requires_grad=False)
  cmat_neg = Variable(torch.sparse.FloatTensor(sp_ind_neg.t(),sp_val_neg,torch.Size([qbf.num_clauses,qbf.num_vars])),requires_grad=False)

  for t in range(10000):  # Don't infinite loop while learning
    # Could cache ground_embs here, as a Variable
    action = select_action(state, ground_embs, cmat_pos=cmat_pos, cmat_neg=cmat_neg)
    print('Chose action %d' % action)
    if ground_embs[action][3]:
      print('Chose an invalid action!')
      ipdb.set_trace()
    state, vars_add, vars_remove, done = env.step(action)
    if len(vars_add): 
      ground_embs[:,3][vars_add] = True
    if len(vars_remove):
      ground_embs[:,3][vars_remove] = False
    if done:
      break
def cadet_main(settings):
  for fname in all_episode_files:
    handle_episode(fname)
    
