import random
import os
import torch
from torch.autograd import Variable
from collections import namedtuple

from settings import *
from qbf_data import *
from cadet_env import *
from rl_utils import *
from rl_types import *
from utils import *


  # We return a ground embedding of (self.num_vars,7), where embedding is 0 - universal, 1 - existential, 2 - pad, 
  # 3 - determinized, 4 - activity, [5,6] - pos/neg determinization
  # 3 is obviously empty here

def get_base_ground(qbf, settings=None):
  if not settings:
    settings = CnfSettings()
  rc = np.zeros([qbf.num_vars,settings['ground_dim']]).astype(float)
  for j, val in enumerate(qbf.var_types):
    rc[j][val] = True
  return rc

def get_input_from_qbf(qbf, settings=None):
  if not settings:
    settings = CnfSettings()
  a = qbf.as_np_dict()  
  rc_i = a['sp_indices']
  rc_v = a['sp_vals']
  sp_ind_pos = torch.from_numpy(rc_i[np.where(rc_v>0)])
  sp_ind_neg = torch.from_numpy(rc_i[np.where(rc_v<0)])
  sp_val_pos = torch.ones(len(sp_ind_pos))
  sp_val_neg = torch.ones(len(sp_ind_neg))
  cmat_pos = Variable(torch.sparse.FloatTensor(sp_ind_pos.t(),sp_val_pos,torch.Size([qbf.num_clauses,qbf.num_vars])))
  cmat_neg = Variable(torch.sparse.FloatTensor(sp_ind_neg.t(),sp_val_neg,torch.Size([qbf.num_clauses,qbf.num_vars])))  
  # if settings['cuda']:
  #   cmat_pos, cmat_neg = cmat_pos.cuda(), cmat_neg.cuda()
  return cmat_pos, cmat_neg

# This gets a tuple from stepping the environment:
# state, vars_add, vars_remove, activities, decision, clause, reward, done = env.step(action)
# And it returns the next observation.

def process_observation(env, last_obs, env_obs, settings=None):
  if not settings:
    settings = CnfSettings()

  if True or env_obs.clause:
    cmat_pos, cmat_neg = get_input_from_qbf(env.qbf, settings)
  else:
    cmat_pos, cmat_neg = last_obs.cmat_pos, last_obs.cmat_neg
  ground_embs = np.copy(last_obs.ground.data.numpy().squeeze())
  if env_obs.decision:
    ground_embs[env_obs.decision[0]][IDX_VAR_POLARITY_POS+1-env_obs.decision[1]] = True
  if len(env_obs.vars_add):
    ground_embs[:,IDX_VAR_DETERMINIZED][env_obs.vars_add] = True
  if len(env_obs.vars_remove):
    ground_embs[:,IDX_VAR_DETERMINIZED][env_obs.vars_remove] = False
    ground_embs[:,IDX_VAR_POLARITY_POS:IDX_VAR_POLARITY_NEG][env_obs.vars_remove] = False
  ground_embs[:,IDX_VAR_ACTIVITY] = env_obs.activities
  state = Variable(torch.from_numpy(env_obs.state).float().unsqueeze(0))
  ground_embs = Variable(torch.from_numpy(ground_embs).float().unsqueeze(0))
  return State(state,cmat_pos,cmat_neg,ground_embs)


def new_episode(env, all_episode_files, settings=None, fname=None, **kwargs):
  if not settings:
    settings = CnfSettings()

  if fname is None:
    total_envs = len(all_episode_files)
    fname = all_episode_files[random.randint(0,total_envs-1)]
  env_id = int(os.path.split(fname)[1].split('_')[0])
  # Set up ground_embeddings and adjacency matrices
  state, vars_add, vars_remove, activities, _, _ , _, _ = env.reset(fname)
  assert(len(state)==settings['state_dim'])
  ground_embs = get_base_ground(env.qbf)
  ground_embs[:,IDX_VAR_DETERMINIZED][vars_add] = True
  ground_embs[:,IDX_VAR_ACTIVITY] = activities

  cmat_pos, cmat_neg = get_input_from_qbf(env.qbf, settings)
  
  state = Variable(torch.from_numpy(state).float().unsqueeze(0))
  ground_embs = Variable(torch.from_numpy(ground_embs).float().unsqueeze(0))
  # if settings['cuda']:
  #   cmat_pos, cmat_neg = cmat_pos.cuda(), cmat_neg.cuda()
  #   state, ground_embs = state.cuda(), ground_embs.cuda()
  rc = State(state,cmat_pos, cmat_neg, ground_embs)
  return rc, env_id


# This currently does not work on batches
def get_determinized(obs):
  return obs.ground.long().data[0][:,IDX_VAR_DETERMINIZED].byte()

def action_allowed(obs, action):
  return not get_determinized(obs)[action]

