import torch
# from torch.distributions import Categorical
import ipdb
import random

from settings import *
from cadet_env import *
from rl_model import *
from qbf_data import *
from utils import *
import torch.nn.utils as tutils

CADET_BINARY = './cadet'

all_episode_files = ['data/mvs.qdimacs']

settings = CnfSettings()
policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
env = CadetEnv(CADET_BINARY, **settings.hyperparameters)

actions_history = []

def select_action(state, ground_embs, **kwargs):
  state = torch.from_numpy(state).float().unsqueeze(0)
  ground_embs = torch.from_numpy(ground_embs).unsqueeze(0).float()
  probs = policy(Variable(state), Variable(ground_embs), batch_size=1, **kwargs)
  # ipdb.set_trace()
  try:
    dist = probs.data.numpy()[0]
    choices = range(len(dist))
    action = np.random.choice(choices, p=dist)
    aug_action = np.where(np.where(dist>0)[0]==action)    # without the 0 probabilities
  except:
    print('Problem in np.random')
    ipdb.set_trace()
  a = probs[probs>0]
  logprob = a.log()[aug_action]
  # ipdb.set_trace()
  # m = Categorical(probs)  
  # action = m.sample()
  # actions_history.append(action)
  entropy = -(a*a.log()).sum()  
  return action.data[0], logprob, entropy.view(1,)
  # return action.data[0], m.log_prob(action), entropy.view(1,)

  # We return a ground embedding of (self.num_vars,7), where embedding is 0 - universal, 1 - existential, 2 - pad, 
  # 3 - determinized, 4 - activity, [5,6] - pos/neg determinization
  # 3 is obviously empty here

def get_base_ground(qbf):
  rc = np.zeros([qbf.num_vars,GROUND_DIM]).astype(float)
  for j, val in enumerate(qbf.var_types):
    rc[j][val] = True
  return rc

def handle_episode(fname):

  # Set up ground_embeddings and adjacency matrices
  state, vars_add, vars_remove, activities, _, _ = env.reset(fname)
  assert(len(state)==settings['state_dim'])
  qbf = env.qbf
  ground_embs = get_base_ground(qbf)
  ground_embs[:,IDX_VAR_DETERMINIZED][vars_add] = True
  ground_embs[:,IDX_VAR_ACTIVITY] = activities


  a = qbf.as_np_dict()
  rc_i = a['sp_indices']
  rc_v = a['sp_vals']
  sp_ind_pos = torch.from_numpy(rc_i[np.where(rc_v>0)])
  sp_ind_neg = torch.from_numpy(rc_i[np.where(rc_v<0)])
  sp_val_pos = torch.ones(len(sp_ind_pos))
  sp_val_neg = torch.ones(len(sp_ind_neg))
  cmat_pos = Variable(torch.sparse.FloatTensor(sp_ind_pos.t(),sp_val_pos,torch.Size([qbf.num_clauses,qbf.num_vars])),requires_grad=False)
  cmat_neg = Variable(torch.sparse.FloatTensor(sp_ind_neg.t(),sp_val_neg,torch.Size([qbf.num_clauses,qbf.num_vars])),requires_grad=False)

  logprobs = []

  for t in range(10000):  # Don't infinite loop while learning
    # Could cache ground_embs here, as a Variable
    action , logprob, ent = select_action(state, ground_embs, cmat_pos=cmat_pos, cmat_neg=cmat_neg)
    logprobs.append((logprob,ent))
    print('Chose action %d' % action)
    if ground_embs[action][IDX_VAR_DETERMINIZED]:
      print('Chose an invalid action!')
      ipdb.set_trace()
    try:
      state, vars_add, vars_remove, activities, decision, done = env.step(action)
    except Exception as e:
      print(e)
      ipdb.set_trace()
    if done:
      # print('Finished. Rewards are:')
      # print(env.rewards)
      # print(discount(env.rewards, settings['gamma']))
      return env.rewards, logprobs
      break
    if decision:
      ground_embs[decision[0]][IDX_VAR_POLARITY_POS+1-decision[1]] = True
    if len(vars_add):
      ground_embs[:,IDX_VAR_DETERMINIZED][vars_add] = True
    if len(vars_remove):
      ground_embs[:,IDX_VAR_DETERMINIZED][vars_remove] = False
      ground_embs[:,IDX_VAR_POLARITY_POS:IDX_VAR_POLARITY_NEG][vars_remove] = False
    ground_embs[:,IDX_VAR_ACTIVITY] = activities

def ts_bonus(s):
  b = 5.
  return b/float(s)

def cadet_main(settings):
  ds = QbfDataset(dirname='data/dataset1/')
  all_episode_files = ds.get_files_list()
  total_envs = len(all_episode_files)

  for iter in range(400):
    rewards = []
    time_steps_this_batch = 0
    logprobs = []
    entropies = []
    time_steps_this_batch = 0
    while time_steps_this_batch < settings['min_timesteps_per_batch']:
      fname = all_episode_files[random.randint(0,total_envs-1)]
      print('Starting episode for file %s' % fname)
      r, log_probs_and_entropy = handle_episode(fname)
      s = len(r)
      r[-1] += ts_bonus(s)
      time_steps_this_batch += s
      # ipdb.set_trace()
      rewards.extend(discount(r, settings['gamma']))
      ep_logprobs, ep_entropy = zip(*log_probs_and_entropy)
      logprobs.extend(ep_logprobs)
      entropies.extend(ep_entropy)
      print('Finished episode for file %s in %d steps' % (fname, s))

    returns = torch.Tensor(rewards)
    batch_entropy = torch.cat(entropies)
    logprobs = torch.cat(logprobs)
    returns = (returns - returns.mean()) / (returns.std() + np.finfo(np.float32).eps)
    loss = (-Variable(returns)*logprobs - 0.0001*batch_entropy).mean()
    optimizer.zero_grad()
    # ipdb.set_trace()
    loss.backward()
    # tutils.clip_grad_norm(policy.parameters(), 40)
    optimizer.step()
    


