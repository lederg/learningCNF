import os.path
import torch
# from torch.distributions import Categorical
import ipdb
import random
import time

from settings import *
from cadet_env import *
from rl_model import *
from qbf_data import *
from utils import *
from episode_reporter import EpisodeReporter
import torch.nn.utils as tutils

CADET_BINARY = './cadet'

all_episode_files = ['data/mvs.qdimacs']

settings = CnfSettings()
policy = Policy()
if settings['cuda']:
  policy = policy.cuda()
reporter = EpisodeReporter()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
env = CadetEnv(CADET_BINARY, **settings.hyperparameters)

actions_history = []

def select_action(state, ground_embs, **kwargs):
  state = Variable(torch.from_numpy(state).float().unsqueeze(0))
  ground_embs = Variable(torch.from_numpy(ground_embs).unsqueeze(0).float())
  if settings['cuda']:
    state, ground_embs = state.cuda(), ground_embs.cuda()
  probs = policy(state, ground_embs, batch_size=1, **kwargs)
  # ipdb.set_trace()
  try:
    if settings['cuda']:
      dist = probs.data.cpu().numpy()[0]
    else:
      dist = probs.data.numpy()[0]
    choices = range(len(dist))
    action = np.random.choice(choices, p=dist)
    if settings['batch_backwards']:
      return action, None, None
    aug_action = np.where(np.where(dist>0)[0]==action)    # without the 0 probabilities
  except:
    print('Problem in np.random')
    ipdb.set_trace()
  a = probs[probs>0]
  logprob = a.log()[aug_action[0][0]]
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


# indices/vals are tensors, clause is a tuple of nparrays, pos/neg

def add_clause(sp_ind_pos, sp_ind_neg, sp_val_pos, sp_val_neg, clause, last_clause):
  rc_pos, rc_neg, rc_val_pos, rc_val_neg = sp_ind_pos, sp_ind_neg, sp_val_pos, sp_val_neg
  if clause[0].size:
    a = np.full((clause[0].size,2),last_clause)
    a[:,1] = clause[0]
    rc_pos = torch.cat([rc_pos,torch.from_numpy(a)])
    rc_val_pos = torch.cat([rc_val_pos,torch.ones(clause[0].size)])

  if clause[1].size:
    a = np.full((clause[1].size,2),last_clause)
    a[:,1] = clause[1]
    rc_neg = torch.cat([rc_neg,torch.from_numpy(a)])
    rc_val_neg = torch.cat([rc_val_neg,torch.ones(clause[1].size)])

  return rc_pos, rc_neg, rc_val_pos, rc_val_neg

def handle_episode(fname):

  # Set up ground_embeddings and adjacency matrices
  state, vars_add, vars_remove, activities, _, _ , _ = env.reset(fname)
  assert(len(state)==settings['state_dim'])
  qbf = env.qbf
  curr_num_clauses = qbf.num_clauses
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
  cmat_pos = Variable(torch.sparse.FloatTensor(sp_ind_pos.t(),sp_val_pos,torch.Size([curr_num_clauses,qbf.num_vars])),requires_grad=False)
  cmat_neg = Variable(torch.sparse.FloatTensor(sp_ind_neg.t(),sp_val_neg,torch.Size([curr_num_clauses,qbf.num_vars])),requires_grad=False)

  if settings['cuda']:
    cmat_pos, cmat_neg = cmat_pos.cuda(), cmat_neg.cuda()
  logprobs = []
  transitions = []

  for t in range(5000):  # Don't infinite loop while learning
    # Could cache ground_embs here, as a Variable
    action , logprob, ent = select_action(state, ground_embs, cmat_pos=cmat_pos, cmat_neg=cmat_neg)
    if settings['batch_backwards']:
      transitions.append((state, ground_embs, cmat_pos, cmat_neg, action))
    else:
      logprobs.append((logprob,ent))
    # print('Chose action %d' % action)
    if ground_embs[action][IDX_VAR_DETERMINIZED]:
      print('Chose an invalid action!')
      ipdb.set_trace()
    try:
      state, vars_add, vars_remove, activities, decision, clause, done = env.step(action)
    except Exception as e:
      print(e)
      ipdb.set_trace()
    if done:
      # print('Finished. Rewards are:')
      # print(env.rewards)
      # print(discount(env.rewards, settings['gamma']))
      return env.rewards, transitions if settings['batch_backwards'] else logprobs
    if clause:
      sp_ind_pos, sp_ind_neg, sp_val_pos, sp_val_neg = add_clause(sp_ind_pos, sp_ind_neg, sp_val_pos, sp_val_neg, 
                                                        clause, curr_num_clauses)
      curr_num_clauses += 1
      cmat_pos = Variable(torch.sparse.FloatTensor(sp_ind_pos.t(),sp_val_pos,torch.Size([curr_num_clauses,qbf.num_vars])))
      cmat_neg = Variable(torch.sparse.FloatTensor(sp_ind_neg.t(),sp_val_neg,torch.Size([curr_num_clauses,qbf.num_vars])))
    if decision:
      ground_embs[decision[0]][IDX_VAR_POLARITY_POS+1-decision[1]] = True
    if len(vars_add):
      ground_embs[:,IDX_VAR_DETERMINIZED][vars_add] = True
    if len(vars_remove):
      ground_embs[:,IDX_VAR_DETERMINIZED][vars_remove] = False
      ground_embs[:,IDX_VAR_POLARITY_POS:IDX_VAR_POLARITY_NEG][vars_remove] = False
    ground_embs[:,IDX_VAR_ACTIVITY] = activities

  ipdb.set_trace()

def ts_bonus(s):
  b = 5.
  return b/float(s)

def cadet_main(settings):
  ds = QbfDataset(dirname='data/dataset1/')
  all_episode_files = ds.get_files_list()
  total_envs = len(all_episode_files)
  total_steps = 0
  for iter in range(400):
    rewards = []
    time_steps_this_batch = 0
    transition_data = []
    total_transitions = []
    time_steps_this_batch = 0
    begin_time = time.time()
    while time_steps_this_batch < settings['min_timesteps_per_batch']:
      fname = all_episode_files[random.randint(0,total_envs-1)]
      r, meta_data = handle_episode(fname)      
      s = len(r)
      # r[-1] += ts_bonus(s)
      time_steps_this_batch += s
      total_steps += s
      reporter.add_stat(int(os.path.split(fname)[1].split('_')[0]),s,sum(r))
      # ipdb.set_trace()
      rewards.extend(discount(r, settings['gamma']))
      if np.isnan(rewards).any():
        ipdb.set_trace()
      transition_data.extend(meta_data)
      # print('Finished episode for file %s in %d steps' % (fname, s))


    end_time = time.time()
    print('Finished batch with total of %d steps in %f seconds' % (time_steps_this_batch, end_time-begin_time))
    if not (iter % 10):
      reporter.report_stats()

    begin_time = time.time()
    if settings['batch_backwards']:
      states, ground_embs, cmat_pos, cmat_neg, actions = zip(*transition_data)
      ipdb.set_trace()
    else:
      logprobs, entropies = zip(*transition_data)
    returns = torch.Tensor(rewards)
    batch_entropy = torch.cat(entropies)
    logprobs = torch.cat(logprobs)
    if settings['cuda']:
      returns = returns.cuda()
    returns = (returns - returns.mean()) / (returns.std() + np.finfo(np.float32).eps)
    loss = (-Variable(returns)*logprobs - 0.0001*batch_entropy).mean()
    optimizer.zero_grad()
    loss.backward()
    if any([(x.grad!=x.grad).data.any() for x in policy.parameters() if x.grad is not None]): # nan in grads
      ipdb.set_trace()
    # tutils.clip_grad_norm(policy.parameters(), 40)
    optimizer.step()
    end_time = time.time()
    print('Backward computation done in %f seconds' % (end_time-begin_time))
    


