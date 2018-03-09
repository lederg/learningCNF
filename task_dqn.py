import os.path
import copy
import torch
import math
# from torch.distributions import Categorical
import ipdb
import random
import time
import itertools

from settings import *
from cadet_env import *
from rl_model import *
from qbf_data import *
from qbf_model import QbfClassifier
from utils import *
from rl_utils import *
from episode_reporter import *
import torch.nn.utils as tutils

CADET_BINARY = './cadet'
SAVE_EVERY = 10
INVALID_ACTION_REWARDS = -100


all_episode_files = ['data/mvs.qdimacs']

settings = CnfSettings()

reporter = DqnEpisodeReporter("{}/{}".format(settings['rl_log_dir'], log_name(settings)))
env = CadetEnv(CADET_BINARY, **settings.hyperparameters)
memory = ReplayMemory(1000000)
exploration = LinearSchedule(1000000, 0.05)
policy = None
target_model = None
optimizer = None
total_steps = 0
time_last_episode = 0
episode_reward = 0
inference_time = []
actions_history = []


def eps_threshold(t):
  return settings['EPS_END'] + (settings['EPS_START'] - settings['EPS_END']) * \
        math.exp(-1. * t / settings['EPS_DECAY'])

def select_action(obs, model=None, testing=False, **kwargs):
  
  begin_time = time.time()
  logits = model(obs, **kwargs)
  inference_time.append(time.time() - begin_time)  
  
  try:
    if testing or random.random() > exploration.value(total_steps):
      action = logits.squeeze().max(0)[1].data   # argmax when testing    
      return action[0]
    else:
      action = np.random.choice(np.where(logits.squeeze().data.numpy()>-10)[0])    
      return int(action)
  
  except Exception as e:
    print(e)
    ipdb.set_trace()

  # We return a ground embedding of (self.num_vars,7), where embedding is 0 - universal, 1 - existential, 2 - pad, 
  # 3 - determinized, 4 - activity, [5,6] - pos/neg determinization
  # 3 is obviously empty here

def get_base_ground(qbf):
  rc = np.zeros([qbf.num_vars,GROUND_DIM]).astype(float)
  for j, val in enumerate(qbf.var_types):
    rc[j][val] = True
  return rc



def new_episode(**kwargs):
  global time_last_episode

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
  if settings['cuda']:
    cmat_pos, cmat_neg = cmat_pos.cuda(), cmat_neg.cuda()
    state, ground_embs = state.cuda(), ground_embs.cuda()
  rc = State(state,cmat_pos, cmat_neg, ground_embs)
  return rc, env_id
  
def dqn_main():
  global policy, optimizer, all_episode_files, total_steps, episode_reward, time_last_episode
  global target_model
  policy = create_policy()
  target_model = create_policy(is_clone=True)
  copy_model_weights(policy,target_model)
  optimizer = optim.Adam(policy.parameters(), lr=1e-2)
  reporter.log_env(settings['rl_log_envs'])
  ds = QbfDataset(fnames=settings['rl_train_data'])
  all_episode_files = ds.get_files_list()
  num_param_updates = 0
  done = True

  for t in itertools.count():            
    if done:
      last_obs, env_id = new_episode()
      cmat_pos, cmat_neg = last_obs.cmat_pos, last_obs.cmat_neg
      if settings['rl_log_all']:
        reporter.log_env(env_id)      


    action = select_action(last_obs,policy)
    if bool(last_obs.ground.long().data[0][action][IDX_VAR_DETERMINIZED]):
      print('Chose an invalid action!')
      reward = INVALID_ACTION_REWARDS
      done = True    
    else:
      try:
        state, vars_add, vars_remove, activities, decision, clause, reward, done = env.step(action)
      except Exception as e:
        print(e)
        ipdb.set_trace()

    total_steps += 1
    episode_reward += reward
    # prepare the next observation
    if not done:
      if clause:
        cmat_pos, cmat_neg = get_input_from_qbf(env.qbf, settings)
      ground_embs = last_obs.ground.data.numpy().squeeze()
      if decision:
        ground_embs[decision[0]][IDX_VAR_POLARITY_POS+1-decision[1]] = True
      if len(vars_add):
        ground_embs[:,IDX_VAR_DETERMINIZED][vars_add] = True
      if len(vars_remove):
        ground_embs[:,IDX_VAR_DETERMINIZED][vars_remove] = False
        ground_embs[:,IDX_VAR_POLARITY_POS:IDX_VAR_POLARITY_NEG][vars_remove] = False
      ground_embs[:,IDX_VAR_ACTIVITY] = activities
      state = Variable(torch.from_numpy(state).float().unsqueeze(0))
      ground_embs = Variable(torch.from_numpy(ground_embs).float().unsqueeze(0))
      obs = State(state,cmat_pos,cmat_neg,ground_embs)
    else:
      s = total_steps - time_last_episode
      reporter.add_stat(env_id,s,episode_reward, total_steps)
      episode_reward = 0
      time_last_episode = total_steps
      obs = None
    memory.push(last_obs,action,obs,reward)
    last_obs = obs

    if t > settings['learning_starts'] and not (t % settings['learning_freq']):
      batch = memory.sample(settings['batch_size'])
      states, actions, next_states, rewards = zip(*batch)
      collated_batch = collate_transitions(batch,settings=settings)
      non_final_mask = settings.ByteTensor(tuple(map(lambda s: s is not None,
                                          next_states)))

      state_action_values = policy(collated_batch.state).gather(1,Variable(collated_batch.action).view(-1,1))
      next_state_values = Variable(settings.zeros(settings['batch_size']))
      next_state_values[non_final_mask] = target_model(collated_batch.next_state).max(1)[0]
      expected_state_action_values = (next_state_values * settings['gamma']) + Variable(collated_batch.reward)
      loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

      optimizer.zero_grad()
      loss.backward()
      clip_to = settings['grad_norm_clipping']
      for param in policy.parameters():
          if param.grad is not None:
            param.grad.data.clamp_(-clip_to, clip_to)
      optimizer.step()

      num_param_updates += 1
      if num_param_updates % settings['target_update_freq'] == 0:
        print('Updating target network (step {})'.format(total_steps))
        copy_model_weights(policy,target_model)



    if not (t % 1000) and t > 0:
      reporter.report_stats()
      print('Epsilon is {}'.format(exploration.value(total_steps)))
    # ipdb.set_trace()

    # print('Finished episode for file %s in %d steps' % (fname, s))

    
    # print('Finished batch with total of %d steps in %f seconds' % (time_steps_this_batch, sum(inference_time)))    
    # if not (i % 10):
    #   reporter.report_stats()
    #   print('Testing all episodes:')
    #   for fname in all_episode_files:
    #     r, _ = handle_episode(fname, model=policy, testing=True)
    #     print('Env %s completed test in %d steps with total reward %f' % (fname, len(r), sum(r)))

    # inference_time.clear()
    # begin_time = time.time()
    # if any([(x.grad!=x.grad).data.any() for x in policy.parameters() if x.grad is not None]): # nan in grads
    #   ipdb.set_trace()
    # # tutils.clip_grad_norm(policy.parameters(), 10)

    # end_time = time.time()
    # print('Backward computation done in %f seconds' % (end_time-begin_time))
    # if i % SAVE_EVERY == 0:
    #   torch.save(policy.state_dict(),'%s/%s_iter%d.model' % (settings['model_dir'],utils.log_name(settings), i))
    


