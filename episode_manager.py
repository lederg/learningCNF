import numpy as np
import torch
import ipdb
from collections import namedtuple
from namedlist import namedlist


from cadet_env import *
from qbf_data import *
from settings import *
from utils import *
from rl_utils import *
from cadet_utils import *


EnvStruct = namedlist('EnvStruct',
                    ['env', 'last_obs', 'episode_memory', 'env_id', 'curr_step'])

MAX_STEP = 2000


def test_one_env(fname, iters=None, threshold=100000, **kwargs):
  s = 0.
  i = 0
  step_counts = []
  if iters is None:
    iters = settings['test_iters']
  for _ in range(iters):
    r, _, _ = handle_episode(fname=fname, **kwargs)
    if settings['restart_in_test']:
      env.restart_cadet(timeout=3)
    if not r:     # If r is None, the episodes never finished
      continue
    if len(r) > 1000:
      print('{} took {} steps!'.format(fname,len(r)))
      # break            
    s += len(r)
    i += 1
    step_counts.append(len(r))

  if i:
    if s/i < threshold:
      print('For {}, average steps: {}'.format(fname,s/i))
    return s/i, float(i)/iters, step_counts
  else:
    return None, 0, None

class EpisodeManager(object):
  def __init__(self, episodes_files, parallelism=20, reporter=None):
    self.settings = CnfSettings()
    self.parallelism = parallelism
    self.episodes_files = episodes_files
    self.envs = []
    self.completed_episodes = []
    self.real_steps = 0
    self.max_reroll = 0
    self.reporter = reporter
    self.bad_episodes = 0
    self.bad_episodes_not_added = 0
    self.INVALID_ACTION_REWARDS = -10

    for i in range(parallelism):
      self.envs.append(EnvStruct(CadetEnv(**self.settings.hyperparameters), None, None, None, None))

  def episode_lengths(self, num=0):
    rc = self.completed_episodes if num==0 else self.completed_episodes[:num]
    return sum([len(x) for x in rc])

  def pop_min(self, num=0):
    if num == 0:
      num = self.settings['min_timesteps_per_batch']
    total = 0
    rc = []
    while len(rc) < num:
      ep = discount_episode(self.completed_episodes.pop(0),self.settings['gamma'])
      rc.extend(ep)

    return rc

  def reset_all(self):
    for envstr in self.envs:
      self.reset_env(envstr)

# This discards everything from the old env
  def reset_env(self, envstr, **kwargs):
    last_obs, env_id = new_episode(envstr.env, self.episodes_files, **kwargs)
    envstr.last_obs = last_obs
    envstr.env_id = env_id
    envstr.curr_step = 0
    envstr.episode_memory = []
    return last_obs

# Step the entire pipeline one step, reseting any new envs. 

  def step_all(self, model):
    step_obs = []
    # if len(self.completed_episodes) > 0:
    #   ipdb.set_trace()
    for envstr in self.envs:
      if not envstr.last_obs or envstr.curr_step > MAX_STEP:
        self.reset_env(envstr)
      step_obs.append(envstr.last_obs)

    obs_batch = collate_observations(step_obs)
    allowed_actions = get_allowed_actions(obs_batch)
    actions, logits = self.select_action(obs_batch, model=model)
    
    # self.settings.LongTensor(np.array(actions)[:,0]).contiguous().view(-1,1)
    for i, envstr in enumerate(self.envs):
      env = envstr.env
      env_id = envstr.env_id
      envstr.episode_memory.append(Transition(step_obs[i],actions[i],None, None, envstr.env_id))
      self.real_steps += 1
      envstr.curr_step += 1
      if allowed_actions[i][actions[i][0]]:
        env_obs = EnvObservation(*env.step(actions[i]))        
        done = env_obs.done
      else:
        print('Chose an invalid action!')
        env.rewards = env.terminate()
        env.rewards = np.append(env.rewards,self.INVALID_ACTION_REWARDS)
        done = True       
      if done:
        try:
          for i,r in enumerate(env.rewards):
            envstr.episode_memory[i].reward = r
        except:
          ipdb.set_trace()
        self.completed_episodes.append(envstr.episode_memory)
        envstr.last_obs = None      # This will mark the env to reset with a new formula
        if env.finished:
          self.reporter.add_stat(env_id,len(envstr.episode_memory),sum(env.rewards), 0, self.real_steps)
        else:        
          print('Env {} did not finish!'.format(env_id))
          self.bad_episodes += 1
          try:
            print(self.reporter.stats_dict[env_id])
            steps = int(np.array([x[0] for x in self.reporter.stats_dict[env_id]]).mean())
            self.reporter.add_stat(env_id,steps,sum(env.rewards), 0, self.real_steps)
            print('Added it with existing steps average: {}'.format(steps))
          except:
            self.bad_episodes_not_added += 1
            print('Average does not exist yet, did not add.')
          print('Total Bad episodes so far: {}. Bad episodes that were not counted: {}'.format(self.bad_episodes,self.bad_episodes_not_added))

      else:
        envstr.last_obs = process_observation(env,envstr.last_obs,env_obs)

    return len(self.completed_episodes)



  def select_action(self, obs_batch, model=None, testing=False, random_test=False, activity_test=False, cadet_test=False, **kwargs):        
    bs = len(obs_batch.ground)
    activities = obs_batch.ground.data.numpy()[:,:,IDX_VAR_ACTIVITY]
    allowed_actions = get_allowed_actions(obs_batch)
    actions = []
    if random_test:
      for allowed in allowed_actions:
        choices = np.where(allowed.numpy())[0]
        actions.append(np.random.choice(choices))

      return actions, None
    elif activity_test:
      for i,act in enumerate(activities):
        if np.any(act):
          actions.append(np.argmax(act))
        else:
          choices = np.where(allowed_actions[i].numpy())[0]
          actions.append(np.random.choice(choices))
      return actions, None
    elif cadet_test:
      return ['?']*bs, None

    logits, values = model(obs_batch, **kwargs)
    
    if testing:
      action = logits.squeeze().max(0)[1].data   # argmax when testing        
      action = action[0]
      if settings['debug_actions']:
        print(obs.state)
        print(logits.squeeze().max(0))
        print('Got action {}'.format(action))
    else:
      probs = F.softmax(logits.contiguous().view(bs,-1)).view(bs,-1,2)
      all_dist = probs.data.cpu().numpy()
      for j, dist in enumerate(all_dist):
        i = 0
        flattened_dist = dist.reshape(-1)
        choices = range(len(flattened_dist))
        while i<1000:
          action = np.random.choice(choices, p=flattened_dist)          
          action = (int(action/2),int(action%2))
          if not self.settings['disallowed_aux'] or action_allowed(obs, action):
            break
          i = i+1
        if i > self.max_reroll:
          self.max_reroll = i
          print('Had to roll {} times to come up with a valid action!'.format(self.max_reroll))
        if i>600:
          # ipdb.set_trace()
          print("Couldn't choose an action within 600 re-samples. Max probability:")            
          # allowed_mass = (probs.view_as(logits).sum(2).data*get_allowed_actions(obs).float()).sum(1)
          # print('total allowed mass is {}'.format(allowed_mass.sum()))
          print(flattened_dist.max())

        actions.append(action)

    return actions, logits

  def test_envs(self, fnames, **kwargs):
    ds = QbfDataset(fnames=fnames)
    print('Testing {} envs..\n'.format(len(ds)))
    all_episode_files = ds.get_files_list()
    totals = 0.
    total_srate = 0.
    total_scored = 0
    rc = {}
    for fname in all_episode_files:
      print('Starting {}'.format(fname))
      # average, srate = test_one_env(fname, **kwargs)
      rc[fname] = test_one_env(fname, **kwargs)
      average = rc[fname][0]
      srate = rc[fname][1]
      total_srate += srate
      if average:
        total_scored += 1
        totals += average        
    if total_scored > 0:
      print("Total average: {}. Success rate: {} out of {}".format(totals/total_scored,total_scored,len(ds)))
      return totals/total_scored
    else:
      return 0.




