import numpy as np
import torch
import time
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
                    ['env', 'last_obs', 'episode_memory', 'env_id', 'fname', 'curr_step', 'active'])

MAX_STEP = 4000


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
    self.debug = False
    self.parallelism = parallelism
    self.episodes_files = episodes_files
    self.packed = self.settings['packed']
    self.envs = []
    self.completed_episodes = []
    self.real_steps = 0
    self.max_reroll = 0
    self.reporter = reporter
    self.bad_episodes = 0
    self.bad_episodes_not_added = 0
    self.INVALID_ACTION_REWARDS = -10

    for i in range(parallelism):
      self.envs.append(EnvStruct(CadetEnv(**self.settings.hyperparameters), None, None, None, None, None, True))

  def check_batch_finished(self):
    if self.settings['normalize_episodes']:
      return not (len(self.completed_episodes) < self.settings['episodes_per_batch'])
    else:
      return not (self.episode_lengths() < self.settings['min_timesteps_per_batch'])
  def episode_lengths(self, num=0):
    rc = self.completed_episodes if num==0 else self.completed_episodes[:num]
    return sum([len(x) for x in rc])

  def pop_min(self, num=0):
    if num == 0:
      num = self.settings['min_timesteps_per_batch']
    rc = []
    while len(rc) < num:
      ep = discount_episode(self.completed_episodes.pop(0),self.settings['gamma'])
      rc.extend(ep)

    return rc

  def pop_min_normalized(self, num=0):
    if num == 0:
      num = self.settings['episodes_per_batch']
    rc = []
    for _ in range(num):
      ep = discount_episode(self.completed_episodes.pop(0),self.settings['gamma'])
      rc.extend(ep)

    return rc

  def reset_all(self):
    for envstr in self.envs:
      self.reset_env(envstr)


  def restart_all(self):
    for envstr in self.envs:      
      envstr.env.stop_cadet(timeout=0)
    time.sleep(2)
    for envstr in self.envs:
      envstr.env.start_cadet()
      envstr.last_obs = None          # This lets step_all know its an "empty" env that got to be reset.

# This discards everything from the old env
  def reset_env(self, envstr, **kwargs):
    last_obs, env_id, fname = new_episode(envstr.env, self.episodes_files, **kwargs)
    envstr.last_obs = last_obs
    envstr.env_id = env_id
    envstr.curr_step = 0
    envstr.fname = fname
    envstr.episode_memory = []    
    return last_obs

# Step the entire pipeline one step, reseting any new envs. 

  def step_all(self, model):
    step_obs = []
    rc = []     # the env structure indices that finished and got to be reset (or will reset automatically next step)
    active_envs = [i for i in range(self.parallelism) if self.envs[i].active]
    for i in active_envs:
      envstr = self.envs[i]
      if not envstr.last_obs or envstr.curr_step > MAX_STEP:
        self.reset_env(envstr)
      step_obs.append(envstr.last_obs)

    if self.packed:
      obs_batch = packed_collate_observations(step_obs)
      vp_ind = obs_batch.pack_indices[1]
    else:
      obs_batch = collate_observations(step_obs)
    allowed_actions = get_allowed_actions(obs_batch,packed=self.packed)
    actions, logits = self.packed_select_action(obs_batch, model=model) if self.packed else self.select_action(obs_batch, model=model)
    
    for i, envnum in enumerate(active_envs):
      envstr = self.envs[envnum]
      env = envstr.env
      env_id = envstr.env_id
      envstr.episode_memory.append(Transition(step_obs[i],actions[i],None, None, envstr.env_id))
      self.real_steps += 1
      envstr.curr_step += 1
      if allowed_actions[vp_ind[i]+actions[i][0]]:
        env_obs = EnvObservation(*env.step(actions[i]))        
        done = env_obs.done
      else:
        print('Chose an invalid action! In the packed version. That was not supposed to happen.')
        env.rewards = env.terminate()
        env.rewards = np.append(env.rewards,self.INVALID_ACTION_REWARDS)
        done = True       
      if done:
        try:
          for j,r in enumerate(env.rewards):
            envstr.episode_memory[j].reward = r
        except:
          ipdb.set_trace()
        self.completed_episodes.append(envstr.episode_memory)
        envstr.last_obs = None      # This will mark the env to reset with a new formula
        rc.append((envnum,True))
        if env.finished:
          self.reporter.add_stat(env_id,len(envstr.episode_memory),sum(env.rewards), 0, self.real_steps)
        else:        
          ipdb.set_trace()
      else:
        if envstr.curr_step > MAX_STEP:
          rc.append((envnum,False))
        envstr.last_obs = process_observation(env,envstr.last_obs,env_obs)

    return rc



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
          # ipdb.set_trace()
          if not self.settings['disallowed_aux'] or allowed_actions[j][action[0]]:
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
        if (2*action[0]+action[1])>len(flattened_dist):
          ipdb.set_trace()

    return actions, logits

  def packed_select_action(self, obs_batch, model=None, testing=False, random_test=False, activity_test=False, cadet_test=False, **kwargs):        
    bs = len(obs_batch.ground)
    activities = obs_batch.ground.data.numpy()[:,IDX_VAR_ACTIVITY]
    allowed_actions = get_allowed_actions(obs_batch, packed=True)
    actions = []
    pack_indices = obs_batch.pack_indices

    if random_test:
      i=0
      while i < len(pack_indices):
        choices = np.where(allowed[pack_indices[i]:pack_indices[i+1]].numpy())[0]
        
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

    logits, values = model(obs_batch, packed=self.packed, **kwargs)
    vp_ind = obs_batch.pack_indices[1]
    for i in range(len(vp_ind)-1):
      ith_allowed = allowed_actions[vp_ind[i]:vp_ind[i+1]]
      ith_logits = logits[vp_ind[i]:vp_ind[i+1]]
      allowed_idx = torch.from_numpy(np.where(ith_allowed.numpy())[0])
      if self.settings['cuda']:
        allowed_idx = allowed_idx.cuda()
      l = ith_logits[allowed_idx]
      probs = F.softmax(l.contiguous().view(1,-1))
      dist = probs.data.cpu().numpy()[0]
      choices = range(len(dist))
      aux_action = np.random.choice(choices, p=dist)
      aux_action = (int(aux_action/2),int(aux_action%2))
      action = (allowed_idx[aux_action[0]], aux_action[1])
      actions.append(action)
      
    return actions, logits

  def test_envs(self, fnames, model, ed=None, iters=10, **kwargs):
    ds = QbfDataset(fnames=fnames)
    print('Testing {} envs..\n'.format(len(ds)))
    all_episode_files = ds.get_files_list()
    totals = 0.
    total_srate = 0.
    total_scored = 0
    rc = {}
    self.restart_all()
    available_envs = list(range(self.parallelism))    
    tasks = []
    for fname in all_episode_files:
      rc[fname] = []
      tasks.extend([fname]*iters)
    while tasks or len(available_envs) < self.parallelism:
      while available_envs and tasks:
        i = available_envs.pop(0)
        fname=tasks.pop(0)
        if self.debug:
          print('Starting {} on Env #{}'.format(fname,i))
        envstr = self.envs[i]
        self.reset_env(envstr,fname=fname)  
      if not tasks:       # We finished the tasks, mark all current available solvers as inactive
        if self.debug:
          print('Tasks are empty. Available envs: {}'.format(available_envs))
        for i in available_envs:
          if self.debug:
              print('Marking Env #{} as inactive'.format(i))
          self.envs[i].active = False      
      finished_envs = self.step_all(model)
      if finished_envs:
        # print('Finished Envs: {}'.format(finished_envs))
        # ipdb.set_trace()
        for i, finished in finished_envs:
          fname = self.envs[i].fname
          if finished:
            if self.debug:
              print('Finished {} on Env #{}'.format(fname,i))
            res = len(self.completed_episodes.pop(0))            
          else:
            print('Env {} took too long!'.format(fname,i))
            ipdb.set_trace()
            res = len(self.envs[i].episode_memory)
          if ed is not None:
            ed.add_stat(fname,res)
          rc[fname].append(res)
          if len(rc[fname]) == iters:
            print('Finished {}, results are: {}, Average/Min are {}/{}'.format(fname,rc[fname],
              np.mean(rc[fname]),min(rc[fname])))
        available_envs.extend([x[0] for x in finished_envs])

    return rc


