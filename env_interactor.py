import numpy as np
import torch
import time
import ipdb
import gc
import os
import sys
import shelve
import cProfile
import tracemalloc
import psutil
import logging
from collections import namedtuple, deque
from namedlist import namedlist

from cadet_env import *
from qbf_data import *
from settings import *
from utils import *
from rl_utils import *
from env_factory import *
from policy_factory import *

# DEF_COST = -1.000e-04

MPEnvStruct = namedlist('EnvStruct',
                    ['env', 'last_obs', 'episode_memory', 'env_id', 'fname', 'curr_step', 'active', 'prev_obs'])


class EnvInteractor:
  def __init__(self, settings, model, ed, name, init_model=None, reporter=None):
    super(EnvInteractor, self).__init__()
    self.name = 'interactor_{}'.format(name)
    self.settings = settings
    self.init_model = init_model
    self.ed = ed
    self.completed_episodes = []
    self.reporter = reporter
    self.max_step = self.settings['max_step']
    self.rnn_iters = self.settings['rnn_iters']
    self.restart_solver_every = self.settings['restart_solver_every']    
    self.check_allowed_actions = self.settings['check_allowed_actions']
    self.envstr = MPEnvStruct(EnvFactory().create_env(), 
        None, None, None, None, None, True, deque(maxlen=self.rnn_iters))        
    self.gmodel = model  
    self.optimizer = opt  
    self.reset_counter = 0
    self.env_steps = 0
    self.real_steps = 0
    self.def_step_cost = self.settings['def_step_cost']
    self.blacklisted_keys = []
    self.whitelisted_keys = []
    global_params = self.gmodel.state_dict()
    for k in global_params.keys():
      if any([x in k for x in self.settings['g2l_blacklist']]):
        self.blacklisted_keys.append(k)    
      if any([x in k for x in self.settings['l2g_whitelist']]):
        self.whitelisted_keys.append(k)    
    self.lmodel = PolicyFactory().create_policy()
    if self.init_model is None:
      self.lmodel.load_state_dict(self.gmodel.state_dict())
    else:
      self.logger.info('Loading model at runtime!')
      self.lmodel.load_state_dict(self.init_model)        
    if self.settings['log_threshold']:
      self.lmodel.shelf_file = shelve.open('thres_proc_{}.shelf'.format(self.name))      
    np.random.seed(int(time.time())+abs(hash(self.name)) % 1000)
    torch.manual_seed(int(time.time())+abs(hash(self.name)) % 1000)
    self.logger = logging.getLogger('EnvInteractor-{}'.format(self.name))
    self.logger.setLevel(eval(self.settings['loglevel']))
    fh = logging.FileHandler('{}.log'.format(self.name))
    fh.setLevel(logging.DEBUG)
    self.logger.addHandler(fh)


# This discards everything from the old env
  def reset_env(self, fname, **kwargs):
    self.reset_counter += 1
    if self.restart_solver_every > 0 and (self.settings['restart_in_test'] or (self.reset_counter % self.restart_solver_every == 0)):
      self.envstr.env.restart_env(timeout=0)
    self.logger.debug("({0}-{1})reset: {2}/{3}, memory: {4:.2f}MB".format(self.name, self.envstr.fname, self.reset_counter, self.envstr.curr_step, self.process.memory_info().rss / float(2 ** 20)))        
    if self.settings['memory_profiling']:
      print("({0}-{1})reset: {2}/{3}, memory: {4:.2f}MB".format(self.name, self.envstr.fname, self.reset_counter, self.envstr.curr_step, self.process.memory_info().rss / float(2 ** 20)))        
      objects = gc.get_objects()
      print('Number of objects is {}'.format(len(objects)))
      del objects

    env_obs = self.envstr.env.new_episode(fname=fname, **kwargs)
    self.envstr.last_obs = self.envstr.env.process_observation(None,env_obs)
    self.envstr.env_id = fname
    self.envstr.curr_step = 0
    self.envstr.fname = fname
    self.envstr.episode_memory = []     
    # Set up the previous observations to be None followed by the last_obs   
    self.envstr.prev_obs.clear()    
    for i in range(self.rnn_iters):
      self.envstr.prev_obs.append(None)
    return self.envstr.last_obs

# Assumes last observation in self.envstr.env.last_obs

  def step(self, **kwargs):
    envstr = self.envstr
    env = envstr.env
    # if not envstr.last_obs or envstr.curr_step > self.max_step:
    #   obs = self.reset_env(fname=self.provider.get_next())
    #   if obs is None:    # degenerate env
    #     self.logger.info('Got degenerate env: {}'.format(envstr.fname))
    #     self.completed_episodes.append(envstr.episode_memory)
    #     return True


    last_obs = collate_observations([envstr.last_obs])
    [action] = self.lmodel.select_action(last_obs, **kwargs)
    envstr.episode_memory.append(Transition(envstr.last_obs,action,None, None, envstr.env_id, envstr.prev_obs))
    allowed_actions = self.lmodel.get_allowed_actions(envstr.last_obs).squeeze() if self.check_allowed_actions else None

    if not self.check_allowed_actions or allowed_actions[action]:
      env_obs = envstr.env.step(self.lmodel.translate_action(action, envstr.last_obs))
      done = env_obs.done
    else:
      print('Chose invalid action, thats not supposed to happen.')
      assert(action_allowed(envstr.last_obs,action))

    self.real_steps += 1
    envstr.curr_step += 1

    if done:
      for j,r in enumerate(env.rewards):
        envstr.episode_memory[j].reward = r
      self.completed_episodes.append(envstr.episode_memory)
      envstr.last_obs = None      # This will mark the env to reset with a new formula
      if env.finished:
        if self.reporter:
          self.reporter.add_stat(envstr.env_id,len(envstr.episode_memory),sum(env.rewards), 0, self.real_steps)
        if self.ed:
          # Once for every episode going into completed_episodes, add it to stats
          self.ed.ed_add_stat(envstr.fname, (len(envstr.episode_memory), sum(env.rewards))) 
      else:        
        ipdb.set_trace()

    else:
      if envstr.curr_step > self.max_step:
        self.logger.info('Environment {} took too long, aborting it.'.format(envstr.fname))
        try:
          for record in envstr.episode_memory:
            record.reward = self.def_step_cost
          env.rewards = [self.def_step_cost]*len(envstr.episode_memory)            
        except:
          ipdb.set_trace()
        if self.ed:
          if 'testing' not in kwargs or not kwargs['testing']:
            self.ed.ed_add_stat(envstr.fname, (len(envstr.episode_memory), sum(env.rewards)))
        if self.settings['learn_from_aborted']:
          self.completed_episodes.append(envstr.episode_memory)
        envstr.last_obs = None
        return True        

      envstr.prev_obs.append(envstr.last_obs)
      envstr.last_obs = env.process_observation(envstr.last_obs,env_obs)

    return done

  def run_episode(self, fname, **kwargs):
    self.lmodel.eval()
    obs = self.reset_env(fname)
    if not obs:   # degenerate episode, return 0 actions taken
      return 0
    rc = False
    i = 0
    while not rc:
      rc = self.step()
      i += 1
    
    return i      

  def run_batch(self, *args, batch_size=0, **kwargs):
    if batch_size == 0:
      batch_size = self.settings['episodes_per_batch']

    total_length = 0
    for i in range(batch_size):
      total_length += self.run_episode(*args, **kwargs)
      if total_length == 0:    # If thats a degenerate episode, just return, the entire batch is degenerate.
        return 0, 0
    return total_length, batch_size

  def collect_batch(self, *args, **kwargs)
    total_length, bs = self.run_batch(*args, **kwargs)
    if total_length == 0:
      return None

    rc = []
    for i in range(bs):
      rc.append(self.completed_episodes.pop(0))

    return rc, total_length

