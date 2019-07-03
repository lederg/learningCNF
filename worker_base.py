import numpy as np
import torch
import time
import ipdb
import os
import sys
import signal
import select
import torch.multiprocessing as mp
import torch.optim as optim
import cProfile
from collections import namedtuple, deque
from namedlist import namedlist

from qbf_data import *
from settings import *
from utils import *
from rl_utils import *
from cadet_utils import *
from episode_data import *
from env_factory import *
from env_interactor import *  


class WorkerBase(mp.Process):
  def __init__(self, settings, model, provider, ed, workers_queue, workers_sem, global_grad_steps, name, reporter=None):
    super(WorkerBase, self).__init__()
    self.name = 'a3c_worker%i' % name
    self.g_grad_steps = global_grad_steps
    self.settings = settings
    self.ed = ed
    self.main_queue = workers_queue
    self.main_sem = workers_sem
    self.completed_episodes = []
    self.reporter = reporter
    self.max_step = self.settings['max_step']
    self.rnn_iters = self.settings['rnn_iters']
    self.training_steps = self.settings['training_steps']
    self.restart_solver_every = self.settings['restart_solver_every']    
    self.check_allowed_actions = self.settings['check_allowed_actions']
    self.envstr = MPEnvStruct(EnvFactory().create_env(), 
        None, None, None, None, None, True, deque(maxlen=self.rnn_iters))
    self.lmodel = model      
    self.reset_counter = 0
    self.env_steps = 0
    self.real_steps = 0
    self.provider = provider
    self.last_grad_step = 0

# This discards everything from the old env
  def reset_env(self, fname, **kwargs):
    self.reset_counter += 1
    if self.restart_solver_every > 0 and (self.settings['restart_in_test'] or (self.reset_counter % self.restart_solver_every == 0)):
      self.envstr.env.restart_env(timeout=0)
    env_obs, env_id = self.envstr.env.new_episode(fname=fname, **kwargs)
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

  def step(self, **kwargs):
    envstr = self.envstr
    env = envstr.env
    if not envstr.last_obs:
      rc = self.reset_env(fname=self.provider.get_next())
      if rc is None:    # degenerate env
        self.completed_episodes.append(envstr.episode_memory)
        return True


    last_obs = collate_observations([envstr.last_obs])
    [action] = self.lmodel.select_action(last_obs, **kwargs)
    # This will turn into a bug the second prev_obs actually contains anything. TBD.
    envstr.episode_memory.append(Transition(densify_obs(envstr.last_obs),action,None, None, envstr.env_id, envstr.prev_obs))
    allowed_actions = self.lmodel.get_allowed_actions(envstr.last_obs).squeeze() if self.check_allowed_actions else None

    if not self.check_allowed_actions or allowed_actions[action]:
      env_obs = envstr.env.step(self.lmodel.translate_action(action,last_obs))
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
        print('Environment {} took too long, aborting it.'.format(envstr.fname))
        try:
          for record in envstr.episode_memory:
            record.reward = DEF_COST
          env.rewards = [DEF_COST]*len(envstr.episode_memory)            
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

  def init_proc(self):
    set_proc_name(str.encode(self.name))
    np.random.seed(int(time.time())+abs(hash(self.name)) % 1000)
    torch.manual_seed(int(time.time())+abs(hash(self.name)) % 1000)
    self.settings.hyperparameters['cuda']=False         # No CUDA in the worker threads


  def run(self):
    self.init_proc()
    if self.settings['profiling']:
      cProfile.runctx('self.run_loop()', globals(), locals(), 'prof_{}.prof'.format(self.name))
    else:
      self.run_loop()


  def run_loop(self):
    pass