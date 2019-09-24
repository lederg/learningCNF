import numpy as np
import torch
import time
import ipdb
import gc
import os
import sys
import signal
import select
import shelve
import torch.multiprocessing as mp
import cProfile
import tracemalloc
import psutil
import logging
import Pyro4
from collections import namedtuple, deque
from namedlist import namedlist

from tick import *
from tick_utils import *
from cadet_env import *
from qbf_data import *
from settings import *
from utils import *
from rl_utils import *
from cadet_utils import *
from episode_data import *
from env_factory import *
from env_interactor import *
from policy_factory import *

# this takes an env_interactor and knows to take a list of files (provider) and test it.

TestResultStruct = namedlist('TestResultStruct',
                    ['time', 'steps', 'reward', 'completed'])



class EnvTester:
  def __init__(self, settings, name, hook_obj=None, ed=None, model=None, init_model=None, **kwargs):
    super(EnvTester, self).__init__()
    self.name = name
    self.settings = settings
    self.init_model = init_model   
    self.hook_obj = hook_obj
    self.logger = utils.get_logger(self.settings, 'EnvTester-{}'.format(self.name), 
                                    'logs/{}_{}.log'.format(log_name(self.settings), self.name))    
    if model is None:
      self.lmodel = PolicyFactory().create_policy(**kwargs)
    else:
      self.lmodel = model
    self.lmodel.logger = self.logger    # override logger object with process-specific one
    self.interactor = EnvInteractor(self.settings, self.lmodel, self.name, **kwargs)
    if self.init_model is None and self.hook_obj is not None:
      self.hook_obj.global_to_local(include_all=True)
    else:
      self.logger.info('Loading model at runtime!')
      statedict = self.lmodel.state_dict()
      numpy_into_statedict(statedict,self.init_model)
      self.lmodel.load_state_dict(statedict)
    if self.settings['log_threshold']:
      self.lmodel.shelf_file = shelve.open('thres_proc_{}.shelf'.format(self.name))      

  def test_envs(self, provider, model=None, ed=None, iters=10, **kwargs):
    max_seconds = int(kwargs['max_seconds'])      
    if model is not None:
      self.logger.info('Setting model at test time')
      self.lmodel = model
      self.interactor.lmodel = model
    print('Testing {} envs..\n'.format(provider.get_total()))
    all_episode_files = self.provider.items
    totals = 0.
    total_srate = 0.
    total_scored = 0
    rc = {}
    kwargs['testing']=True
    self.restart_all()
    available_envs = list(range(self.parallelism))    
    tasks = []
    for fname in all_episode_files:
      tasks.extend([fname]*iters)
    while tasks:      
      fname=tasks.pop(0)        
      self.logger.debug('Starting {}'.format(fname))
      episode_length, finished = self.interactor.run_episode(fname, **kwargs)
      print('Finished {}'.format(fname))
      if episode_length == 0:
        rc[fname].append((0,0,0, True))
        continue
      ep = self.completed_episodes.pop(0)
      total_reward = sum([x.reward for x in ep])                  
      total_time = self.interactor.envstr.end_time - self.interactor.envstr.start_time
      res = TestResultStruct(total_time,episode_length,total_reward,finished)
      if fname not in rc.keys():
        rc[fname] = []
      rc[fname].append(res)
      if len(rc[fname]) == iters:
        mean_steps = np.mean([x.steps for x in rc[fname]])
        mean_reward = np.mean([x.reward for x in rc[fname]])
        mean_time = np.mean([x.time for x in rc[fname]])
        self.logger.info('Finished {}, results are: {}, Averages (time,steps,reward) are {},{},'.format(fname,rc[fname],
          mean_time,mean_steps,mean_reward))      

    return rc
