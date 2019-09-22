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
from env_trainer import *
from worker_base import *
from dispatcher import *

class NodeWorkerBase(WorkerBase, IEnvTrainerHook):
  def __init__(self, settings, provider, name, **kwargs):
    super(NodeWorkerBase, self).__init__(settings, name, **kwargs)
    self.name = 'NodeWorkerBase%i' % name
    self.settings = settings
    self.kwargs = kwargs
    self.is_training = (not self.settings['do_not_learn'])
    self.training_steps = self.settings['training_steps']    
    self.trainer = EnvTrainerLoop(self.settings,provider,name, self, **kwargs)
    self.dispatcher = ObserverDispatcher()
    self.whitelisted_keys = []

  def global_to_local(self, **kwargs):
    global_params = self.node_sync.get_state_dict(**kwargs)
    self.trainer.lmodel.load_state_dict(global_params,strict=False)
    self.last_grad_steps = self.node_sync.g_grad_steps

  def update_from_worker(self, grads, state_dict)
    self.node_sync.update_grad_and_step(grads)
    # self.logger.info('Grad steps taken before step are {}'.format(self.node_sync.g_grad_steps-self.last_grad_steps))
    if self.whitelisted_keys:    
      local_params = {}
      z = state_dict
      for k in self.whitelisted_keys:
        local_params[k] = z[k]
      self.node_sync.set_state_dict(local_params)


  def init_proc(self, **kwargs):
    super(NodeWorkerBase, self).init_proc(**kwargs)
    self.node_sync = Pyro4.core.Proxy("PYRONAME:{}.node_sync".format(self.settings['pyro_name'])) 
    self.reporter = Pyro4.core.Proxy("PYRONAME:{}.reporter".format(self.settings['pyro_name']))
    self.trainer.init_proc(reporter=self.reporter, **kwargs)
    global_params = self.trainer.lmodel.state_dict()
    for k in global_params.keys():
      if any([x in k for x in self.settings['l2g_whitelist']]):
        self.whitelisted_keys.append(k)    
    
  def run_loop(self):
    clock = GlobalTick()
    SYNC_STATS_EVERY = 1
    # SYNC_STATS_EVERY = 5+np.random.randint(10)
    total_step = 0    
    local_env_steps = 0
    global_steps = 0
    while global_steps < self.training_steps:
      clock.tick()
      self.dispatcher.notify('new_batch')
      num_env_steps, num_episodes = self.trainer.train_step(training=self.is_training, **self.kwargs)
      total_step += 1
      local_env_steps += len(num_env_steps)            
      if total_step % SYNC_STATS_EVERY == 0:      
        self.node_sync.mod_g_grad_steps(SYNC_STATS_EVERY)
        self.node_sync.mod_g_episodes(max(lenvec)+1)
        self.node_sync.mod_g_steps(local_env_steps)
        local_env_steps = 0
        global_steps = self.node_sync.g_grad_steps


