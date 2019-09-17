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


# DEF_COST = -1.000e-04
BREAK_CRIT_LOGICAL = 1
BREAK_CRIT_TECHNICAL = 2

MPEnvStruct = namedlist('EnvStruct',
                    ['env', 'last_obs', 'episode_memory', 'env_id', 'fname', 'curr_step', 'active', 'prev_obs', 'start_time'])


class IEnvTrainerHook:
  def post_train(self):
    pass
  def global_to_local(self,**kwargs):
    pass

class EnvTrainer:
  def __init__(self, settings, provider, name, hook_obj, ed=None, init_model=None):
    super(EnvTrainer, self).__init__()
    self.index = name
    self.name = 'a3c_worker%i' % name
    self.settings = settings
    self.init_model = init_model
    self.ed = ed
    self.hook_obj = hook_obj
    self.memory_cap = self.settings['memory_cap']
    self.minimum_episodes = self.settings['minimum_episodes']
    self.gamma = self.settings['gamma']
    self.provider = provider    
    self.last_grad_steps = 0

  def init_proc(self, **kwargs):
    set_proc_name(str.encode(self.name))
    np.random.seed(int(time.time())+abs(hash(self.name)) % 1000)
    torch.manual_seed(int(time.time())+abs(hash(self.name)) % 1000)    
    self.reporter = Pyro4.core.Proxy("PYRONAME:{}.reporter".format(self.settings['pyro_name']))
    self.node_sync = Pyro4.core.Proxy("PYRONAME:{}.node_sync".format(self.settings['pyro_name']))    
    self.logger = utils.get_logger(self.settings, 'WorkerEnv-{}'.format(self.name), 
                                    'logs/{}_{}.log'.format(log_name(self.settings), self.name))
    self.settings.hyperparameters['cuda']=False         # No CUDA in the worker threads
    self.lmodel = PolicyFactory().create_policy(**kwargs)
    self.lmodel.logger = self.logger    # override logger object with process-specific one
    self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.lmodel.parameters()))
    self.interactor = EnvInteractor(self.settings, self.lmodel, name, reporter=self.reporter, **kwargs):    
    self.blacklisted_keys = []
    self.whitelisted_keys = []
    global_params = self.lmodel.state_dict()
    for k in global_params.keys():
      if any([x in k for x in self.settings['g2l_blacklist']]):
        self.blacklisted_keys.append(k)    
      if any([x in k for x in self.settings['l2g_whitelist']]):
        self.whitelisted_keys.append(k)    
    if self.init_model is None:
      self.hook_obj.global_to_local(include_all=True)
    else:
      self.logger.info('Loading model at runtime!')
      statedict = self.lmodel.state_dict()
      numpy_into_statedict(statedict,self.init_model)
      self.lmodel.load_state_dict(statedict)
    self.process = psutil.Process(os.getpid())
    if self.settings['log_threshold']:
      self.lmodel.shelf_file = shelve.open('thres_proc_{}.shelf'.format(self.name))      

  def train(self,transition_data, **kwargs):
    # print('train: batch size is {} and reward is {}'.format(len(transition_data),sum([t.reward for t in transition_data])))
    if sum([t.reward for t in transition_data]) == 0:
    # if len(transition_data) == self.settings['episodes_per_batch']*(self.settings['max_step']+1):
      self.logger.info('A lost batch, no use training')
      return
    if self.settings['do_not_learn']:
      return
    need_sem = False
    if len(transition_data) >= self.settings['episodes_per_batch']*(self.settings['max_step']+1)*self.settings['batch_size_threshold']:
      self.logger.info('Large batch encountered. Acquiring batch semaphore')
      need_sem = True
    self.lmodel.train()
    mt = time.time()
    loss, logits = self.lmodel.compute_loss(transition_data, **kwargs)
    mt1 = time.time()
    self.logger.info('Loss computation took {} seconds on {} with length {}'.format(mt1-mt,self.provider.get_next(),len(transition_data)))
    self.optimizer.zero_grad()      # Local model grads are being zeros here!
    loss.backward()
    mt2 = time.time()
    self.logger.info('Backward took {} seconds'.format(mt2-mt1))
    grads = [x.grad for x in self.lmodel.parameters()]
    self.node_sync.update_grad_and_step(grads)
    # self.logger.info('Grad steps taken before step are {}'.format(self.node_sync.g_grad_steps-self.last_grad_steps))
    z = self.lmodel.state_dict()
    # We may want to sync that

    local_params = {}
    for k in self.whitelisted_keys:
      local_params[k] = z[k]
    self.node_sync.set_state_dict(local_params)

  def run(self):
    self.init_proc()
    if self.settings['memory_profiling']:
      tracemalloc.start(25)
    if self.settings['profiling']:
      cProfile.runctx('self.run_loop()', globals(), locals(), 'prof_{}.prof'.format(self.name))
    else:
      self.run_loop()

# Returns the number of steps taken (total length of batch, in steps)

  def train_step(self, **kwargs):
    self.lmodel.eval()
    self.hook_obj.global_to_local()      
    begin_time = time.time()
    curr_formula = 
    total_episodes = 0
    eps, ns = self.interactor.collect_batch(curr_formula, **kwargs)
    total_inference_time = time.time() - begin_time
    num_episodes = len(eps)
    transition_data = flatten([discount_episode(x,self.gamma,self.settings) for x in eps])
    lenvec = flatten([[i]*len(eps[i]) for i in range(num_episodes)])
    if ns == 0:
      self.logger.info('Degenerate batch, ignoring')
      if self.settings['autodelete_degenerate']:
        self.provider.delete_item(curr_formula)
        self.logger.debug('After deleting degenerate formula, total number of formulas left is {}'.format(self.provider.get_total()))
      return 0, 0
    elif num_episodes < self.minimum_episodes:
      self.logger.info('too few episodes ({}), dropping batch'.format(num_episodes))
      return 0, 0
    self.logger.info('Forward pass in {} ({}) got batch with length {} ({}) in {} seconds. Ratio: {}'.format(self.name,transition_data[0].formula,len(transition_data),num_episodes,total_inference_time,len(transition_data)/total_inference_time))
    # After the batch is finished, advance the iterator
    begin_time = time.time()
    self.train(transition_data,lenvec=lenvec)
    total_train_time = time.time() - begin_time
    self.logger.info('Backward pass in {} done in {} seconds!'.format(self.name,total_train_time))
    return len(transition_data), num_episodes
