import numpy as np
import torch
import time
import ipdb
import os
import sys
import signal
import select
import shelve
import torch.multiprocessing as mp
import torch.optim as optim
import cProfile
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

DEF_COST = -1.000e-04

MPEnvStruct = namedlist('EnvStruct',
                    ['env', 'last_obs', 'episode_memory', 'env_id', 'fname', 'curr_step', 'active', 'prev_obs'])


class WorkerEnv(mp.Process):
  def __init__(self, settings, model, opt, provider, ed, global_steps, global_grad_steps, global_episodes, name, reporter=None):
    super(WorkerEnv, self).__init__()
    self.name = 'a3c_worker%i' % name
    self.g_steps = global_steps
    self.g_grad_steps = global_grad_steps
    self.g_episodes = global_episodes
    self.settings = settings
    # self.ds = ds
    self.ed = ed
    self.completed_episodes = []
    self.reporter = reporter
    self.max_step = self.settings['max_step']
    self.rnn_iters = self.settings['rnn_iters']
    self.training_steps = self.settings['training_steps']
    self.restart_solver_every = self.settings['restart_solver_every']    
    self.check_allowed_actions = self.settings['check_allowed_actions']
    self.envstr = MPEnvStruct(EnvFactory().create_env(), 
        None, None, None, None, None, True, deque(maxlen=self.rnn_iters))
    # self.gnet, self.opt = gnet, opt
    # self.lnet = Net(N_S, N_A)           # local network
    self.gmodel = model  
    self.optimizer = opt  
    self.reset_counter = 0
    self.env_steps = 0
    self.real_steps = 0
    self.provider = provider
    self.blacklisted_keys = []
    self.whitelisted_keys = []
    global_params = self.gmodel.state_dict()
    for k in global_params.keys():
      if any([x in k for x in self.settings['g2l_blacklist']]):
        self.blacklisted_keys.append(k)    
      if any([x in k for x in self.settings['l2g_whitelist']]):
        self.whitelisted_keys.append(k)    

# This discards everything from the old env
  def reset_env(self, fname, **kwargs):
    self.reset_counter += 1
    if self.restart_solver_every > 0 and (self.settings['restart_in_test'] or (self.reset_counter % self.restart_solver_every == 0)):
      self.envstr.env.restart_env(timeout=0)
    # if not fname:
    #   if not self.reset_counter % 200:
    #     self.ds.recalc_weights()
    #   (fname,) = self.ds.weighted_sample()
    # last_obs, env_id = self.envstr.env.new_episode(fname=fname, **kwargs)
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
    if not envstr.last_obs or envstr.curr_step > self.max_step:
      self.reset_env(fname=self.provider.get_next())   
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

  def discount_episode(self, ep):
    def compute_baseline(formula):
      d = self.ed.get_data()
      if not formula in d.keys() or len(d[formula]) < 3:
        latest_stats = list(x for y in d.values() for x in y[-20:])
        _, r = zip(*latest_stats)        
        return np.mean(r)

      stats = d[formula]
      steps, rewards = zip(*stats)
      return np.mean(rewards[-20:-1])

    gamma = self.settings['gamma']    
    baseline = compute_baseline(ep[0].formula) if self.settings['stats_baseline'] else 0
    _, _, _,rewards, *_ = zip(*ep)
    r = discount(rewards, gamma) - baseline
    return [Transition(transition.state, transition.action, None, rew, transition.formula, transition.prev_obs) for transition, rew in zip(ep, r)]

  def check_batch_finished(self):
    if self.settings['episodes_per_batch']:
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
    i=0
    while len(rc) < num:
      ep = self.discount_episode(self.completed_episodes.pop(0))      
      rc.extend(ep)
      i += 1

    return rc, i

  def pop_min_normalized(self, num=0):
    if num == 0:
      num = self.settings['episodes_per_batch']
    rc = []
    rc_len = []
    for i in range(num):
      ep = self.discount_episode(self.completed_episodes.pop(0))
      rc.extend(ep)
      rc_len.extend([i]*len(ep))
    return rc, rc_len

  def init_proc(self):
    set_proc_name(str.encode(self.name))
    np.random.seed(int(time.time())+abs(hash(self.name)) % 1000)
    torch.manual_seed(int(time.time())+abs(hash(self.name)) % 1000)
    self.settings.hyperparameters['cuda']=False         # No CUDA in the worker threads
    self.lmodel = create_policy(settings=self.settings)
    self.lmodel.load_state_dict(self.gmodel.state_dict())
    if self.settings['log_threshold']:
      self.lmodel.shelf_file = shelve.open('thres_proc_{}.shelf'.format(self.name))      

  def train(self,transition_data, **kwargs):
    if self.settings['do_not_learn']:
      return
    self.lmodel.train()
    mt = time.time()
    loss, logits = self.lmodel.compute_loss(transition_data, **kwargs)
    mt1 = time.time()
    print('Loss computation took {} seconds'.format(mt1-mt))
    self.optimizer.zero_grad()
    loss.backward()
    mt2 = time.time()
    print('Backward took {} seconds'.format(mt2-mt1))
    # torch.nn.utils.clip_grad_norm_(self.lmodel.parameters(), self.settings['grad_norm_clipping'])
    for lp, gp in zip(self.lmodel.parameters(), self.gmodel.parameters()):
        gp._grad = lp.grad
    self.optimizer.step()
    global_params = self.gmodel.state_dict()
    for k in self.blacklisted_keys:
      global_params.pop(k,None)
    self.lmodel.load_state_dict(global_params,strict=False)
    z = self.lmodel.state_dict()

    # We may want to sync that

    local_params = {}
    for k in self.whitelisted_keys:
      local_params[k] = z[k]
    self.gmodel.load_state_dict(local_params,strict=False)





  def run(self):
    if self.settings['profiling']:
      cProfile.runctx('self.run_loop()', globals(), locals(), 'prof_{}.prof'.format(self.name))
    else:
      self.run_loop()


  def run_loop(self):
    self.init_proc()
    clock = GlobalTick()
    SYNC_STATS_EVERY = 1
    # SYNC_STATS_EVERY = 5+np.random.randint(10)
    total_step = 0
    total_eps = 0
    local_env_steps = 0
    global_steps = 0
    # self.episodes_files = self.ds.get_files_list()
    while global_steps < self.training_steps:
      self.lmodel.eval()
      begin_time = time.time()
      rc = False
      while (not rc) or (not self.check_batch_finished()):
        if self.settings['log_threshold']:
          k = '{}_{}'.format(global_steps,len(self.completed_episodes))
          print('setting key to {}'.format(k))
          self.lmodel.shelf_key = k
          self.lmodel.shelf_file.sync()
        rc = self.step()
      total_inference_time = time.time() - begin_time
      transition_data, lenvec = self.pop_min_normalized() if self.settings['episodes_per_batch'] else self.pop_min()
      print('Forward pass in {} got batch with length {} in {} seconds. Ratio: {}'.format(self.name,len(transition_data),total_inference_time,len(transition_data)/total_inference_time))
      # After the batch is finished, advance the iterator
      self.provider.reset()
      self.reset_env(fname=self.provider.get_next())
      begin_time = time.time()
      self.train(transition_data,lenvec=lenvec)
      total_train_time = time.time() - begin_time
      print('Backward pass in {} done in {} seconds!'.format(self.name,total_train_time))

      # Sync to global step counts
      total_step += 1
      clock.tick()
      local_env_steps += len(transition_data)
      total_eps += (max(lenvec)+1)
      if total_step % SYNC_STATS_EVERY == 0:
        with self.g_grad_steps.get_lock():
          self.g_grad_steps.value += SYNC_STATS_EVERY
        with self.g_episodes.get_lock():
          self.g_episodes.value += total_eps
          total_eps = 0
        with self.g_steps.get_lock():
          self.g_steps.value += local_env_steps
          local_env_steps = 0
          global_steps = self.g_grad_steps.value




