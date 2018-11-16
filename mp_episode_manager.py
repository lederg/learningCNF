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


from cadet_env import *
from qbf_data import *
from settings import *
from utils import *
from rl_utils import *
from cadet_utils import *
from episode_data import *

# DEF_COST = -1.000e-04

MPEnvStruct = namedlist('EnvStruct',
                    ['env', 'last_obs', 'episode_memory', 'env_id', 'fname', 'curr_step', 'active', 'prev_obs'])


class WorkerEnv(mp.Process):
  def __init__(self, settings, model, opt, ds, ed, global_steps, global_grad_steps, global_episodes, name, reporter=None):
    super(WorkerEnv, self).__init__()
    self.name = 'a3c_worker%i' % name
    self.g_steps = global_steps
    self.g_grad_steps = global_grad_steps
    self.g_episodes = global_episodes
    self.settings = settings
    self.ds = ds
    self.ed = ed
    self.completed_episodes = []
    self.reporter = reporter
    self.max_step = self.settings['max_step']
    self.rnn_iters = self.settings['rnn_iters']
    self.training_steps = self.settings['training_steps']
    self.restart_cadet_every = self.settings['restart_cadet_every']
    self.envstr = MPEnvStruct(CadetEnv(**self.settings.hyperparameters), 
        None, None, None, None, None, True, deque(maxlen=self.rnn_iters))
    # self.gnet, self.opt = gnet, opt
    # self.lnet = Net(N_S, N_A)           # local network
    self.gmodel = model  
    self.optimizer = opt  
    self.reset_counter = 0
    self.env_steps = 0
    self.real_steps = 0
    self.def_step_cost = self.settings['def_step_cost']
    self.ProviderClass = eval(self.settings['episode_provider'])
    self.provider = iter(self.ProviderClass(self.ds))


# This discards everything from the old env
  def reset_env(self, fname=None, **kwargs):
    self.reset_counter += 1
    if self.settings['restart_in_test'] or (self.reset_counter % self.restart_cadet_every == 0):
      self.envstr.env.restart_cadet(timeout=0)
    if not fname:
      if not self.reset_counter % 200:
        self.ds.recalc_weights()
      (fname,) = self.ds.weighted_sample()
    last_obs, env_id = new_episode(self.envstr.env, fname=fname, **kwargs)
    self.envstr.last_obs = last_obs
    self.envstr.env_id = fname
    self.envstr.curr_step = 0
    self.envstr.fname = fname
    self.envstr.episode_memory = []     
    # Set up the previous observations to be None followed by the last_obs   
    self.envstr.prev_obs.clear()    
    for i in range(self.rnn_iters):
      self.envstr.prev_obs.append(None)
    return last_obs

  def step(self, **kwargs):
    envstr = self.envstr
    env = envstr.env
    if not envstr.last_obs or envstr.curr_step > self.max_step:
      self.reset_env(fname=next(self.provider))    
    [action] = self.lmodel.select_action(envstr.last_obs, **kwargs)
    envstr.episode_memory.append(Transition(envstr.last_obs,action,None, None, envstr.env_id, envstr.prev_obs))
    allowed_actions = self.lmodel.get_allowed_actions(envstr.last_obs).squeeze()

    if allowed_actions[action]:
      env_obs = EnvObservation(*envstr.env.step(self.lmodel.translate_action(action)))        
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
      envstr.last_obs = process_observation(env,envstr.last_obs,env_obs)


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
    i=0
    for _ in range(num):
      ep = self.discount_episode(self.completed_episodes.pop(0))
      rc.extend(ep)

    return rc, num

  def init_proc(self):
    set_proc_name(str.encode(self.name))
    np.random.seed(int(time.time())+abs(hash(self.name)) % 1000)
    torch.manual_seed(int(time.time())+abs(hash(self.name)) % 1000)
    self.lmodel = create_policy(settings=self.settings)
    self.lmodel.load_state_dict(self.gmodel.state_dict())


  def compute_loss(self, transition_data):
    lambda_disallowed = self.settings['lambda_disallowed']
    lambda_value = self.settings['lambda_value']
    lambda_aux = self.settings['lambda_aux']
    disallowed_loss = 0.
    _, _, _, rewards, *_ = zip(*transition_data)
    collated_batch = collate_transitions(transition_data,settings=self.settings)
    logits, values, _, aux_losses = self.lmodel(collated_batch.state, prev_obs=collated_batch.prev_obs)
    allowed_actions = Variable(self.lmodel.get_allowed_actions(collated_batch.state))
    if self.settings['cuda']:
      allowed_actions = allowed_actions.cuda()
    # unpacked_logits = unpack_logits(logits, collated_batch.state.pack_indices[1])
    effective_bs = len(logits)    

    if self.settings['masked_softmax']:
      allowed_mask = allowed_actions.float()      
      probs, debug_probs = masked_softmax2d(logits,allowed_mask)
    else:
      probs = F.softmax(logits, dim=1)
    all_logprobs = safe_logprobs(probs)
    if self.settings['disallowed_aux']:        # Disallowed actions are possible, so we add auxilliary loss
      aux_probs = F.softmax(logits,dim=1)
      disallowed_actions = Variable(allowed_actions.data^1).float()      
      disallowed_mass = (aux_probs*disallowed_actions).sum(1)
      disallowed_loss = disallowed_mass.mean()
      print('Disallowed loss is {}'.format(disallowed_loss))

    returns = self.settings.FloatTensor(rewards)
    if self.settings['ac_baseline']:
      adv_t = returns - values.squeeze().data      
      value_loss = mse_loss(values.squeeze(), Variable(returns))    
      print('Value loss is {}'.format(value_loss.data.numpy()))
      print('Value Auxilliary loss is {}'.format(sum(aux_losses).data.numpy()))
      # if i>0 and i % 60 == 0:
      #   vecs = {'returns': returns.numpy(), 'values': values.squeeze().data.numpy()}        
      #   pprint_vectors(vecs)
    else:
      adv_t = returns
      value_loss = 0.
    actions = collated_batch.action    
    logprobs = all_logprobs.gather(1,Variable(actions).view(-1,1)).squeeze()
    entropies = (-probs*all_logprobs).sum(1)    
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + float(np.finfo(np.float32).eps))
    if self.settings['normalize_episodes']:
      episodes_weights = normalize_weights(collated_batch.formula.cpu().numpy())
      adv_t = adv_t*self.settings.FloatTensor(episodes_weights)    
    pg_loss = (-Variable(adv_t)*logprobs).mean()
    total_aux_loss = sum(aux_losses) if aux_losses else 0.
    loss = pg_loss + lambda_value*value_loss + lambda_disallowed*disallowed_loss + lambda_aux*total_aux_loss
    return loss

  def train(self,transition_data):
    if self.settings['do_not_learn']:
      return
    self.lmodel.train()
    loss = self.compute_loss(transition_data)
    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.lmodel.parameters(), self.settings['grad_norm_clipping'])
    for lp, gp in zip(self.lmodel.parameters(), self.gmodel.parameters()):
        gp._grad = lp.grad
    self.optimizer.step()
    self.lmodel.load_state_dict(self.gmodel.state_dict())


  def run(self):
    if self.settings['profiling']:
      cProfile.runctx('self.run_loop()', globals(), locals(), 'prof_{}.prof'.format(self.name))
    else:
      self.run_loop()


  def run_loop(self):
    self.init_proc()
    SYNC_STATS_EVERY = 5+np.random.randint(10)
    total_step = 0
    total_eps = 0
    local_env_steps = 0
    global_steps = 0
    self.episodes_files = self.ds.get_files_list()
    while global_steps < self.training_steps:
      self.lmodel.eval()
      begin_time = time.time()
      rc = False
      while (not rc) or (not self.check_batch_finished()):
        rc = self.step()
      total_inference_time = time.time() - begin_time
      transition_data, num_eps = self.pop_min_normalized() if self.settings['episodes_per_batch'] else self.pop_min()
      # After the batch is finished, advance the iterator
      self.provider.reset()
      self.reset_env(fname=next(self.provider))
      print('Forward pass in {} got batch with length {} in {} seconds!'.format(self.name,len(transition_data),total_inference_time))
      begin_time = time.time()
      self.train(transition_data)
      total_train_time = time.time() - begin_time
      print('Backward pass in {} done in {} seconds!'.format(self.name,total_train_time))

      # Sync to global step counts
      total_step += 1
      local_env_steps += len(transition_data)
      total_eps += num_eps
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




