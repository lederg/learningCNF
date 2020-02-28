from pysat.solvers import Minisat22, Glucose3, SharpSAT
from pysat.formula import CNF
from subprocess import Popen, PIPE, STDOUT
from collections import deque
from namedlist import namedlist
from scipy.sparse import csr_matrix
import select
import threading
from IPython.core.debugger import Tracer
import time
import logging
import pickle
import tracemalloc
import torch.multiprocessing as mp
import utils
import pysolvers
import traceback
from gym import *
from gym import spaces
from settings import *
from qbf_data import *
from envbase import *
from rl_types import *
from reduce_base_provider import *

LOG_SIZE = 200
DEF_STEP_REWARD = -0.01     # Temporary reward until Pash sends something from minisat

class SharpSpace(gym.Space):
  def contains(self, x):
    return True

  @property
  def shape(self):
    return ()

class SharpActiveEnv:
  EnvObservation = namedlist('SharpEnvObservation', 
                              ['gss', 'vfeatures', 'cfeatures', 'indices_row', 'indices_col', 'reward', 'done'],
                              default=None)

  def __init__(self, server=None, settings=None, **kwargs):
    self.settings = settings if settings else CnfSettings()    
    self.debug = debug
    self.tail = deque([],LOG_SIZE)
    self.solver = None
    self.server = server
    self.current_step = 0    
    self.disable_gnn = self.settings['disable_gnn']
    self.formulas_dict = {}
    self._name = 'SharpEnv'

  @property
  def name(self):
    return self._name
  
  # def load_formula(self, fname):
  #   if fname not in self.formulas_dict.keys():
  #     self.formulas_dict[fname] = CNF(fname)
  #     print('Lazily loaded {} in process {}_{}'.format(fname,self._name,os.getpid()))
  #   return self.formulas_dict[fname]

  def start_solver(self, fname=None):
    
    def thunk(*args):
      return self.__callback(*args)
    if self.solver is None:
      self.solver = SharpSAT()
    else:
      self.solver.delete()
      self.solver.new(branching_oracle= {"branching_cb": thunk})
    self.current_step = 0
    if fname:
      try:
        f1 = self.settings.formula_cache.load_formula(fname)
        self.solver.append_formula(f1.clauses)
        del f1
        return True
      except:
        return False
    else:
      print('Got no filename!!')
      return False

  def __callback(self, row, col, data):
    self.current_step += 1
    if self.disable_gnn:
      adj_matrix = None
      vlabels = None
    else:
      vlabels = m.get_lit_labels()
      adj_matrix = csr_matrix((data, (row, col)))
    if not self.server:
      log.info('Running a test version of SharpEnv')
      ind = np.argmax(vlabels[:,1])
      pick = ind + 1 if (vlabels[ind][0] < vlabels[ind + 1][0]) else ind
      return pick
      
    else:
      try:
        return self.server.callback(vlabels, None, adj_matrix)
      except Exception as e:
        print('SharpEnv: Gah, an exception: {}'.format(e))
        raise e

class SharpEnvProxy(EnvBase):
  def __init__(self, config):
    self.settings = config['settings']
    if not self.settings:
      self.settings = CnfSettings()
    self.state_dim = self.settings['state_dim']
    self.observation_space = SharpSpace()
    self.action_space = spaces.Discrete(self.settings['max_variables'])
    self.queue_in = config['queue_in']
    self.queue_out = config['queue_out']
    self.provider = config['provider']
    self.rewards = []
    self.current_step = 0
    self.finished = False
    self.def_step_cost = self.settings['def_step_cost']    
    self.completion_reward = self.settings['sharp_completion_reward']
    self.max_step = self.settings['max_step']    
    self.disable_gnn = self.settings['disable_gnn']
    self.server_pid = config['server_pid']
    self.logger = utils.get_logger(self.settings, 'SharpEnvProxy')

  def step(self, action):
    self.queue_out.put((EnvCommands.CMD_STEP,action))
    ack, rc = self.queue_in.get()  
    assert ack==EnvCommands.ACK_STEP, 'Expected ACK_STEP'
    env_obs = SharpActiveEnv.EnvObservation(*rc)
    self.finished = env_obs.done
    if env_obs.reward:
      self.rewards.append(env_obs.reward)
    self.current_step += 1
    # if env_obs.done:
    #   print('Env returning DONE, number of rewards is {}'.format(len(self.rewards)))
    return env_obs, r, env_obs.done or self.check_break(), {}

  def reset(self):
    fname = self.provider.get_next()
    # print('reset: Got formula: {}'.format(fname))
    self.finished = False
    self.current_step = 0
    self.rewards = []
    self.queue_out.put((EnvCommands.CMD_RESET,fname))
    ack, rc = self.queue_in.get()
    assert ack==EnvCommands.ACK_RESET, 'Expected ACK_RESET'    
    if rc != None:
      return SharpActiveEnv.EnvObservation(*rc)

  def exit(self):
    self.queue_out.put((EnvCommands.CMD_EXIT,None))
    ack, rc = self.queue_in.get()
    assert ack==EnvCommands.ACK_EXIT, 'Expected ACK_EXIT'

    # For the gym interface, the env itself decides whether to abort.

  def check_break(self):    
    return (self.current_step > self.max_step)

  def new_episode(self, fname, **kwargs):    
    return self.reset(fname)        

class SatEnvServer(threading.Thread):
  def __init__(self, env, settings=None):
    super(SatEnvServer, self).__init__()
    self.settings = settings if settings else CnfSettings()
    self.state_dim = self.settings['state_dim']    
    self.env = env
    self.env.server = self
    self.queue_in = mp.Queue()
    self.queue_out = mp.Queue()
    self.cmd = None
    self.current_fname = None
    self.last_reward = 0
    self.last_orig_clause_size = 0
    self.do_lbd = self.settings['do_lbd']
    self.disable_gnn = self.settings['disable_gnn']
    if self.settings['sat_min_reward']:
      self.winning_reward = -self.settings['sat_min_reward']*self.settings['sat_reward_scale']*self.settings['sat_win_scale']
    else:
      self.winning_reward = self.settings['sat_winning_reward']*self.settings['sat_reward_scale']
    self.total_episodes = 0
    self.uncache_after_batch = self.settings['uncache_after_batch']
    self.logger = utils.get_logger(self.settings, 'SatEnvServer')    

  def proxy(self, **kwargs):
    config = kwargs
    config['queue_out'] = self.queue_in
    config['queue_in'] = self.queue_out
    config['server_pid'] = self.pid
    return SharpEnvProxy(config)

  def run(self):
    print('Env {} on pid {}'.format(self.env.name, os.getpid()))
    set_proc_name(str.encode('{}_{}'.format(self.env.name,os.getpid())))
    # if self.settings['memory_profiling']:
    #   tracemalloc.start(25)    
    while True:
      # if self.settings['memory_profiling'] and (self.total_episodes % 10 == 1):    
      # if self.settings['memory_profiling']:
      #   snapshot = tracemalloc.take_snapshot()
      #   top_stats = snapshot.statistics('lineno')
      #   print("[ Top 20 in {}]".format(self.name))
      #   for stat in top_stats[:20]:
      #       print(stat)            
      #   print('Number of cached formulas: {}'.format(len(self.env.formulas_dict.keys())))
      #   print(self.env.formulas_dict.keys())


      if self.cmd == EnvCommands.CMD_RESET:
        # We get here only after a CMD_RESET aborted a running episode and requested a new file.
        fname = self.current_fname        
      else:
        self.cmd, fname = self.queue_in.get()
        if self.cmd == EnvCommands.CMD_EXIT:
          print('Got CMD_EXIT 1')
          self.queue_out.put((EnvCommands.ACK_EXIT,None))
          break
        assert self.cmd == EnvCommands.CMD_RESET, 'Unexpected command {}'.format(self.cmd)
      if self.uncache_after_batch and  fname != self.current_fname:
        self.settings.formula_cache.delete_key(self.current_fname)
      self.current_fname = fname

      # This call does not return until the episodes is done. Messages are going to be exchanged until then through
      # the __callback method

      if self.env.start_solver(fname):
        self.env.solver.solve()
      else:
        print('Skipping {}'.format(fname))

      if self.cmd == EnvCommands.CMD_STEP:
        last_step_reward = -(self.env.get_reward() - self.last_reward)      
        # We are here because the episode successfuly finished. We need to mark done and return the rewards to the client.
        msg = self.env.EnvObservation(state=np.zeros(self.state_dim), reward=self.winning_reward+last_step_reward, done=True)
        # msg = self.env.EnvObservation(None, None, None, None, None, None, self.winning_reward+last_step_reward, True)
        self.queue_out.put((EnvCommands.ACK_STEP,tuple(msg)))
        self.total_episodes += 1

      elif self.cmd == EnvCommands.CMD_RESET:
        if self.env.current_step == 0:
          print('Degenerate episode on {}'.format(fname))
          self.cmd = None
          self.queue_out.put((EnvCommands.ACK_RESET,None))
          # This is a degenerate episodes with no GC
        else:
          pass
          # We are here because the episode was aborted. We can just move on, the client already has everything.
      elif self.cmd == EnvCommands.CMD_EXIT:
        print('Got CMD_EXIT 2')
        self.queue_out.put((EnvCommands.ACK_EXIT,None))
        break


  def callback(self, vlabels, cl_label_arr, adj_matrix):
    self.env.current_step += 1
    # print('clabels shape: {}'.format(cl_label_arr.shape))
    state = self.env.get_global_state()
    # print('reward is {}'.format(self.env.get_reward()))
    msg = self.env.EnvObservation(state, None, None, adj_matrix, vlabels, cl_label_arr, None, False)
    if not self.disable_gnn:      
      msg.orig_clause_labels = self.env.get_clabels()
      if self.cmd == EnvCommands.CMD_RESET or (self.last_orig_clause_size and len(msg.orig_clause_labels) < self.last_orig_clause_size):
        msg.orig_clauses = self.env.get_orig_clauses()
        self.last_orig_clause_size = len(msg.orig_clause_labels)
    if self.cmd == EnvCommands.CMD_RESET:
      # if not self.disable_gnn:
      #   msg.orig_clauses = self.env.get_orig_clauses()
      self.last_reward = self.env.get_reward()
      ack = EnvCommands.ACK_RESET
    elif self.cmd == EnvCommands.CMD_STEP:
      last_reward = self.env.get_reward()
      msg.reward = -(last_reward - self.last_reward)
      # print('Got reward: {}'.format(msg.reward))
      self.last_reward = last_reward
      ack = EnvCommands.ACK_STEP
    elif self.cmd == EnvCommands.CMD_EXIT:
      print('self.cmd is CMD_EXIT, yet we are in the callback again!')
      return None
    else:
      assert True, 'Invalid last command detected'

    self.queue_out.put((ack,tuple(msg)))
    self.cmd, rc = self.queue_in.get()
    if self.cmd == EnvCommands.CMD_STEP:
      # We got back an action
      return rc
    elif self.cmd == EnvCommands.CMD_RESET:
      # We were asked to abort the current episode. Notify the solver and continue as usual
      if self.uncache_after_batch and rc != self.current_fname:
        self.settings.formula_cache.delete_key(self.current_fname)
      self.current_fname = rc
      self.env.solver.terminate()
      return None
    elif self.cmd == EnvCommands.CMD_EXIT:
      print('Got CMD_EXIT')
      self.env.solver.terminate()
      return None

