from pysat.solvers import Minisat22
from pysat.formula import CNF
from subprocess import Popen, PIPE, STDOUT
from collections import deque
from namedlist import namedlist
from scipy.sparse import csr_matrix
import select
import ipdb
import time
import logging
import pickle
import multiprocessing as mp
from settings import *
from qbf_data import *
from envbase import *
from rl_types import *

LOG_SIZE = 200
DEF_STEP_REWARD = -0.01     # Temporary reward until Pash sends something from minisat
log = mp.get_logger()


CLABEL_LEARNED = 0
CLABEL_LBD = 3
CLABEL_LOCKED = 5

class SatActiveEnv:
  EnvObservation = namedlist('SatEnvObservation', 
                              ['orig_clauses', 'orig_clause_labels', 'learned_clauses', 'vlabels', 'clabels', 'reward', 'done'])

  def __init__(self, debug=False, server=None, settings=None, **kwargs):
    self.settings = settings if settings else CnfSettings()    
    self.debug = debug
    self.tail = deque([],LOG_SIZE)
    self.solver = None
    self.server = server
    self.current_step = 0
    self.reduce_base = int(self.settings['sat_reduce_base'])
    self.disable_gnn = self.settings['disable_gnn']
    self.formulas_dict = {}
    self._name = 'SatEnv'

  @property
  def name(self):
    return self._name
  
  def load_formula(self, fname):
    if fname not in self.formulas_dict.keys():
      self.formulas_dict[fname] = CNF(fname)
    return self.formulas_dict[fname]

  def start_solver(self, fname=None):
    
    def thunk(cl_label_arr, rows_arr, cols_arr, data_arr):      
      return self.__callback(cl_label_arr, rows_arr, cols_arr, data_arr, DEF_STEP_REWARD)
      

    if self.solver is None:
      print('reduce_base is {}'.format(self.reduce_base))
      # self.solver = Minisat22(callback=thunk)
      self.solver = Minisat22(callback=thunk, reduce_base=self.reduce_base)
    else:
      self.solver.delete()
      self.solver.new(callback=thunk, reduce_base=self.reduce_base)
    if fname:
      f1 = self.load_formula(fname)
      self.solver.append_formula(f1.clauses)
    self.current_step = 0

  def get_orig_clauses(self):
    return self.solver.get_cl_arr()

  def get_vlabels(self):
    return self.solver.get_var_labels()

  def get_clabels(self, learned=False):
    return self.solver.get_cl_labels(learned)

  def get_global_state(self):
    return self.solver.get_solver_state()

  def get_reward(self):
    return self.solver.reward()

  def __callback(self, cl_label_arr, rows_arr, cols_arr, data_arr, reward):
    self.current_step += 1
    if self.disable_gnn:
      adj_matrix = None
    else:            
      adj_matrix = csr_matrix((data_arr, (rows_arr, cols_arr)))
    if not self.server:
      log.info('Running a test version of SatEnv')
      utility = cl_label_arr[:,3] # just return the lbd
      ipdb.set_trace()
      return utility
    else:
      try:
        return self.server.callback(self.get_vlabels(), cl_label_arr, adj_matrix, reward)
      except Exception as e:
        print('Gah, an exception: {}'.format(e))

class SatEnvProxy(EnvBase):
  def __init__(self, queue_in, queue_out,settings=None):
    self.settings = settings if settings else CnfSettings()
    self.queue_in = queue_in
    self.queue_out = queue_out
    self.state = torch.zeros(8)
    self.orig_clabels = None
    self.rewards = None
    self.finished = False
    self.reward_scale = self.settings['sat_reward_scale']
    self.disable_gnn = self.settings['disable_gnn']

  def step(self, action):
    self.queue_out.put((EnvCommands.CMD_STEP,action))    
    ack, rc = self.queue_in.get()  
    assert ack==EnvCommands.ACK_STEP, 'Expected ACK_STEP'
    env_obs = SatActiveEnv.EnvObservation(*rc)
    self.finished = env_obs.done    
    if env_obs.reward:
      r = env_obs.reward / self.reward_scale
      self.rewards.append(r)
    return env_obs

  def reset(self, fname):
    self.finished = False
    self.rewards = []
    self.queue_out.put((EnvCommands.CMD_RESET,fname))
    ack, rc = self.queue_in.get()
    assert ack==EnvCommands.ACK_RESET, 'Expected ACK_RESET'    
    if rc != None:
      return SatActiveEnv.EnvObservation(*rc)


  def new_episode(self, fname, settings=None, **kwargs):
    if not settings:
      settings = CnfSettings()
    env_id = os.path.split(fname)[1]
    # Set up ground_embeddings and adjacency matrices
    obs = self.reset(fname)    
    return obs, env_id

  def process_observation(self, last_obs, env_obs, settings=None):
    if not settings:
      settings = CnfSettings()

    if env_obs == None:
      return None
    self.orig_clabels = [] if self.disable_gnn else env_obs.orig_clause_labels
    if env_obs.orig_clauses is not None:
      self.orig_clauses = None if self.disable_gnn else csr_to_pytorch(env_obs.orig_clauses)
    learned_clauses = None if self.disable_gnn else csr_to_pytorch(env_obs.learned_clauses)
    cmat = None if self.disable_gnn else Variable(concat_sparse(self.orig_clauses,learned_clauses))
    all_clabels = torch.from_numpy(env_obs.clabels if self.disable_gnn else np.concatenate([self.orig_clabels,env_obs.clabels])).float()



    # Replace the first index of the clabels with a marker for orig/learned

    all_clabels[:,0]=0
    all_clabels[-len(env_obs.clabels):,0]=1

    # Take log of vlabels[:,3]
    activities = env_obs.vlabels[:,3]+10
    env_obs.vlabels[:,3]=np.log(activities)
    clabels = Variable(all_clabels)
    # vlabels = Variable(torch.from_numpy(env_obs.vlabels[1:]).float())   # Remove first (zero) row
    vlabels = Variable(torch.from_numpy(env_obs.vlabels).float())   # Remove first (zero) row
    # ipdb.set_trace()
    vmask = last_obs.vmask if last_obs else None
    cmask = last_obs.cmask if last_obs else None
    state = Variable(self.state.unsqueeze(0))    
    num_orig_clauses = len(self.orig_clabels)
    num_learned_clauses = len(env_obs.clabels)

    return State(state,cmat, vlabels, clabels, vmask, cmask, (num_orig_clauses,num_orig_clauses+num_learned_clauses))

class SatEnvServer(mp.Process):
  def __init__(self, env, settings=None):
    super(SatEnvServer, self).__init__()
    self.settings = settings if settings else CnfSettings()
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
    self.winning_reward = self.settings['sat_winning_reward']*self.settings['sat_reward_scale']

  def proxy(self):
    return SatEnvProxy(self.queue_out, self.queue_in)

  def run(self):
    print('Env {} on pid {}'.format(self.env.name, os.getpid()))
    # print('You seeing that?!?!')
    set_proc_name(str.encode('{}_{}'.format(self.env.name,os.getpid())))
    while True:
      if self.cmd == EnvCommands.CMD_RESET:
        # We get here only after a CMD_RESET aborted a running episode and requested a new file.
        fname = self.current_fname
      else:
        self.cmd, fname = self.queue_in.get()
        assert self.cmd == EnvCommands.CMD_RESET, 'Unexpected command {}'.format(self.cmd)
      self.current_fname = None
      # print('Env object is {}'.format(self.env))
      # print('The solver is {}'.format(self.env.solver))
      self.env.start_solver(fname)

      # This call does not return until the episodes is done. Messages are going to be exchanged until then through
      # the __callback method

      self.env.solver.solve()

      if self.cmd == EnvCommands.CMD_STEP:
        last_step_reward = -(self.env.get_reward() - self.last_reward)      
        # We are here because the episode successfuly finished. We need to mark done and return the rewards to the client.
        msg = self.env.EnvObservation(None, None, None, None, None, self.winning_reward+last_step_reward, True)
        self.queue_out.put((EnvCommands.ACK_STEP,tuple(msg)))

      elif self.cmd == EnvCommands.CMD_RESET:
        if self.env.current_step == 0:
          if self.settings['debug']:
            print('Degenerate episode on {}'.format(fname))
          self.cmd = None
          self.queue_out.put((EnvCommands.ACK_RESET,None))
          # This is a degenerate episodes with no GC
        else:
          pass
          # We are here because the episode was aborted. We can just move on, the client already has everything.
      elif self.cmd == EnvCommands.CMD_EXIT:
        break


  def callback(self, vlabels, cl_label_arr, adj_matrix, reward):
    self.env.current_step += 1    
    msg = self.env.EnvObservation(None, None, adj_matrix, vlabels, cl_label_arr, None, False)
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
    else:
      assert True, 'Invalid last command detected'

    self.queue_out.put((ack,tuple(msg)))
    self.cmd, rc = self.queue_in.get()
    if self.cmd == EnvCommands.CMD_STEP:
      # We got back an array of decisions
      return rc
    elif self.cmd == EnvCommands.CMD_RESET:
      # We were asked to abort the current episode. Notify the solver and continue as usual
      self.env.solver.terminate()
      self.current_fname = rc
      return None
    elif self.cmd == EnvCommands.CMD_EXIT:
      self.env.solver.terminate()
      log.info('Got CMD_EXIT')
      return None

