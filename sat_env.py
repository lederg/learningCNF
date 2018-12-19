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

LOG_SIZE = 200
DEF_STEP_REWARD = -0.01     # Temporary reward until Pash sends something from minisat
WINNING_REWARD = 1
log = mp.get_logger()

class SatActiveEnv:
  EnvObservation = namedlist('SatEnvObservation', 
                              ['orig_clauses', 'learned_clauses', 'vlabels', 'clabels', 'reward', 'done'])

  def __init__(self, debug=False, server=None, **kwargs):
    self.debug = debug
    self.tail = deque([],LOG_SIZE)
    self.solver = None
    self.server = server
    self.current_step = 0
    self._name = 'SatEnv'

  @property
  def name(self):
    return self._name
  
  def start_solver(self, fname=None):
    def thunk(cl_label_arr, rows_arr, cols_arr, data_arr):      
      return self.__callback(cl_label_arr, rows_arr, cols_arr, data_arr, DEF_STEP_REWARD)
      

    if self.solver is None:
      self.solver = Minisat22(callback=thunk)
    else:
      self.solver.delete()
      self.solver.new(callback=thunk)
    if fname:
      f1 = CNF(fname)
      self.solver.append_formula(f1.clauses)
    self.current_step = 0

  def get_orig_clauses(self):
    return self.solver.get_cl_arr()

  def __callback(self, cl_label_arr, rows_arr, cols_arr, data_arr, reward):
    self.current_step += 1
    adj_matrix = csr_matrix((data_arr, (rows_arr, cols_arr)))
    # ipdb.set_trace()
    if not self.server:
      log.info('Running a test version of SatEnv')
      utility = cl_label_arr[:,3] # just return the lbd
      ipdb.set_trace()
      return utility
    else:
      log.info('Calling server callback')
      print('huh. server callback is {}'.format(self.server))
      try:
        return self.server.callback(cl_label_arr, adj_matrix, reward)
      except Exception as e:
        print('Gah, an exception: {}'.format(e))

class SatEnvProxy(EnvBase):
  def __init__(self, queue_in, queue_out):
    self.queue_in = queue_in
    self.queue_out = queue_out

  def step(self, action):
    self.queue_out.put((EnvCommands.CMD_STEP,action))
    ack, rc = self.queue_in.get()
    assert ack==EnvCommands.ACK_STEP, 'Expected ACK_STEP'
    return SatActiveEnv.EnvObservation(*rc)

  def reset(self, fname):
    self.queue_out.put((EnvCommands.CMD_RESET,fname))
    ack, rc = self.queue_in.get()
    assert ack==EnvCommands.ACK_RESET, 'Expected ACK_RESET'
    return SatActiveEnv.EnvObservation(*rc)

  def new_episode(self, fname, settings=None, **kwargs):
    if not settings:
      settings = CnfSettings()
    try:
      env_id = int(os.path.split(fname)[1].split('_')[-2])
    except:
      env_id = os.path.split(fname)[1]
    # Set up ground_embeddings and adjacency matrices
    obs = self.reset(fname)    
    return obs, env_id

  def process_observation(self, last_obs, env_obs, settings=None):
    if not settings:
      settings = CnfSettings()

    orig_clauses = env_obs.orig_clauses
    ipdb.set_trace()
    return State(state,cmat_pos, cmat_neg, ground_embs, clabels, None, None)

class SatEnvServer(mp.Process):
  def __init__(self, env):
    super(SatEnvServer, self).__init__()
    self.env = env
    self.env.server = self
    self.queue_in = mp.Queue()
    self.queue_out = mp.Queue()
    self.cmd = None
    self.current_fname = None

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
        # We are here because the episode successfuly finished. We need to mark done and return the rewards to the client.
        msg = self.env.EnvObservation(None, None, None, None, WINNING_REWARD, True)
        self.queue_out.put((ACK_STEP,tuple(msg)))

      elif self.cmd == EnvCommands.CMD_RESET:
        if self.env.current_step == 0:
          pass
          # This is a degenerate episodes with no GC
        else:
          pass
          # We are here because the episode was aborted. We can just move on, the client already has everything.
      elif self.cmd == EnvCommands.CMD_EXIT:
        break


  def callback(self, cl_label_arr, adj_matrix, reward):
    print('self is {}'.format(self))
    self.env.current_step += 1
    print('Here 1')
    msg = self.env.EnvObservation(None, adj_matrix, None, cl_label_arr, reward, False)
    print('Here 2')
    if self.cmd == EnvCommands.CMD_RESET:
      # If this is the reply to a RESET add all existing (permanent) clauses
      msg.orig_clauses = self.env.get_orig_clauses()
      print('Here 3')
      ack = EnvCommands.ACK_RESET
    elif self.cmd == EnvCommands.CMD_STEP:
      ack = EnvCommands.ACK_STEP
    else:
      assert True, 'Invalid last command detected'

    self.queue_out.put((ack,tuple(msg)))
    print('Here 4')
    self.cmd, rc = self.queue_in.get()
    print('Here 5')
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

