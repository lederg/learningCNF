from pysat.solvers import Minisat22
from pysat.formula import CNF
from subprocess import Popen, PIPE, STDOUT
from collections import deque
from scipy.sparse import csr_matrix
import select
import ipdb
import time
import logging
import multiprocessing as mp
from settings import *
from qbf_data import *
from envbase import *

LOG_SIZE = 200
DEF_STEP_REWARD = -1			# Temporary reward until Pash sends something from minisat
log = mp.get_logger()

class SatActiveEnv:
	def __init__(self, debug=False, server=None, **kwargs):
		self.debug = debug
		self.tail = deque([],LOG_SIZE)
		self.solver = None
		self.server = server
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
			self.solver.new(callback=thunk)
		if fname:
			f1 = CNF(fname)
			self.solver.append_formula(f1.clauses)

	def __callback(self, cl_label_arr, rows_arr, cols_arr, data_arr, reward):
		# adj_matrix = csr_matrix((data_arr, (rows_arr, cols_arr)))

		if not self.server:
			log.info('Running a test version of SatEnv')
			utility = cl_label_arr[:,3] # just return the lbd
			ipdb.set_trace()
			return utility
		else:
			return self.server.__callback(cl_label_arr, rows_arr, cols_arr, data_arr, reward)

	def step(self, action):
		return None

class SatEnvProxy(EnvBase):
	def __init__(self, queue_in, queue_out):
		self.queue_in = queue_in
		self.queue_out = queue_out

	def step(self, action):
		self.queue_out.put((EnvCommands.CMD_STEP,action))
		ack, rc = self.queue_in.get()
		assert ack==EnvCommands.ACK_STEP, 'Expected ACK_STEP'
		return rc

	def reset(self, fname):
		self.queue_out.put((EnvCommands.CMD_RESET,fname))
		ack, rc = self.queue_in.get()
		assert ack==EnvCommands.ACK_RESET, 'Expected ACK_RESET'
		return rc

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

	def handle_reset(self, fname):
		self.env.start_solver(fname)

	def run(self):
		print('Env {} on pid {}'.format(self.env.name, os.getpid()))
		set_proc_name(str.encode('{}_{}'.format(self.env.name,os.getpid())))
		while True:
			if self.cmd == EnvCommands.CMD_RESET:
				fname = self.current_fname
			else:
				self.cmd, fname = self.queue_in.get()
				assert self.cmd == EnvCommands.CMD_RESET, 'Unexpected command {}'.format(self.cmd)
			self.current_fname = None
			self.env.start_solver(fname)

			# This call does not return until the episodes is done. Messages are going to be exchanged until then through
			# the __callback method

			self.env.solver.solve()

			if self.cmd == EnvCommands.CMD_STEP:
				pass
				# We are here because the episode successfuly finished. We need to mark done and return the rewards to the client.
			elif self.cmd == EnvCommands.CMD_RESET:
				pass
				# We are here because the episode was aborted.


	def __callback(self, *args):
		if self.cmd == EnvCommands.CMD_RESET:
			# If this is the reply to a RESET add all existing (permanent) clauses
			args += self.env.solver.get_cl_arr()
			ack = EnvCommands.ACK_RESET
		elif self.cmd == EnvCommands.CMD_STEP:
			ack = EnvCommands.ACK_STEP
		else:
			assert True, 'Invalid last command detected'

		self.queue_out.put((ack,args))
		self.cmd, rc = self.queue_in.get()
		if self.cmd == EnvCommands.CMD_STEP:
			return rc
		elif self.cmd == EnvCommands.CMD_RESET:
			self.current_fname = rc
		elif self.cmd == EnvCommands.CMD_EXIT:


