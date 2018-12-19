import os
import multiprocessing as mp
from enum import Enum
from utils import set_proc_name

class EnvCommands(Enum):
	CMD_RESET = 1
	CMD_STEP = 2
	CMD_EXIT = 3
	ACK_RESET = 4
	ACK_STEP = 5
	ACK_EXIT = 6

class EnvBase(object):
	def __init__(self):
		pass

	def step(self,action):
		pass

	def reset(self,fname):
		pass

	def restart_env(self, timeout=0):
		pass

	@property
	def name(self):
		return 'EnvBase'

