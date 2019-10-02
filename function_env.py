import ipdb
import utils

from settings import *
from envbase import *
from rl_types import *

class FunctionEnv(EnvBase):
	"""docstring for FunctionEnv"""
	def __init__(self, settings=None, func=None, init_x=0):
		super(FunctionEnv, self).__init__()
		self.settings = settings if settings else CnfSettings()
    self.logger = utils.get_logger(self.settings, 'FunctionEnv')
    self.init_x = init_x    
    self.max_step=20
    self.reset()
    if func is None:
    	self.func = lambda x: x**x
    else:
    	self.func = func

  def step(self, action):
  	reward = self.func(self.x+action) - self.func(self.x)
  	env.rewards.append(reward)
  	self.x += action
  	self.local_step += 1
  	return self.x

	def reset(self):
		self.x = self.init_x
		self.local_step = 0
		self.rewards = 0

		return self.x

  def new_episode(self, *args, **kwargs):    
    return self.reset()

	def process_observation(self, last_obs, env_obs, settings=None):
		return env_obs


