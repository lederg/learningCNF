from IPython.core.debugger import Tracer
import utils
import random

from namedlist import namedlist
from settings import *
from envbase import *
from dispatcher import *
from rl_types import *

class EmptyEnv(EnvBase):
  EnvObservation = namedlist('EnvObservation', ['state', 'reward', 'done'], default=None)
  """docstring for EmptyEnv"""
  def __init__(self, settings=None, func=None, init_x=None, **kwargs):
    super(EmptyEnv, self).__init__()
    self.settings = settings if settings else CnfSettings()
    self.logger = utils.get_logger(self.settings, 'EmptyEnv')

  def step(self, action):
    return self.EnvObservation(0,random.uniform(-2,2), random.random() < 0.08)

  def reset(self):
    return self.EnvObservation(0,0,False)

  def new_episode(self, *args, **kwargs):    
    return self.reset()

  def process_observation(self, last_obs, env_obs, settings=None):
    return self.EnvObservation(self.settings.FloatTensor([env_obs.state]),env_obs.reward,env_obs.done)
