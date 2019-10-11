from IPython.core.debugger import Tracer
import time
import logging

from settings import *
from cadet_env import CadetEnv
from sat_env import *
from function_env import *
from empty_env import *

class EnvFactory:
  def __init__(self, settings=None, **kwargs):
    if not settings:
      settings = CnfSettings()
    self.settings = settings

  def create_env(self, envtype=None, **kwargs):
    if not envtype:
      envtype = self.settings['solver']

    if envtype == 'cadet':
      return CadetEnv(**self.settings.hyperparameters)
    elif envtype == 'minisat':
      satenv = SatActiveEnv(**kwargs)
      satserv = SatEnvServer(satenv)
      log.info('Starting minisat server')
      satserv.start()
      return satserv.proxy()
    elif envtype == 'function':
      return FunctionEnv(**kwargs)
    elif envtype == 'empty':
      return EmptyEnv(**kwargs)

    else:
      log.error('Unknown env type: {}'.format(envtype))
      return None
