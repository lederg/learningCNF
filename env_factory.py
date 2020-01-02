from IPython.core.debugger import Tracer
import time
import logging


from settings import *
from cadet_env import CadetEnv
from sat_env import *
from function_env import *
from empty_env import *
from episode_data import *
from formula_utils import *

class EnvFactory:
  def __init__(self, settings=None, **kwargs):
    if not settings:
      settings = CnfSettings()
    self.settings = settings

  def create_env(self, envtype=None, **kwargs):
    if not envtype:
      envtype = self.settings['solver']
    kwargs['settings'] = self.settings
    if envtype == 'cadet':
      return CadetEnv(**self.settings.hyperparameters)
    elif envtype == 'minisat':
      satenv = SatActiveEnv(**kwargs)
      satserv = SatEnvServer(satenv)
      log.info('Starting minisat server')
      satserv.start()
      return satserv.proxy(**kwargs)
    elif envtype == 'function':
      return FunctionEnv(**kwargs)
    elif envtype == 'empty':
      return EmptyEnv(**kwargs)

    else:
      log.error('Unknown env type: {}'.format(envtype))
      return None

def env_creator(env_config):
    settings = CnfSettings(env_config['settings'])
    settings.hyperparameters['cuda']=False
    if settings['preload_formulas']:
        settings.formula_cache.load_files(provider.items)  
    envfac = EnvFactory()
    pcls = eval(settings['episode_provider'])
    provider=pcls(env_config['formula_dir'])
    settings.formula_cache = FormulaCache()
    env = envfac.create_env(provider=provider, oracletype='lbd_threshold')
    return env
