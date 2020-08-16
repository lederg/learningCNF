# Code in this file is copied and adapted from
# https://github.com/openai/evolution-strategies-starter.
import ipdb
import gym
import numpy as np
import torch
import ray
import ray.experimental.tf_utils
from ray.rllib.evaluation.sampler import _unbatch_tuple_actions
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.filter import get_filter
from ray.util.sgd.utils import TimerStat
from custom_rllib_utils import *
from rllib_sharp_models import SharpModel
from rllib_sat_models import SatActivityModel
from clause_model import ClausePredictionModel
from settings import *
from graph_utils import *

def rollout(policy, env, fname, timestep_limit=None, add_noise=False):
  """Do a rollout.

  If add_noise is True, the rollout will take noisy actions with
  noise drawn from that stream. Otherwise, no action noise will be added.
  """
  timers = {k: TimerStat() for k in ["reset", "compute", "step"]}
  policy.model.curr_fname = fname           # This is an ugly hack for the extra high level information awareness
  env_timestep_limit = policy.settings['max_step']+10
  timestep_limit = (env_timestep_limit if timestep_limit is None else min(
    timestep_limit, env_timestep_limit))
  rews = []
  t = 0
  with timers['reset']:
    observation = env.reset(fname=fname)
  for _ in range(timestep_limit or 999999):
    with timers['compute']:
      ac = policy.compute(observation)[0]
    with timers['step']:
      observation, rew, done, _ = env.step(ac)
    rews.append(rew)
    t += 1
    if done:
      break
  rews = np.array(rews, dtype=np.float32)  
  return rews, t


class TorchGNNPolicy:
  def __init__(self, model, preprocessor, observation_filter):
    self.settings = CnfSettings()
    self.model = model
    self.num_params = np.sum([np.prod(x.shape) for x in self.model.parameters()])
    self.preprocessor = preprocessor
    self.observation_filter = get_filter(observation_filter,
                                         self.preprocessor.shape)

  def get_weights(self):
    return self.get_flat_weights()

  def get_flat_weights(self):
    pre_flat = {k: v.cpu() for k, v in self.model.state_dict().items()}
    rc = torch.cat([v.view(-1) for k,v in self.model.state_dict().items()],dim=0)
    return rc.numpy()

  def set_weights(self, weights):
    curr_weights = torch.from_numpy(weights)
    curr_dict = self.model.state_dict()
    for k,v in curr_dict.items():
      total = np.prod(v.shape)      
      curr_dict[k] = curr_weights[:total].view_as(v)
      curr_weights = curr_weights[total:]

    self.model.load_state_dict(curr_dict)

  def get_filter(self):
    return self.observation_filter

  def set_filter(self, observation_filter):
    self.observation_filter = observation_filter

class SharpPolicy(TorchGNNPolicy):
  def __init__(self, preprocessor,
             observation_filter):

    super(SharpPolicy, self).__init__(SharpModel(), preprocessor, observation_filter)
    self.settings = CnfSettings()
    self.time_hack = self.settings['sharp_time_random']

  def compute(self, observation):
    if self.settings['sharp_vanilla_policy']:
      return [-1]    
    if self.settings['sharp_random_policy']:
      return [np.random.randint(observation.ground.shape[0])]
    logits, _ = self.model(observation, state=None, seq_lens=None)
    l = logits.detach().numpy()
    if self.time_hack:
      indices = np.where(l.reshape(-1)==l.max())[0]
      action = [np.random.choice(indices)]
    else:
      dist = TorchCategoricalArgmax(logits, self.model)    
      action = dist.sample().numpy()
    return action

class SATPolicy(TorchGNNPolicy):
  def __init__(self, preprocessor,
             observation_filter):

    super(SATPolicy, self).__init__(SatActivityModel(), preprocessor, observation_filter)
    self.settings = CnfSettings()

# Returning logits for learnt clauses

  def compute(self, observation):
    input_dict = {
      'gss': observation.state,
      'graph': graph_from_arrays(observation.vlabels,observation.clabels,observation.adj_arrays)
    }
    scores, _ = self.model(input_dict, state=None, seq_lens=None)
    return scores.detach().numpy()
