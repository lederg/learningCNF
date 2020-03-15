# Code in this file is copied and adapted from
# https://github.com/openai/evolution-strategies-starter.

import gym
import numpy as np
import torch
import ipdb
import ray
import ray.experimental.tf_utils
from ray.rllib.evaluation.sampler import _unbatch_tuple_actions
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.filter import get_filter
from custom_rllib_utils import *
from settings import *

def rollout(policy, env, timestep_limit=None, add_noise=False):
  """Do a rollout.

  If add_noise is True, the rollout will take noisy actions with
  noise drawn from that stream. Otherwise, no action noise will be added.
  """

  env_timestep_limit = policy.settings['max_step']+10
  timestep_limit = (env_timestep_limit if timestep_limit is None else min(
    timestep_limit, env_timestep_limit))
  rews = []
  t = 0
  observation = env.reset()
  for _ in range(timestep_limit or 999999):
    ac = policy.compute(observation, add_noise=add_noise)[0]
    observation, rew, done, _ = env.step(ac)
    rews.append(rew)
    t += 1
    if done:
      break
  rews = np.array(rews, dtype=np.float32)
  return rews, t


class TorchGNNPolicy:
  def __init__(self, action_space, obs_space, preprocessor,
             observation_filter, model_options, action_noise_std):        
    self.settings = CnfSettings()
    self.action_space = action_space
    self.action_noise_std = action_noise_std
    self.preprocessor = preprocessor
    self.observation_filter = get_filter(observation_filter,
                                         self.preprocessor.shape)
    self.dist_class = TorchCategoricalArgmax
    # Policy network.
    # dist_class, dist_dim = ModelCatalog.get_action_dist(
    #     self.action_space, model_options, dist_type="deterministic")
    self.model = ModelCatalog.get_model_v2(obs_space, action_space, action_space.n, model_options, "torch")
    # dist = dist_class(model.outputs, model)
    # self.sampler = dist.sample()

    # self.variables = ray.experimental.tf_utils.TensorFlowVariables(
    #     model.outputs, self.sess)

    self.num_params = np.sum([np.prod(x.shape) for x in self.model.parameters()])

  def compute(self, observation, add_noise=False, update=True):
    # observation = self.preprocessor.transform(observation)
    # observation = self.observation_filter(observation[None], update=update)
    logits, _ = self.model(observation, state=None, seq_lens=None)
    dist = self.dist_class(logits, self.model)
    action = dist.sample().numpy()
    if add_noise and isinstance(self.action_space, gym.spaces.Box):
      action += np.random.randn(*action.shape) * self.action_noise_std            
    return action

  def get_weights(self):
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
