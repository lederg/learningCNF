from config import *
from settings import *
import os
import numpy as np
import pickle
import itertools
import argparse
import pandas as pd
import plotly.express as px
import torch.multiprocessing as mp
import ray
from IPython.core.debugger import Tracer
from collections import Counter
from matplotlib import pyplot as plt
from ray.tune.registry import register_env
import ray.rllib.agents.ppo as ppo
from ray.rllib.agents import a3c
from ray.rllib.agents.a3c.a3c_torch_policy import *
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.evaluation.rollout_worker import *
from ray.rllib.evaluation.worker_set import *
from ray.tune.logger import pretty_print

from dispatcher import *
from episode_data import *
from policy_factory import *
from test_envs import *
from rllib_sat_models import *


# In[2]:
steps_counter = 0
settings = CnfSettings(cfg())


def get_settings_from_file(fname):
	conf = load_config_from_file(fname)
	for (k,v) in conf.items():
		settings.hyperparameters[k]=v

def my_postprocess(info):
	global steps_counter
	settings = CnfSettings()
	episode = info["episode"]
	batch = info["post_batch"]
	episode.custom_metrics["length"] = batch.count
	steps_counter += batch.count
	# steps_counter += info['post_batch']['obs'].shape[0]
	if steps_counter > int(settings['min_timesteps_per_batch']):
		steps_counter = 0
		ObserverDispatcher().notify('new_batch')

class RLLibTrainer():
	def __init__(self):
		self.settings = CnfSettings()		
		self.clock = GlobalTick()
		self.logger = utils.get_logger(self.settings, 'rllib_trainer', 'logs/{}_rllib_trainer.log'.format(log_name(self.settings)))
		self.settings.formula_cache = FormulaCache()
		self.training_steps = self.settings['training_steps']        
		register_env("sat_env", env_creator)
		ModelCatalog.register_custom_model("sat_model", SatThresholdModel)
		ray.init()

	def main(self):    
		if self.settings['do_not_run']:
			print('Not running. Printing settings instead:')
			print(self.settings.hyperparameters)
			return
		# config = ppo.DEFAULT_CONFIG.copy()
		config = a3c.DEFAULT_CONFIG.copy()
		config["num_gpus"] = 0
		config["num_workers"] = int(self.settings['parallelism'])
		config["eager"] = False
		config["sample_async"] = False
		config["batch_mode"]='complete_episodes'
		config["sample_batch_size"]=int(self.settings['min_timesteps_per_batch'])
		config["train_batch_size"]=int(self.settings['min_timesteps_per_batch'])
		config['gamma'] = float(self.settings['gamma'])
		config['lr'] = float(self.settings['init_lr'])
		if settings['use_seed']:
			config['seed'] = int(settings['use_seed'])
		config["callbacks"] = {'on_postprocess_traj': my_postprocess}
		config["env_config"]={'settings': settings.hyperparameters.copy(), 'formula_dir': self.settings['rl_train_data']}
		trainer = a3c.A3CTrainer(config=config, env="sat_env")
		print('Running for {} iterations..'.format(self.training_steps))
		for i in range(self.training_steps):
			result = trainer.train()
			print(pretty_print(result))
			if i % 100 == 0:
				checkpoint = trainer.save()
				print("checkpoint saved at", checkpoint)


def rllib_main():	
	rllib_trainer = RLLibTrainer()
	rllib_trainer.main()

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Process some params.')
	parser.add_argument('params', metavar='N', type=str, nargs='*', help='an integer for the accumulator')
	parser.add_argument('-s', '--settings', type=str, help='settings file') 
	args = parser.parse_args()
	get_settings_from_file(args.settings)
	for param in args.params:
		k, v = param.split('=')
		settings.hyperparameters[k]=v
	rllib_main()



# config["num_envs"]=1
# if settings['preload_formulas']:
#     settings.formula_cache.load_files(provider.items)  
# settings.hyperparameters['loglevel']='logging.INFO'
# settings.hyperparameters['sat_min_reward']=-100
# settings.hyperparameters['max_step']=300
# settings.hyperparameters['min_timesteps_per_batch']=100


# config["model"] = {"custom_model": "sat_model"}
# config["use_pytorch"] = True


# Can optionally call trainer.restore(path) to load a checkpoint.

# # w = RolloutWorker(env_creator=env_creator, policy=A3CTorchPolicy, batch_mode='complete_episodes', policy_config=config)
# workers = WorkerSet(
#     policy=A3CTFPolicy,
#     env_creator=env_creator,
#     num_workers=2, 
#     trainer_config=config
#     )

# # In[7]:

# a = ray.get(workers.remote_workers()[0].sample.remote())
# print('total steps: {}'.format(len(a['obs'])))
# b = ray.get(workers.remote_workers()[0].sample.remote())
# print('total steps: {}'.format(len(b['obs'])))