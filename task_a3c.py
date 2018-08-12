import os
import os.path
import torch
# from torch.distributions import Categorical
import ipdb
import pdb
import random
import time

from multiprocessing.managers import BaseManager
import torch.multiprocessing as mp
from shared_adam import SharedAdam
from settings import *
from cadet_env import *
from rl_model import *
from new_policies import *
from qbf_data import *
from qbf_model import QbfClassifier
from utils import *
from rl_utils import *
from cadet_utils import *
from episode_reporter import *
from mp_episode_manager import *
from episode_data import *
import torch.nn.utils as tutils

settings = CnfSettings()

# Units of 30 seconds

UNIT_LENGTH = 30

# 2 = 1 Minute
REPORT_EVERY = 2
SAVE_EVERY = 20

init_lr = settings['init_lr']
desired_kl = settings['desired_kl']
stepsize = settings['init_lr']
curr_lr = init_lr
max_reroll = 0

class MyManager(BaseManager): pass
MyManager.register('EpisodeData',EpisodeData)

def a3c_main():
  if settings['do_not_run']:
    print('Not running. Printing settings instead:')
    print(settings.hyperparameters)
    return
  total_steps = 0
  global_steps = mp.Value('i', 0)
  global_grad_steps = mp.Value('i', 0)
  manager = MyManager()
  reporter = PGReporterServer(PGEpisodeReporter("{}/{}".format(settings['rl_log_dir'], log_name(settings)), settings, tensorboard=settings['report_tensorboard']))
  manager.start()
  reporter.start()
  ed = manager.EpisodeData(name=settings['name'], fname=settings['base_stats'])
  ds = QbfCurriculumDataset(fnames=settings['rl_train_data'], ed=ed)
  all_episode_files = ds.get_files_list()
  policy = create_policy()
  policy.share_memory()
  optimizer = SharedAdam(filter(lambda p: p.requires_grad, policy.parameters()), lr=stepsize)    
  max_iterations = len(ds)*100  
  num_steps = len(ds)*15000
  lr_schedule = PiecewiseSchedule([
                                       (0,                   init_lr),
                                       (num_steps / 10, init_lr),
                                       (num_steps / 5,  init_lr * 0.5),
                                       (num_steps / 3,  init_lr * 0.25),
                                       (num_steps / 2,  init_lr * 0.1),
                                  ],
                                  outside_value=init_lr * 0.02) 

  kl_schedule = PiecewiseSchedule([
                                       (0,                   desired_kl),
                                       (num_steps / 10, desired_kl),
                                       (num_steps / 5,  desired_kl * 0.5),
                                       (num_steps / 3,  desired_kl * 0.25),
                                       (num_steps / 2,  desired_kl * 0.1),
                                  ],
                                  outside_value=desired_kl * 0.02) 

  workers = [WorkerEnv(settings,policy,optimizer,ds,ed,global_steps, global_grad_steps, i, reporter=reporter.proxy()) for i in range(settings['parallelism'])]  
  print('Running for {} iterations..'.format(max_iterations))
  for w in workers:
    w.start()  
    # Change learning rate according to KL

  i = 0
  set_proc_name(str.encode('a3c_main'))

  while True:
    time.sleep(UNIT_LENGTH)
    if i % REPORT_EVERY == 0 and i>0:
      reporter.proxy().report_stats(global_steps.value, len(all_episode_files))
    if i % SAVE_EVERY == 0 and i>0:
      torch.save(policy.state_dict(),'%s/%s_step%d.model' % (settings['model_dir'],utils.log_name(settings), global_steps.value))
      ed.save_file()      
    i += 1
