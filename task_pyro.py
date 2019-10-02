import os
import os.path
import torch
# from torch.distributions import Categorical
import ipdb
import pdb
import random
import utils
import time
import tracemalloc
import signal
import logging
import socket
import Pyro4
import Pyro4.core
import Pyro4.naming
import Pyro4.socketutil


from multiprocessing.managers import BaseManager
import torch.multiprocessing as mp
from shared_adam import SharedAdam
from settings import *
from rl_model import *
from new_policies import *
from qbf_data import *
from utils import *
from rl_utils import *
from episode_reporter import *
from mp_episode_manager import *
from episode_data import *
from formula_utils import *
from policy_factory import *
from node_sync import *
from node_worker import *

Pyro4.config.SERVERTYPE = "multiplex"
Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.SERIALIZERS_ACCEPTED = ["json", "marshal", "serpent", "pickle"]
Pyro4.config.POLLTIMEOUT = 3


settings = CnfSettings()
hostname = settings['pyro_host'] if settings['pyro_host'] else socket.gethostname()
my_ip = Pyro4.socketutil.getIpAddress(None, workaround127=True)
first_time = time.time()
last_time = first_time

# Units of 30 seconds

UNIT_LENGTH = 30

# 2 = 1 Minute
REPORT_EVERY = 2
SAVE_EVERY = 20
TEST_EVERY = settings['test_every']
init_lr = settings['init_lr']
desired_kl = settings['desired_kl']
curr_lr = init_lr
max_reroll = 0
round_num = 0

def handleSIGCHLD(a,b):
  os.waitpid(-1, os.WNOHANG)

class MyManager(BaseManager): 
  pass
# MyManager.register('EpisodeData',EpisodeData)
MyManager.register('wsync',WorkersSynchronizer)

def pyro_main():
  utils.seed_all(settings,'myname')
  # mp.set_start_method('forkserver')
  signal.signal(signal.SIGCHLD, handleSIGCHLD)
  if settings['do_not_run']:
    print('Not running. Printing settings instead:')
    print(settings.hyperparameters)
    return
  logger = utils.get_logger(settings, 'task_pyro', 'logs/{}_a3c_main_{}.log'.format(log_name(settings),my_ip))
  manager = MyManager()
  manager.start()
  main_node = False
  ns = None
  try:
    ns = Pyro4.locateNS(host=settings['pyro_host'], port=settings['pyro_port'])
    nsyncuri = ns.lookup("{}.node_sync".format(settings['pyro_name']))
    if not nsyncuri:
      main_node = True
  except:
    main_node = True

    # This lets certain models change their operation!
  settings.hyperparameters['main_node'] = main_node

  if main_node:    
    logger.info('Main node starting up for cluster {}, experiment {}, ip {}'.format(settings['pyro_name'],settings['name'],my_ip))
    pyrodaemon = Pyro4.core.Daemon(host=hostname)
    settings.pyrodaemon = pyrodaemon
    if ns is None:
      nameserverUri, nameserverDaemon, broadcastServer = Pyro4.naming.startNS(host=(hostname if settings['pyro_host'] else my_ip), port=settings['pyro_port'])
      assert broadcastServer is not None, "expect a broadcast server to be created"
      logger.info("got a Nameserver, uri={}".format(nameserverUri))
      ns = nameserverDaemon.nameserver
      pyrodaemon.combine(nameserverDaemon)
      pyrodaemon.combine(broadcastServer)
    settings.ns = ns
    reporter = PGEpisodeReporter("{}/{}".format(settings['rl_log_dir'], log_name(settings)), settings, tensorboard=settings['report_tensorboard'])
    node_sync = NodeSync(settings)
    reporter_uri = pyrodaemon.register(reporter)
    node_sync_uri = pyrodaemon.register(node_sync)
    ns.register("{}.reporter".format(settings['pyro_name']), reporter_uri)
    ns.register("{}.node_sync".format(settings['pyro_name']), node_sync_uri)
    # nameserverDaemon.nameserver.register("{}.reporter".format(settings['pyro_name']), reporter_uri)
    # nameserverDaemon.nameserver.register("{}.node_sync".format(settings['pyro_name']), node_sync_uri)

  else:
    logger.info('client node starting up for cluster {}, experiment {}, ip {}'.format(settings['pyro_name'],settings['name'],my_ip))
    reporter = Pyro4.core.Proxy("PYRONAME:{}.reporter".format(settings['pyro_name']))
    node_sync = Pyro4.core.Proxy("PYRONAME:{}.node_sync".format(settings['pyro_name']))  

  total_steps = 0
  wsync = manager.wsync()
  batch_sem = mp.Semaphore(settings['batch_sem_value'])
  ed = None
  if settings['sat_balanced_override']:
    logger.info('Overriding flag: Using BalancedEpisodeProvider')
    provider = BalancedEpisodeProvider.from_sat_dirs(settings['rl_train_data'],settings['sat_balanced_sat'], settings['sat_balanced_unsat'])
  else:
    ProviderClass = eval(settings['episode_provider'])
    provider = ProviderClass(settings['rl_train_data'])
  settings.formula_cache = FormulaCache()
  if settings['preload_formulas']:
    settings.formula_cache.load_files(provider.items)

  num_steps = 100000000
  curr_lr = init_lr
  lr_schedule = PiecewiseSchedule([
                                       (0,                   init_lr),
                                       (num_steps / 20, init_lr),
                                       (num_steps / 10, init_lr * 0.75),
                                       (num_steps / 5,  init_lr * 0.5),
                                       (num_steps / 3,  init_lr * 0.25),
                                       (num_steps / 2,  init_lr * 0.1),
                                  ],
                                  outside_value = init_lr * 0.02) 

  kl_schedule = PiecewiseSchedule([
                                       (0,                   desired_kl),
                                       (num_steps / 10, desired_kl),
                                       (num_steps / 5,  desired_kl * 0.5),
                                       (num_steps / 3,  desired_kl * 0.25),
                                       (num_steps / 2,  desired_kl * 0.1),
                                  ],
                                  outside_value=desired_kl * 0.02) 

  mp.set_sharing_strategy('file_system')
  workers = {}
  if settings['parallelism']=='auto':
    cpunum = psutil.cpu_count() - 2
    mem = psutil.virtual_memory().total / (2**20) # (in MBs)
    cap = settings['memory_cap']
    settings['parallelism'] = min(cpunum, int(mem/cap))
    self.logger.info('auto-parallelism set to {}'.format(settings['parallelism']))
  for _ in range(settings['parallelism']):
    wnum = node_sync.get_worker_num()
    # workers[wnum] = WorkerEnv(settings, provider, ed, wnum, wsync, batch_sem, init_model=None)
    workers[wnum] = NodeWorker(settings, provider, wnum)
  print('Running with {} workers...'.format(len(workers)))
  for _,w in workers.items():
    w.start()  

  pval = None
  main_proc = psutil.Process(os.getpid())
  set_proc_name(str.encode('a3c_main'))

  def common_loop_tasks():
    global round_num

    # Restart worker processes cooperatively
    while wsync.get_total() > 0:
      w = wsync.pop()
      j = w[0]
      logger.info('restarting worker {}'.format(j))
      # workers[j] = WorkerEnv(settings, provider, ed, j, wsync, batch_sem, init_model=w[1])
      workers[j] = NodeWorker(settings,provider,j)
      workers[j].start()

    # Restart worker processes uncooperatively
    
    total_workers_num = 0    
    for j,w in workers.items():
      worker_running = True
      if psutil.pid_exists(w.pid):        
        worker_proc = psutil.Process(w.pid)        
        logger.info('Worker {} (pid: {}) status: {}, on host {}'.format(j,w.pid,worker_proc.status(),my_ip))
        total_workers_num += 1
        if not worker_proc.is_running() or worker_proc.status() is 'zombie':
          worker_running = False
      else:
        worker_running = False
      if not worker_running:
        logger.info('restarting worker (probably killed by oom) {}'.format(j))
        # env_pid = w.envstr.env.server_pid
        if settings['is_sat']:
          env_pid = w.trainer.interactor.envstr.env.server_pid
          if psutil.pid_exists(env_pid):
            logger.info('defunct SatEnv on pid {}, killing it'.format(env_pid))
            os.kill(env_pid, signal.SIGTERM)
        # workers[j] = WorkerEnv(settings, provider, ed, j, wsync, batch_sem)
        workers[j] = NodeWorker(settings, provider, j)
        workers[j].start()
    logger.info('Total number of running workers: {}'.format(total_workers_num))
    try:
      total_mem = main_proc.memory_info().rss / float(2 ** 20)
      children = main_proc.children(recursive=True)
      for child in children:
        child_mem = child.memory_info().rss / float(2 ** 20)
        total_mem += child_mem
        logger.info('Child pid is {}, name is {}, mem is {}'.format(child.pid, child.name(), child_mem))
      logger.info('Total memory on host {} is {}'.format(my_ip, total_mem))
    except:       # A child could already be dead due to a race. Just ignore it this round.
      pass

    round_num += 1

  def main_node_tasks():
    global last_time
    curr_time = time.time()
    if curr_time-last_time > UNIT_LENGTH:
      logger.info('Round {}'.format(round_num))
      last_time = curr_time
      common_loop_tasks()
      gsteps = node_sync.g_steps
      policy = node_sync.gmodel
      if round_num % REPORT_EVERY == 0 and round_num>0:
        reporter.report_stats(gsteps, provider.get_total(), pval)
        eps = node_sync.g_episodes
        logger.info('Average number of simulated episodes per time unit: {} ({}/{})'.format(eps/round_num,eps,round_num))
      if round_num % SAVE_EVERY == 0 and round_num>0:
        torch.save(policy.state_dict(),'%s/%s_step%d.model' % (settings['model_dir'],utils.log_name(settings), gsteps))
        if ed is not None:
          ed.save_file()
      if round_num % TEST_EVERY == 0 and round_num>0:      
        em.test_envs(settings['rl_test_data'], policy, iters=1, training=False)
      if settings['rl_decay']:
        new_lr = lr_schedule.value(gsteps)
        if new_lr != curr_lr:
          node_sync.update_lr(new_lr)
          print('setting new learning rate to {}'.format(new_lr))
          curr_lr = new_lr

    return True

  if main_node:
    pyrodaemon.requestLoop(main_node_tasks)

    nameserverDaemon.close()
    broadcastServer.close()
    pyrodaemon.close()
  else:    
    while True:
      time.sleep(UNIT_LENGTH)
      common_loop_tasks()

