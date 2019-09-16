import os
import numpy as np
import torch.multiprocessing as mp
from torch.multiprocessing import Manager
from tensorboard_logger import configure, log_value
import utils
from utils import *
from settings import *
import ipdb
import shelve
import logging
import Pyro4


DEF_WINDOW = 100
CMD_EXIT = 1
CMD_REPORT = 2
CMD_ADDSTAT = 3


class AbstractReporter(object):
  def __init__(self, fname, settings, tensorboard=False):
    self.tensorboard = tensorboard
    self.settings = settings
    self.log_value = log_value
    self.shelf_name = fname+'.shelf'
    self.tb_configured = False
    self.fname = fname
  
class SPReporter(AbstractReporter):
  def __init__(self, fname, settings, tensorboard=False):
    super(SPReporter, self).__init__(fname, settings, tensorboard)
    self.stats = []
    self.stats_dict = {}
    self.ids_to_log = set()
    self.settings = settings
    self.report_window = self.settings['report_window']
    self.short_window = self.settings['short_window']
    self.report_last_envs = self.settings['report_last_envs']
    self.logger = utils.get_logger(self.settings, 'episode_reporter', 'logs/{}_reporter.log'.format(log_name(self.settings)))

class MPReporter(AbstractReporter):
  def __init__(self, fname, settings, tensorboard=False):
    super(MPReporter, self).__init__(fname, settings, tensorboard)
    self.manager = Manager()
    self.manager.start()
    self.stats = self.manager.list()
    self.stats_dict = self.manager.dict()
    print('Created shared objects in reporter')
    # self.ids_to_log = set()
    

class PGReporterServer(mp.Process):
  def __init__(self, reporter):
    super(PGReporterServer, self).__init__()
    self.reporter = reporter
    self.queue = mp.Queue()

  def proxy(self):
    return PGProxyReporter(self.queue)

  def run(self):
    obj = None
    self.reporter.logger.info('Reporter on pid {}'.format(os.getpid()))
    set_proc_name(str.encode('a3c_reporter'))
    while True:
      cmd, obj = self.queue.get()
      if cmd == CMD_ADDSTAT:
        self.reporter.add_stat(*obj)
      elif cmd == CMD_REPORT:
        print('Reporting Stats!')
        self.reporter.report_stats(*obj)
      elif cmd == CMD_EXIT:
        print('Received CMD_EXIT')
        break

  

class PGProxyReporter(AbstractReporter):  
  def __init__(self, queue):    
    # super(PGProxyReporter, self).__init__()
    self.queue = queue

  def add_stat(self, *args):
    self.queue.put((CMD_ADDSTAT,args))

  def report_stats(self, *args):
    self.queue.put((CMD_REPORT,args))

@Pyro4.expose
class PGEpisodeReporter(SPReporter):  
  def add_stat(self, env_id, steps, reward, entropy, total_steps):
    self.stats = self.stats+[[env_id,steps,reward, entropy]]
    if not env_id in self.stats_dict.keys():
      self.stats_dict[env_id] = []

    # Add entropy as well     
    self.stats_dict[env_id] = self.stats_dict[env_id] + [(steps, reward, entropy)]

  def __len__(self):
    return len(self.stats)

  def log_env(self,env):
    pass

  def report_stats(self, total_steps, total_envs, pval=None):
    if not self.stats:
      self.logger.info('report_stats: No episodes registered yet, skipping')
      return    
    _, steps, rewards, ents = zip(*self.stats)    
    # self.report_window = total_envs*60
    self.logger.info('Total episodes so far: %d' % len(steps))
    self.logger.info('Total steps learned from so far: %d' % sum(steps))
    self.logger.info('Total rewards so far: %f' % sum(rewards))
    # ipdb.set_trace()
    totals = sorted([(k,len(val), *zip(*val)) for k, val in self.stats_dict.items()],key=lambda x: -x[1])
    if self.settings['report_uniform']:
      all_stats = [x[1] for x in self.stats_dict.items()]
      windowed_stats = []
      for x in all_stats:
        windowed_stats.append(np.array(x[-self.short_window:]))
      # windowed_stats = [np.array(x[-100:]) for x in all_stats]
      mean_rewards = [x.mean(axis=0)[1] for x in windowed_stats]
      mean_steps = [x.mean(axis=0)[0] for x in windowed_stats]
      self.logger.info('Mean steps for the last {} instances per episode: {}'.format(self.short_window,np.mean(mean_steps)))
      self.logger.info('Mean reward for the last {} instances per episode: {}'.format(self.short_window,np.mean(mean_rewards)))
    else:     
      self.logger.info('Mean steps for the last {} episodes: {}'.format(self.report_window,np.mean(steps[-self.report_window:])))
      self.logger.info('Mean reward for the last {} episodes: {}'.format(self.report_window,np.mean(rewards[-self.report_window:])))

    # if sum(steps)+1000 < total_steps:     # Not all episodes are actually used (parallelism/on-policy pg)
    #   total_steps = sum(steps)

    self.logger.info('Data for the {} most common envs:'.format(self.report_last_envs))
    for vals in totals[:self.report_last_envs]:
      s = vals[2][-self.short_window:]
      r = vals[3][-self.short_window:]
      self.logger.info('Env {} appeared {} times, with moving (100) reward mean/std {}/{}:'.format(vals[0], vals[1], np.mean(r), np.std(r)))
      self.logger.info('Env {} appeared {} times, with moving (100) steps mean/std {}/{}:'.format(vals[0], vals[1], np.mean(s), np.std(s)))
      self.logger.info(r)
      self.logger.info(s)
      self.logger.info('\n\n')
    

    if self.settings['rl_shelve_it']:
      with shelve.open(self.shelf_name) as db:
        db['stats'] = self.stats
        db['stats_dict'] = self.stats_dict

    if self.tensorboard:
      if not self.tb_configured:
        print('Configuring tensorboard for the first time with pid {}'.format(os.getpid()))
        configure(self.fname, flush_secs=5)
        self.tb_configured = True

      if self.settings['report_uniform']:
        log_value('Mean steps for last {} episodes'.format(self.short_window), np.mean(mean_steps), total_steps)
        log_value('Mean reward for last {} episodes'.format(self.short_window), np.mean(mean_rewards), total_steps)
      else:
        log_value('Mean steps for last {} episodes'.format(self.report_window), np.mean(steps[-self.report_window:]), total_steps)
        log_value('Mean reward for last {} episodes'.format(self.report_window), np.mean(rewards[-self.report_window:]), total_steps)
      if pval:
        log_value('pval', pval, total_steps)

      print('Total steps are {}'.format(total_steps))
    return len(steps)
    # for id in self.ids_to_log:
    #   try:
    #     stats = self.stats_dict[id][-DEF_WINDOW:]
    #   except:
    #     continue
    #   steps, rewards, entropy = zip(*stats)
    #   log_value('env {} #steps'.format(id), np.mean(steps), total_steps)
    #   log_value('env {} entropy'.format(id), np.mean(entropy), total_steps)
    #   log_value('env {} rewards'.format(id), np.mean(rewards), total_steps)



class DqnEpisodeReporter(SPReporter): 
  def log_env(self, id):
    if type(id) is list:
      for x in id:
        self.log_env(x)
    else:
      self.ids_to_log.add(id)

  def add_stat(self, env_id, steps, reward, total_steps):
    self.stats.append([env_id,steps,reward])
    if not env_id in self.stats_dict.keys():
      self.stats_dict[env_id] = []

    # Add entropy as well     
    self.stats_dict[env_id].append((steps, reward))
    self.total_steps = total_steps

  def __len__(self):
    return len(self.stats)


  def report_stats(self, total_steps):
    _, steps, rewards = zip(*self.stats)
    print('[{}] Total episodes so far: {}'.format(self.total_steps, len(steps)))
    print('[{}] Total steps so far: {}'.format(self.total_steps, sum(steps)))
    print('[{}] Total rewards so far: {}'.format(self.total_steps, sum(rewards)))
    totals = sorted([(k,len(val), *zip(*val)) for k, val in self.stats_dict.items()],key=lambda x: -x[1])

    print('Data for the 10 most common envs:')
    for vals in totals[:10]:
      s = vals[2][-DEF_WINDOW:]
      r = vals[3][-DEF_WINDOW:]
      print('Env %s appeared %d times, with moving (100) mean/std %f/%f:' % (str(vals[0]), vals[1], np.mean(r), np.std(r)))
      print(r)
      print(s)
      print('\n\n')

    if not self.tensorboard:
      return
    for id in self.ids_to_log:
      try:
        stats = self.stats_dict[id][-DEF_WINDOW:]
      except:
        continue
      steps, rewards = zip(*stats)
      log_value('env {} #steps'.format(id), np.mean(steps), self.total_steps)
      log_value('env {} rewards'.format(id), np.mean(rewards), self.total_steps)
