import time
import ipdb
import cProfile
from enum import Enum
import logging

from settings import *
from functional_env import *


class WorkerCommands(Enum):
  CMD_EXIT = 1
  CMD_TASK = 2
  ACK_EXIT = 3
  ACK_TASK = 4

class FunctionalWorkerBase(mp.Process):
  def __init__(self, settings, task_queue_in, task_queue_out, index):    
    super(FunctionalWorkerBase, self).__init__()
    self.index = index
    self.name = 'func_worker_{}'.format(index)
    self.settings = settings
    self.task_queue_in = task_queue_in
    self.task_queue_out = task_queue_out

    self.logger = logging.getLogger('func_worker_{}'.format(index))
    self.logger.setLevel(eval(self.settings['loglevel']))    
    fh = logging.FileHandler('func_worker_{}.log'.format(index), mode='w')
    fh.setLevel(logging.DEBUG)
    self.logger.addHandler(fh)    

  def init_proc(self):
    set_proc_name(str.encode(self.name))
    np.random.seed(int(time.time())+abs(hash(self.name)) % 1000)
    torch.manual_seed(int(time.time())+abs(hash(self.name)) % 1000)
    self.settings.hyperparameters['cuda']=False         # No CUDA in the worker processes


  def run(self):
    self.init_proc()
    if self.settings['profiling']:
      cProfile.runctx('self.run_loop()', globals(), locals(), 'prof_{}.prof'.format(self.name))
    else:
      self.run_loop()


  def run_loop(self):
    while True:
      cmd, params, cookie = self.task_queue_in.get()
      if cmd == WorkerCommands.CMD_EXIT:
        self.task_queue_out.put((WorkerCommands.ACK_EXIT,None, cookie))
        return
      elif cmd == WorkerCommands.CMD_TASK:
        rc = self.do_task(params)
        self.task_queue_out.put((WorkerCommands.ACK_TASK,rc, cookie))        
      else:
        self.logger.error('Received unknown WorkerCommands!')
        return
        
  def do_task(self, params):
    pass
