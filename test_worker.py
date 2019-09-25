import time
import random
from functional_worker_base import *

class TestWorker(FunctionalWorkerBase):
  def __init__(self, func_env, *args, **kwargs):    
    super(TestWorker, self).__init__(*args, **kwargs)

  def do_task(self, params):
  	n = random.randint(1,10)
  	print('Got task on index {}, sleeping {}'.format(self.index,n))
  	time.sleep(n)
  	return n