import os
import numpy as np
import torch
import torch.multiprocessing as mp
import time
import ipdb
import pickle
import itertools
from collections import namedtuple
from namedlist import namedlist


from cadet_env import *
from qbf_data import *
from settings import *
from utils import *
from rl_utils import *
from cadet_utils import *

ED_RECALC_THRESHOLD = 50

class EpisodeData(object):
  def __init__(self, name=None, fname=None):
    self.settings = CnfSettings()
    self.stats_cover = False
    self.flist = None
    self.data_ = {}    
    self.total_stats = 0
    self.last_recalc = 0
    self.__weights_vector = None
    if fname is not None:
      self.load_file(fname)
    elif name is not None:
      self.name = name
    else:
      self.name = self.settings['name']
    if self.settings['mp']:
      print('EpisodeData: MultiProcessing: {} (pid: {})'.format(self.settings['mp'],os.getpid()))
      set_proc_name(str.encode('a3c_ed'))

  def ed_add_stat(self, key, stat):
    if type(stat) is not list:
      return self.ed_add_stat(key,[stat])
    if key not in self.data_.keys():
      self.data_[key] = []
    self.data_[key].extend(stat)
    self.total_stats += len(stat)

  def set_file_list(self,flist):
    self.flist = flist

  def load_file(self, fname):
    with open(fname,'rb') as f:
      self.name, self.data_ = pickle.load(f)
    self.total_stats = sum([len(x) for x in self.data_.values()])

  def save_file(self, fname=None):
    if not fname:
      fname = 'eps/{}.eps'.format(self.name)
    with open(fname,'wb') as f:
      pickle.dump((self.name,self.data_),f)

  def recalc_weights(self):
    if not self.flist:
      return None    
    if self.total_stats - self.last_recalc < ED_RECALC_THRESHOLD and self.__weights_vector is not None:
      return self.__weights_vector
    stats = [self.data_[x] if x in self.data_.keys() else [] for x in self.flist]
    not_seen = np.array([0 if (x and len(x) > 1) else 1 for x in stats])
    if self.stats_cover or not not_seen.any():
      if not self.stats_cover:
        print('Covered dataset!')
        self.stats_cover = True
      z = [list(zip(*x)) for x in stats]
      steps, rewards = zip(*z)
      m1, m2 = np.array([[np.mean(x[-60:]), np.std(x[-60:])] for x in steps]).transpose()
      m2 = (m2 - m2.mean()) / (m2.std() + float(np.finfo(np.float32).eps))
      m2 = m2 - m2.min() + 1      
      rc = (self.settings['episode_cutoff'] - m1).clip(0)*m2
    # If we don't have at least >1 attempts on all environments, try the ones that are still missing.
    else:      
      print('Number of unseen formulas is {}'.format(not_seen.sum()))
      rc = not_seen
    
    self.__weights_vector = epsilonize(rc / rc.sum(),0.01)
    self.last_recalc = self.total_stats
    # print('Recalculating formulas weights in EpisodeData, length of vector is {}'.format(len(self.__weights_vector)))
    # print('Largest value is {}'.format(self.__weights_vector.max()))

    return self.__weights_vector

class QbfCurriculumDataset(Dataset):
  def __init__(self, fnames=None, ed=None, max_variables=MAX_VARIABLES, max_clauses=MAX_CLAUSES):
    self.settings = CnfSettings()
    self.samples = []
    self.max_vars = max_variables
    self.max_clauses = max_clauses       
    self.stats_cover = False
    self.use_cl = self.settings['use_curriculum']
    
    self.ed = ed if ed else EpisodeData()    
    if fnames:
      if type(fnames) is list:
        self.load_files(fnames)
      else:
        self.load_files([fnames])

    self.ed.set_file_list(self.get_files_list())

  def load_dir(self, directory):
    self.load_files([join(directory, f) for f in listdir(directory)])

  def load_files(self, files):
    only_files = [x for x in files if os.path.isfile(x)]
    only_dirs = [x for x in files if os.path.isdir(x)]
    for x in only_dirs:
      self.load_dir(x)
    rc = map(f2qbf,only_files)
    self.samples.extend([x for x in rc if x and x.num_vars <= self.max_vars and x.num_clauses < self.max_clauses\
                                                         and x.num_clauses > 0 and x.num_vars > 0])
    
    self.ed.set_file_list(self.get_files_list())
    print ('Just set the file list from pid ({}):'.format(os.getpid()))
    print(self.get_files_list()[:10])
    try:
      del self.__weights_vector
    except:
      pass
    return len(self.samples)

  @property
  def weights_vector(self):
    try:
      return self.__weights_vector
    except:
      pass

    self.__weights_vector = self.recalc_weights() if self.use_cl else np.ones(len(self.samples)) / len(self.samples)
    return self.__weights_vector


  def recalc_weights(self):
    if not self.use_cl:      
      return self.__weights_vector
    
    self.__weights_vector = self.ed.recalc_weights()
    # print('recalculated weights vector in pid ({})'.format(os.getpid()))
    # print(self.get_files_list()[:10])

    return self.__weights_vector

  def weighted_sample(self, n=1):
    choices = np.random.choice(len(self.samples),size=n,p=self.weights_vector)
    return [self.samples[i].qcnf['fname'] for i in choices]

  def load_file(self,fname):
    if os.path.isdir(fname):
      self.load_dir(fname)
    else:
      self.load_files([fname])

  def get_files_list(self):
    return [x.qcnf['fname'] for x in self.samples]

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    return self.samples[idx].as_np_dict()


class AbstractEpisodeProvider(object):
  def __init__(self,flist):    
    self.items = self.load_files(flist)
    self.settings = CnfSettings()

  def load_dir(self, directory):
    return self.load_files([join(directory, f) for f in listdir(directory)])

  def load_files(self, files):
    if type(files) is not list:
      files = [files]
    only_files = [x for x in files if os.path.isfile(x)]
    only_dirs = [x for x in files if os.path.isdir(x)]
    return only_files if not only_dirs else only_files + list(itertools.chain.from_iterable([self.load_dir(x) for x in only_dirs]))

  def reset(self, **kwargs):
    pass

  def sample(self):
    pass

  def get_next(self):
    pass

  def get_total(self):
    return len(self.items)

  def __iter__(self):
    return self.get_next()

  def __len__(self):
    return self.get_total()

class UniformEpisodeProvider(AbstractEpisodeProvider):
  def __init__(self,ds):
    super(UniformEpisodeProvider, self).__init__(ds) 
    self.current = self.sample()

  def sample(self):
    return np.random.choice(self.items)

  def reset(self, **kwargs):
    self.current = self.sample()

  def get_next(self):
    return self.current

  def __iter__(self):
    return self.get_next()

class OnePassProvider(AbstractEpisodeProvider):
  def __init__(self,ds):
    super(OnePassProvider, self).__init__(ds) 
    print('items: {}'.format(len(self.items)))
    self.current = 0

  def sample(self):    
    if self.current >= len(self.items):
      print('OnePassProvider: Finished ({})'.format(self.current))
      return None
    else:
      return self.items[self.current]

  def reset(self, **kwargs):
    self.current += 1

  def get_next(self):
    return self.sample()

  def __iter__(self):
    return self.get_next()

    
class RandomEpisodeProvider(AbstractEpisodeProvider):
  def sample(self):
    return np.random.choice(self.items)
  def get_next(self):
    return self.sample()

