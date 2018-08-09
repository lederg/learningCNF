import numpy as np
import torch
import time
import ipdb
import pickle
from collections import namedtuple
from namedlist import namedlist


from cadet_env import *
from qbf_data import *
from settings import *
from utils import *
from rl_utils import *
from cadet_utils import *


class EpisodeData(object):
  def __init__(self, name=None, fname=None):
    self.settings = CnfSettings()
    self.data_ = {}
    if fname is not None:
      self.load_file(fname)
    elif name is not None:
      self.name = name
    else:
      self.name = self.settings['name']

  def add_stat(self, key, stat):
    if type(stat) is not list:
      return self.add_stat(key,[stat])
    if key not in self.data_.keys():
      self.data_[key] = []
    self.data_[key].extend(stat)

  def get_data(self):
    return self.data_

  def load_file(self, fname):
    with open(fname,'rb') as f:
      self.name, self.data_ = pickle.load(f)

  def save_file(self, fname=None):
    if not fname:
      fname = 'eps/{}.eps'.format(self.name)
    with open(fname,'wb') as f:
      pickle.dump((self.name,self.data_),f)


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
    d = self.ed.get_data()
    stats = [d[x] if x in d.keys() else [] for x in self.get_files_list()]
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
