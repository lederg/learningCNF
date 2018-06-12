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
    self.data = {}
    if fname is not None:
      self.load_file(fname)
    elif name is not None:
      self.name = name
    else:
      self.name = self.settings['name']

  def add_stat(self, key, stat):
    if type(stat) is not list:
      return self.add_stat(key,[stat])
    if key not in self.data.keys():
      self.data[key] = []
    self.data[key].extend(stat)

  def load_file(self, fname):
    with open(fname,'rb') as f:
      self.name, self.data = pickle.load(f)

  def save_file(self, fname=None):
    if not fname:
      fname = 'eps/{}.eps'.format(self.name)
    with open(fname,'wb') as f:
      pickle.dump((self.name,self.data),f)


class QbfCurriculumDataset(Dataset):
  def __init__(self, fnames=None, ed=None, max_variables=MAX_VARIABLES, max_clauses=MAX_CLAUSES):
    self.settings = CnfSettings()
    self.samples = []
    self.max_vars = max_variables
    self.max_clauses = max_clauses       
    self.stats_cover = False
    
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

    self.__weights_vector = self.recalc_weights()
    return self.__weights_vector


  def recalc_weights(self):
    stats = [self.ed.data[x] if x in self.ed.data.keys() else [] for x in self.get_files_list()]

    # If we don't have at least >1 attempts on all environments, try the ones that are still missing.
    if not self.stats_cover:
      if not [] in stats:
        self.stats_cover = True
      else:        
        self.__weights_vector = np.array([0 if (x and len(x) > 1) else 1 for x in stats])
        return
    
    moments = [[np.mean(x), np.std(x)] for x in stats]
    m1 = np.mean(moments[:,0])
    m2 = moments[:,1]

    return self.__weights_vector

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
