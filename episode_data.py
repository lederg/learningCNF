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
	  else:
	  	self.name	= name

	def add_stat(self, key, stat):
		if type(stat) is not list:
			return self.add_stat(key,[stat])
		if key not in self.data.keys():
			self.data[key] = []
		self.data[key].extend(stat)

	def load_file(self, fname):
		with open(fname,'rb') as f:
			self.name, self.data = pickle.load(f)

	def save_file(self, fname):
		with open(fname,'wb') as f:
			pickle.save((self.name,self.data),f)