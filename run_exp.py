import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from model import *
from datautils import *
import utils
import time
import numpy as np
import ipdb
import pdb
from tensorboard_logger import configure, log_value
from sacred import Experiment
from training import *

# EX_NAME = 'trenery_4'

# ex = Experiment(EX_NAME)

@ex.config
def cfg():
	hyperparams = {
		'exp_name': EX_NAME,
		'time': int(time.time()),
	    'embedding_dim': 6,
	    'ground_dim': 3,
	    'max_clauses': 3, 
	    'max_variables': 3, 
	    'num_ground_variables': 3, 
	    'data_mode': DataMode.SAT,
	    'dataset': 'boolean8',
	    'model_dir': 'saved_models',
	    'max_iters': 5,
	    'batch_size': 4,
	    'val_size': 100, 
	    # 'classifier_type': 'GraphLevelClassifier',
	    'classifier_type': 'EqClassifier',
	    'combinator_type': 'SymmetricSumCombine',	    
	    'ground_combinator_type': 'DummyGroundCombinator',	    
	    'gru_bias': False,
	    'use_ground': False,
	    'split': False,
	    'cuda': False
	}

	def_settings = CnfSettings(hyperparams)
	data_mode = hyperparams['data_mode']


	DS_TRAIN_FILE = 'expressions-synthetic/split/%s-trainset.json' % hyperparams['dataset']
	DS_VALIDATION_FILE = 'expressions-synthetic/split/%s-validationset.json' % hyperparams['dataset']
	DS_TEST_FILE = 'expressions-synthetic/split/%s-testset.json' % hyperparams['dataset']


@ex.automain
def main(DS_TRAIN_FILE, DS_VALIDATION_FILE, data_mode):
	ds1 = CnfDataset(DS_TRAIN_FILE,4000,mode=data_mode)
	ds2 = CnfDataset(DS_VALIDATION_FILE,1000,mode=data_mode)
	print('Classes from validation set:')
	print(ds2.labels)
	train(ds1,ds2)
