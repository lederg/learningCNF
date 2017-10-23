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
	    # 'max_clauses': 12, 
	    'max_variables': 3, 
	    'num_ground_variables': 3, 
	    'data_mode': DataMode.SAT,
	    'dataset': 'boolean10',
	    'model_dir': 'saved_models',
	    # 'base_model': 'saved_models/run_bigsat_50_4_nc2_bs40_ed4_iters8__1508199570_epoch200.model',
	    'base_model': None,
	    'base_mode': BaseMode.ALL,
	    'max_iters': 8,
	    'batch_size': 64,
	    'val_size': 100, 
	    'threshold': 10,
	    'init_lr': 0.0011,
	    # 'init_lr': 0.0004,
	    'decay_lr': 0.055,
	    'decay_num_epochs': 6,
	    'cosine_margin': 0,
	    # 'classifier_type': 'BatchGraphLevelClassifier',
	    # 'classifier_type': 'BatchEqClassifier',
	    'classifier_type': 'TopLevelClassifier',
	    'combinator_type': 'SymmetricSumCombine',	    
	    'ground_combinator_type': 'DummyGroundCombinator',	    
	    # 'ground_combinator_type': 'GroundCombinator',	    
	    'encoder_type': 'BatchEncoder',	    
	    'embedder_type': 'TopVarEmbedder',	    
	    # 'embedder_type': 'GraphEmbedder',	    
	    # 'gru_bias': False,
	    'gru_bias': True,
	    'use_ground': False,
	    'moving_ground': True, 
	    'split': False,
	    # 'cuda': True, 
	    'cuda': False,
	    'reset_on_save': False
	}

	def_settings = CnfSettings(hyperparams)
	data_mode = hyperparams['data_mode']
	threshold = hyperparams['threshold']


	DS_TRAIN_FILE = 'expressions-synthetic/split/%s-trainset.json' % hyperparams['dataset']
	DS_VALIDATION_FILE = 'expressions-synthetic/split/%s-validationset.json' % hyperparams['dataset']
	DS_TEST_FILE = 'expressions-synthetic/split/%s-testset.json' % hyperparams['dataset']


@ex.automain
def main(DS_TRAIN_FILE, DS_VALIDATION_FILE, DS_TEST_FILE, data_mode, threshold):
	# ds1 = CnfDataset.from_eqparser(DS_TRAIN_FILE,mode=data_mode, threshold=threshold)
	# ds2 = CnfDataset.from_eqparser(DS_VALIDATION_FILE, threshold=0, ref_dataset=ds1, mode=data_mode)
	ns1 = CnfDataset.from_dimacs('data/train_big_3/sat/', 'data/train_big_3/unsat/', max_size=200)
	ns2 = CnfDataset.from_dimacs('data/validation_3/sat/', 'data/validation_3/unsat/', max_size=200)
	# ds1 = CnfDataset(DS_TRAIN_FILE,threshold,mode=data_mode, num_max_clauses=12)
	# ds2 = CnfDataset(DS_VALIDATION_FILE, threshold, ref_dataset=ds1, mode=data_mode, num_max_clauses=12)
	# ds3 = CnfDataset(DS_TEST_FILE, threshold, ref_dataset=ds1, mode=data_mode)
	# print('Classes from validation set:')
	# print(ds2.labels)
	# train(ds1,ds2)

	print('max_variables: %d' % ns1.max_variables)
	train(ns1,ns2)
