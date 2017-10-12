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
	    'embedding_dim': 64,
	    'ground_dim': 4,
	    # 'max_clauses': 12, 
	    'max_variables': 3, 
	    'num_ground_variables': 3, 
	    'data_mode': DataMode.NORMAL,
	    'dataset': 'boolean8',
	    'model_dir': 'saved_models',
	    # 'base_model': 'saved_models/run_siamese_test_1_nc2_bs64_ed64_iters6__1507265148_epoch25.model',
	    'base_model': None,
	    'base_mode': BaseMode.EMBEDDING,
	    'max_iters': 1,
	    'batch_size': 64,
	    'val_size': 200, 
	    'threshold': 5000,
	    'init_lr': 0.001,
	    # 'init_lr': 0.0004,
	    'decay_lr': 0.07,
	    'decay_num_epochs': 2,
	    'cosine_margin': 0,
	    # 'classifier_type': 'BatchGraphLevelClassifier',
	    # 'classifier_type': 'BatchEqClassifier',
	    'classifier_type': 'TopLevelClassifier',
	    'combinator_type': 'SymmetricSumCombine',	    
	    # 'ground_combinator_type': 'DummyGroundCombinator',	    
	    'ground_combinator_type': 'GroundCombinator',	    
	    'encoder_type': 'BatchEncoder',	    
	    # 'embedder_type': 'TopVarEmbedder',	    
	    'embedder_type': 'GraphEmbedder',	    
	    'gru_bias': False,
	    'use_ground': True,
	    'moving_ground': False, 
	    'split': False,
	    # 'cuda': True, 
	    'cuda': False,
	    'reset_on_save': True
	}

	def_settings = CnfSettings(hyperparams)
	data_mode = hyperparams['data_mode']
	threshold = hyperparams['threshold']


	DS_TRAIN_FILE = 'expressions-synthetic/split/%s-trainset.json' % hyperparams['dataset']
	DS_VALIDATION_FILE = 'expressions-synthetic/split/%s-validationset.json' % hyperparams['dataset']
	DS_TEST_FILE = 'expressions-synthetic/split/%s-testset.json' % hyperparams['dataset']


@ex.automain
def main(DS_TRAIN_FILE, DS_VALIDATION_FILE, DS_TEST_FILE, data_mode, threshold):
	ds1 = CnfDataset.from_eqparser(DS_TRAIN_FILE,mode=data_mode, threshold=threshold)
	ds2 = CnfDataset.from_eqparser(DS_VALIDATION_FILE, threshold=threshold, ref_dataset=ds1, mode=data_mode)
	# ds1 = CnfDataset(DS_TRAIN_FILE,threshold,mode=data_mode, num_max_clauses=12)
	# ds2 = CnfDataset(DS_VALIDATION_FILE, threshold, ref_dataset=ds1, mode=data_mode, num_max_clauses=12)
	# ds3 = CnfDataset(DS_TEST_FILE, threshold, ref_dataset=ds1, mode=data_mode)
	print('Classes from validation set:')
	print(ds2.labels)
	train(ds1,ds2)
