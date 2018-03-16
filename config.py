import time
import ipdb
from utils import BaseMode
from sacred import Experiment

ex = Experiment('REINFORCE')

@ex.config
def cfg():	
	name = 'DEF_NAME'
	exp_time = int(time.time())
	state_dim = 35
	embedding_dim = 20
	ground_dim = 6
	max_variables = 200 
	max_clauses = 600
	num_ground_variables = 0
	data_dir = 'data/'
	dataset = 'boolean8'
	model_dir = 'saved_models'
	# base_model = 'run_qbf_base3_bs64_ed30_iters7__1519801625_epoch15.model'
	# base_model = 'run_qbf_base4_bs64_ed40_iters7__1519842678_epoch200.model'
	# base_model = 'run_rl_env718_bs64_ed30_iters7__1519807551_iter200.model'
	# base_model = 'run_rl_env718_test2_bs64_ed30_iters7__1519810118_iter100.model'
	base_model = None
	base_mode = BaseMode.ALL
	# base_mode = BaseMode.EMBEDDING
	max_iters = 6
	batch_size = 128
	val_size = 100 
	threshold = 10
	init_lr = 1e-04
	# init_lr = 0.001
	# 'init_lr = 0.0004
	decay_lr = 0.055
	decay_num_epochs = 6
	cosine_margin = 0
	# 'classifier_type = 'BatchGraphLevelClassifier'
	# 'classifier_type = 'BatchEqClassifier'
	classifier_type = 'TopLevelClassifier'
	combinator_type = 'SymmetricSumCombine'	    
	# ground_combinator_type = 'DummyGroundCombinator'
	ground_combinator_type = 'GroundCombinator'	
	# encoder_type = 'BatchEncoder'	    
	encoder_type = 'QbfEncoder'	    
	# 'embedder_type = 'TopVarEmbedder'	    
	embedder_type = 'GraphEmbedder'	    
	# 'negate_type = 'minus'
	negate_type = 'regular'
	sparse = True
	# sparse = False
	gru_bias = False
	use_ground = False
	moving_ground = False 
	split = False
	# cuda = True 
	cuda = False
	reset_on_save = False
	run_task='train'
	do_not_run=False

# RL - PG

	gamma=0.99
	policy_dim1 = 20
	policy_dim2 = 10
	# policy_dim1 = 100
	# policy_dim2 = 50
	min_timesteps_per_batch = 400
	batch_backwards = False					# Are we going to re-feed all states into the network in batch (True) or do the naive solution (False)
	# greedy_rewards = True
	greedy_rewards = False
	rl_log_dir = 'runs_cadet'
	# rl_train_data = 'data/single_qbf/'
	rl_train_data = 'data/single_qbf/616_SAT.qdimacs'
	rl_log_envs = [616]
	rl_log_all = True
	# rl_clip_episode_at = 100

# RL - DQN

	EPS_START = 0.9
	EPS_END = 0.03
	EPS_DECAY = 2000
	learning_starts = 50000
	replay_size = 400000
	learning_freq=4
	target_update_freq=3000
	grad_norm_clipping=2
	pre_bias = True
	# pre_bias = False
	invalid_bias = -1000
	# invalid_bias = 0
	report_tensorboard = False
# Localization

	max_edges = 20


	
	DS_TRAIN_FILE = 'expressions-synthetic/split/%s-trainset.json' % dataset
	DS_VALIDATION_FILE = 'expressions-synthetic/split/%s-validationset.json' % dataset
	DS_TEST_FILE = 'expressions-synthetic/split/%s-testset.json' % dataset

	# return vars()

