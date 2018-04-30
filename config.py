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
	ground_dim = 8
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
	# base_model = 'run_test2_1_bs128_ed40_iters1__1521592635_iter4600.model'
	base_model = None
	base_mode = BaseMode.ALL
	# base_mode = BaseMode.EMBEDDING
	max_iters = 1
	batch_size = 128
	val_size = 100 
	threshold = 10
	init_lr = 0.0006
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
	ggn_core = 'WeightedNegInnerIteration'
	# ggn_core = 'FactoredInnerIteration'
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
	do_not_learn = False
	restart_cadet_every = 3500
	non_linearity = 'F.relu'
	test_every = 1000
# RL - PG

	gamma=0.99	
	entropy_alpha = 0.000
	policy = 'Policy'
	policy_dim1 = 30
	policy_dim2 = 15
	# policy_dim1 = 100
	# policy_dim2 = 50
	batch_backwards = False					# Are we going to re-feed all states into the network in batch (True) or do the naive solution (False)
	# greedy_rewards = True
	greedy_rewards = False
	rl_log_dir = 'runs_cadet'
	# rl_train_data = 'data/candidate_qbf/'
	# rl_train_data = 'data/test2_qbf/299_UNSAT.qdimacs'
	# rl_train_data = 'data/sat_med_500/sat_med__42_UNSAT.dimacs'
	rl_train_data = 'data/qbf_easy_train/'
	rl_validation_data = 'data/qbf_easy_test'
	rl_test_data = 'data/old/medium_gen2/'
	rl_log_envs = []
	rl_log_all = False
	rl_decay = False
	# rl_clip_episode_at = 100
	debug_actions = False
	debug = False
	use_old_rewards = False					#  Must be false if using old cadet
	leaky=False
	cadet_binary = './cadet'
	# cadet_binary = './old_cadet' 			
	do_test = True
	test_iters = 100
	fresh_seed = False  						# Use a fresh seed in cadet
	adaptive_lr = False
	desired_kl = 1e-6
	min_timesteps_per_batch = 400
	ac_baseline = False
	use_global_state = True
	clause_learning = True
	vars_set = True
	use_gru = True
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
	report_tensorboard = True
# Localization

	max_edges = 20


	
	DS_TRAIN_FILE = 'expressions-synthetic/split/%s-trainset.json' % dataset
	DS_VALIDATION_FILE = 'expressions-synthetic/split/%s-validationset.json' % dataset
	DS_TEST_FILE = 'expressions-synthetic/split/%s-testset.json' % dataset

	# return vars()

