import time
import ipdb
from utils import BaseMode
from sacred import Experiment

ex = Experiment('REINFORCE')

@ex.config
def cfg():	
	name = 'DEF_NAME'
	exp_time = int(time.time())
	state_dim = 36
	embedding_dim = 20  # deprecated
	# vemb_dim = 16
	# cemb_dim = 16
	vemb_dim = 8
	cemb_dim = 64
	ground_dim = 8  # deprecated
	vlabel_dim = 8
	clabel_dim = 1
	max_variables = 200 
	max_clauses = 600
	num_ground_variables = 0  # deprecated
	data_dir = 'data/'
	dataset = 'boolean8'
	model_dir = 'saved_models'
	base_model = None
	base_mode = BaseMode.ITERS 
	base_iters = 1
	base_stats = None
	# base_mode = BaseMode.EMBEDDING
	max_iters = 1
	max_step = 500			# Maximum steps per formula
	batch_size = 128  # deprecated
	val_size = 100  # deprecated?
	threshold = 10  # deprecated
	init_lr = 0.0006
	# init_lr = 0.001
	# 'init_lr = 0.0004
	decay_lr = 0.055
	decay_num_epochs = 6  # deprecated
	cosine_margin = 0  # deprecated
	# 'classifier_type = 'BatchGraphLevelClassifier'
	# 'classifier_type = 'BatchEqClassifier'
	classifier_type = 'TopLevelClassifier'  # deprecated
	combinator_type = 'SymmetricSumCombine'  # deprecated
	# ground_combinator_type = 'DummyGroundCombinator'
	ground_combinator_type = 'GroundCombinator'  # deprecated?
	# encoder_type = 'BatchEncoder'
	encoder_type = 'QbfEncoder'  # deprecated
	ggn_core = 'WeightedNegInnerIteration'  # deprecated
	# ggn_core = 'FactoredInnerIteration'
	# 'embedder_type = 'TopVarEmbedder'
	embedder_type = 'GraphEmbedder'  # deprecated
	# 'negate_type = 'minus'
	negate_type = 'regular'  # deprecated
	sparse = True  # deprecated
	# sparse = False
	gru_bias = False  # deprecated
	use_ground = False  # deprecated
	moving_ground = False  # deprecated
	split = False  # deprecated
	# cuda = True 
	cuda = False
	reset_on_save = False
	run_task = 'train'
	do_not_run = False
	do_not_learn = False
	restart_cadet_every = 3500
	restart_in_test = False
	non_linearity = 'F.relu'
	test_every = 1000
# RL - PG
	static_ed='eps/random_easy_train.eps'
	gamma = 0.99
	entropy_alpha = 0.000
	policy = 'NewDoublePolicy'
	policy_dim1 = 64
	policy_dim2 = 32
	batch_backwards = False  # deprecated					# Are we going to re-feed all states into the network in batch (True) or do the naive solution (False)
	# greedy_rewards = True
	greedy_rewards = False
	rl_log_dir = 'runs_cadet'
	rl_train_data = 'data/qbf_easy_train/'
	rl_validation_data = 'data/qbf_easy_test'
	rl_test_data = 'data/qbf_hard_test/'
	rl_log_envs = []
	rl_log_all = False
	rl_shelve_it = False
	rl_decay = False
	# rl_clip_episode_at = 100
	debug_actions = False
	debug = False
	use_old_rewards = True					#  Must be false if using old cadet
	use_vsids_rewards = False
	leaky = False  # deprecated
	cadet_binary = './cadet'
	# cadet_binary = './old_cadet' 			
	do_test = False
	test_iters = 100
	fresh_seed = False  						# Use a fresh seed in cadet
	follow_kl = False
	desired_kl = 1e-6
	min_timesteps_per_batch = 400  # replace by number of formulas per batch after normalization
	episodes_per_batch = 32
	ac_baseline = False
	stats_baseline = False
	use_global_state = True
	clause_learning = True
	vars_set = True
	use_gru = False  # deprecated
	use_bn = False
	use_curriculum = False
	normalize_episodes = False
	parallelism = 1
	full_pipeline = True
	packed = True
	masked_softmax = True 					# This chooses actions only from allowed actions
	slim_state=False
	episode_cutoff = 200

	grad_norm_clipping = 2
	pre_bias = False  # deprecate
	disallowed_aux = False
	lambda_disallowed = 1.
	invalid_bias = -1000  # deprecate
	# invalid_bias = 0
	report_tensorboard = True

	EPS_START = 0.9  # deprecated
	EPS_END = 0.03  # deprecated
	EPS_DECAY = 2000  # deprecated

# RL - DQN
	learning_starts = 50000
	replay_size = 400000
	learning_freq = 4
	target_update_freq = 3000

# Localization
	max_edges = 20

	DS_TRAIN_FILE = 'expressions-synthetic/split/%s-trainset.json' % dataset
	DS_VALIDATION_FILE = 'expressions-synthetic/split/%s-validationset.json' % dataset
	DS_TEST_FILE = 'expressions-synthetic/split/%s-testset.json' % dataset

	# return vars()

