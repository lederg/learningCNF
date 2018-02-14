import time
import ipdb

def cfg():	
	exp_time = int(time.time())
	state_dim = 30
	embedding_dim = 10
	ground_dim = 3
	policy_dim1 = 100
	policy_dim2 = 50
	max_variables = 200 
	max_clauses = 600
	num_ground_variables = 3 	
	data_dir = 'data/'
	dataset = 'boolean8'
	model_dir = 'saved_models'
	# 'base_model = 'saved_models/run_bigsat_50_4_nc2_bs40_ed4_iters8__1508199570_epoch200.model'
	base_model = None
	# base_mode = BaseMode.ALL
	max_iters = 12
	batch_size = 1			# for RL
	# batch_size = 64
	val_size = 100 
	threshold = 10
	init_lr = 0.001
	# 'init_lr = 0.0004
	decay_lr = 0.055
	decay_num_epochs = 6
	cosine_margin = 0
	# 'classifier_type = 'BatchGraphLevelClassifier'
	# 'classifier_type = 'BatchEqClassifier'
	classifier_type = 'TopLevelClassifier'
	combinator_type = 'SymmetricSumCombine'	    
	ground_combinator_type = 'DummyGroundCombinator'	    
	# 'ground_combinator_type = 'GroundCombinator'	
	encoder_type = 'BatchEncoder'	    
	# 'embedder_type = 'TopVarEmbedder'	    
	embedder_type = 'GraphEmbedder'	    
	# 'negate_type = 'minus'
	negate_type = 'regular'
	sparse = True
	# sparse = False
	gru_bias = False
	use_ground = True
	moving_ground = False 
	split = False
	# 'cuda = True 
	cuda = False
	reset_on_save = False


	run_task='train'

	max_edges = 20


	
	DS_TRAIN_FILE = 'expressions-synthetic/split/%s-trainset.json' % dataset
	DS_VALIDATION_FILE = 'expressions-synthetic/split/%s-validationset.json' % dataset
	DS_TEST_FILE = 'expressions-synthetic/split/%s-testset.json' % dataset

	return vars()

