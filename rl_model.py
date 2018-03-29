import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import init as nn_init
import torch.nn.functional as F
import torch.optim as optim
import utils
import numpy as np
from collections import namedtuple
import ipdb
# import pdb
from qbf_data import *
from batch_model import FactoredInnerIteration, GraphEmbedder
from qbf_model import QbfEncoder
from settings import *

INVALID_BIAS = -1000
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class Policy(nn.Module):
	def __init__(self, encoder=None, **kwargs):
		super(Policy, self).__init__()
		self.settings = kwargs['settings'] if 'settings' in kwargs.keys() else CnfSettings()				
		self.state_dim = self.settings['state_dim']
		self.embedding_dim = self.settings['embedding_dim']
		self.ground_dim = self.settings['ground_dim']
		self.policy_dim1 = self.settings['policy_dim1']
		self.policy_dim2 = self.settings['policy_dim2']		
		if self.settings['ac_baseline']:
			self.graph_embedder = GraphEmbedder(settings=self.settings)
			self.value_score = nn.Linear(self.state_dim+self.embedding_dim,1)
		if encoder:
			print('Bootstraping Policy from existing encoder')
			self.encoder = encoder
		else:
			self.encoder = QbfEncoder(**self.settings.hyperparameters)
		# self.linear1 = nn.Linear(self.state_dim+self.ground_dim, self.policy_dim1)
		self.linear1 = nn.Linear(self.state_dim+self.embedding_dim+self.ground_dim, self.policy_dim1)
		self.linear2 = nn.Linear(self.policy_dim1,self.policy_dim2)
		self.invalid_bias = nn.Parameter(self.settings.FloatTensor([self.settings['invalid_bias']]))
		self.action_score = nn.Linear(self.policy_dim2,1)
		if self.settings['leaky']:
			self.activation = F.leaky_relu
		else:
			self.activation = F.relu
		self.saved_log_probs = []
	
	# state is just a (batched) vector of fixed size state_dim which should be expanded. 
	# ground_embeddings are batch * max_vars * ground_embedding

	# cmat_net and cmat_pos are already "batched" into a single matrix

	def forward(self, obs, **kwargs):
		state = obs.state
		ground_embeddings = obs.ground
		cmat_pos = obs.cmat_pos		
		cmat_neg = obs.cmat_neg

		if self.settings['cuda']:
			cmat_pos, cmat_neg = cmat_pos.cuda(), cmat_neg.cuda()
			state, ground_embeddings = state.cuda(), ground_embeddings.cuda()			
		size = ground_embeddings.size()
		self.batch_size=size[0]
		if 'vs' in kwargs.keys():
			vs = kwargs['vs']		
		else:						
			rc = self.encoder(ground_embeddings, cmat_pos=cmat_pos, cmat_neg=cmat_neg, **kwargs)
			vs = rc.view(self.batch_size,-1,self.embedding_dim)
		reshaped_state = state.view(self.batch_size,1,self.state_dim).expand(self.batch_size,size[1],self.state_dim)
		inputs = torch.cat([reshaped_state, vs,ground_embeddings],dim=2).view(-1,self.state_dim+self.embedding_dim+self.ground_dim)
		# inputs = torch.cat([reshaped_state, ground_embeddings],dim=2).view(-1,self.state_dim+self.ground_dim)
		# inputs = ground_embeddings.view(-1,self.ground_dim)
		
		outputs = self.action_score(self.activation(self.linear2(self.activation(self.linear1(inputs))))).view(self.batch_size,-1)
		# outputs = outputs-value		# Advantage
		# outputs = self.action_score(self.activation(self.linear1(inputs))).view(self.batch_size,-1)		

		if self.settings['pre_bias']:
			missing = (1-ground_embeddings[:,:,IDX_VAR_UNIVERSAL])*(1-ground_embeddings[:,:,IDX_VAR_EXISTENTIAL])
			valid = (1-(1-missing)*(1-ground_embeddings[:,:,IDX_VAR_DETERMINIZED]))*self.invalid_bias
			outputs = outputs + valid
		if self.settings['ac_baseline']:
			graph_embedding = self.graph_embedder(vs,batch_size=len(vs))
			value = self.value_score(torch.cat([state,graph_embedding],dim=1))
		else:
			value = None
		return outputs, value

		# rc = F.softmax(valid_outputs)
		# return rc
