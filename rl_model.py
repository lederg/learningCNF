import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import init as nn_init
import torch.nn.functional as F
import torch.optim as optim
import utils
import numpy as np
import ipdb
# import pdb
from qbf_data import *
from batch_model import FactoredInnerIteration, GraphEmbedder
from settings import *

INVALID_BIAS = -1000

class Policy(nn.Module):
	def __init__(self, **kwargs):
		super(Policy, self).__init__()
		self.settings = kwargs['settings'] if 'settings' in kwargs.keys() else CnfSettings()				
		self.state_dim = self.settings['state_dim']
		self.embedding_dim = self.settings['embedding_dim']
		self.ground_dim = self.settings['ground_dim']
		self.policy_dim1 = self.settings['policy_dim1']
		self.policy_dim2 = self.settings['policy_dim2']		
		self.graph_embedder = GraphEmbedder(settings=self.settings)
		self.encoder = QbfEncoder(**self.settings.hyperparameters)
		self.linear1 = nn.Linear(self.state_dim+self.embedding_dim+self.ground_dim, self.policy_dim1)
		self.linear2 = nn.Linear(self.policy_dim1,self.policy_dim2)
		self.invalid_bias = nn.Parameter(self.settings.FloatTensor([INVALID_BIAS]))
		self.action_score = nn.Linear(self.policy_dim2,1)
		self.activation = F.relu
		self.saved_log_probs = []
	
	# state is just a (batched) vector of fixed size state_dim which should be expanded. 
	# ground_embeddings are batch * max_vars * ground_embedding

	# kwargs includes 'input', 'cmat_pos', 'cmat_neg', the latter two already in the correct format.

	def forward(self, state, ground_embeddings, **kwargs):		
		if 'batch_size' in kwargs:
			self.batch_size=kwargs['batch_size']
		size = ground_embeddings.size()
		if 'vs' in kwargs.keys():
			vs = kwargs['vs']		
		else:			
			# vs = self.encoder(**a).view(self.settings['batch_size'],self.max_variables,self.embedding_dim)		
			rc = self.encoder(ground_embeddings,**kwargs)
			vs = rc.view(self.batch_size,-1,self.embedding_dim)
			# vs = rc.view(-1,self.embedding_dim)
		reshaped_state = state.view(self.batch_size,1,self.state_dim).expand(self.batch_size,size[1],self.state_dim)
		inputs = torch.cat([reshaped_state, vs,ground_embeddings],dim=2).view(-1,self.state_dim+self.embedding_dim+self.ground_dim)

		outputs = self.action_score(self.activation(self.linear2(self.activation(self.linear1(inputs))))).view(self.batch_size,-1)
		# outputs = self.action_score(self.activation(self.linear1(inputs))).view(self.batch_size,-1)		
		valid_outputs = outputs + (1-(1-ground_embeddings[:,:,2])*(1-ground_embeddings[:,:,3]))*self.invalid_bias
		rc = F.softmax(valid_outputs)
		return rc

class QbfEncoder(nn.Module):
	def __init__(self, **kwargs):
		super(QbfEncoder, self).__init__() 
		self.settings = kwargs['settings'] if 'settings' in kwargs.keys() else CnfSettings()
		self.debug = False
		self.ground_dim = GROUND_DIM
		# self.batch_size = 1
		self.batch_size = self.settings['batch_size']
		self.embedding_dim = self.settings['embedding_dim']		
		self.expand_dim_const = Variable(self.settings.zeros(1), requires_grad=False)
		self.max_iters = self.settings['max_iters']
		self.inner_iteration = FactoredInnerIteration(None, **kwargs)			
		self.forward_pos_neg = nn.Parameter(self.settings.FloatTensor(2,self.embedding_dim,self.embedding_dim))
		nn_init.normal(self.forward_pos_neg)		
		self.backwards_pos_neg = nn.Parameter(self.settings.FloatTensor(2,self.embedding_dim,self.embedding_dim))		
		nn_init.normal(self.backwards_pos_neg)		
					
	def expand_ground_to_state(self,v):
		dconst = self.expand_dim_const.expand(len(v),self.embedding_dim - self.ground_dim)
		return torch.cat([v,dconst],dim=1)
	
# This should probably be factored into a base class, its basically the same as for BatchEncoder

# ground_embeddings are (batch,maxvars,ground_dim)

	
	def forward(self, ground_embeddings, **kwargs):
		variables = []
		clauses = []		
		size = ground_embeddings.size()
		if 'batch_size' in kwargs:
			self.batch_size=kwargs['batch_size']
			assert(self.batch_size==size[0])		
		f_vars = None
		f_clauses = None
		# ipdb.set_trace()
		v = self.expand_ground_to_state(ground_embeddings.view(-1,self.ground_dim)).view(1,-1).transpose(0,1)
		variables = v.view(-1,self.embedding_dim*size[1],1)
		
		# Inner iteration expects (batch*,embedding_dim*maxvar,1) for variables

		if self.debug:
			print('Variables:')
			print(variables)
			pdb.set_trace()
		# Start propagation
		
		for i in range(self.max_iters):
			# print('Starting iteration %d' % i)
			if (variables != variables).data.any():
				print('Variables have nan!')
				pdb.set_trace()
			variables = self.inner_iteration(variables, f_vars, f_clauses, ground_vars=ground_embeddings, 
						v_block = self.forward_pos_neg, c_block=self.backwards_pos_neg, max_variables=size[1], **kwargs)
			# variables = self.inner_iteration(variables, v_mat, c_mat, ground_vars=ground_variables, v_block = self.forward_pos_neg, c_block=self.backwards_pos_neg, old_forward=True)
			if self.debug:
				print('Variables:')
				print(variables)
				pdb.set_trace()
			# if (variables[0]==variables[1]).data.all():
			# 	print('Variables identical on (inner) iteration %d' % i)
		# aux_losses = Variable(torch.zeros(len(variables)))
		return torch.squeeze(variables)
