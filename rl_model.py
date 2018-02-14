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
from batch_model import FactoredInnerIteration, GroundCombinator, DummyGroundCombinator, GraphEmbedder
from settings import *


class Policy(nn.Module):
	def __init__(self, **kwargs):
		super(Policy, self).__init__()
		self.settings = kwargs['settings'] if 'settings' in kwargs.keys() else CnfSettings()
		self.max_variables = self.settings['max_variables']
		self.max_clauses = self.settings['max_clauses']
		self.state_dim = self.settings['state_dim']
		self.embedding_dim = self.settings['embedding_dim']
		self.ground_dim = self.settings['ground_dim']
		self.policy_dim1 = self.settings['policy_dim1']
		self.policy_dim2 = self.settings['policy_dim2']
		self.qbf = QbfBase(sparse=self.settings['sparse'])
		self.graph_embedder = GraphEmbedder(settings=self.settings)
		self.encoder = QbfEncoder(**self.settings.hyperparameters)
		self.linear1 = nn.Linear(self.state_dim+self.embedding_dim, self.policy_dim1)
		# self.linear2 = nn.Linear(self.policy_dim1,self.policy_dim2)
		self.action_score = nn.Linear(self.policy_dim1,1)
		self.activation = F.relu
		self.saved_log_probs = []

	# (Re)load the encoder
	def re_init_qbf_base(self, qbf):
		self.qbf = qbf
		assert(self.max_variables >= self.qbf.num_vars)
		assert(self.max_clauses >= self.qbf.num_clau(batched) ses)
		# self.encoder.recreate_ground(self.qbf.var_types)

	def create_ground_labels(self,var_types):
		base_annotations = self.settings.zeros([self.max_variables, self.ground_dim])
		for i,val in enumerate(var_types):
			base_annotations[i][val] = True

		return base_annotations


	def get_data_from_qbf(self):
		b = self.qbf.as_tensor_dict()		
		if self.settings['cuda']:
			func = lambda x: x.cuda() 
		else:
			func = lambda x: x
		rc = {}
		rc['input'] = func(Variable(b['v2c'].transpose(0,1).unsqueeze(0), requires_grad=False))
		rc['cmat_pos'] = func(Variable(b['sp_v2c_pos'], requires_grad=False))
		rc['cmat_neg'] = func(Variable(b['sp_v2c_neg'], requires_grad=False))
		rc['ground_embeddings'] = func(Variable(self.create_ground_labels(self.qbf.var_types)))
		return rc

	# state is just a (batched) vector of fixed size state_dim which should be expanded. 
	# ground_embeddings are batch * max_vars * ground_embedding

	# kwargs includes 'input', 'cmat_pos', 'cmat_neg', the latter two already in the correct format.

	def forward(self, state, **kwargs):		
		if 'vs' in kwargs.keys():
			vs = kwargs['vs']
		else:			
			# vs = self.encoder(**a).view(self.settings['batch_size'],self.max_variables,self.embedding_dim)		
			vs = self.encoder(**a).view(-1,self.embedding_dim)		
		inputs = torch.cat([state.expand(len(vs),len(state)), vs],dim=1)

		# Note, the softmax is on dimension 0, along the batch. We compute it on all candidate actions.

		# rc = F.softmax(self.action_scores(self.activation(self.linear2(self.activation(self.linear1(inputs))))), dim=0)
		outputs = self.action_scores(self.activation(self.linear1(inputs))).view(-1,self.max_variables)
		rc = F.softmax(outputs)
		return rc

class QbfEncoder(nn.Module):
	def __init__(self, **kwargs):
		super(QbfEncoder, self).__init__() 
		self.settings = kwargs['settings'] if 'settings' in kwargs.keys() else CnfSettings()
		self.debug = False
		self.ground_dim = 3					# 1 - existential, 2 - universal, 3 (not yet used) - determinized
		# self.batch_size = 1
		self.batch_size = self.settings['batch_size']
		self.embedding_dim = self.settings['embedding_dim']
		self.max_variables = self.settings['max_variables']
		self.expand_dim_const = Variable(self.settings.zeros(1).expand(self.batch_size*self.max_variables,self.embedding_dim - self.ground_dim), requires_grad=False)
		self.max_iters = self.settings['max_iters']
		self.inner_iteration = FactoredInnerIteration(self.get_ground_embeddings, **kwargs)			
		self.forward_pos_neg = nn.Parameter(self.settings.FloatTensor(2,self.embedding_dim,self.embedding_dim))
		nn_init.normal(self.forward_pos_neg)		
		self.backwards_pos_neg = nn.Parameter(self.settings.FloatTensor(2,self.embedding_dim,self.embedding_dim))		
		nn_init.normal(self.backwards_pos_neg)
		self.ground_annotations = None
		
	def recreate_ground(self, var_types):
		self.ground_annotations = self.expand_ground_to_state(self.create_ground_labels(var_types))

	def fix_annotations(self):
		if self.settings['cuda']:
			self.ground_annotations = self.ground_annotations.cuda()			
		else:
			self.ground_annotations = self.ground_annotations.cpu()
			
	def expand_ground_to_state(self,v):
		return torch.cat([v,self.expand_dim_const],dim=1)

	def get_ground_embeddings(self):		
		return self.ground_annotations.view(1,-1).transpose(0,1)

	def get_block_matrix(self, blocks, indices):		
		rc = []
		for a in indices:
			rc.append(torch.cat([torch.cat([blocks[x] for x in i],dim=1) for i in a.long()]))

		return torch.stack(rc)


# This should probably be factored into a base class, its basically the same as for BatchEncoder

# ground_embeddings are (batch,maxvars,embedding_dim)

	
	def forward(self, input, ground_embeddings, **kwargs):
		variables = []
		clauses = []
		if self.settings['sparse']:			
			f_vars = None
			f_clauses = None
		else:
			f_vars = input
			f_clauses = f_vars.transpose(1,2)				
		v = self.expand_ground_to_state(ground_embeddings.view(-1,self.embedding_dim)).view(1,-1).transpose(0,1)
		variables = v.view(-1,self.embedding_dim*self.max_variables,1)
		
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
			variables = self.inner_iteration(variables, f_vars, f_clauses, ground_vars=ground_embeddings, v_block = self.forward_pos_neg, c_block=self.backwards_pos_neg, **kwargs)
			# variables = self.inner_iteration(variables, v_mat, c_mat, ground_vars=ground_variables, v_block = self.forward_pos_neg, c_block=self.backwards_pos_neg, old_forward=True)
			if self.debug:
				print('Variables:')
				print(variables)
				pdb.set_trace()
			# if (variables[0]==variables[1]).data.all():
			# 	print('Variables identical on (inner) iteration %d' % i)
		# aux_losses = Variable(torch.zeros(len(variables)))
		return torch.squeeze(variables)
