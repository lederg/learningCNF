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
from settings import *
from batch_model import FactoredInnerIteration, GroundCombinator, DummyGroundCombinator



class QbfEncoder(nn.Module):
	def __init__(self, embedding_dim, num_ground_variables, max_iters, cvars, **kwargs):
		super(QbfEncoder, self).__init__() 
		self.settings = kwargs['settings'] if 'settings' in kwargs.keys() else CnfSettings()
		self.debug = False
		self.ground_dim = 3					# 1 - existential, 2 - universal, 3 (not yet used) - determinized
		self.batch_size = 1					# Not using self.settings['batch_size'] for now, cuz its RL, no batch
		self.max_variables = self.settings['max_variables']
		self.embedding_dim = embedding_dim		
		self.expand_dim_const = Variable(self.settings.zeros([self.max_variables,self.embedding_dim - self.ground_dim]), requires_grad=False)
		self.max_iters = max_iters		
					
		
		self.ground_annotations = self.expand_ground_to_state(self.create_ground_labels(cvars))
		self.inner_iteration = FactoredInnerIteration(self.get_ground_embeddings, embedding_dim, num_ground_variables=0, **kwargs)			
		self.forward_pos_neg = nn.Parameter(self.settings.FloatTensor(2,self.embedding_dim,self.embedding_dim))
		nn_init.normal(self.forward_pos_neg)		
		self.backwards_pos_neg = nn.Parameter(self.settings.FloatTensor(2,self.embedding_dim,self.embedding_dim))		
		nn_init.normal(self.backwards_pos_neg)
		
	def fix_annotations(self):
		if self.settings['cuda']:
			self.ground_annotations = self.ground_annotations.cuda()			
		else:
			self.ground_annotations = self.ground_annotations.cpu()
			
	def create_ground_labels(self,cvars):
		base_annotations = self.settings.zeros([self.max_variables, self.ground_dim])
		for i in cvars:			
			base_annotations[i-1][0] = int(not cvars[i]['universal'])
			base_annotations[i-1][1] = int(cvars[i]['universal'])

		return Variable(base_annotations, requires_grad=False)

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
	
	def forward(self, input, **kwargs):
		variables = []
		clauses = []
		if self.settings['sparse']:			
			f_vars = None
			f_clauses = None
		else:
			f_vars = input
			f_clauses = f_vars.transpose(1,2)		
		v = self.get_ground_embeddings()		
		variables = v.expand(len(input),v.size(0),1).contiguous()
		ground_variables = variables.view(-1,self.embedding_dim)[:,:self.ground_dim]

		if self.debug:
			print('Variables:')
			print(variables)
			pdb.set_trace()
		# Start propagation

		self.inner_iteration.re_init()
		
		for i in range(self.max_iters):
			# print('Starting iteration %d' % i)
			if (variables != variables).data.any():
				print('Variables have nan!')
				pdb.set_trace()
			variables = self.inner_iteration(variables, f_vars, f_clauses, ground_vars=ground_variables, v_block = self.forward_pos_neg, c_block=self.backwards_pos_neg, **kwargs)
			# variables = self.inner_iteration(variables, v_mat, c_mat, ground_vars=ground_variables, v_block = self.forward_pos_neg, c_block=self.backwards_pos_neg, old_forward=True)
			if self.debug:
				print('Variables:')
				print(variables)
				pdb.set_trace()
			# if (variables[0]==variables[1]).data.all():
			# 	print('Variables identical on (inner) iteration %d' % i)
		aux_losses = Variable(torch.zeros(len(variables)))
		return torch.squeeze(variables), aux_losses
