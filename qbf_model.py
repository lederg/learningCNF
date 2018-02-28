import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import init as nn_init
import torch.nn.functional as F
import numpy as np
import ipdb

import utils
from qbf_data import *
from batch_model import FactoredInnerIteration, GraphEmbedder
from settings import *


class QbfClassifier(nn.Module):
	def __init__(self, encoder=None, embedder=None, **kwargs):
		super(QbfClassifier, self).__init__()        
		self.settings = kwargs['settings'] if 'settings' in kwargs.keys() else CnfSettings()
		# self.embedding_dim = self.settings['embedding_dim']
		# self.max_variables = self.settings['max_variables']
		if encoder:
			self.encoder = encoder
			# self.encoder.fix_annotations()
		else:
			self.encoder = QbfEncoder(**kwargs)
		if embedder:
			self.embedder = embedder
		else:
			self.embedder = GraphEmbedder(**kwargs)
		self.softmax_layer = nn.Linear(self.encoder.embedding_dim,2)

	def forward(self, input, **kwargs):
		embeddings = self.encoder(input, **kwargs)
		enc = self.embedder(embeddings, batch_size=len(input), **kwargs)
		return self.softmax_layer(enc)     # variables are 1-based


class QbfEncoder(nn.Module):
	def __init__(self, **kwargs):
		super(QbfEncoder, self).__init__() 
		self.settings = kwargs['settings'] if 'settings' in kwargs.keys() else CnfSettings()
		self.debug = False
		self.ground_dim = self.settings['ground_dim']
		# self.batch_size = 1
		self.batch_size = self.settings['batch_size']
		self.embedding_dim = self.settings['embedding_dim']		
		self.expand_dim_const = Variable(self.settings.zeros(1), requires_grad=False)
		self.max_iters = self.settings['max_iters']
		self.inner_iteration = FactoredInnerIteration(**kwargs)			
		self.forward_pos_neg = nn.Parameter(self.settings.FloatTensor(2,self.embedding_dim,self.embedding_dim))
		nn_init.normal(self.forward_pos_neg)		
		self.backwards_pos_neg = nn.Parameter(self.settings.FloatTensor(2,self.embedding_dim,self.embedding_dim))		
		nn_init.normal(self.backwards_pos_neg)		
					
	def expand_ground_to_state(self,v):
		# ipdb.set_trace()
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
			variables = self.inner_iteration(variables, f_vars, f_clauses, batch_size=size[0], ground_vars=ground_embeddings, 
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
