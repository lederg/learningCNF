import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
import numpy as np
import ipdb


class TestDup(nn.Module):
	def __init__(self, param1):
		super(TestDup, self).__init__()
		self.extra_embedding = nn.Embedding(1, param1, max_norm=1.)				
		self.layer1 = nn.Linear(param1*3,param1)

	@property
	def false(self):
		return self.extra_embedding(Variable(torch.LongTensor([0]), requires_grad=False))

	def forward(self, inputs):
		return self.layer1(torch.cat([self.false,self.false,inputs],dim=1))
		

class TestAdd(nn.Module):
	def __init__(self, param1):
		super(TestAdd, self).__init__()
		self.coefficient = nn.Parameter(torch.Tensor([param1]))

	def forward(self, input):
		return self.coefficient * input[0] + input[1]

class ResidualCombine(nn.Module):
	def __init__(self, input_size, embedding_dim):
		super(ResidualCombine, self).__init__()        
		self.layer1 = nn.Linear(input_size*embedding_dim,embedding_dim)
		self.layer2 = nn.Linear(input_size*embedding_dim,embedding_dim)

	def forward(self, input):
		try:
			out = utils.normalize(F.sigmoid(self.layer1(input)) + self.layer2(input))
		except Exception as e:
			print(e)
			ipdb.set_trace()
		return out


# class VariableIteration(nn.Module):
# 	def __init__(self, embedding_dim, max_clauses, max_variables, num_ground_variables, max_iters):


class InnerIteration(nn.Module):
	def __init__(self, embedding_dim, max_clauses, max_variables, num_ground_variables, split=True, permute=True, **kwargs):
		super(InnerIteration, self).__init__()        
		self.embedding_dim = embedding_dim
		self.max_variables = max_variables
		self.max_clauses = max_clauses
		self.split = split
		self.permute = permute
		self.num_ground_variables = num_ground_variables
		self.negation = nn.Linear(embedding_dim, embedding_dim)   # add non-linearity?		
		self.extra_embedding = nn.Embedding(1, embedding_dim, max_norm=1.)				
		self.clause_combiner = ResidualCombine(max_clauses,embedding_dim)
		self.variable_combiner = ResidualCombine(max_variables,embedding_dim)

	@property
	def false(self):
		return self.extra_embedding(Variable(torch.LongTensor([0]), requires_grad=False))

	@property
	def true(self):
		return self.negation(self.false)

	def prepare_clauses(self, clauses):
		if self.permute:  	
			rc = torch.cat(utils.permute_seq(clauses),dim=1)
			if not self.split:
				return rc
			else:
				org = torch.cat(clauses,1)			# split
				return torch.cat([org,rc])	
		else:
			return torch.cat(clauses,dim=1)

 # i is the index of the special variable (the current one)
	def prepare_variables(self, variables, curr_variable):
		tmp = variables.pop(curr_variable)
		if self.permute:    		
			rc = [tmp] + utils.permute_seq(variables)	    		    	
			try:
				perm = torch.cat(rc,1)
			except RuntimeError:
				ipdb.set_trace()
			if not self.split:
				return perm
			else:
				org = torch.cat([tmp] + variables,1)        # splitting batch
				return torch.cat([org,perm])
		else:
			rc = [tmp] + variables
			return torch.cat(rc,1)



	def _forward_clause(self, variables, clause, i):
		c_vars = []
		for j in range(self.max_variables):
			if j<len(clause):								# clause is a list of tensors
				l=clause[j]									# l is a tensored floaty integer
				ind = torch.abs(l)-1       					# variables in clauses are 1-based and negative if negated
				v = torch.stack(variables)[ind.data][0] 	# tensored variables (to be indexed by tensor which is inside a torch variable..gah)
				# ipdb.set_trace()
				if (ind==i).data.all():
					ind_in_clause = j
				if (l < 0).data.all():
					v = self.negation(v)
			else:
				v = self.false.expand_as(variables[0])
			c_vars.append(v)
		return self.variable_combiner(self.prepare_variables(c_vars,ind_in_clause))

	def forward(self, variables, formula):
		out_embeddings = []		
		for i,clauses in enumerate(formula):
			# print('Clauses for variable %d: %d' % (i+1, len(clauses)))
			if clauses:
				clause_embeddings = [self._forward_clause(variables,c, i) for c in clauses]
				true_embeddings = [self.true.expand_as(clause_embeddings[0])]*(self.max_clauses-len(clauses))
				out_embeddings.append(self.clause_combiner(self.prepare_clauses(clause_embeddings+true_embeddings)))		
			else:
				out_embeddings.append(variables[i])

		return out_embeddings


class Encoder(nn.Module):
	def __init__(self, embedding_dim, num_ground_variables, max_iters, **kwargs):
		super(Encoder, self).__init__()        
		self.embedding_dim = embedding_dim		
		self.max_iters = max_iters		
		self.num_ground_variables = num_ground_variables		
		self.embedding = nn.Embedding(num_ground_variables, embedding_dim, max_norm=1.)				
		self.tseitin_embedding = nn.Embedding(1, embedding_dim, max_norm=1.)		
		self.inner_iteration = InnerIteration(embedding_dim, num_ground_variables=num_ground_variables, **kwargs)		
	
# input is one training sample (a formula), we'll permute it a bit at every iteration and possibly split to create a batch

	@property
	def tseitin(self):
		return self.tseitin_embedding(Variable(torch.LongTensor([0])))

	def forward(self, input):
		variables = []        
		for i in range(len(input)):
			if i<self.num_ground_variables:
				variables.append(self.embedding(Variable(torch.LongTensor([i]))))
			else:
				variables.append(utils.normalize(self.tseitin))

		# ipdb.set_trace()

		for i in range(self.max_iters):
			# print('Starting iteration %d' % i)
			variables = self.inner_iteration(variables, input)

		# We add loss on each variable embedding to encourage different elements in the batch to stay close. 

		if self.inner_iteration.split:
			aux_losses = [(v - v.mean(dim=0).expand_as(v)).norm(dim=1).sum() for v in variables]            
		else:
			aux_losses = Variable(torch.zeros(len(variables)))
		return variables, aux_losses


class EqClassifier(nn.Module):
	def __init__(self, num_classes, **kwargs):
		super(EqClassifier, self).__init__()        
		self.num_classes = num_classes
		self.encoder = Encoder(**kwargs)
		self.softmax_layer = nn.Linear(self.encoder.embedding_dim,num_classes)

	def forward(self, input, output_ind):
		embeddings, aux_losses = self.encoder(input)
		return self.softmax_layer(embeddings[output_ind.data[0]-1]), aux_losses     # variables are 1-based
		# return F.relu(self.softmax_layer(embeddings[output_ind.data[0]-1])), aux_losses     # variables are 1-based
		
