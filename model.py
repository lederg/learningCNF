import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

torch.manual_seed(1)


class ResidualCombine(nn.Module):
    def __init__(self, input_size, embedding_dim):
    	self.layer1 = nn.linear(input_size*embedding_dim,embedding)
    	self.layer2 = nn.linear(input_size*embedding_dim,embedding)

    def forward(self, input):
    	out = F.normalize(F.sigmoid(self.layer1(input)) + self.layer2(input))
    	return out


class Encoder(nn.Module):
    def __init__(self, embedding_dim, max_clauses, max_variables, num_ground_variables, num_variables):
        super(EncoderRNN, self).__init__()        
        self.embedding_dim = embedding_dim
        self.max_variables = max_variables
        self.max_clauses = max_clauses
        self.num_ground_variables = num_ground_variables
        self.negation = nn.linear(embedding_dim, embedding_dim)
        self.embedding = nn.Embedding(num_ground_variables, embedding_dim)
        self.extra_embedding = nn.Embedding(2, embedding_dim)    # one for the ground tseitin variables, one for False.
        self.false = self.extra_embedding(0)
        self.tseitin = self.extra_embedding(1)
        self.clause_combiner = ResidualCombine(max_clauses,embedding_dim)
        self.variable_combiner = ResidualCombine(max_variables,embedding_dim)


 # We permute the clauses and concatenate them

    def prepare_clauses(self, clauses):    	
    	rc = torch.cat(np.random.permutation(clauses),dim=1)
    	return rc

 # i is the index of the special variable (the current one)
    def prepare_variables(self, variables, i):    	
    	tmp = variables.pop(i)
    	rc = [tmp] + np.random.permutation(variables)
    	return torch.cat(rc,1)



    def _forward_clause(variables, clause, i):
    	c_vars = []    	
    	for j in range(self.max_variables):


    	for j,l in enumerate(clause):
    		ind = np.abs(l)-1		# variables in clauses are 1-based and negative if negated
    		v = variables[ind]
    		if ind==i:
    			ind_in_clause = j
    		if l < 0:
    			v = self.negation(v)
    		c_vars.append(v)
    	return self.variable_combiner(self.prepare_variables(c_vars,ind_in_clause))


    def _forward_iteration(variables, formula):
    	out_embeddings = []
    	for i,clauses in enumerate(formula):
    		clause_embeddings = [self._forward_clause(variables,c, i) for c in clauses]
    		out_embeddings.append(self.clause_combiner(self.prepare_clauses(clause_embeddings)))

    	return out_embeddings

# input is one training sample, 

    def forward(self, input):    	
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result
