import torch
import itertools
import numpy as np

def normalize(input, p=2, dim=1, eps=1e-12):
	return input / input.norm(p, dim).clamp(min=eps).expand_as(input)

def formula_to_input(formula):
	rc = []
	for f in formula:
		rc.append([torch.LongTensor(x) for x in f])
	return rc

def permute_seq(inp):
	# inp is a sequence of tensors, we return a random permutation

	p = list(itertools.permutations(inp))
	i = np.random.randint(len(p))
	return list(p[i])