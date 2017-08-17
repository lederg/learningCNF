import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from model import *
from datautils import *
import utils
import numpy as np

torch.manual_seed(1)

TRAIN_FILE = fname = 'expressions-synthetic/boolean5.json'

# a^b -> c
# 1 -3
# 2 -3
# -1 -2 3

train_formula = [[[1,-3],[-1,-2,3]],
				 [[2,-3],[-1,-2,3]], 
				 [[1,-3],[2,-3],[-1,-2,3]]]



hyperparams = {
	'embedding_dim': 16,
	'max_clauses': 3, 
	'max_variables': 3, 
	'num_ground_variables': 3, 
	'max_iters': 2,
	'split': False
}

ds = CnfDataset(fname)
hyperparams['num_classes'] = ds.num_classes
sampler = torch.utils.data.sampler.WeightedRandomSampler(ds.weights_vector, 1)
a = EqClassifier(**hyperparams)
# sampler = torch.utils.data.sampler.WeightedRandomSampler(ds.weights_vector, len(ds))
trainloader = torch.utils.data.DataLoader(ds, batch_size=len(ds), sampler = sampler)
input = utils.formula_to_input(train_formula)
out, aux_losses = a.forward(input)