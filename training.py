import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import *
import utils
import numpy as np

torch.manual_seed(1)

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
	'max_iters': 2
}

input = utils.formula_to_input(train_formula)
a = Encoder(**hyperparams)
