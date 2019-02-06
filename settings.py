import torch
from torch.autograd import Variable
from utils import Singleton

class CnfSettings(metaclass=Singleton):
	def __init__(self, hyperparameters=None):
		self.hyperparameters = hyperparameters
		self.LongTensor = torch.cuda.LongTensor if hyperparameters['cuda'] else torch.LongTensor
		self.FloatTensor = torch.cuda.FloatTensor if hyperparameters['cuda'] else torch.FloatTensor
		self.ByteTensor = torch.cuda.ByteTensor if hyperparameters['cuda'] else torch.ByteTensor
		self.expand_dim_const = Variable(self.zeros(1), requires_grad=False)

	def __getitem__(self, key):
		return self.hyperparameters[key]

	def zeros(self,size):
		rc = torch.zeros(size)
		return rc.cuda() if self.hyperparameters['cuda'] else rc

	def cudaize(self, func, *args, **kwargs):
		rc = func(*args,**kwargs)
		return rc.cuda() if self.hyperparameters['cuda'] else rc

	def cudaize_var(self, var):
		return var.cuda() if self.hyperparameters['cuda'] else var