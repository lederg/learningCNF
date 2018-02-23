import os.path
import torch
# from torch.distributions import Categorical
import ipdb
import random
import time

from settings import *
from qbf_train import *
from utils import *
import torch.nn.utils as tutils

settings = CnfSettings()

def qbf_train_main():
	ds = QbfDataset(dirname='data/dataset1/')
	model = QbfClassifier()
	optimizer = optim.SGD(model.parameters(), lr=settings['init_lr'], momentum=0.9)

	train(ds,model,optimizer=optimizer)