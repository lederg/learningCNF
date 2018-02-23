import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from qbf_model import *
from qbf_data import *
import re
import utils
import time
import numpy as np
import ipdb
from enum import Enum
from tensorboard_logger import configure, log_value

NUM_EPOCHS = 400
PRINT_LOSS_EVERY = 100


criterion = nn.CrossEntropyLoss()

def train(ds, model, optimizer=None, iters=None, ds_validate=None):
  settings = CnfSettings()
  if optimizer is None:
    optimizer = optim.SGD(model.parameters(), lr=settings['init_lr'], momentum=0.9)

  sampler = torch.utils.data.sampler.RandomSampler(ds)
  trainloader = torch.utils.data.DataLoader(ds, batch_size=settings['batch_size'], sampler = sampler, collate_fn = qbf_collate)

  get_step = lambda x,y: x*len(trainloader)+y
  start_epoch=0
  current_time = time.time()

  for epoch in range(start_epoch,NUM_EPOCHS):
    for i, data in enumerate(trainloader, 0):
      labels = Variable(data['label'])
      cmat_pos = Variable(data['sp_v2c_pos'])
      cmat_neg = Variable(data['sp_v2c_neg'])
      ground = Variable(data['ground']).float()
      running_loss = 0.
      total_correct = 0.
      effective_bs = len(ground)
      if  effective_bs != settings['batch_size']:
        print('Trainer gave us shorter batch!!')
      if settings.hyperparameters['cuda']:
        labels = labels.cuda()
        ground, cmat_pos, cmat_neg = ground.cuda(), cmat_pos.cuda(), cmat_neg.cuda()

      print('iteration %d beginning...' % i)
      # forward + backward + optimize

      optimizer.zero_grad()
      outputs = model(ground, cmat_pos=cmat_pos, cmat_neg=cmat_neg)
      loss = criterion(outputs, labels)   # + torch.sum(aux_losses)
      try:
        loss.backward()
      except RuntimeError as e:
        print('Woah, something is going on')
        print(e)
      optimizer.step()
      # print statistics
      running_loss += loss.data[0]
      correct = outputs.max(dim=1)[1]==labels
      num_correct = torch.nonzero(correct.data).size()


      if i % PRINT_LOSS_EVERY == PRINT_LOSS_EVERY-1:                
        new_time = time.time()                
        print('Average time per mini-batch, %f' % ((new_time-current_time) / PRINT_LOSS_EVERY))
        current_time = new_time
        print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, running_loss / PRINT_LOSS_EVERY))
        running_loss = 0.0
        print('[%d, %5d] training accuracy: %.3f' %
              (epoch + 1, i + 1, total_correct / (PRINT_LOSS_EVERY*settings['batch_size'])))
        total_correct = 0.0

      if iters is not None and get_step(epoch,i) > iters:
        return
