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
import ipdb

torch.manual_seed(1)

TRAIN_FILE = 'expressions-synthetic/boolean5.json'
PRINT_LOSS_EVERY = 100
NUM_EPOCHS = 20

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
    'max_iters': 1,
    'split': False
}

def train(fname):
    ds = CnfDataset(fname)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(ds.weights_vector, len(ds))
    trainloader = torch.utils.data.DataLoader(ds, batch_size=1, sampler = sampler)
    # dataiter = iter(trainloader)
    hyperparams['num_classes'] = ds.num_classes
    hyperparams['max_clauses'] = ds.max_clauses

    net = EqClassifier(**hyperparams)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs, no batching unfortunately...          
            ds_idx = data['idx_in_dataset'][0]
            # if ds_idx==541:
            #     ipdb.set_trace()
            inputs = utils.formula_to_input(data['sample'])
            topvar = torch.abs(Variable(data['topvar'], requires_grad=False))
            labels = Variable(data['label'], requires_grad=False)
            print('Processing sample from dataset with index %d' % ds_idx)
            print(ds[ds_idx]['orig_sample']['clauses'])
            # if ds_idx in [194]:
            #     print('Skipping index %d' % ds_idx)
            #     continue

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, aux_losses = net(inputs, topvar)
            # ipdb.set_trace()
            loss = criterion(outputs, labels)   # + torch.sum(aux_losses)
            try:
                loss.backward()
            except RuntimeError as e:
                print('Woah, something is going on')
                print(e)
                ipdb.set_trace()            
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % PRINT_LOSS_EVERY == PRINT_LOSS_EVERY-1:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / PRINT_LOSS_EVERY))
                running_loss = 0.0

    print('Finished Training')
