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
from tensorboard_logger import configure, log_value

torch.manual_seed(1)

TRAIN_FILE = 'expressions-synthetic/boolean5.json'
PRINT_LOSS_EVERY = 100
NUM_EPOCHS = 400
LOG_EVERY = 10


# a^b -> c
# 1 -3
# 2 -3
# -1 -2 3

train_formula = [[[1,-3],[-1,-2,3]],
                 [[2,-3],[-1,-2,3]], 
                 [[1,-3],[2,-3],[-1,-2,3]]]



hyperparams = {
    'embedding_dim': 32,
    'max_clauses': 3, 
    'max_variables': 3, 
    'num_ground_variables': 3, 
    'max_iters': 3,
    'split': False,
    'cuda': False
}

def train(fname):
    ds = CnfDataset(fname,70)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(ds.weights_vector, len(ds))
    trainloader = torch.utils.data.DataLoader(ds, batch_size=1, sampler = sampler)
    # dataiter = iter(trainloader)
    print('%d classes, %d samples'% (ds.num_classes,len(ds)))
    hyperparams['num_classes'] = ds.num_classes
    hyperparams['max_clauses'] = ds.max_clauses

    net = EqClassifier(**hyperparams)
    if hyperparams['cuda']:
        net.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    batch_size = 4
    get_step = lambda x,y: x*len(ds)+y

    configure("runs/run-1234", flush_secs=5)

    do_step = True
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
            if hyperparams['cuda']:
                inputs, topvar, labels = Variable(inputs.cuda()), Variable(topvar.cuda()), Variable(labels.cuda())
            # print('Processing sample from dataset with index %d' % ds_idx)
            # print(ds[ds_idx]['orig_sample']['clauses'])
            # if ds_idx in [194]:
            #     print('Skipping index %d' % ds_idx)
            #     continue

            # zero the parameter gradients
            if do_step:
                optimizer.zero_grad()

            do_step = i>0 and i % batch_size == 0
            # forward + backward + optimize
            outputs, aux_losses = net(inputs, topvar)
            loss = criterion(outputs, labels)   # + torch.sum(aux_losses)
            try:
                loss.backward()
            except RuntimeError as e:
                print('Woah, something is going on')
                print(e)
                ipdb.set_trace()            
            if do_step:
                optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if get_step(epoch,i) % LOG_EVERY == 0:
                log_value('loss',loss.data[0],get_step(epoch,i))
            if i % PRINT_LOSS_EVERY == PRINT_LOSS_EVERY-1:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / PRINT_LOSS_EVERY))
                running_loss = 0.0
                print('Outputs are:')
                print(outputs)
                print('And labels:')
                print(labels)
                # ipdb.set_trace()            


    print('Finished Training')
