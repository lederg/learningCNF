import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from model import *
from datautils import *
import utils
import time
import numpy as np
import ipdb
from tensorboard_logger import configure, log_value
from sacred import Experiment

torch.manual_seed(1)

# TRAIN_FILE = 'expressions-synthetic/boolean5.json'

TRAIN_FILE = 'expressions-synthetic/split/boolean5-trainset.json'
VALIDATION_FILE = 'expressions-synthetic/split/boolean5-validationset.json'
TEST_FILE = 'expressions-synthetic/split/boolean5-testset.json'

DS_TRAIN_TEMPLATE = 'expressions-synthetic/split/%s-trainset.json'
DS_VALIDATION_TEMPLATE = 'expressions-synthetic/split/%s-validationset.json'
DS_TEST_TEMPLATE = 'expressions-synthetic/split/%s-testset.json'

PRINT_LOSS_EVERY = 100
# NUM_EPOCHS = 400
NUM_EPOCHS = 4
LOG_EVERY = 10
SAVE_EVERY = 1000

# a^b -> c
# 1 -3
# 2 -3
# -1 -2 3

train_formula = [[[1,-3],[-1,-2,3]],
                 [[2,-3],[-1,-2,3]], 
                 [[1,-3],[2,-3],[-1,-2,3]]]



hyperparams = {
    'embedding_dim': 4,
    'max_clauses': 3, 
    'max_variables': 3, 
    'num_ground_variables': 3, 
    'dataset': 'boolean8',
    'model_dir': 'saved_models',
    'max_iters': 6,
    'batch_size': 4,
    'val_size': 100, 
    # 'classifier_type': 'GraphLevelClassifier',
    'classifier_type': 'EqClassifier',
    'combinator_type': 'SimpleCombinator',
    'use_ground': False,
    'split': False,
    'cuda': False
}

# ds = CnfDataset(fname,50)

def_settings = CnfSettings(hyperparams)



def do_experiment():
    ds1 = CnfDataset(DS_TRAIN_TEMPLATE % def_settings['dataset'],7000,mode=DataMode.TF)
    ds2 = CnfDataset(DS_TRAIN_TEMPLATE % def_settings['dataset'],7000,mode=DataMode.TF)


def log_name(settings):
    return 'run_%s_%s_nc%d_bs%d_ed%d_iters%d__%s' % (settings['classifier_type'], settings['dataset'], settings['num_classes'], 
        settings['batch_size'], settings['embedding_dim'], 
        settings['max_iters'], int(time.time()))
    

def test(model, ds, **kwargs):
    settings = kwargs['settings'] if 'settings' in kwargs.keys() else CnfSettings()
    criterion = nn.CrossEntropyLoss()
    if 'weighted_test' in kwargs and kwargs['weighted_test']:
        sampler = torch.utils.data.sampler.WeightedRandomSampler(ds.weights_vector, len(ds))
        vloader = torch.utils.data.DataLoader(ds, batch_size=1, sampler = sampler, pin_memory=settings['cuda'])
    else:
        sampler = torch.utils.data.sampler.RandomSampler(ds)
        vloader = torch.utils.data.DataLoader(ds, batch_size=1, sampler = sampler)
    total_loss = 0
    total_correct = 0
    for _,data in zip(range(settings['val_size']),vloader):
        inputs = utils.formula_to_input(data['sample'])
        topvar = torch.abs(Variable(data['topvar'], requires_grad=False))
        labels = Variable(data['label'], requires_grad=False)
        if settings.hyperparameters['cuda']:
                topvar, labels = topvar.cuda(), labels.cuda()
                inputs = [[[x.cuda() for x in y] for y in t] for t in inputs]
        outputs, aux_losses = model(inputs, topvar)
        loss = criterion(outputs, labels)   # + torch.sum(aux_losses)
        correct = (outputs.max() == outputs[:,labels.data[0]]).data.all()   # Did we get it?
        total_correct += 1 if correct else 0
        total_loss += loss

    return total_loss, total_correct / settings['val_size']


def train(ds, ds_validate=None):
    settings = CnfSettings()
    sampler = torch.utils.data.sampler.WeightedRandomSampler(ds.weights_vector, len(ds))
    trainloader = torch.utils.data.DataLoader(ds, batch_size=1, sampler = sampler, pin_memory=settings['cuda'])
    # dataiter = iter(trainloader)
    print('%d classes, %d samples'% (ds.num_classes,len(ds)))
    settings.hyperparameters['num_classes'] = ds.num_classes
    settings.hyperparameters['max_clauses'] = ds.max_clauses

    cl_type = eval(settings['classifier_type'])
    net = cl_type(**hyperparams)
    if settings.hyperparameters['cuda']:
        net.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    batch_size = settings['batch_size']
    get_step = lambda x,y: x*len(ds)+y

    configure("runs/%s" % log_name(settings), flush_secs=5)

    do_step = True
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs, no batching unfortunately...          
            ds_idx = data['idx_in_dataset'][0]
            # if ds_idx==541:
            #     ipdb.set_trace()
            inputs = utils.formula_to_input(data['sample'])
            topvar = Variable(data['topvar'], requires_grad=False)
            labels = Variable(data['label'], requires_grad=False)
            if settings.hyperparameters['cuda']:
                topvar, labels = topvar.cuda(), labels.cuda()
                inputs = [[[x.cuda() for x in y] for y in t] for t in inputs]
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
                # print('Outputs are:')
                # print(outputs)
                # print('And labels:')
                # print(labels)
                if ds_validate:
                    v_loss, v_acc = test(net, ds_validate, weighted_test=True)
                    v_loss = v_loss.data.numpy() if not settings['cuda'] else v_loss.cpu().data.numpy()
                    print('Validation loss %f, accuracy %f' % (v_loss,v_acc))
                    log_value('validation_loss',v_loss,get_step(epoch,i))
                    log_value('validation_accuracy',v_acc,get_step(epoch,i))

            if i>0 and i % SAVE_EVERY == 0:
                torch.save(net.state_dict(),'%s/%s.model' % (settings['model_dir'],log_name(settings)))


    print('Finished Training')
