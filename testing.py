import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from datautils import *
from settings import *
import utils

def test(model, ds: CnfDataset, **kwargs):
    test_bs = 5
    # test_bs = settings['batch_size']
    settings = kwargs['settings'] if 'settings' in kwargs.keys() else CnfSettings()
    criterion = nn.CrossEntropyLoss()
    if 'weighted_test' in kwargs and kwargs['weighted_test']:
        sampler = torch.utils.data.sampler.WeightedRandomSampler(ds.weights_vector, len(ds))
        vloader = torch.utils.data.DataLoader(ds, batch_size=test_bs, sampler = sampler, pin_memory=settings['cuda'])
    else:
        sampler = torch.utils.data.sampler.RandomSampler(ds)
        vloader = torch.utils.data.DataLoader(ds, batch_size=test_bs, sampler = sampler)
    total_loss = 0
    total_correct = 0
    total_iters = 0
    print('Begin testing, number of mini-batches is %d' % len(vloader))

    for _,data in zip(range(settings['val_size']),vloader):
        inputs = (data['variables'], data['clauses'])
        if  len(inputs[0]) != test_bs:
            print('Trainer gave us no batch!!')
            continue
        topvar = torch.abs(Variable(data['topvar'], requires_grad=False))
        labels = Variable(data['label'], requires_grad=False)
        if settings.hyperparameters['cuda']:
                topvar, labels = topvar.cuda(), labels.cuda()
                inputs = [t.cuda() for t in inputs]        
        outputs, aux_losses = model(inputs, output_ind=topvar, batch_size=test_bs)
        loss = criterion(outputs, labels)   # + torch.sum(aux_losses)
        correct = outputs.max(dim=1)[1]==labels
        num_correct = torch.nonzero(correct.data).size()
        if len(num_correct):
            total_correct += num_correct[0]
        total_loss += loss
        total_iters += 1
        print('Testing, iteration %d, total_correct = %d' % (total_iters, total_correct))

    return total_loss, total_correct / (total_iters*test_bs)

def get_embeddings(model, ds: CnfDataset, **kwargs):
    test_bs = 5
    # test_bs = settings['batch_size']
    settings = kwargs['settings'] if 'settings' in kwargs.keys() else CnfSettings()    
    sampler = torch.utils.data.sampler.SequentialSampler(ds)
    vloader = torch.utils.data.DataLoader(ds, batch_size=test_bs, sampler = sampler)
    
    total_iters = 0
    print('Begin forward embedding, number of mini-batches is %d' % len(vloader))

    for data in vloader:
        inputs = (data['variables'], data['clauses'])
        if  len(inputs[0]) != test_bs:
            print('Short batch!')            
        topvar = torch.abs(Variable(data['topvar'], requires_grad=False))
        labels = Variable(data['label'], requires_grad=False)
        if settings.hyperparameters['cuda']:
                topvar, labels = topvar.cuda(), labels.cuda()
                inputs = [t.cuda() for t in inputs]        
        outputs, aux_losses = model.encoder(inputs, output_ind=topvar, batch_size=test_bs)
        enc = self.embedder(embeddings,output_ind=topvar, batch_size=test_bs)        
        total_iters += 1
        

    return total_loss, total_correct / (total_iters*test_bs)
