import torch
import itertools
import numpy as np
from torch.autograd import Variable
import getopt
import sys
from pprint import pprint



class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
    

def normalize(input, p=2, dim=1, eps=1e-20):
    return input / input.norm(p, dim).clamp(min=eps).expand_as(input)

def formula_to_input(formula):
    try:
        return [[Variable(x, requires_grad=False) for x in y] for y in formula]    
    except:
        return [[Variable(torch.LongTensor([x]), requires_grad=False) for x in y] for y in formula]    


def permute_seq(inp):
    # inp is a sequence of tensors, we return a random permutation

    p = list(itertools.permutations(inp))
    i = np.random.randint(len(p))
    return list(p[i])


def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7, decay_rate=0.1):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * ((1-decay_rate)**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0 and epoch > 0:
        print('LR is set to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return optimizer
