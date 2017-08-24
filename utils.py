import torch
import itertools
import numpy as np
from torch.autograd import Variable

def normalize(input, p=2, dim=1, eps=1e-12):
    return input / input.norm(p, dim).clamp(min=eps).expand_as(input)

def formula_to_input(formula):
    try:
        return [[[Variable(x, requires_grad=False) for x in y] for y in t] for t in formula]    
    except:
        return [[[Variable(torch.LongTensor([x]), requires_grad=False) for x in y] for y in t] for t in formula]    


def permute_seq(inp):
    # inp is a sequence of tensors, we return a random permutation

    p = list(itertools.permutations(inp))
    i = np.random.randint(len(p))
    return list(p[i])


def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0 and epoch > 0:
        print('LR is set to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return optimizer