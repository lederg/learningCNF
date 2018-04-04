import torch
import itertools
import numpy as np
from torch.autograd import Variable
import scipy.signal
import getopt
import sys
from pprint import pprint
import functools
import weakref
from enum import Enum

class BaseMode(Enum):
    ALL = 1
    EMBEDDING = 2


def where(cond, x_1, x_2):
    cond = cond.float()    
    return (cond * x_1) + ((1-cond) * x_2)

def copy_model_weights(model1, model2):
    model2.load_state_dict(model1.state_dict())

def memoized_method(*lru_args, **lru_kwargs):
    def decorator(func):
        @functools.wraps(func)
        def wrapped_func(self, *args, **kwargs):
            # We're storing the wrapped method inside the instance. If we had
            # a strong reference to self the instance would never die.
            self_weak = weakref.ref(self)
            @functools.wraps(func)
            @functools.lru_cache(*lru_args, **lru_kwargs)
            def cached_method(*args, **kwargs):
                return func(self_weak(), *args, **kwargs)
            setattr(self, func.__name__, cached_method)
            return cached_method(*args, **kwargs)
        return wrapped_func
    return decorator

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
    

def convert_var(v, reverse=False):
    s = sign(v)
    val = abs(v)

def log_name(settings):
    return 'run_%s_bs%d_ed%d_iters%d__%s' % (settings['name'], 
        settings['batch_size'], settings['embedding_dim'], 
        settings['max_iters'], settings['exp_time'])


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


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7, decay_rate=0.1):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * ((1-decay_rate)**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0 and epoch > 0:
        print('LR is set to {}'.format(lr))
        set_lr(optimizer,lr)

    return optimizer

def discount(x, gamma):
    """
    Compute discounted sum of future values
    out[i] = in[i] + gamma * in[i+1] + gamma^2 * in[i+2] + ...
    """
    return scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]

# cross-product

def dict_product(dicts):
    """    
    >>> list(dict_product(dict(number=[1,2], character='ab')))
    [{'character': 'a', 'number': 1},
     {'character': 'a', 'number': 2},
     {'character': 'b', 'number': 1},
     {'character': 'b', 'number': 2}]
    """
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))

    
class EnvIdGen(metaclass=Singleton):
    def __init__(self, initial_id=10000):
        self.initial_id = initial_id

    def get_id(self):
        rc = self.initial_id
        self.initial_id += 1
        return rc
