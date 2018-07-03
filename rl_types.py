from collections import namedtuple
from namedlist import namedlist

State = namedtuple('State', 
                    ['state', 'cmat_pos', 'cmat_neg', 'ground', 'clabels', 'vmask', 'cmask'])


# pack_indices for different formulas is  (clause_indices, variable_indices), where each is [0,s_1,..,s_n] for n formulas

PackedState = namedtuple('PackedState', 
                    ['state', 'cmat_pos', 'cmat_neg', 'ground', 'clabels', 'pack_indices'])

Transition = namedlist('Transition',
                        ['state', 'action', 'next_state', 'reward', 'formula'])


EnvObservation = namedtuple('EnvObservation', 
                    ['state', 'vars_add', 'vars_remove', 'activities', 'decision', 'clause', 
                    	'reward', 'vars_set', 'done'])
