from collections import namedtuple
from namedlist import namedlist

State = namedtuple('State', 
                    ['state', 'cmat_pos', 'cmat_neg', 'ground', 'clabels'])

Transition = namedlist('Transition',
                        ['state', 'action', 'next_state', 'reward', 'formula'])


EnvObservation = namedtuple('EnvObservation', 
                    ['state', 'vars_add', 'vars_remove', 'activities', 'decision', 'clause', 
                    	'reward', 'vars_set', 'done'])
