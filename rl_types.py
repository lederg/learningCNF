from collections import namedtuple

State = namedtuple('State', 
                    ['state', 'cmat_pos', 'cmat_neg', 'ground'])

Transition = namedtuple('Transition',
                        ['state', 'action', 'next_state', 'reward'])


EnvObservation = namedtuple('EnvObservation', 
                    ['state', 'vars_add', 'vars_remove', 'activities', 'decision', 'clause', 
                    	'reward', 'vars_set', 'done'])
