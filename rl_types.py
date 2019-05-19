from collections import namedtuple
from namedlist import namedlist

State = namedtuple('State', 
                    ['state','cmat', 'ground', 'clabels', 'vmask', 'cmask', 'ext_data'])

DenseState = namedtuple('DenseState', 
                    ['state','cmat_ind', 'cmat_val', 'cmat_size', 'ground', 'clabels', 'vmask', 'cmask', 'ext_data'])


# pack_indices for different formulas is  (clause_indices, variable_indices), where each is [0,s_1,..,s_n] for n formulas

PackedState = namedtuple('PackedState', 
                    ['state', 'cmat_pos', 'cmat_neg', 'ground', 'clabels', 'pack_indices'])

Transition = namedlist('Transition',
                        ['state', 'action', 'next_state', 'reward', 'formula', 'prev_obs'])

# # Inner struct for EpisodeManager and MPEpisodeManager
# EnvStruct = namedlist('EnvStruct',
#                     ['env', 'last_obs', 'episode_memory', 'env_id', 'fname', 'curr_step', 'active', 'prev_obs', 'start_time', 'end_time'])

EmptyState = State(state=None, cmat=None, ground=None, clabels=None, vmask=None, cmask=None, ext_data=None)