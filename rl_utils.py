import torch
from torch.autograd import Variable
from settings import *
from rl_model import *
from collections import namedtuple


State = namedtuple('State', 
                    ['state', 'cmat_pos', 'cmat_neg', 'ground'])

Transition = namedtuple('Transition',
                        ['state', 'action', 'next_state', 'reward'])


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def get_input_from_qbf(qbf, settings=None):
  if not settings:
    settings = CnfSettings()
  a = qbf.as_np_dict()  
  rc_i = a['sp_indices']
  rc_v = a['sp_vals']
  sp_ind_pos = torch.from_numpy(rc_i[np.where(rc_v>0)])
  sp_ind_neg = torch.from_numpy(rc_i[np.where(rc_v<0)])
  sp_val_pos = torch.ones(len(sp_ind_pos))
  sp_val_neg = torch.ones(len(sp_ind_neg))
  cmat_pos = Variable(torch.sparse.FloatTensor(sp_ind_pos.t(),sp_val_pos,torch.Size([qbf.num_clauses,qbf.num_vars])))
  cmat_neg = Variable(torch.sparse.FloatTensor(sp_ind_neg.t(),sp_val_neg,torch.Size([qbf.num_clauses,qbf.num_vars])))  
  if settings['cuda']:
    cmat_pos, cmat_neg = cmat_pos.cuda(), cmat_neg.cuda()
  return cmat_pos, cmat_neg


def create_policy(settings=None, is_clone=False):
  if not settings:
    settings = CnfSettings()
  base_model = settings['base_model']
  if base_model and not is_clone:
    if settings['base_mode'] == BaseMode.ALL:
      policy = Policy()
      policy.load_state_dict(torch.load('{}/{}'.format(settings['model_dir'],base_model)))
    else:
      model = QbfClassifier()
      model.load_state_dict(torch.load('{}/{}'.format(settings['model_dir'],base_model)))
      encoder=model.encoder
      policy = Policy(encoder=encoder)
  else:
    policy = Policy()
  if settings['cuda']:
    policy = policy.cuda()

  return policy

class Schedule(object):
    def value(self, t):
        """Value of the schedule at time t"""
        raise NotImplementedError()

class ConstantSchedule(object):
    def __init__(self, value):
        """Value remains constant over time.
        Parameters
        ----------
        value: float
            Constant value of the schedule
        """
        self._v = value

    def value(self, t):
        """See Schedule.value"""
        return self._v

def linear_interpolation(l, r, alpha):
    return l + alpha * (r - l)

class PiecewiseSchedule(object):
    def __init__(self, endpoints, interpolation=linear_interpolation, outside_value=None):
        """Piecewise schedule.
        endpoints: [(int, int)]
            list of pairs `(time, value)` meanining that schedule should output
            `value` when `t==time`. All the values for time must be sorted in
            an increasing order. When t is between two times, e.g. `(time_a, value_a)`
            and `(time_b, value_b)`, such that `time_a <= t < time_b` then value outputs
            `interpolation(value_a, value_b, alpha)` where alpha is a fraction of
            time passed between `time_a` and `time_b` for time `t`.
        interpolation: lambda float, float, float: float
            a function that takes value to the left and to the right of t according
            to the `endpoints`. Alpha is the fraction of distance from left endpoint to
            right endpoint that t has covered. See linear_interpolation for example.
        outside_value: float
            if the value is requested outside of all the intervals sepecified in
            `endpoints` this value is returned. If None then AssertionError is
            raised when outside value is requested.
        """
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._interpolation = interpolation
        self._outside_value = outside_value
        self._endpoints      = endpoints

    def value(self, t):
        """See Schedule.value"""
        for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= t and t < r_t:
                alpha = float(t - l_t) / (r_t - l_t)
                return self._interpolation(l, r, alpha)

        # t does not belong to any of the pieces, so doom.
        assert self._outside_value is not None
        return self._outside_value

class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p            = final_p
        self.initial_p          = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction  = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)

def collate_observations(batch, settings=None):
  if not settings:
    settings = CnfSettings()
  c_size = max([x.cmat_neg.size(0) for x in batch if x])
  v_size = max([x.cmat_neg.size(1) for x in batch if x])
  states = []
  ind_pos = []
  ind_neg = []
  val_pos = []
  val_neg = []
  all_embs = []
  i=0
  for b in batch:
    if b:
      states.append(b.state)
      ind_pos.append(b.cmat_pos.data._indices() + settings.LongTensor([i*c_size,i*v_size]).view(2,1))
      ind_neg.append(b.cmat_neg.data._indices() + settings.LongTensor([i*c_size,i*v_size]).view(2,1))
      val_pos.append(b.cmat_pos.data._values())
      val_neg.append(b.cmat_neg.data._values())
      embs = b.ground.squeeze()
      l = len(embs)
      if l < v_size:
        embs = torch.cat([embs,settings.zeros([v_size-l,settings['ground_dim']])])
      all_embs.append(embs)
      i += 1    
      # states.append(Variable(settings.zeros([1,settings['state_dim']])))
      # all_embs.append(Variable(settings.zeros([v_size,settings['ground_dim']])))
  cmat_pos = torch.sparse.FloatTensor(torch.cat(ind_pos,1),torch.cat(val_pos),torch.Size([c_size*i,v_size*i]))
  cmat_neg = torch.sparse.FloatTensor(torch.cat(ind_neg,1),torch.cat(val_neg),torch.Size([c_size*i,v_size*i]))
  if settings['cuda']:
    cmat_pos, cmat_neg = cmat_pos.cuda(), cmat_neg.cuda()
  return State(torch.cat(states),Variable(cmat_pos),Variable(cmat_neg),torch.stack(all_embs))

def collate_transitions(batch, settings=None):  
  if not settings:
    settings = CnfSettings()

  obs1 = collate_observations([x.state for x in batch], settings)
  obs2 = collate_observations([x.next_state for x in batch], settings)
  rews = settings.FloatTensor([x.reward for x in batch])
  actions = settings.LongTensor([x.action for x in batch])
  return Transition(obs1,actions,obs2,rews)
