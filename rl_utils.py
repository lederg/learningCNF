import ipdb
import copy
import torch
from torch.autograd import Variable
from settings import *
from rl_types import *
from cadet_utils import *
from rl_model import *
from new_policies import Actor1Policy

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

    def __getitem__(self, idx):
      return self.memory[idx]

    def __len__(self):
        return len(self.memory)


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

def discount_episode(ep, gamma, settings=None):
  if not settings:
    settings = CnfSettings()
    
  _, _, _,rewards, _ = zip(*ep)
  r = discount(rewards, gamma)
  return [Transition(transition.state, transition.action, None, rew, transition.formula) for transition, rew in zip(ep, r)]

def collate_observations(batch, settings=None):
  if batch.count(None) == len(batch):
    return State(None, None, None, None, None, None, None)
  if not settings:
    settings = CnfSettings()
  bs = len(batch)
  c_size = max([x.cmat_neg.size(0) for x in batch if x])
  v_size = max([x.cmat_neg.size(1) for x in batch if x])
  states = []
  ind_pos = []
  ind_neg = []
  val_pos = []
  val_neg = []
  all_embs = []
  all_clabels = []
  i=0
  vmask = settings.zeros((len(batch),v_size))
  cmask = settings.zeros((len(batch),c_size))
  for i, b in enumerate(batch):
    if b:      
      states.append(b.state)
      ind_pos.append(b.cmat_pos.data._indices() + torch.LongTensor([i*c_size,i*v_size]).view(2,1))
      ind_neg.append(b.cmat_neg.data._indices() + torch.LongTensor([i*c_size,i*v_size]).view(2,1))
      val_pos.append(b.cmat_pos.data._values())
      val_neg.append(b.cmat_neg.data._values())
      embs = b.ground.squeeze()
      clabels = b.clabels.t()                            # 1*num_clauses  ==> num_clauses*1 (1 is for dim now)
      l = len(embs)
      vmask[i][:l]=1
      if l < v_size:
        embs = torch.cat([embs,torch.zeros([v_size-l,settings['ground_dim']])])
      all_embs.append(embs)
      l = len(clabels)
      cmask[i][:l]=1
      if l < c_size:        
        clabels = torch.cat([clabels,torch.zeros([c_size-l,settings['clabel_dim']])])
      all_clabels.append(clabels)
      i += 1    
      # states.append(Variable(settings.zeros([1,settings['state_dim']])))
      # all_embs.append(Variable(settings.zeros([v_size,settings['ground_dim']])))
  cmat_pos = torch.sparse.FloatTensor(torch.cat(ind_pos,1),torch.cat(val_pos),torch.Size([c_size*i,v_size*i]))
  cmat_neg = torch.sparse.FloatTensor(torch.cat(ind_neg,1),torch.cat(val_neg),torch.Size([c_size*i,v_size*i]))
  # if settings['cuda']:
  #   cmat_pos, cmat_neg = cmat_pos.cuda(), cmat_neg.cuda()
  return State(torch.cat(states),Variable(cmat_pos),Variable(cmat_neg),torch.stack(all_embs), torch.stack(all_clabels),
                vmask, cmask)

def packed_collate_observations(batch, settings=None):
  if batch.count(None) == len(batch):
    return PackedState(None, None, None, None, None, None)
  if not settings:
    settings = CnfSettings()
  bs = len(batch)
  c_size = max([x.cmat_neg.size(0) for x in batch if x])
  v_size = max([x.cmat_neg.size(1) for x in batch if x])
  states = []
  ind_pos = []
  ind_neg = []
  val_pos = []
  val_neg = []
  all_embs = []
  all_clabels = []
  v_indices = [0]
  c_indices = [0]
  
  for b in batch:
    if b:      
      states.append(b.state)
      ind_pos.append(b.cmat_pos.data._indices() + torch.LongTensor([c_indices[-1],v_indices[-1]]).view(2,1))
      ind_neg.append(b.cmat_neg.data._indices() + torch.LongTensor([c_indices[-1],v_indices[-1]]).view(2,1))
      val_pos.append(b.cmat_pos.data._values())
      val_neg.append(b.cmat_neg.data._values())
      c_indices.append(c_indices[-1] + b.cmat_pos.size(0))
      v_indices.append(v_indices[-1] + b.cmat_pos.size(1))
      embs = b.ground.squeeze()
      clabels = b.clabels.t()                            # 1*num_clauses  ==> num_clauses*1 (1 is for dim now)      
      all_embs.append(embs)
      all_clabels.append(clabels)
  cmat_pos = torch.sparse.FloatTensor(torch.cat(ind_pos,1),torch.cat(val_pos),torch.Size([c_indices[-1],v_indices[-1]]))
  cmat_neg = torch.sparse.FloatTensor(torch.cat(ind_neg,1),torch.cat(val_neg),torch.Size([c_indices[-1],v_indices[-1]]))
  return PackedState(torch.cat(states),Variable(cmat_pos),Variable(cmat_neg),torch.cat(all_embs), torch.cat(all_clabels), (c_indices, v_indices))

def collate_transitions(batch, settings=None, packed=False):  
  if not settings:
    settings = CnfSettings()

  collate_fn = packed_collate_observations if packed else collate_observations
  obs1 = collate_fn([x.state for x in batch], settings)
  obs2 = collate_fn([x.next_state for x in batch], settings)
  rews = settings.FloatTensor([x.reward for x in batch])
  actions = settings.LongTensor([x.action for x in batch])
  # formulas = settings.LongTensor([x.formula for x in batch])
  formulas = [x.formula for x in batch]
  return Transition(obs1,actions,obs2,rews, formulas)

def create_policy(settings=None, is_clone=False):
  if not settings:
    settings = CnfSettings()
  base_model = settings['base_model']
  policy_class = eval(settings['policy'])
  if base_model and not is_clone:
    print('Loading parameters from {}'.format(base_model))
    if settings['base_mode'] == BaseMode.ALL:
      policy = policy_class(settings=settings)
      policy.load_state_dict(torch.load('{}/{}'.format(settings['model_dir'],base_model)))
    elif settings['base_mode'] == BaseMode.ITERS:
      base_iters = settings['base_iters']
      if base_iters != settings['max_iters']:
        base_settings = copy.deepcopy(settings)
        base_settings.hyperparameters['max_iters'] = base_iters
      else:
        base_settings = settings
      base_policy = policy_class(settings=base_settings)
      base_policy.load_state_dict(torch.load('{}/{}'.format(settings['model_dir'],base_model)))      
      policy = policy_class(settings=settings)
      policy.encoder.copy_from_encoder(base_policy.encoder, freeze=True)
    else:
      model = QbfClassifier()
      model.load_state_dict(torch.load('{}/{}'.format(settings['model_dir'],base_model)))
      encoder=model.encoder
      policy = policy_class(settings=settings,encoder=encoder)
  else:
    policy = policy_class(settings=settings)
  if settings['cuda']:
    policy = policy.cuda()

  return policy

def unpack_logits(logits, vp_ind, settings=None):
  if not settings:
    settings = CnfSettings()  
  sizes = [vp_ind[i+1] - vp_ind[i] for i in range(len(vp_ind)-1)]
  vmax = max(sizes)
  zero_seed = Variable(settings.zeros(1))
  rc = []
  for i, l in enumerate(sizes):
    if vmax-l:
      rc.append(torch.cat([logits[vp_ind[i]:vp_ind[i+1]], zero_seed.expand(vmax-l,2)]))
    else:
      rc.append(logits[vp_ind[i]:vp_ind[i+1]])    
  return torch.stack(rc)

def masked_softmax2d(A, mask):
  # ipdb.set_trace()
  antimask = (1-mask)*(-1e20)
  A_max = torch.max(A*mask+antimask,dim=1,keepdim=True)[0]
  B = A*mask - A_max  
  A_exp = torch.exp(torch.clamp(B*mask,max=20))
  A_exp = A_exp * mask 
  Z = torch.sum(A_exp,dim=1,keepdim=True)
  A_softmax = A_exp / Z
  return A_softmax, Z

def masked_softmax2d_loop(A, mask):
  A_max = []
  for i, row in enumerate(A):
    A_max.append(torch.max(row[mask[i]==1])[0])
  A_max = torch.cat(A_max).unsqueeze(1)
  A_exp = torch.exp(torch.clamp(A*mask-A_max,max=20))
  A_exp = A_exp * mask 
  Z = torch.sum(A_exp,dim=1,keepdim=True)
  A_softmax = A_exp / Z
  return A_softmax, Z


def safe_logprobs(probs, settings=None, thres=1e-4):
  if not settings:
    settings = CnfSettings()  
  zero_probs = Variable(settings.zeros(probs.size()))
  fake_probs = zero_probs + 100
  aug_probs = torch.stack([fake_probs, probs])
  index_probs = (probs>thres).long().unsqueeze(0)
  aug_logprobs = torch.stack([zero_probs,aug_probs.gather(0,index_probs).squeeze().log()])
  all_logprobs = aug_logprobs.gather(0,index_probs).squeeze()
  return all_logprobs

def compute_kl(logits, old_logits):
  s = logits.size(0)
  old_logits = old_logits.view(s,-1)
  logits = logits.view(s,-1)
  totals = F.softmax(old_logits,dim=1) * (F.log_softmax(old_logits,dim=1) - F.log_softmax(logits,dim=1))
  return totals.sum(1).data
