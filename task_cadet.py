from settings import *
from cadet_env import *
from rl_model import *
import ipdb

CADET_BINARY = './cadet'

all_episode_files = ['data/mvs.qdimacs']

settings = CnfSettings()
policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
env = CadetEnv(CADET_BINARY, **settings.hyperparameters)

def select_action(state, actions):
  state = torch.from_numpy(state).float().unsqueeze(0)
  actions = torch.from_numpy(actions).unsqueeze(0)
  probs = policy(Variable(state), Variable(actions))
  m = Categorical(probs)
  ipdb.set_trace()
  action = m.sample()
  policy.saved_log_probs.append(m.log_prob(action))
  return action.data[0]


def handle_episode(fname):
  state, cands, _ = env.reset(fname)
  policy.re_init_qbf_base(env.qbf)
  for t in range(10000):  # Don't infinite loop while learning
    # ipdb.set_trace()
    action = select_action(state, cands)
    state, cands, done = env.step(action)        
    if done:
      break
def cadet_main(settings):
  for fname in all_episode_files:
    handle_episode(fname)
    
