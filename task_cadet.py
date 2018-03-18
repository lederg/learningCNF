import os.path
import torch
# from torch.distributions import Categorical
import ipdb
import random
import time

from settings import *
from cadet_env import *
from rl_model import *
from qbf_data import *
from qbf_model import QbfClassifier
from utils import *
from rl_utils import *
from cadet_utils import *
from episode_reporter import *
import torch.nn.utils as tutils

CADET_BINARY = './cadet'
SAVE_EVERY = 10
INVALID_ACTION_REWARDS = -100


all_episode_files = ['data/mvs.qdimacs']

settings = CnfSettings()

reporter = PGEpisodeReporter("{}/{}".format(settings['rl_log_dir'], log_name(settings)))
env = CadetEnv(CADET_BINARY, **settings.hyperparameters)
exploration = LinearSchedule(1, 1.)
total_steps = 0
inference_time = []
all_logits = []


def select_action(obs, model=None, testing=False, **kwargs):    
  logits = model(obs, **kwargs)
  logprob = None
  if testing:
    action = logits.squeeze().max(0)[1].data   # argmax when testing    
    action = action[0]
  else:
    probs = F.softmax(logits)
    dist = probs.data.cpu().numpy()[0]
    choices = range(len(dist))
    action = np.random.choice(choices, p=dist) 


    aug_action = np.where(np.where(dist>0)[0]==action)
    a = probs[probs>0]
    logprob = a.log()[aug_action[0][0]]   
  return action, logits, logprob



def handle_episode(no_naive=False, **kwargs):
  global total_steps
  episode_memory = ReplayMemory(5000)

  last_obs, env_id = new_episode(env,all_episode_files, **kwargs)
  if settings['rl_log_all']:
    reporter.log_env(env_id)
  entropies = []

  for t in range(5000):  # Don't infinite loop while learning
    begin_time = time.time()
    action, logits, logprob = select_action(last_obs, **kwargs)
    if not no_naive:
      all_logits.append(logprob)
    probs = F.softmax(logits.data)
    a = probs[probs>0].data
    entropy = -(a*a.log()).sum()
    entropies.append(entropy)
    inference_time.append(time.time()-begin_time)
    if action_allowed(last_obs,action):
      try:
        env_obs = EnvObservation(*env.step(action))
        state, vars_add, vars_remove, activities, decision, clause, reward, done = env_obs
      except Exception as e:
        print(e)
        ipdb.set_trace()
    else:
      print('Chose an invalid action!')
      # rewards = np.zeros(len(episode_memory)+1) # stupid hack
      # rewards[-1] = INVALID_ACTION_REWARDS
      done = True
    episode_memory.push(last_obs,action,None, None)
    total_steps += 1
    if done:
      break
    obs = process_observation(env,last_obs,env_obs)
    last_obs = obs

  return episode_memory, env_id, np.mean(entropies)

def ts_bonus(s):
  b = 5.
  return b/float(s)

def cadet_main():
  global all_episode_files, total_steps, all_logits

  policy = create_policy()
  optimizer = optim.Adam(policy.parameters(), lr=1e-3)
  # optimizer = optim.SGD(policy.parameters(), lr=settings['init_lr'], momentum=0.9)
  # optimizer = optim.RMSprop(policy.parameters())
  reporter.log_env(settings['rl_log_envs'])
  ds = QbfDataset(fnames=settings['rl_train_data'])
  all_episode_files = ds.get_files_list()  
  for i in range(3000):
    rewards = []
    time_steps_this_batch = 0
    transition_data = []
    total_transitions = []
    time_steps_this_batch = 0
    begin_time = time.time()
    while time_steps_this_batch < settings['min_timesteps_per_batch']:      
      episode, env_id, entropy = handle_episode(model=policy)
      if settings['rl_log_all']:
        reporter.log_env(env_id)      
      s = len(episode)
      time_steps_this_batch += s      
      # episode is a list of half-empty Tranitions - (obs, action, None, None), we want to turn it to (obs,action,None, None)
      reporter.add_stat(env_id,s,sum(env.rewards), entropy, total_steps)
      r = discount(env.rewards, settings['gamma'])
      transition_data.extend([Transition(transition.state, transition.action, None, rew) for transition, rew in zip(episode, r)])
    
    print('Finished batch with total of %d steps in %f seconds' % (time_steps_this_batch, sum(inference_time)))
    if not (i % 10) and i>0:
      reporter.report_stats()
      print('Testing all episodes:')
      for fname in all_episode_files:
        _, _ , _= handle_episode(model=policy, testing=True, fname=fname, no_naive=True)
        r = env.rewards
        print('Env %s completed test in %d steps with total reward %f' % (fname, len(r), sum(r)))

    inference_time.clear()
    begin_time = time.time()
    states, actions, next_states, rewards = zip(*transition_data)
    collated_batch = collate_transitions(transition_data,settings=settings)
    logits = policy(collated_batch.state)
    returns = torch.Tensor(rewards)
    # batch_entropy = torch.cat(entropies)
    batch_entropy = 0.
    logprobs = F.softmax(logits).gather(1,Variable(collated_batch.action).view(-1,1)).log()
    naive_logprobs = torch.cat(all_logits).view(-1,1)
    all_logits = []
    if settings['cuda']:
      returns = returns.cuda()
    returns = (returns - returns.mean()) / (returns.std() + np.finfo(np.float32).eps)
    loss = (-Variable(returns)*logprobs - 0.0*batch_entropy).sum()
    naive_loss = (-Variable(returns)*naive_logprobs - 0.0*batch_entropy).sum()
    # ipdb.set_trace()
    optimizer.zero_grad()
    naive_loss.backward()
    if any([(x.grad!=x.grad).data.any() for x in policy.parameters() if x.grad is not None]): # nan in grads
      ipdb.set_trace()
    # tutils.clip_grad_norm(policy.parameters(), 40)
    optimizer.step()
    end_time = time.time()
    print('Backward computation done in %f seconds' % (end_time-begin_time))
    if i % SAVE_EVERY == 0:
      torch.save(policy.state_dict(),'%s/%s_iter%d.model' % (settings['model_dir'],utils.log_name(settings), i))
    


