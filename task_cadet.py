import os.path
import torch
# from torch.distributions import Categorical
import ipdb
import pdb
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

reporter = PGEpisodeReporter("{}/{}".format(settings['rl_log_dir'], log_name(settings)), tensorboard=settings['report_tensorboard'])
env = CadetEnv(CADET_BINARY, **settings.hyperparameters)
exploration = LinearSchedule(1, 1.)
total_steps = 0
inference_time = []



def select_action(obs, model=None, testing=False, random_test=False, activity_test=False, **kwargs):    
  activities = obs.ground.data.numpy()[0,:,IDX_VAR_ACTIVITY]
  if random_test or (activity_test and not np.any(activities)):
    choices = np.where(1-obs.ground.data.numpy()[0,:,IDX_VAR_DETERMINIZED])[0]
    action = np.random.choice(choices)
    return action, None
  elif activity_test:    
    action = np.argmax(activities)
    return action, None

  logits = model(obs, **kwargs)
  
  if testing:
    action = logits.squeeze().max(0)[1].data   # argmax when testing        
    action = action[0]
    if settings['debug_actions']:
      print(obs.state)
      print(logits.squeeze().max(0))
      print('Got action {}'.format(action))
  else:
    probs = F.softmax(logits)
    dist = probs.data.cpu().numpy()[0]
    choices = range(len(dist))
    action = np.random.choice(choices, p=dist) 
  return action, logits



def handle_episode(**kwargs):
  global total_steps
  episode_memory = ReplayMemory(5000)

  last_obs, env_id = new_episode(env,all_episode_files, **kwargs)
  if settings['rl_log_all']:
    reporter.log_env(env_id)
  entropies = []

  for t in range(5000):  # Don't infinite loop while learning
    begin_time = time.time()
    action, logits = select_action(last_obs, **kwargs)    
    if logits is not None:
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
    if not 'testing' in kwargs:
      total_steps += 1
    if done:
      break
    obs = process_observation(env,last_obs,env_obs)
    last_obs = obs

  return episode_memory, env_id, np.mean(entropies) if entropies else None

def ts_bonus(s):
  b = 5.
  return b/float(s)

def cadet_main():
  global all_episode_files, total_steps

  random_test_envs()
  total_steps = 0
  policy = create_policy()
  optimizer = optim.Adam(policy.parameters(), lr=settings['init_lr'])
  # optimizer = optim.SGD(policy.parameters(), lr=settings['init_lr'], momentum=0.9)
  # optimizer = optim.RMSprop(policy.parameters())
  reporter.log_env(settings['rl_log_envs'])
  ds = QbfDataset(fnames=settings['rl_train_data'])
  all_episode_files = ds.get_files_list()  
  for i in range(5000):
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
      reporter.report_stats(total_steps, len(all_episode_files))
      print('Testing all episodes:')
      for fname in all_episode_files:
        _, _ , _= handle_episode(model=policy, testing=True, fname=fname)
        r = env.rewards
        print('Env %s completed test in %d steps with total reward %f' % (fname, len(r), sum(r)))
    inference_time.clear()

    if settings['do_not_learn']:
      continue
    begin_time = time.time()
    states, actions, next_states, rewards = zip(*transition_data)
    collated_batch = collate_transitions(transition_data,settings=settings)
    logits = policy(collated_batch.state)
    returns = torch.Tensor(rewards)
    # batch_entropy = torch.cat(entropies)
    batch_entropy = 0.
    logprobs = F.softmax(logits).gather(1,Variable(collated_batch.action).view(-1,1)).log().squeeze()        
    if settings['cuda']:
      returns = returns.cuda()
    returns = (returns - returns.mean()) / (returns.std() + np.finfo(np.float32).eps)
    loss = (-Variable(returns)*logprobs - 0.0*batch_entropy).sum()    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm(policy.parameters(), settings['grad_norm_clipping'])
    if any([(x.grad!=x.grad).data.any() for x in policy.parameters() if x.grad is not None]): # nan in grads
      ipdb.set_trace()
    # tutils.clip_grad_norm(policy.parameters(), 40)
    optimizer.step()
    end_time = time.time()
    print('Backward computation done in %f seconds' % (end_time-begin_time))
    if i % SAVE_EVERY == 0:
      torch.save(policy.state_dict(),'%s/%s_step%d.model' % (settings['model_dir'],utils.log_name(settings), total_steps))
    

def random_test_one_env(fname, iters=100, threshold=100000, **kwargs):
  s = 0.
  for _ in range(iters):
    r, _, _ = handle_episode(fname=fname, **kwargs)
    if len(r) > 1000:
      print('{} took {} steps!'.format(fname,len(r)))
      break
    s += len(r)

  if s/iters < threshold:
    print('For {}, average random steps: {}'.format(fname,s/iters))
  return s/iters



def random_test_envs():
  ds = QbfDataset(fnames=settings['rl_train_data'])
  all_episode_files = ds.get_files_list()
  totals_rand = 0.
  totals_act = 0.
  for fname in all_episode_files:
    totals_rand += random_test_one_env(fname, random_test=True)
  print("random average: {}".format(totals_rand/len(all_episode_files)))
  # for fname in all_episode_files:
  #   totals_act += random_test_one_env(fname, activity_test=True)

  # print("activity-based average: {}".format(totals_act/len(all_episode_files)))

