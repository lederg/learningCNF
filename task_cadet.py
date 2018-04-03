import os.path
import torch
# from torch.distributions import Categorical
import ipdb
import pdb
import random
import time
from tensorboard_logger import configure, log_value

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

SAVE_EVERY = 500
INVALID_ACTION_REWARDS = -100
TEST_EVERY = 500

all_episode_files = ['data/mvs.qdimacs']

settings = CnfSettings()

reporter = PGEpisodeReporter("{}/{}".format(settings['rl_log_dir'], log_name(settings)), tensorboard=settings['report_tensorboard'])
env = CadetEnv(**settings.hyperparameters)
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

  logits, values = model(obs, **kwargs)
  
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

  if settings['do_test']:
    test_envs(random_test=True)
  if settings['do_not_run']:
    return
  total_steps = 0
  mse_loss = nn.MSELoss()
  stepsize = settings['init_lr']
  policy = create_policy()
  optimizer = optim.Adam(policy.parameters(), lr=stepsize)  
  # optimizer = optim.SGD(policy.parameters(), lr=settings['init_lr'], momentum=0.9)
  # optimizer = optim.RMSprop(policy.parameters())
  reporter.log_env(settings['rl_log_envs'])
  ds = QbfDataset(fnames=settings['rl_train_data'])
  all_episode_files = ds.get_files_list()
  old_logits = None
  max_steps = len(ds)*100
  print('Running for {} steps..'.format(max_steps))
  for i in range(max_steps):
    rewards = []
    transition_data = []
    total_transitions = []
    time_steps_this_batch = 0
    begin_time = time.time()
    while time_steps_this_batch < settings['min_timesteps_per_batch']:      
      episode, env_id, entropy = handle_episode(model=policy)
      s = len(episode)
      total_steps += s
      time_steps_this_batch += s      
      if settings['rl_log_all']:
        reporter.log_env(env_id)      
      # episode is a list of half-empty Tranitions - (obs, action, None, None), we want to turn it to (obs,action,None, None)
      reporter.add_stat(env_id,s,sum(env.rewards), entropy, total_steps)
      r = discount(env.rewards, settings['gamma'])
      transition_data.extend([Transition(transition.state, transition.action, None, rew) for transition, rew in zip(episode, r)])
    
    print('Finished batch with total of %d steps in %f seconds' % (time_steps_this_batch, sum(inference_time)))
    if not (i % 250) and i>0:
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
    logits, values = policy(collated_batch.state)
    probs = F.softmax(logits)    
    all_logprobs = safe_logprobs(probs)
    returns = torch.Tensor(rewards)
    if settings['ac_baseline']:
      adv_t = returns - values.squeeze().data
      value_loss = mse_loss(values, Variable(returns))    
      print('Value loss is {}'.format(value_loss.data.numpy()))
    else:
      adv_t = returns
      value_loss = 0.
    logprobs = all_logprobs.gather(1,Variable(collated_batch.action).view(-1,1)).squeeze()            
    entropies = (-probs*all_logprobs).sum(1)
    if settings['cuda']:
      returns = returns.cuda()
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + np.finfo(np.float32).eps)
    pg_loss = (-Variable(adv_t)*logprobs - settings['entropy_alpha']*entropies).sum()
    loss = pg_loss + value_loss
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm(policy.parameters(), settings['grad_norm_clipping'])
    if any([(x.grad!=x.grad).data.any() for x in policy.parameters() if x.grad is not None]): # nan in grads
      print('NaN in grads!')
      ipdb.set_trace()
    # tutils.clip_grad_norm(policy.parameters(), 40)
    optimizer.step()
    end_time = time.time()
    print('Backward computation done in %f seconds' % (end_time-begin_time))


    # Change learning rate according to KL

    if settings['adaptive_lr']:
      old_logits = logits
      logits, _ = policy(collated_batch.state)    
      kl = compute_kl(logits.data,old_logits.data)
      kl = kl.mean()
      if kl > settings['desired_kl'] * 2: 
        stepsize /= 1.5
        print('stepsize -> %s'%stepsize)
        utils.set_lr(optimizer,stepsize)
      elif kl < settings['desired_kl'] / 2: 
        stepsize *= 1.5
        print('stepsize -> %s'%stepsize)
        utils.set_lr(optimizer,stepsize)
      else:
        print('stepsize OK')




    if i % SAVE_EVERY == 0 and i>0:
      torch.save(policy.state_dict(),'%s/%s_step%d.model' % (settings['model_dir'],utils.log_name(settings), total_steps))
    if i % TEST_EVERY == 0 and i>0:
      if settings['rl_validation_data']:
        print('Testing envs:')
        val_average = test_envs(fnames=settings['rl_validation_data'], model=policy)
        log_value('Validation', val_average, total_steps)
        test_average = test_envs(fnames=settings['rl_test_data'], model=policy)
        log_value('Test', test_average, total_steps)

  
def test_one_env(fname, iters=100, threshold=100000, **kwargs):
  s = 0.
  for _ in range(iters):
    r, _, _ = handle_episode(fname=fname, **kwargs)
    if len(r) > 1000:
      print('{} took {} steps!'.format(fname,len(r)))
      break
    s += len(r)

  if s/iters < threshold:
    print('For {}, average steps: {}'.format(fname,s/iters))
  return s/iters

def test_envs(fnames=settings['rl_train_data'], **kwargs):
  ds = QbfDataset(fnames=fnames)
  all_episode_files = ds.get_files_list()
  totals = 0.
  totals_act = 0.
  for fname in all_episode_files:
    print('Starting {}'.format(fname))
    totals += test_one_env(fname, **kwargs)
  print("Total average: {}".format(totals/len(all_episode_files)))
  return totals/len(all_episode_files)
