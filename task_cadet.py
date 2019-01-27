import os.path
import torch
# from torch.distributions import Categorical
import ipdb
import pdb
import random
import time
from tensorboard_logger import configure, log_value

from shared_adam import SharedAdam
from settings import *
from cadet_env import *
from rl_model import *
from new_policies import *
from qbf_data import *
from qbf_model import QbfClassifier
from utils import *
from rl_utils import *
from cadet_utils import *
from episode_reporter import *
from episode_manager import *
from episode_data import *
import torch.nn.utils as tutils

all_episode_files = ['data/mvs.qdimacs']

settings = CnfSettings()

SAVE_EVERY = 500
INVALID_ACTION_REWARDS = -10
TEST_EVERY = settings['test_every']
REPORT_EVERY = 10

reporter = PGEpisodeReporter("{}/{}".format(settings['rl_log_dir'], log_name(settings)), settings, tensorboard=settings['report_tensorboard'])
env = CadetEnv(**settings.hyperparameters)
exploration = LinearSchedule(1, 1.)
total_steps = 0
real_steps = 0
inference_time = []
total_inference_time = 0
lambda_disallowed = settings['lambda_disallowed']
lambda_value = settings['lambda_value']
lambda_aux = settings['lambda_aux']
init_lr = settings['init_lr']
desired_kl = settings['desired_kl']
curr_lr = init_lr
max_reroll = 0


# def select_action(obs, model=None, testing=False, max_test=False, random_test=False, activity_test=False, cadet_test=False, **kwargs):    
#   global max_reroll, real_steps
#   activities = obs.ground.data.numpy()[0,:,IDX_VAR_ACTIVITY]
#   if random_test:
#     choices = np.where(get_allowed_actions(obs).squeeze().numpy())[0]
#     action = np.random.choice(choices)
#     return action, None
#   elif activity_test:
#     if np.any(activities):
#       action = np.argmax(activities)
#     else:
#       choices = np.where(get_allowed_actions(obs).squeeze().numpy())[0]
#       action = choices[0]
#     return action, None
#   elif cadet_test:
#     return '?', None

#   if False and not (real_steps % 500):
#     logits, values = model(obs, do_debug=True, **kwargs)
#   else:
#     logits, values = model(obs, **kwargs)
  
#   if max_test:
#     action = logits.squeeze().max(0)[1].data   # argmax when testing        
#     action = action[0]
#     if settings['debug_actions']:
#       print(obs.state)
#       print(logits.squeeze().max(0))
#       print('Got action {}'.format(action))
#   elif testing or settings['packed']:       # Don't resample, just choose only from the real data.
#     allowed = torch.from_numpy(np.where(get_allowed_actions(obs).squeeze().numpy())[0])
#     l = logits.squeeze()[allowed]
#     probs = F.softmax(l.contiguous().view(1,-1))
#     dist = probs.data.cpu().numpy()[0]
#     choices = range(len(dist))
#     aux_action = np.random.choice(choices, p=dist)
#     if len(logits.size()) > 2:    # logits is  > 1-dimensional, action must be > 1-dimensional too      
#       aux_action = (int(aux_action/2),int(aux_action%2))
#       action = (allowed[aux_action[0]], aux_action[1])
#   else:
#     # probs = F.softmax(logits)
#     probs = F.softmax(logits.contiguous().view(1,-1))
#     real_steps += 1
#     # if not (real_steps % 500):
#     #   ipdb.set_trace()
#     dist = probs.data.cpu().numpy()[0]
#     choices = range(len(dist))
#     i = 0
#     while i<1000:
#       action = np.random.choice(choices, p=dist)
#       if len(logits.size()) > 2:    # logits is  > 1-dimensional, action must be > 1-dimensional too      
#         action = (int(action/2),int(action%2))
#       if not settings['disallowed_aux'] or action_allowed(obs, action):
#         break
#       i = i+1
#     if i > max_reroll:
#       max_reroll = i
#       print('Had to roll {} times to come up with a valid action!'.format(max_reroll))
#     if i>600:
#       # ipdb.set_trace()
#       print("Couldn't choose an action within 600 re-samples. Printing probabilities:")            
#       allowed_mass = (probs.view_as(logits).sum(2).data*get_allowed_actions(obs).float()).sum(1)
#       print('total allowed mass is {}'.format(allowed_mass.sum()))
#       print(dist.max())
#     if i==1000 and testing:     # If we're testing, take random action
#       print('Could not choose an action in testing. Choosing at random!')
#       choices = np.where(get_allowed_actions(obs).squeeze().numpy())[0]
#       action = np.random.choice(choices)
#       return action, None



#   return action, logits.contiguous()

# def handle_episode(**kwargs):
#   episode_memory = ReplayMemory(5000)

#   last_obs, env_id, _ = new_episode(env,all_episode_files, **kwargs)
#   if last_obs is None:      # env solved in 0 steps
#     print('Env {} solved in 0 steps, removing'.format(env_id))
#     return None, None, None
#   if settings['rl_log_all']:
#     reporter.log_env(env_id)
#   entropies = []

#   for t in range(4000):  # Don't infinite loop while learning
#     begin_time = time.time()
#     action, logits = select_action(last_obs, **kwargs)    
#     if logits is not None:
#       probs = F.softmax(logits.data.view(1,-1))
#       a = probs[probs>0].data
#       entropy = -(a*a.log()).sum()
#       entropies.append(entropy)
#     inference_time.append(time.time()-begin_time)
#     if action_allowed(last_obs,action):
#       # try:
#         # ipdb.set_trace()
#       env_obs = EnvObservation(*env.step(action))        
#       state, vars_add, vars_remove, activities, decision, clause, reward, vars_set, done = env_obs      
#       # except Exception as e:
#       #   print(e)
#       #   ipdb.set_trace()
#     else:      
#       print('Chose an invalid action!')
#       print('Entropy for this step: {}'.format(entropy))
#       env.rewards = env.terminate()
#       env.rewards = np.append(env.rewards,INVALID_ACTION_REWARDS)
#       done = True
#     episode_memory.push(last_obs,action,None, None, env_id)
#     # if t % 1500 == 0 and t > 0:
#     #   print('In env {}, step {}'.format(env.current_fname,t))
#     if done:
#       return episode_memory, env_id, np.mean(entropies) if entropies else None
#     obs = process_observation(env,last_obs,env_obs)
#     last_obs = obs

#   print('Env {} took too long, aborting!'.format(env.current_fname))  
#   return None, None, None


def ts_bonus(s):
  b = 5.
  return b/float(s)

def cadet_main():

  def train_model(transition_data):
    policy.train()
    loss, logits = policy.compute_loss(transition_data)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), settings['grad_norm_clipping'])
    if any([(x.grad!=x.grad).data.any() for x in policy.parameters() if x.grad is not None]): # nan in grads
      print('NaN in grads!')
      ipdb.set_trace()
    # tutils.clip_grad_norm(policy.parameters(), 40)
    optimizer.step()
    return logits
  global all_episode_files, total_steps, curr_lr  
  if settings['do_test']:
    test_envs(cadet_test=True, iters=1)
  if settings['do_not_run']:
    print('Not running. Printing settings instead:')
    print(settings.hyperparameters)
    return


    
  total_steps = 0
  bad_episodes = 0
  bad_episodes_not_added = 0
  mse_loss = nn.MSELoss()
  stepsize = settings['init_lr']
  policy = create_policy()
  optimizer = SharedAdam(filter(lambda p: p.requires_grad, policy.parameters()), lr=stepsize)  
  # optimizer = optim.SGD(policy.parameters(), lr=settings['init_lr'], momentum=0.9)
  # optimizer = optim.RMSprop(policy.parameters())
  reporter.log_env(settings['rl_log_envs'])
  ed = EpisodeData(name=settings['name'], fname=settings['base_stats'])
  ProviderClass = eval(settings['episode_provider'])
  provider = iter(ProviderClass(settings['rl_train_data']))
  # ds = QbfCurriculumDataset(fnames=settings['rl_train_data'], ed=ed)
  em = EpisodeManager(provider, ed=ed, parallelism=settings['parallelism'],reporter=reporter)
  # all_episode_files = ds.get_files_list()
  old_logits = None
  disallowed_loss = 0.
  max_iterations = len(provider)*2000
  settings.env = env
  num_steps = len(provider)*15000
  lr_schedule = PiecewiseSchedule([
                                       (0,                   init_lr),
                                       (num_steps / 10, init_lr),
                                       (num_steps / 5,  init_lr * 0.5),
                                       (num_steps / 3,  init_lr * 0.25),
                                       (num_steps / 2,  init_lr * 0.1),
                                  ],
                                  outside_value=init_lr * 0.02) 

  kl_schedule = PiecewiseSchedule([
                                       (0,                   desired_kl),
                                       (num_steps / 10, desired_kl),
                                       (num_steps / 5,  desired_kl * 0.5),
                                       (num_steps / 3,  desired_kl * 0.25),
                                       (num_steps / 2,  desired_kl * 0.1),
                                  ],
                                  outside_value=desired_kl * 0.02) 

  print('Running for {} iterations..'.format(max_iterations))
  for i in range(max_iterations):
    rewards = []
    transition_data = []
    total_transitions = []
    total_envs = []
    time_steps_this_batch = 0
    begin_time = time.time()
    policy.eval()

    if True or settings['parallelism'] > 1:
      while not em.check_batch_finished():
        em.step_all(policy)
      transition_data = em.pop_min_normalized() if settings['episodes_per_batch'] else em.pop_min()
      total_steps = em.real_steps
      if not settings['full_pipeline']:     # We throw away all incomplete episodes to keep it on-policy
        em.reset_all()
      if len(transition_data) == settings['episodes_per_batch']*(settings['max_step']+1):
        print('A lost batch, moving on')
        continue

    total_inference_time = time.time() - begin_time
    print('Finished batch with total of %d steps in %f seconds' % (len(transition_data), total_inference_time))
    if not (i % REPORT_EVERY) and i>0:
      reporter.report_stats(total_steps, len(all_episode_files))
      # print('Testing all episodes:')
      # for fname in all_episode_files:
      #   _, _ , _= handle_episode(model=policy, testing=True, fname=fname)
      #   r = env.rewards
      #   print('Env %s completed test in %d steps with total reward %f' % (fname, len(r), sum(r)))
    inference_time.clear()

    if settings['do_not_learn']:
      continue
    begin_time = time.time()
    logits = train_model(transition_data)
    end_time = time.time()
    # print('Backward computation done in %f seconds' % (end_time-begin_time))


    # Change learning rate according to KL

    if settings['follow_kl']:
      old_logits = logits
      policy.eval()
      logits, *_ = policy(collated_batch.state)    
      kl = compute_kl(logits.data.contiguous().view(effective_bs,-1),old_logits.data.contiguous().view(effective_bs,-1))
      kl = kl.mean()      
      print('desired kl is {}, real one is {}'.format(settings['desired_kl'],kl))
      curr_kl = kl_schedule.value(total_steps)
      if kl > curr_kl * 2: 
        stepsize /= 1.5
        print('stepsize -> %s'%stepsize)
        utils.set_lr(optimizer,stepsize)
      elif kl < curr_kl / 2: 
        stepsize *= 1.5
        print('stepsize -> %s'%stepsize)
        utils.set_lr(optimizer,stepsize)
      else:
        print('stepsize OK')

    elif settings['rl_decay']:
      new_lr = lr_schedule.value(total_steps)
      if new_lr != curr_lr:
        utils.set_lr(optimizer,new_lr)
        curr_lr = new_lr



    if settings['restart_solver_every'] and not (i % settings['restart_solver_every']) and i > 0:      
      em.restart_all()

    if i % SAVE_EVERY == 0 and i>0:
      torch.save(policy.state_dict(),'%s/%s_step%d.model' % (settings['model_dir'],utils.log_name(settings), total_steps))
      ed.save_file()
    if i % TEST_EVERY == 0 and i>0:
      if settings['rl_validation_data']:
        print('Testing envs:')
        rc = em.test_envs(settings['rl_validation_data'], policy, iters=2)
        z = np.array(list(rc.values()))
        val_average = z.mean()
        print('Average on {}: {}'.format(settings['rl_validation_data'],val_average))
        log_value('Validation', val_average, total_steps)
      if settings['rl_test_data']:        
        rc = em.test_envs(settings['rl_test_data'], policy, iters=2)
        z = np.array(list(rc.values()))        
        test_average = z.mean()
        print('Average on {}: {}'.format(settings['rl_test_data'],test_average))
        log_value('Test', test_average, total_steps)
        # print('\n\n\nResults on VSIDS policy:\n\n\n')
        # val_average = test_envs(fnames=settings['rl_validation_data'], model=policy, activity_test=True, iters=1)
        # test_average = test_envs(fnames=settings['rl_test_data'], model=policy, activity_test=True, iters=1)
        # print('\n\n\nResults on optimal policy:\n\n\n')
        # val_average = test_envs(model=policy, testing=True, iters=1)
        # val_average = test_envs(fnames=settings['rl_validation_data'], model=policy, testing=True, iters=1)
        # test_average = test_envs(fnames=settings['rl_test_data'], model=policy, testing=True, iters=1)





  
def test_one_env(fname, iters=None, threshold=100000, **kwargs):
  s = 0.
  i = 0
  step_counts = []
  if iters is None:
    iters = settings['test_iters']
  for _ in range(iters):
    r, _, _ = handle_episode(fname=fname, **kwargs)
    if settings['restart_in_test']:
      env.restart_cadet(timeout=0)
    if not r:     # If r is None, the episodes never finished
      continue
    if len(r) > 1000:
      print('{} took {} steps!'.format(fname,len(r)))
      # break            
    s += len(r)
    i += 1
    step_counts.append(len(r))

  if i:
    if s/i < threshold:
      print('For {}, average/min steps: {}/{}'.format(fname,s/i,min(step_counts)))
    return s/i, float(i)/iters, step_counts
  else:
    return None, 0, None


def test_envs(fnames=settings['rl_train_data'], **kwargs):
  ds = QbfDataset(fnames=fnames)
  print('Testing {} envs..\n'.format(len(ds)))
  all_episode_files = ds.get_files_list()
  totals = 0.
  mins = 0.
  total_srate = 0.
  total_scored = 0
  rc = {}
  for fname in all_episode_files:
    print('Starting {}'.format(fname))
    # average, srate = test_one_env(fname, **kwargs)
    rc[fname] = test_one_env(fname, **kwargs)
    average = rc[fname][0]
    srate = rc[fname][1]    
    total_srate += srate
    if average:
      total_scored += 1
      totals += average        
      mins += min(rc[fname][2])
  if total_scored > 0:
    print("Total average: {}. Success rate: {} out of {}".format(totals/total_scored,total_scored,len(ds)))
    print("Average min: {}.".format(mins/total_scored))
    return totals/total_scored
  else:
    return 0.
