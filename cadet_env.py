from subprocess import Popen, PIPE, STDOUT
import select
import ipdb
import time
from settings import *
from qbf_data import *

DEF_GREEDY_ALPHA = 0.01
MAX_EPISODE_LENGTH = 200
DQN_DEF_COST = -0.2
# DQN_DEF_COST = 0.
BINARY_SUCCESS = 1.
def require_init(f, *args, **kwargs): 
  def inner(instance, *args, **kwargs):
    assert(instance.cadet_proc != None)
    return f(instance,*args,**kwargs)
  return inner


# Cadet actions are 1-based. The CadetEnv exposes 0-based actions
    
class CadetEnv:
  def __init__(self, cadet_binary, debug=False, greedy_rewards=False, **kwargs):
    self.cadet_binary = cadet_binary
    self.debug = debug
    self.cadet_proc = Popen([self.cadet_binary,  '--rl', '--cegar'], stdout=PIPE, stdin=PIPE, stderr=STDOUT, universal_newlines=True)
    self.poll_obj = select.poll()
    self.poll_obj.register(self.cadet_proc.stdout, select.POLLIN)  
    self.qbf = QbfBase(**kwargs)
    self.done = True      
    self.greedy_rewards = greedy_rewards
    self.greedy_alpha = DEF_GREEDY_ALPHA if self.greedy_rewards else 0.
    


  def eat_initial_output(self):
    self.read_line_with_timeout()
    self.read_line_with_timeout()


  def write(self, val):    
    self.cadet_proc.stdin.write(val)
    self.cadet_proc.stdin.flush()


  def terminate(self):
    if not self.done:
      if self.debug:
        print('interrupting mid-episode!')
      self.write_action(-1)
      a = self.read_line_with_timeout()
      self.done = True
      rewards = np.asarray(list(map(float,a.split()[1:])))
      return rewards

  def reset(self, fname):    
    self.terminate()
    self.qbf.reload_qdimacs(fname)    # This holds our own representation of the qbf graph
    self.vars_deterministic = np.zeros(self.qbf.num_vars)
    self.total_vars_deterministic = np.zeros(self.qbf.num_vars)    
    self.activities = np.zeros(self.qbf.num_vars)
    self.max_rewards = self.qbf.num_existential
    self.timestep = 0
    self.running_reward = []

    self.write(fname+'\n')
    self.done = False
    return self.read_state_update()     # Initial state


  def read_line_with_timeout(self, timeout=10.):
    return self.cadet_proc.stdout.readline()
    entry = time.time()
    curr = entry
    line = ''
    while curr < entry + timeout:
      p = self.poll_obj.poll(0) 
      if p:
        c = self.cadet_proc.stdout.read(1)
        if c == '\n':
          # ipdb.set_trace()
          return line
        line += c
      else:
        # print('Poll negative..')
        pass
      curr = time.time()

    return None

  # This is where we go from 0-based to 1-based
  def write_action(self, a):
    self.write('%d\n' % (a+1))


  '''
  Returns:

    state - A bunch of numbers describing general solver state
    candidates - an array of (0-based) available actions in the current state
    done - Is this the end of the episode?

  '''
  def read_state_update(self):
    self.vars_deterministic.fill(0)
    self.activities.fill(0)
    clause = None
    reward = None
    decision = None
    while True:
      pos_vars = np.where(self.vars_deterministic>0)[0]
      neg_vars = np.where(self.vars_deterministic<0)[0]      
      a = self.read_line_with_timeout()
      if not a or a == '\n': continue
      if self.debug:
        print(a)
      if a == 'UNSAT\n':
        a = self.read_line_with_timeout()     # refutation line
        a = self.read_line_with_timeout()     # rewards
        self.rewards = np.asarray(list(map(float,a.split()[1:])))
        if np.isnan(self.rewards).any():
          if np.isnan(self.rewards[:-1]).any():
            ipdb.set_trace()
          else:
            self.rewards[-1]=BINARY_SUCCESS
        self.done = True
        state = None
        break
      elif a == 'SAT\n':
        a = self.read_line_with_timeout()     # rewards
        self.rewards = np.asarray(list(map(float,a.split()[1:])))                
        if np.isnan(self.rewards).any():
          if np.isnan(self.rewards[:-1]).any():
            ipdb.set_trace()
          else:
            self.rewards[-1]=BINARY_SUCCESS
        self.done = True
        state = None
        break

      elif a[0] == 'u':
        update = int(a[3:])-1     # Here we go from 1-based to 0-based
        if a[1] == '+':
          self.vars_deterministic[update] = 1
          self.total_vars_deterministic[update] = 1
        else:
          self.vars_deterministic[update] = -1
          self.total_vars_deterministic[update] = 0
      elif a[0] == 'd':
        decision = [int(x) for x in a[2:].split(',')]
        decision[0] -= 1
      elif a[0] == 's':
        state = np.array([float(x) for x in a[2:].split(',')])
        break
      elif a[0] == 'a':
        b = a[2:].split(',')
        update = int(b[0])-1
        activity = float(b[1])
        self.activities[update] = activity
      elif a[0] == 'c':
        b = [int(x) for x in a[2:].split()]
        self.qbf.add_clause(b)
        clause = (np.array([abs(x)-1 for x in b if x > 0]), np.array([abs(x)-1 for x in b if x < 0]))
      elif self.debug:
        print('Got unprocessed line: %s' % a)
        if a.startswith('Error'):
          return

    if self.timestep > 0:      
      greedy_reward = np.count_nonzero(self.total_vars_deterministic) - self.last_total_determinized
      self.running_reward.append(greedy_reward)
      reward = BINARY_SUCCESS if self.done else DQN_DEF_COST 
      if self.greedy_rewards:
        reward += self.greedy_alpha*self.running_reward[-1]
    self.last_total_determinized = np.count_nonzero(self.total_vars_deterministic)
    if sum(self.running_reward) < 0:
      ipdb.set_trace()
    if self.done:
      self.rewards = self.rewards + self.greedy_alpha*np.asarray(self.running_reward)

    # on-line rewards, for Q-learning

    return state, pos_vars, neg_vars, self.activities, decision, clause, reward, self.done

  def step(self, action):
    assert(not self.done)
    self.timestep += 1
    if self.greedy_rewards and self.timestep > MAX_EPISODE_LENGTH:      
      rewards = self.terminate() + self.greedy_alpha*np.asarray(self.running_reward)
      self.rewards = np.concatenate([rewards, [DQN_DEF_COST]])    # Average action
      return None, None, None, None, None, None, DQN_DEF_COST, True
    self.write_action(action)
    return self.read_state_update()
            
