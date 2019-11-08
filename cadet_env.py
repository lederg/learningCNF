from subprocess import Popen, PIPE, STDOUT
from collections import deque
from collections import namedtuple
import select
from IPython.core.debugger import Tracer
import time
from settings import *
from qbf_data import *
from cadet_utils import *

DEF_GREEDY_ALPHA = 0.01
MAX_EPISODE_LENGTH = 200
DQN_DEF_COST = -0.2
# DQN_DEF_COST = 0.
BINARY_SUCCESS = 1.
LOG_SIZE = 100
def require_init(f, *args, **kwargs): 
  def inner(instance, *args, **kwargs):
    assert(instance.cadet_proc != None)
    return f(instance,*args,**kwargs)
  return inner


# Cadet actions are 1-based. The CadetEnv exposes 0-based actions
    
class CadetEnv:
  def __init__(self, cadet_binary='./cadet', debug=False, greedy_rewards=False, slim_state=False,
                use_old_rewards = False, fresh_seed = False, clause_learning=True, vars_set=True, 
                use_vsids_rewards = False, def_step_cost = -1e-4, cadet_completion_reward=1., logger=None, settings=None, **kwargs):
    self.EnvObservation = namedtuple('EnvObservation', 
                    ['state', 'vars_add', 'vars_remove', 'activities', 'decision', 'clause', 
                      'reward', 'vars_set', 'done'])
    
    self.settings = settings if settings else CnfSettings()
    self.cadet_binary = cadet_binary
    self.debug = debug
    self.qbf = QbfBase(**kwargs)
    self.aag_qcnf = CombinedGraph1Base(**kwargs)
    self.greedy_rewards = greedy_rewards
    self.clause_learning = clause_learning
    self.vars_set = vars_set
    self.fresh_seed = fresh_seed
    self.use_old_rewards = use_old_rewards
    self.use_vsids_rewards = use_vsids_rewards
    self.slim_state = slim_state
    self.def_step_cost = def_step_cost
    self.cadet_completion_reward = cadet_completion_reward
    self.logger = logger
    self.greedy_alpha = DEF_GREEDY_ALPHA if self.greedy_rewards else 0.    
    self.tail = deque([],LOG_SIZE)
    self.use_activities = self.settings['cadet_use_activities']
    self.start_cadet()
    

  def start_cadet(self):
    cadet_params = ['--rl', '--cegar', '--sat_by_qbf', '--rl_reward_per_decision', '{}'.format(self.def_step_cost), 
                      '--rl_completion_reward', '{}'.format(self.cadet_completion_reward)]
    if not self.use_old_rewards:
      if self.debug:
        print('Using new rewards!')
      cadet_params.append('--rl_advanced_rewards')
    if self.fresh_seed:
      if self.debug:
        print('Using fresh seed!')
      cadet_params.append('--fresh_seed')
    if self.slim_state:
      if self.debug:
        print('Using slim_state!')
      cadet_params.append('--rl_slim_state')
    if self.use_vsids_rewards:
      if self.debug:
        print('Using vsids rewards!!')
      cadet_params.append('--rl_vsids_rewards')

    if self.debug:
      print(' '.join([self.cadet_binary,  *cadet_params]))
    self.cadet_proc = Popen([self.cadet_binary,  *cadet_params], stdout=PIPE, stdin=PIPE, stderr=STDOUT, universal_newlines=True)
    self.poll_obj = select.poll()
    self.poll_obj.register(self.cadet_proc.stdout, select.POLLIN)  
    self.finished = False
    self.done = True      
    self.current_fname = None

  def stop_cadet(self, timeout):
    assert(self.cadet_proc != None)
    self.cadet_proc.terminate()
    time.sleep(timeout)
    if self.cadet_proc.poll() != None:
      self.cadet_proc.kill()
    self.cadet_proc = None
    self.poll_obj = None

  def restart_env(self, timeout=5):
    if self.debug:
      print('Stopping cadet...')
    self.stop_cadet(timeout)
    if self.debug:
      print('Restarting cadet...')
    self.start_cadet()

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
      if self.debug:
        print(a)
      self.done = True
      rewards = np.asarray(list(map(float,a.split()[1:])))
      return rewards

  def reset(self, fname): 
    self.terminate()
    if self.debug:
      print('Starting Env {}'.format(fname))
    # if fname == 'data/huge_gen1/small-bug1-fixpoint-3.qdimacs':
    #   Tracer()()
    
    ############################################
    if type(fname) == str:
#        import ipdb
#        ipdb.set_trace()
        fname = (None, fname)
    if type(fname) == tuple and len(fname) == 2:
        self.aag_qcnf.load_paired_files(aag_fname = fname[0], qcnf_fname = fname[1])
        self.qbf.reload_qdimacs(fname[1])
        fname = fname[1]
    else:
        raise Exception 
    ############################################
    
    self.vars_deterministic = np.zeros(self.qbf.num_vars)
    self.total_vars_deterministic = np.zeros(self.qbf.num_vars)    
    self.activities = np.zeros(self.qbf.num_vars)
    self.max_rewards = self.qbf.num_existential
    self.timestep = 0
    self.finished = False
    self.running_reward = []
    self.rewards = None

    self.write(fname+'\n')
    self.done = False
    self.current_fname = fname
    rc = self.read_state_update()     # Initial state
    return rc

  def read_line_with_timeout(self, timeout=10.):
    return self.cadet_proc.stdout.readline()
    # entry = time.time()
    # curr = entry
    # line = ''
    # while curr < entry + timeout:
    #   p = self.poll_obj.poll(0) 
    #   if p:
    #     c = self.cadet_proc.stdout.read(1)
    #     if c == '\n':
    #       # Tracer()()
    #       return line
    #     line += c
    #   else:
    #     # print('Poll negative..')
    #     pass
    #   curr = time.time()

    # return None

  # This is where we go from 0-based to 1-based
  def write_action(self, a):
    if self.debug:
      print('Writing action {}'.format(a))
    if a in ['?','r']:
      self.write('{}\n'.format(a))
      return
    if type(a) is tuple:
      cadet_action = int(a[0]) + 1
      if a[1]:
        cadet_action = -cadet_action
    else:
      cadet_action = int(a)+1
    self.write('%d\n' % cadet_action)


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
    vars_set = []
    while True:
      a = self.read_line_with_timeout()
      if not a or a == '\n': continue
      self.tail.append(a)
      if self.debug:
        print(a)
      if False and a == 'UNSAT\n':
        if self.cadet_binary != './cadet':
          a = self.read_line_with_timeout()     # refutation line
        a = self.read_line_with_timeout()     # rewards
        self.rewards = np.asarray(list(map(float,a.split()[1:])))
        if np.isnan(self.rewards).any():
          if np.isnan(self.rewards[:-1]).any():
            Tracer()()
          else:
            self.rewards[-1]=BINARY_SUCCESS
        self.done = True
        state = None
        break
      elif False and a == 'SAT\n':
        Tracer()()
        a = self.read_line_with_timeout()     # rewards
        self.rewards = np.asarray(list(map(float,a.split()[1:])))                
        if np.isnan(self.rewards).any():
          if np.isnan(self.rewards[:-1]).any():
            Tracer()()
          else:
            self.rewards[-1]=BINARY_SUCCESS
        self.done = True
        state = None
        break
      elif a.startswith('rewards') or a.startswith('SATrewards') or a.startswith('UNSATrewards'):
        self.rewards = np.asarray(list(map(float,a.split()[1:])))
        if self.debug:
          Tracer()()
        if np.isnan(self.rewards).any():
          if np.isnan(self.rewards[:-1]).any():
            Tracer()()
          else:
            self.rewards[-1]=BINARY_SUCCESS
        self.done = True
        self.finished = True
        state = None
        if self.debug:
          print('Successfuly finished episode in {} steps!'.format(self.timestep))
        break
      elif a[0] == 'u' and a[1] != 'c':   # New cadet has 'uc'
        update = int(a[3:])-1     # Here we go from 1-based to 0-based
        if a[1] == '+':
          self.vars_deterministic[update] = 1
          self.total_vars_deterministic[update] = 1
        elif a[1] == '-':
          self.vars_deterministic[update] = -1
          self.total_vars_deterministic[update] = 0
      elif a.startswith('delete_clause'):
        if not self.clause_learning:
          continue
        cid = int(a.split()[1])
        # print('Removing clause id {}'.format(cid))
        # print(a)
        self.qbf.remove_clause(cid)
        clause = True
      elif a[0] == 'd':
        decision = [int(x) for x in a[2:].split(',')]
        decision[0] -= 1
      elif a[0] == 's':
        state = np.array([float(x) for x in a[2:].split(',')])
        if self.debug:
          print('Got state!')
        break
      elif a[0] == 'a':
        b = a[2:].split(',')
        update = int(b[0])-1
        activity = float(b[1])
        self.activities[update] = activity
      elif a[0] == 'v' and self.vars_set:
        v, pol = a.split(' ')[1:]        
        vars_set.append((int(v)-1,int(pol)))
      elif self.timestep > 0 and a[0] == 'c':
        if a.startswith('conflict'):
          continue
        elif a.startswith('clause'):      # new cadet version
          if not self.clause_learning:
            continue
          c = a.split()
          cid = int(c[1])
          b = [int(x) for x in c[4:]]
          self.qbf.add_clause(b,cid)
        else:
          print('This version is too old')
          Tracer()()
          b = [int(x) for x in a[2:].split()]        
          
        # clause = (np.array([abs(x)-1 for x in b if x > 0]), np.array([abs(x)-1 for x in b if x < 0]))
        clause = True
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
    # if self.timestep > 200:
    #   self.debug = True
    if sum(self.running_reward) < -1:
      Tracer()()
    if self.done:
      self.rewards = self.rewards + self.greedy_alpha*np.asarray(self.running_reward)

    pos_vars = np.where(self.vars_deterministic>0)[0]
    neg_vars = np.where(self.vars_deterministic<0)[0]      

    return self.EnvObservation(state, pos_vars, neg_vars, self.activities, decision, clause, reward, np.array(vars_set), self.done)

  # This gets a tuple from stepping the environment:
  # state, vars_add, vars_remove, activities, decision, clause, reward, done = env.step(action)
  # And it returns the next observation.

  def process_observation(self, last_obs, env_obs, settings=None):

    if not env_obs:
      return None
      
    if env_obs.clause or not last_obs:
      # If learned/derived/removed a clause, or this is the first observation, then UPDATE THE GRAPH.
      cmat = get_input_from_qbf(self.qbf, self.settings, False) # Do not split
      clabels = Variable(torch.from_numpy(self.qbf.get_clabels()).float().unsqueeze(0)).t()
      
#      if self.cnf_and_aag:
      G = last_obs.ext_data if last_obs else self.aag_qcnf.G      
      try:
          lit_embs = G.nodes['literal'].data['lit_embs']
      except:
          lit_embs = []
      try:
          clause_embs = G.nodes['clause'].data['clause_embs']
      except:
          clause_embs = []

      extra_clauses = self.qbf.extra_clauses.copy()
      for k in extra_clauses:
          extra_clauses[k] = [convert_qdimacs_lit(l) for l in extra_clauses[k]]
        
      # update the combined graph
      self.aag_qcnf.create_DGL_graph(
              qcnf_base = self.qbf,
              aag_forward_edges = self.aag_qcnf.aag_forward_edges, 
              aag_backward_edges = self.aag_qcnf.aag_backward_edges,
              qcnf_forward_edges = self.aag_qcnf.qcnf_forward_edges, 
              qcnf_backward_edges = self.aag_qcnf.qcnf_backward_edges,
              extra_clauses = extra_clauses, # UPDATE: learned/derived clauses
              removed_old_clauses = self.qbf.removed_old_clauses, # UPDATE: removed old clauses
              lit_embs = lit_embs, 
              clause_embs = clause_embs
      )
    else: # no changed clauses AND last_obs, graph is the same
      cmat, clabels = last_obs.cmat, last_obs.clabels
      
      # no action for combined graph
    ### Get the combined graph G and the literal embeddings
    if last_obs:
      ground_embs = np.copy(last_obs.ground.data.numpy().squeeze())
      vmask = last_obs.vmask
      cmask = last_obs.cmask
      
      G = self.aag_qcnf.G if env_obs.clause else last_obs.ext_data
      lit_embs = G.nodes['literal'].data['lit_embs']
    else:      
      ground_embs = self.qbf.get_base_embeddings()
      vmask = None
      cmask = None
      
      G = self.aag_qcnf.G
      lit_embs = G.nodes['literal'].data['lit_embs']

    if env_obs.decision:
      ground_embs[env_obs.decision[0]][IDX_VAR_POLARITY_POS+1-env_obs.decision[1]] = True
      
      ## FIXME: signed lit ??
#      lit = vars_to_lits([env_obs.decision[0]],env_obs.decision[1])[0]
#      lit_embs[lit][IDX_VAR_POLARITY_POS+1-env_obs.decision[1]] = 1 
    if len(env_obs.vars_add):
      ground_embs[:,IDX_VAR_DETERMINIZED][env_obs.vars_add] = True
      
#      lit_embs[:,IDX_VAR_DETERMINIZED][vars_to_lits(env_obs.vars_add)] = 1
    if len(env_obs.vars_remove):
      ground_embs[:,IDX_VAR_DETERMINIZED][env_obs.vars_remove] = False
      ground_embs[:,IDX_VAR_POLARITY_POS:IDX_VAR_POLARITY_NEG][env_obs.vars_remove] = False
      
#      lit_embs[:,IDX_VAR_DETERMINIZED][vars_to_lits(env_obs.vars_remove)] = 0
#      lit_embs[:,IDX_VAR_POLARITY_POS:IDX_VAR_POLARITY_NEG][vars_to_lits(env_obs.vars_remove)] = 0
    if self.use_activities:
      ground_embs[:,IDX_VAR_ACTIVITY] = env_obs.activities
      
#      lit_embs[:,IDX_VAR_ACTIVITY] = Vars01_to_Lits01(env_obs.activities)
    if len(env_obs.vars_set):
      x = env_obs.vars_set
      pos_idx = x[:,0][np.where(x[:,1]==1)[0]]
      ground_embs[:,IDX_VAR_SET_POS][pos_idx] = True
      neg_idx = x[:,0][np.where(x[:,1]==-1)[0]]
      ground_embs[:,IDX_VAR_SET_NEG][neg_idx] = True
      not_yet_idx = x[:,0][np.where(x[:,1]==0)[0]]
      ground_embs[:,IDX_VAR_SET_POS:IDX_VAR_SET_NEG][not_yet_idx] = False 
      
#      lit_embs[:,IDX_VAR_SET_POS][vars_to_lits(pos_idx,1)] = 1
#      lit_embs[:,IDX_VAR_SET_NEG][vars_to_lits(neg_idx,-1)] = 1
#      lit_embs[:,IDX_VAR_SET_POS:IDX_VAR_SET_NEG][vars_to_lits(not_yet_idx)] = 0
      
#    import ipdb
#    ipdb.set_trace()

    state = Variable(torch.from_numpy(env_obs.state).float().unsqueeze(0))
    ground_embs = Variable(torch.from_numpy(ground_embs).float().unsqueeze(0))
    return State(state, cmat, ground_embs, clabels, vmask, cmask, G)
    


# This returns already a State (higher-level) observation, not EnvObs.

  def new_episode(self, fname, settings=None, **kwargs):
    try:
      obs = self.reset(fname)
      # state, vars_add, vars_remove, activities, _, _ , _, vars_set, _ = self.reset(fname)
      if obs.state is not None:   # Env solved in 0 steps
        return obs
    except Exception as e:
      print('Error reseting with file {}'.format(fname))
      print(e)
#      import traceback
#      print("***", traceback.format_exc())

  def step(self, action):
    assert(not self.done)
    # if self.greedy_rewards and self.timestep > MAX_EPISODE_LENGTH:      
    #   rewards = self.terminate() + self.greedy_alpha*np.asarray(self.running_reward)
    #   self.rewards = np.concatenate([rewards, [DQN_DEF_COST]])    # Average action
    #   return None, None, None, None, None, None, DQN_DEF_COST, None, True    
    self.timestep += 1
    self.write_action(action)
    return self.read_state_update()
            
