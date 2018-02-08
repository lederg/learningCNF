from subprocess import Popen, PIPE, STDOUT
import ipdb
from qbf_data import *

def require_init(f, *args, **kwargs): 
  def inner(instance, *args, **kwargs):
    assert(instance.cadet_proc != None)
    return f(instance,*args,**kwargs)
  return inner


# Cadet actions are 1-based. The CadetEnv exposes 0-based actions
    
class CadetEnv:
  def __init__(self, cadet_binary, debug=False, **kwargs):
    self.cadet_binary = cadet_binary
    self.debug = debug
    self.cadet_proc = Popen([self.cadet_binary,  '--rl'], stdout=PIPE, stdin=PIPE, stderr=PIPE, universal_newlines=True)
    self.qbf = QbfBase(**kwargs)
    self.done = True


  def eat_initial_output(self):
    self.cadet_proc.stdout.readline()
    self.cadet_proc.stdout.readline()


  def write(self, val):    
    self.cadet_proc.stdin.write(val)
    self.cadet_proc.stdin.flush()

  def reset(self, fname):
    if not self.done:
      print('interrupting mid-episode!')
      self.write_action(-1)
    self.qbf.reload_qdimacs(fname)    # This holds our own representation of the qbf graph
    self.vars_deterministic = np.zeros(self.qbf.num_vars)
    self.write(fname+'\n')
    self.done = False
    return self.read_state_update()     # Initial state


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
    var_updates_add = []
    var_updates_remove = []
    while True:
      a = self.cadet_proc.stdout.readline()
      if not a: continue
      if a == 'UNSAT\n':
        self.done = True
        return None, None, True
      elif a[0] == 'u':
        update = int(a[3:])-1     # Here we go from 1-based to 0-based
        if a[1] == '+':
          var_updates_add.append(update)
        else:
          var_updates_remove.append(update)
      elif a[0] == 's':
        state = [float(x) for x in a[2:].split(',')]
        break
      else:        
        print('Got unprocessed line: %s' % a[:-1])
      
    if var_updates_add:
      self.vars_deterministic[np.asarray(var_updates_add)] = 1
    if var_updates_remove:
      self.vars_deterministic[np.asarray(var_updates_remove)] = 0
    return state, np.where(self.vars_deterministic==0), self.done

  def step(self, action):
    assert(not self.done)
    self.write_action(action)
    return self.read_state_update()
            
