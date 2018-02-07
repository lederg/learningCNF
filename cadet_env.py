from subprocess import Popen, PIPE, STDOUT

def require_init(f, *args, **kwargs):	
	def inner(instance, *args, **kwargs):
		assert(instance.cadet_proc != None)
		return f(instance,*args,**kwargs)
	return inner
		
class CadetEnv:
	def __init__(self, cadet_binary):
		self.cadet_binary = cadet_binary
		self.cadet_proc = None

	def init_episode(self, fname):
		self.cadet_proc = Popen([self.cadet_binary,  '--rl'], stdout=PIPE, stdin=PIPE, stderr=PIPE, universal_newlines=True)
    self.cadet_proc.stdin.write(fname+'\n')
    return self.read_state_update()     # Initial state

  @require_init
  def reinit_eposide(self, fname):
    self.write_action(0)
    self.cadet_proc.stdin.flush()
    self.cadet_proc.stdin.write(fname+'\n')
    self.cadet_proc.stdin.flush()
    return self.read_state_update()     # Initial state

  @require_init
  def write_action(self, a):
    self.cadet_proc.stdin.write('%d\n' % a)
    self.cadet_proc.stdin.flush()

	@require_init
	def read_state_update(self):
    var_updates_add = []
    var_updates_remove = []
    while True:
        a = self.cadet_proc.stdout.readline()
        if a[0] == 'u':
            update = int(a[3:])
            if a[1] == '+':
                var_updates_add.append(update)
            else:
                var_updates_remove.append(update)
        elif a[0] == 's':
            state = [float(x) for x in a[2:].split(',')]
            break;
    
    return var_updates_add, var_updates_remove, state

  @require_init
  def act(self, action):
  	self.write_action(action)
  	return self.read_state_update()
            
