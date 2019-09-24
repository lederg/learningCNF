import argparse
import ipdb
import time
import json

from config import *                                                             
from qbf_data import *
from rl_model import *
settings = CnfSettings(cfg())
from episode_data import *
from task_cadet import *
from qbf_model import *
from dispatch_utils import *
from policy_factory import *
from get_run import get_experiment_config

HOSTNAME = 'russell.eecs.berkeley.edu:27017'
DBNAME = 'rl_exp'
MONGO_MACHINE = 'russell'


def load_config_from_file(fname):
  rc = {}
  trivial = ['base_mode', 'seed', 'exp_time']
  with open(fname,'r') as f:
    fcfg = json.load(f)
    for k in fcfg.keys():
      if k not in trivial and (k not in settings.hyperparameters.keys() or settings[k] != fcfg[k]):
        rc[k] = fcfg[k]
  return rc



def main():

  parser = argparse.ArgumentParser(description='Process some params.')
  parser.add_argument('params', metavar='N', type=str, nargs='*', help='an integer for the accumulator')
  parser.add_argument('--host', type=str, help='Host address') 
  parser.add_argument('-d', '--db', type=str, default='rl_exp', help='Database name')    
  parser.add_argument('-e', '--experiment', type=str, default='', help='Get settings automatically from experiment')    
  parser.add_argument('-f', '--file', type=str, default='', help='Load settings from config file') 
  parser.add_argument('-o', '--output', type=str, default='', help='Output file name')    
  parser.add_argument('-r', '--random', action='store_true', default=False, help='Random test') 
  parser.add_argument('-v', '--vsids', action='store_true', default=False, help='VSIDS test') 
  parser.add_argument('-p', '--parallelism', type=int, default=1, help='Use # processes') 
  parser.add_argument('-i', '--iterations', type=int, default=10, help='Average # iterations per formula') 
  parser.add_argument('-t', '--seconds', type=int, default=0, help='Number of seconds per formula') 
  parser.add_argument('-s', '--steps', type=int, default=500, help='Maximum # of steps before quitting') 
  args = parser.parse_args()

  assert(len(args.params)>1)
  testdir = args.params[0]
  model_name = args.params[1]

  if args.host:
      hostname = args.host
  else :
      hostname = get_mongo_addr(MONGO_MACHINE)+':27017'
  
  conf = None
  if args.experiment:
    rc = get_experiment_config(args.experiment,hostname,args.db)
    if len(rc.keys()) > 1:
      print('Multiple experiments found, choose one:')
      s = '\n'.join(['{}. {}'.format(x,y) for (x,y) in zip(range(len(rc.keys())),rc.keys())])
      inp = input(s+'\n')
      exp = list(iter(rc.keys()))[int(inp)]
      conf = rc[exp]
    elif len(rc.keys()):
      s = list(rc.values())
      conf = s[0]
    else:
      print('{} not found!')
      return
  elif args.file:
    conf = load_config_from_file(args.file)      
  if conf:
    print('Updating settings with:')
    pprint(conf)
    for (k,v) in conf.items():
      settings.hyperparameters[k]=v

# By now we have all settings except the base model
  # settings.hyperparameters['restart_solver_every'] = 50
  settings.hyperparameters['base_model']=model_name
  settings.hyperparameters['parallelism']=args.parallelism
  settings.hyperparameters['max_step']=args.steps
  settings.hyperparameters['sat_gc_freq']='glucose'
  # settings.hyperparameters['preload_formulas']=False
  # settings.hyperparameters['debug']=True


  # ipdb.set_trace()
  policy = PolicyFactory().create_policy()  
  policy.eval()
  ProviderClass = eval(settings['episode_provider'])
  provider = ProviderClass(testdir)  
  if settings['preload_formulas']:
    settings.formula_cache = FormulaCache()
    settings.formula_cache.load_files(provider.items)  

  em = EpisodeManager(provider, parallelism=args.parallelism,reporter=reporter)
  start_time = time.time()
  kwargs = {'cadet_test': args.vsids, 'random_test': args.random}
  if args.parallelism > 1:
    rc = em.mp_test_envs(fnames=testdir,model=policy, iters=args.iterations, **kwargs)
  else:
    rc = em.test_envs(fnames=testdir,model=policy, iters=args.iterations, max_seconds=args.seconds, **kwargs)
  end_time = time.time()
  print('Entire test took {} seconds'.format(end_time-start_time))  
  z = np.array([x for (x) in rc.values()]).squeeze()
  vals = sorted(z.astype(float).tolist())
  if args.output:
    with open(args.output,'w') as f:
      f.write('number_of_formulas      decisions\n')
      for i,val in enumerate(vals):
        f.write('{}\t{}\n'.format(i,val))

if __name__=='__main__':
    main()
