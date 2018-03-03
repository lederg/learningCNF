import argparse
import ipdb
import subprocess
import json

from dispatch_utils import *

def_params = {
    'batch_size': 32, 
    'max_iters': 12
}

NAME = 'DEBUG_TEST'
LOCAL_CMD = ['python', 'run_exp.py']
MONGO_MACHINE = 'aws01'
MONGO_SFX = ':27017:'
EXP_CNF_SFX = 'graph_exp'
EXP_QBF_SFX = 'qbf_exp'
EXP_RL_SFX = 'rl_exp'

def main():
	parser = argparse.ArgumentParser(description='Process some params.')
	parser.add_argument('params', metavar='N', type=str, nargs='*',
	                    help='an integer for the accumulator')
	parser.add_argument('--name', type=str, help='Experiment name')
	parser.add_argument('-f', '--file', type=str, help='Settings')	
	parser.add_argument('-c', '--command', type=str, default='run_exp.py', help='Settings')	
	args = parser.parse_args()

	if args.name is None:
		print('Name is NOT optional')
		exit()
	name = machine_name(args.name)
	params = args.params

# override params, cmdargs > json file > def_params > params defined in source code.

	if args.file:
		with open(args.file,'r') as f:
			# load dict
			d = json.load(f)
			for (i,v) in d.items():
				def_params[i]=v



	for k in params:
		a, b = k.split('=')
		def_params[a]=b
	ipdb.set_trace()

	mongo_addr = get_mongo_addr(MONGO_MACHINE)+MONGO_SFX
	if args.command == 'run_exp.py':
		mongo_addr += EXP_CNF_SFX
	elif args.command == 'qbf_exp.py':
		mongo_addr += EXP_QBF_SFX
	elif args.command == 'reinforce_exp.py':
		mongo_addr += EXP_RL_SFX
	a = ['%s=%s' % i for i in def_params.items()]
	a.insert(0, 'with')
	a.insert(0, '--name %s' % str(args.name))
	a.insert(0, '-m %s' % mongo_addr)
	a.insert(0, '-c %s' % args.command)


	print('Provisioning machine %s...' % name)
	provision_machine(name)
	print('Machine %s created, running experiment...' % name)
	execute_machine(name," ".join(a))
	print('Experiment finished, removing machine %s ...' % name)
	remove_machine(name)

if __name__=='__main__':
	main()
