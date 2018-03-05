import argparse
import ipdb
import subprocess
import json

from dispatch_utils import *

def_params = {
    
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
	parser.add_argument('-f', '--file', type=str, help='Settings file')	
	parser.add_argument('-c', '--command', type=str, default='reinforce_exp.py', help='Command to run (eg: qbf_exp.py)')	
	parser.add_argument('-t', '--instance-type', type=str, help='instance type (eg: t2.xlarge)')	
	parser.add_argument('-m', '--machine', type=str, help='machine name (eg: exp_dqn)')	
	parser.add_argument('--rm', action='store_true', default=False, help='Delete after experiment is done')	
	args = parser.parse_args()

	if args.name is None:
		print('Name is NOT optional')
		exit()
	machine_name = args.machine if args.machine else machine_name(args.name)
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
	a.insert(0, '%s' % args.command)


	if not machine_exists(machine_name):
		print('Provisioning machine %s...' % machine_name)
		provision_machine(machine_name,args.instance_type)
	print('Running experiment %s on machine %s...' % (args.command,machine_name))
	execute_machine(machine_name," ".join(a))
	print('Experiment finished')
	if args.rm:
		print('Removing machine %s ...' % machine_name)
		remove_machine(machine_name)

if __name__=='__main__':
	main()
