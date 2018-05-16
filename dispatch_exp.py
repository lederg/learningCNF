import argparse
import functools
import ipdb
import subprocess
import json
import asyncio
import signal

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

all_machines = []

loop = asyncio.get_event_loop()

def cleanup_handler(signame):
	print('Got ctrl-c, cleaning up synchronously')
	loop.run_until_complete(loop.shutdown_asyncgens())
	loop.stop()
	loop.close()
	for mname in all_machines:
		remove_machine(mname)

	exit()

def main():
	parser = argparse.ArgumentParser(description='Process some params.')
	parser.add_argument('params', metavar='N', type=str, nargs='*',
	                    help='an integer for the accumulator')
	parser.add_argument('--name', type=str, help='Experiment name')
	parser.add_argument('-f', '--file', type=str, help='Settings file')	
	parser.add_argument('-c', '--command', type=str, default='reinforce_exp.py', help='Command to run (eg: qbf_exp.py)')	
	parser.add_argument('-t', '--instance-type', type=str, help='instance type (eg: t2.xlarge)')	
	parser.add_argument('-m', '--machine', type=str, help='machine name (eg: exp_dqn)')	
	parser.add_argument('--commit', type=str, default='rl', help='commit to load')	
	parser.add_argument('-n', '--num', type=int, default=1, help='Number of concurrent experiments')	
	parser.add_argument('--rm', action='store_true', default=False, help='Delete after experiment is done')	
	args = parser.parse_args()

	if args.name is None:
		print('Name is NOT optional')
		exit()
	base_mname = args.machine if args.machine else machine_name(args.name)
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
	else: 
		mongo_addr += 'unknown'

	all_params = ['%s=%s' % i for i in def_params.items()]	
	all_executions = []

	for i in range(args.num):
		a = all_params.copy()
		if all_params:
			a.insert(0, 'with')
		a.insert(0, '--name %s-%d' % (str(args.name),i))
		a.insert(0, '-m %s' % mongo_addr)
		a.insert(0, '%s' % args.command)

		mname = base_mname+'-{}'.format(i)
		p = async_dispatch_chain(mname,a, args.instance_type, args.rm, args.commit)
		all_executions.append(p)

	for signame in ('SIGINT', 'SIGTERM'):
		loop.add_signal_handler(getattr(signal, signame), functools.partial(cleanup_handler, signame))
	loop.run_until_complete(asyncio.gather(*all_executions))
	loop.close()
	# else:
	# 	execute_machine(mname," ".join(a))
	print('Experiment finished')
	# if args.rm:
	# 	print('Removing machine %s ...' % mname)
	# 	remove_machine(mname)


async def async_dispatch_chain(mname, params, instance_type, rm, commit_name):
	all_machines.append(mname)
	if not machine_exists(mname):
		print('Provisioning machine %s...' % mname)
		rc = await async_provision_machine(mname,instance_type,commit_name)
	else:
		print('Machine already exists, hmm...')
	print('Running experiment %s on machine %s...' % (params[0],mname))
	p = await async_execute_machine(mname," ".join(params))
	print('Experiment {} finished!'.format(mname))
	if rm:
		print('Removing machine %s ...' % mname)
		await async_remove_machine(mname)
		print('Removed machine %s ...' % mname)

if __name__=='__main__':
	main()
