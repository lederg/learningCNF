import argparse
import ipdb
import subprocess

from dispatch_utils import *

def_params = {
    'batch_size': 128, 
    'max_iters': 8
}

NAME = 'DEBUG_TEST'
LOCAL_CMD = ['python', 'run_exp.py']
MONGO_MACHINE = 'aws01'
MONGO_SFX = ':27017:graph_exp'

def main():
	parser = argparse.ArgumentParser(description='Process some params.')
	parser.add_argument('params', metavar='N', type=str, nargs='+',
	                    help='an integer for the accumulator')
	parser.add_argument('--name', type=str, help='Experiment name')	
	args = parser.parse_args()

	if args.name is None:
		print('Name is NOT optional')
		exit()
	name = machine_name(args.name)
	params = args.params

# override default params

	for k in params:
		a, b = k.split('=')
		def_params[a]=b

	mongo_addr = get_mongo_addr(MONGO_MACHINE, MONGO_SFX)
	a = ['%s=%s' % i for i in def_params.items()]
	a.insert(0, 'with')
	a.insert(0, '--name %s' % str(args.name))
	a.insert(0, '-m %s' % mongo_addr)
	" ".join(a)
	ipdb.set_trace()

if __name__=='__main__':
	main()
