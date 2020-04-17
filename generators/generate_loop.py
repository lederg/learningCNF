import os
import ipdb
import random
import string
import ray
import logging
import argparse
import pickle
import itertools
import shutil
import numpy as np

from os import listdir
from pysat.formula import CNF
from pysat._fileio import FileObject

from sharp_wrapped_filter import *
from word_sampler import *

def random_string(n):
  return ''.join([random.choice(string.ascii_letters + string.digits) for _ in range(n)])

@ray.remote
def generate_datum(fname, dest, step):
	from utils.supervised import capture
	# with FileObject(fname, mode='r', compression='use_ext') as fobj:
	# 	formula_str = fobj.fp.read()
	# 	formula = CNF(from_string=formula_str)

	print('generate_darum called with {}'.format(fname))
	cnf = CNF(from_file=fname)
	try:
		rc = capture(cnf,step)
		print('capture finished')
		with open('{}/{}_step_{}.pickle'.format(dest,os.path.basename(fname),step),'wb') as f:
			pickle.dump(rc,f)
	except Exception as e:
		print('capture threw exception')
		pass
	return True

def get_sampler(config):
	if config['sampler'] == 'word':
		return WordSampler(config)
	else:
		assert False, 'WHHAAT?'

def get_filter(config):
	if config['filter'] == 'sharp':
		return SharpFilter(config)
	elif config['filter'] == 'true':
		return TrueFilter(config)
	elif config['filter'] == 'false':
		return FalseFilter(config)
	else:
		assert False, 'WHHAAT?'

@ray.remote
def generate_from_sampler(config):
	sampler = get_sampler(config)
	fltr = get_filter(config)
	done = False
	while not done:	
		try:
			candidate = sampler.sample()
		except Exception as e:
			print('Gah, Exception:')
			print(e)
			continue
		if fltr.filter(candidate):
			shutil.move(candidate,config['dir']+'/'+os.path.basename(candidate))
			done = True
		else:
			os.remove(candidate)

	return candidate

def generate_dataset(args):
	assert args.n, "Must define target number"
	dst = args.destination_dir
	config = {
		'sampler': args.sampler, 
		'filter': args.filter,
		'dir': dst
	}
	for p in args.params:
		k, v = p.split('=')
		config[k]=v
	try:
		os.mkdir(dst)
	except:
		pass
	results = [generate_from_sampler.remote(config) for _ in range(args.n)]
	vals = ray.get(results)
	print('Finished')

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Reject Sampling for CNF files.')
	parser.add_argument('params', metavar='N', type=str, nargs='*', help='an integer for the accumulator')
	parser.add_argument('-d', '--destination_dir', type=str, default=os.curdir, help='destination directory')
	parser.add_argument('-s', '--sampler', type=str, default='word', help='Sampler (generator)')
	parser.add_argument('-f', '--filter', type=str, default='sharp', help='Filter')
	parser.add_argument('-n', type=int, default=0, help='Number of formulas to generate')
	parser.add_argument('-p', '--parallelism', type=int, default=1, help='number of cores to use (Only if not in cluster mode)')
	parser.add_argument('-c', '--cluster', action='store_true', default=False, help='run in cluster mode')
	args = parser.parse_args()
	if args.cluster:
		ray.init(address='auto', redis_password='blabla')
	else:
		ray.init(num_cpus=args.parallelism)

	generate_dataset(args)
