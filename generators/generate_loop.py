import os
import ipdb
import random
import string
import ray
import logging
import argparse
import pickle
import itertools
import numpy as np

from os import listdir
from pysat.formula import CNF
from pysat._fileio import FileObject

from sharp_wrapped_filter import *

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

def get_filter(config):


@ray.remote
def generate_from_sampler(config):
		for sample in sampler:
			fname = '{}_{}'.format(fname_prefix,random_string(8))
			print('generated {}'.format(fname))
			with open('{}/{}.pickle'.format(dest,fname),'wb') as f:
				pickle.dump(sample,f)
	except Exception as e:
		print('capture threw exception')
		print(e)
		pass
	return True

def 


def generate_dataset(args):
	assert args.n, "Must define target number"
	dst = args.destination_dir
	try:
		os.mkdir(dst)
	except:
		pass
	results = [generate_from_sampler.remote(dst) for _ in range(args.n)]
	vals = ray.get(results)
	print('Finished')

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Reject Sampling for CNF files.')
	# parser.add_argument('params', metavar='N', type=str, nargs='*', help='an integer for the accumulator')
	parser.add_argument('-d', '--destination_dir', type=str, help='destination directory')
	parser.add_argument('-n', type=int, default=0, help='hard cap on number of formulas')
	parser.add_argument('-p', '--parallelism', type=int, default=1, help='number of cores to use (Only if not in cluster mode)')
	parser.add_argument('-c', '--cluster', action='store_true', default=False, help='run in cluster mode')
	args = parser.parse_args()
	if args.cluster:
		ray.init(address='auto', redis_password='blabla')
	else:
		ray.init(num_cpus=args.parallelism)

	generate_dataset(args)
