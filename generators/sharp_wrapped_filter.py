from filter_base import *
from sharpsat_filter import *
from pysat.solvers import SharpSAT

class SharpFilter(FilterBase):
  def __init__(self, config):
  	time_max = int(config.get('time_max', 2))
  	steps_max = int(config.get('steps_max', 1000))
  	self._filter = SharpSATFilter(time_max=time_max, steps_max=steps_max)
    
  def filter(self, fname: FileName) -> bool:
  	# Check for degenerate
  	with open(fname, 'r') as f:
  		z = f.readline()
  		if z.startswith('p cnf 0 1'):
		  	print('{} is degenerate'.format(fname))
  			return False
  	sharpSAT = SharpSAT(time_budget = self._filter.time_max, use_timer= True)
  	count = sharpSAT.solve(fname)
  	rc = not self._filter.filter(sharpSAT)
  	print('{}: {}'.format(fname,rc))
  	return rc

