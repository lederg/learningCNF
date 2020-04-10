from filter_base import *
from sharpsat_filter import *
from pysat.solvers import SharpSAT

class SharpFilter(FilterBase):
  def __init__(self, config):
    self._filter = SharpSATFilter()
    
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

