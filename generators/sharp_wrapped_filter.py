from filter_base import *
from sharpsat_filter import *
from pysat.solvers import SharpSAT

class SharpFilter(FilterBase):
  def __init__(self, **kwargs):
    self._filter = SharpSATFilter(**kwargs)
    
  def filter(self, fname):
    sharpSAT = SharpSAT(time_budget = self._filter.time_max, use_timer= True)
    count = sharpSAT.solve(fname)
    return not self._filter.filter(sharpSAT)

