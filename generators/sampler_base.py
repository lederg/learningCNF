import aiger as A
import aiger_cnf as ACNF

from generators.gentypes import FileName
from generators.cnf_tools import write_to_file

class SamplerBase:
	def __init__(self, config):
		self.dir = config.get('dir', '.')

	def write_expression(self, e, fname):
		c = ACNF.aig2cnf(e)
		maxvar = max([max(x) for x in c.clauses])
		write_to_file(maxvar, c.clauses, fname)
	# This returns a filename!
	def sample(self) -> FileName:
		raise NotImplementedError
