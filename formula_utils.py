from pysat.formula import CNF
from pysat._fileio import FileObject
from settings import *

class FormulaCache(object):
	def __init__(self, settings=None):
		if settings:
			self.settings = settings
		else:
			self.settings = CnfSettings()
		self.preload_cnf = self.settings['preload_cnf']
		self.formulas_dict = {}

	def load_files(self,flist):
		for fname in flist:
			with FileObject(fname, mode='r', compression='use_ext') as fobj:
				formula_str = fobj.fp.read()				
				self.formulas_dict[fname] = CNF(from_string=formula_str) if self.preload_cnf else formula_str
				print('loaded {}'.format(fname))

	def load_formula(self,fname):
		if fname not in self.formulas_dict.keys():
			print('Loading {} in runtime!'.format(fname))
			self.load_files([fname])
		formula = self.formulas_dict[fname]
		rc = formula if self.preload_cnf else CNF(from_string=formula)
		return rc