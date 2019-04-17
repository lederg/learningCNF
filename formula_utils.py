from pysat.formula import CNF
from pysat._fileio import FileObject

class FormulaCache(object):
	def __init__(self):
		self.formulas_dict = {}

	def load_files(self,flist):
		for fname in flist:
			with FileObject(fname, mode='r', compression='use_ext') as fobj:
				self.formulas_dict[fname] = fobj.fp.read()
				print('loaded {}'.format(fname))

	def load_formula(self,fname):
		if fname not in self.formulas_dict.keys():
			print('Loading {} in runtime!'.format(fname))
			self.load_files([fname])
		return CNF(from_string=self.formulas_dict[fname])