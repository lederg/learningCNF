from gentypes import FileName

class FilterBase:
	def __init__(self, config):
		pass

	def filter(self, fname: FileName) -> bool:
		raise NotImplementedError


class TrueFilter(FilterBase):
	def filter(self, fname: FileName) -> bool:
		return True

class FalseFilter(FilterBase):
	def filter(self, fname: FileName) -> bool:
		return False
