from gentypes import FileName

class FilterBase:
	def filter(self, fname: FileName) -> bool:
		raise NotImplementedError
