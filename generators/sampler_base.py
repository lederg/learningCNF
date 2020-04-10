from gentypes import FileName

class SamplerBase:
	def __init__(self, config):
		pass

	# This returns a filename!
	def sample(self) -> FileName:
		raise NotImplementedError
